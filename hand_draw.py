#!/usr/bin/env python3
"""
Air-Draw — hand drawing in the air, with shape snapping and a float-away finish.

Rewrite of hand_draw.py focused on three things: smoothness, accuracy, and
running well on laptop iGPUs as well as dGPUs.

WHAT ACTUALLY MAKES IT FASTER
-----------------------------
Be sceptical of anything that claims to "run MediaPipe on your GPU" from Python.
The pip wheels ship a CPU (XNNPACK) delegate on Windows and on most Linux
builds; there is no usable GPU delegate for the hand model there. So the wins
here are real ones:

  1. Inference runs in its own thread on a DOWNSCALED copy of the frame
     (default 320 px wide). Landmarks are normalised, so they map back to full
     resolution for free. This is the single biggest saving - roughly 4x less
     work than feeding it a 720p frame.
  2. model_complexity=0 (the "lite" hand model) - about 2x faster than the
     default, and for a single fingertip you cannot see the difference.
  3. Capture runs in its own thread with a 1-frame buffer, so a slow USB
     camera can never stall the render loop.
  4. The render loop is DECOUPLED from inference. Drawing stays at camera
     framerate even when the model is only managing 25 fps, and a One Euro
     filter plus a small velocity extrapolation covers the gap.
  5. Strokes are drawn INCREMENTALLY (one new segment per frame) onto a
     persistent layer, instead of re-stroking the whole polyline every frame.
  6. Compositing is dirty-rect only, and uses a `max` (screen) blend - integer
     SIMD, no float math, no per-frame full-frame allocations.
  7. The glow is a small blurred ROI around each new segment, cached into the
     layer. No full-frame blur, ever.
  8. MediaPipe's own draw_landmarks() is replaced with 6 batched polylines.
  9. Animation is TIME-based, not frame-based, so the float-away looks
     identical whether you are getting 20 fps on an iGPU or 90 on a dGPU.

LAPTOP iGPU vs dGPU, HONESTLY
-----------------------------
There is no GPU delegate for MediaPipe's hand model in the pip wheels on
Windows or most Linux builds, so the model runs on the CPU whichever GPU you
have - a discrete card does not speed it up. And the image work here is already
well under a millisecond a frame (run --bench and see), so pushing it onto the
iGPU through OpenCV's OpenCL path would cost more in transfers than it saves.
A flag promising otherwise would be a placebo, so there isn't one.

What genuinely differs between these machines is thermal and power headroom,
so that is what to tune:

  thin / fanless iGPU     --infer-width 256 --max-fps 30 --infer-fps 20
  ordinary laptop         (the defaults)
  plugged in, want crisp  --infer-width 384 --complexity 1 --max-fps 0

--infer-fps decouples tracking rate from render rate: at 20 the drawing still
renders at 30-60 fps and only the model idles, which is where the watts go.

CONTROLS
--------
  Point (index up, middle down) ....... pen down
  Two fingers up ...................... move cursor, no ink
  Open palm (hold ~0.5s) .............. clear canvas
  ESC / q ... quit          c ... clear canvas    z ... undo last stroke
  [ / ] ..... colour        - / = ... brush size  s ... save a PNG
  h ......... toggle HUD    l ... hand skeleton   g ... toggle glow

Finish a recognisable shape and it snaps to a perfect one, flashes, then
drifts upward and fades out.

An unrecognised stroke is kept as freehand ink, so nothing you draw is lost.

Headless checks (no webcam needed):
  python hand_draw.py --selftest       recogniser accuracy on synthetic strokes
  python hand_draw.py --demo out.mp4   render the float-away to video
  python hand_draw.py --demo out.png   ...or to a contact sheet
  python hand_draw.py --bench          time the render pipeline on this machine
"""

from __future__ import annotations

import argparse
import math
import os
import platform
import sys
import threading
import time
from dataclasses import dataclass
from typing import List, Optional, Tuple

# Quiet the absl/glog spam MediaPipe emits on import. Must precede the import.
os.environ.setdefault("GLOG_minloglevel", "2")
os.environ.setdefault("TF_CPP_MIN_LOG_LEVEL", "3")
os.environ.setdefault("ABSL_MIN_LOG_LEVEL", "2")

import cv2
import numpy as np

# --------------------------------------------------------------------------
# Palette (BGR)
# --------------------------------------------------------------------------
PALETTE: List[Tuple[int, int, int]] = [
    (255, 226, 64),    # neon cyan
    (150, 90, 255),    # hot pink
    (120, 255, 150),   # acid green
    (60, 190, 255),    # amber
    (255, 120, 140),   # violet
    (245, 245, 255),   # white
]

# MediaPipe hand landmark indices
WRIST = 0
FINGER_MCP = [5, 9, 13, 17]
FINGER_PIP = [6, 10, 14, 18]
FINGER_TIP = [8, 12, 16, 20]
THUMB_TIP, THUMB_MCP = 4, 2
SKELETON = [
    [0, 1, 2, 3, 4],
    [0, 5, 6, 7, 8],
    [9, 10, 11, 12],
    [13, 14, 15, 16],
    [0, 17, 18, 19, 20],
    [5, 9, 13, 17],
]


# --------------------------------------------------------------------------
# One Euro filter  (Casiez, Roussel & Vogel 2012)
# --------------------------------------------------------------------------
def _alpha(cutoff: float, dt: float) -> float:
    tau = 1.0 / (2.0 * math.pi * cutoff)
    return 1.0 / (1.0 + tau / dt)


class OneEuro:
    """Adaptive low-pass: heavy smoothing when still, light when moving fast.

    A plain EMA has to choose between jitter and lag. This one varies its
    cutoff with the estimated speed, so a resting fingertip stops shaking
    without the pen feeling like it is dragging behind your hand.
    """

    __slots__ = ("mincutoff", "beta", "dcutoff", "_x", "_dx", "_t")

    def __init__(self, mincutoff: float = 1.3, beta: float = 0.012, dcutoff: float = 1.0):
        self.mincutoff = mincutoff
        self.beta = beta
        self.dcutoff = dcutoff
        self.reset()

    def reset(self) -> None:
        self._x = None
        self._dx = 0.0
        self._t = None

    @property
    def velocity(self) -> float:
        """Filtered speed in px/s - reused for lag compensation and taper."""
        return self._dx

    def __call__(self, x: float, t: float) -> float:
        if self._t is None:
            self._t, self._x, self._dx = t, x, 0.0
            return x
        dt = t - self._t
        if dt <= 0.0:
            return self._x
        dt = min(max(dt, 1.0 / 240.0), 0.2)
        self._t = t
        a_d = _alpha(self.dcutoff, dt)
        self._dx += a_d * ((x - self._x) / dt - self._dx)
        cutoff = self.mincutoff + self.beta * abs(self._dx)
        a = _alpha(cutoff, dt)
        self._x += a * (x - self._x)
        return self._x


class OneEuroPoint:
    __slots__ = ("fx", "fy")

    def __init__(self, mincutoff: float = 1.3, beta: float = 0.012):
        self.fx = OneEuro(mincutoff, beta)
        self.fy = OneEuro(mincutoff, beta)

    def reset(self) -> None:
        self.fx.reset()
        self.fy.reset()

    def __call__(self, x: float, y: float, t: float) -> Tuple[float, float]:
        return self.fx(x, t), self.fy(y, t)

    @property
    def velocity(self) -> Tuple[float, float]:
        return self.fx.velocity, self.fy.velocity


# --------------------------------------------------------------------------
# Threaded capture - latest frame wins, never queue
# --------------------------------------------------------------------------
class CameraStream:
    """Grabs frames on a daemon thread and keeps only the newest one.

    Queuing frames is what makes hand tracking feel laggy: the driver buffers
    a few, you consume them in order, and you end up drawing where your hand
    was a third of a second ago. Dropping stale frames is always the right
    call for interactive work.
    """

    def __init__(self, src=0, width=1280, height=720, fps=60, mirror=True):
        backend = 0
        system = platform.system()
        if system == "Windows":
            backend = cv2.CAP_DSHOW      # much lower open/latency than MSMF
        elif system == "Linux":
            backend = cv2.CAP_V4L2
        self.cap = cv2.VideoCapture(src, backend) if backend else cv2.VideoCapture(src)
        if not self.cap.isOpened() and backend:
            self.cap = cv2.VideoCapture(src)   # fall back to whatever works

        # MJPG matters: many UVC webcams cap at ~10 fps for 720p raw YUY2 but
        # will happily deliver 30-60 fps of MJPG.
        try:
            self.cap.set(cv2.CAP_PROP_FOURCC, cv2.VideoWriter_fourcc(*"MJPG"))
        except Exception:
            pass
        self.cap.set(cv2.CAP_PROP_FRAME_WIDTH, width)
        self.cap.set(cv2.CAP_PROP_FRAME_HEIGHT, height)
        self.cap.set(cv2.CAP_PROP_FPS, fps)
        try:
            self.cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)
        except Exception:
            pass

        self.mirror = mirror
        self._lock = threading.Lock()
        self._frame = None
        self._seq = 0
        self._t = 0.0
        self._stop = threading.Event()
        self._thread = threading.Thread(target=self._loop, daemon=True)

    def start(self) -> "CameraStream":
        if not self.cap.isOpened():
            raise RuntimeError("Could not open the camera. Is another app using it?")
        self._thread.start()
        # Block briefly for the first frame so the caller can size its buffers.
        deadline = time.time() + 5.0
        while time.time() < deadline:
            if self.read() is not None:
                return self
            time.sleep(0.01)
        raise RuntimeError("Camera opened but delivered no frames.")

    def _loop(self) -> None:
        while not self._stop.is_set():
            ok, frame = self.cap.read()
            if not ok or frame is None:
                time.sleep(0.005)
                continue
            if self.mirror:
                # Mirror once, here, so display and inference share one
                # coordinate system. Saves a flip and a whole class of bugs.
                frame = cv2.flip(frame, 1)
            now = time.perf_counter()
            with self._lock:
                self._frame = frame
                self._seq += 1
                self._t = now

    def read(self):
        """-> (frame, seq, capture_time) or None. Frame is a fresh array."""
        with self._lock:
            if self._frame is None:
                return None
            return self._frame, self._seq, self._t

    def release(self) -> None:
        self._stop.set()
        if self._thread.is_alive():
            self._thread.join(timeout=1.0)
        self.cap.release()


# --------------------------------------------------------------------------
# Threaded hand tracker
# --------------------------------------------------------------------------
class HandTracker:
    """Runs MediaPipe Hands off the render thread, on a downscaled frame."""

    def __init__(self, infer_width=320, complexity=0, det_conf=0.6, trk_conf=0.5,
                 threaded=True):
        import mediapipe as mp  # imported late so --selftest works without it

        self._mp = mp
        self.hands = mp.solutions.hands.Hands(
            static_image_mode=False,
            max_num_hands=1,
            model_complexity=complexity,
            min_detection_confidence=det_conf,
            min_tracking_confidence=trk_conf,
        )
        self.infer_width = infer_width
        self.threaded = threaded
        self._lock = threading.Lock()
        self._result = None           # (landmarks (21,3) float32, t, seq)
        self._seq = 0
        self._pending = None
        self._new = threading.Event()
        self._stop = threading.Event()
        self.infer_ms = 0.0
        self._thread = None
        if threaded:
            self._thread = threading.Thread(target=self._loop, daemon=True)
            self._thread.start()

    # -- internals ---------------------------------------------------------
    def _infer(self, frame, t) -> None:
        h, w = frame.shape[:2]
        if w > self.infer_width:
            scale = self.infer_width / float(w)
            small = cv2.resize(frame, (self.infer_width, max(1, int(round(h * scale)))),
                               interpolation=cv2.INTER_AREA)
        else:
            small = frame
        rgb = cv2.cvtColor(small, cv2.COLOR_BGR2RGB)
        rgb.flags.writeable = False          # lets MediaPipe skip a copy
        t0 = time.perf_counter()
        res = self.hands.process(rgb)
        dt = (time.perf_counter() - t0) * 1000.0

        lm = None
        if res.multi_hand_landmarks:
            hand = res.multi_hand_landmarks[0]
            lm = np.empty((21, 3), np.float32)
            for i, p in enumerate(hand.landmark):
                lm[i, 0] = p.x
                lm[i, 1] = p.y
                lm[i, 2] = p.z
        with self._lock:
            self._seq += 1
            self._result = (lm, t, self._seq)
            self.infer_ms += 0.15 * (dt - self.infer_ms)

    def _loop(self) -> None:
        while not self._stop.is_set():
            if not self._new.wait(timeout=0.1):
                continue
            self._new.clear()
            with self._lock:
                job = self._pending
                self._pending = None
            if job is None:
                continue
            try:
                self._infer(job[0], job[1])
            except Exception:      # never let a bad frame kill the thread
                pass

    # -- public ------------------------------------------------------------
    def submit(self, frame, t) -> None:
        if not self.threaded:
            self._infer(frame, t)
            return
        with self._lock:
            self._pending = (frame, t)     # newest job replaces any stale one
        self._new.set()

    def latest(self):
        with self._lock:
            return self._result

    def close(self) -> None:
        self._stop.set()
        self._new.set()
        if self._thread is not None:
            self._thread.join(timeout=1.0)
        self.hands.close()


# --------------------------------------------------------------------------
# Gesture recognition
# --------------------------------------------------------------------------
IDLE, DRAW, MOVE, CLEAR = "idle", "draw", "move", "clear"


class HandPose:
    """Scale- and rotation-invariant finger state with per-finger hysteresis.

    The original `landmark[tip].y < landmark[pip].y` test only works when the
    hand is upright; tilt your wrist and it falls apart. Measuring the joint
    angle at the PIP instead is invariant to how the hand is rotated in the
    image plane, and normalising by palm size makes it invariant to distance
    from the camera.
    """

    ENTER = -0.45      # cos(angle) below this  -> finger counts as extended
    EXIT = -0.12       # ...and stays extended until it rises above this

    def __init__(self):
        self.extended = np.zeros(5, bool)     # thumb, index, middle, ring, pinky
        self.palm = 1.0

    def update(self, lm_px: np.ndarray) -> np.ndarray:
        mcp = lm_px[FINGER_MCP]
        pip = lm_px[FINGER_PIP]
        tip = lm_px[FINGER_TIP]
        v1 = mcp - pip
        v2 = tip - pip
        denom = (np.linalg.norm(v1, axis=1) * np.linalg.norm(v2, axis=1)) + 1e-6
        cos = (v1 * v2).sum(1) / denom      # straight finger -> ~ -1

        prev = self.extended[1:]
        self.extended[1:] = np.where(prev, cos < self.EXIT, cos < self.ENTER)

        # Thumb has a different joint layout; distance to the middle-finger
        # knuckle separates tucked from splayed far more reliably than angle.
        d_tip = np.linalg.norm(lm_px[THUMB_TIP] - lm_px[9])
        d_mcp = np.linalg.norm(lm_px[THUMB_MCP] - lm_px[9])
        self.extended[0] = d_tip > d_mcp * (0.98 if self.extended[0] else 1.12)

        self.palm = float(np.linalg.norm(lm_px[WRIST] - lm_px[9])) or 1.0
        return self.extended

    def gesture(self) -> str:
        thumb, index, middle, ring, pinky = self.extended
        if index and not middle and not ring and not pinky:
            return DRAW
        if index and middle and not ring and not pinky:
            return MOVE
        if index and middle and ring and pinky and thumb:
            return CLEAR
        return IDLE


class GestureFSM:
    """Debounces gestures so a single bad frame cannot break a stroke.

    Asymmetric on purpose: pen-down is quick (2 agreeing samples) because
    latency there is felt directly, pen-up is slower (4) because a dropped
    frame mid-stroke should not chop your circle in half.
    """

    def __init__(self, down_frames=2, up_frames=4, clear_hold=0.5):
        self.state = IDLE
        self._cand = IDLE
        self._count = 0
        self.down_frames = down_frames
        self.up_frames = up_frames
        self.clear_hold = clear_hold
        self._clear_since = None

    def update(self, raw: str, now: float) -> Tuple[str, bool]:
        if raw == self._cand:
            self._count += 1
        else:
            self._cand, self._count = raw, 1

        need = self.down_frames if self._cand == DRAW else self.up_frames
        if self._count >= need and self._cand != self.state:
            self.state = self._cand

        # Clearing wipes the canvas, so it needs to be deliberate: hold it.
        fired = False
        if self.state == CLEAR:
            if self._clear_since is None:
                self._clear_since = now
            elif now - self._clear_since >= self.clear_hold:
                fired = True
                self._clear_since = now + 1e9      # once per gesture
        else:
            self._clear_since = None
        return self.state, fired

    def clear_progress(self, now: float) -> float:
        if self._clear_since is None or self._clear_since > now:
            return 0.0
        return min(1.0, (now - self._clear_since) / self.clear_hold)


# --------------------------------------------------------------------------
# Polyline helpers
# --------------------------------------------------------------------------
def _dedup(pts: np.ndarray, eps: float = 1.0) -> np.ndarray:
    keep = [0]
    last = pts[0]
    for i in range(1, len(pts)):
        if abs(pts[i, 0] - last[0]) + abs(pts[i, 1] - last[1]) > eps:
            keep.append(i)
            last = pts[i]
    return pts[keep]


def _resample(pts: np.ndarray, n: int):
    seg = np.linalg.norm(np.diff(pts, axis=0), axis=1)
    cum = np.concatenate(([0.0], np.cumsum(seg)))
    total = float(cum[-1])
    if total <= 1e-6:
        return None, 0.0
    target = np.linspace(0.0, total, n)
    out = np.empty((n, 2), np.float32)
    out[:, 0] = np.interp(target, cum, pts[:, 0])
    out[:, 1] = np.interp(target, cum, pts[:, 1])
    return out, total


def _smooth(pts: np.ndarray, k: int, closed: bool) -> np.ndarray:
    if k < 3:
        return pts
    ker = np.ones(k, np.float32) / k
    mode = "wrap" if closed else "edge"
    pad = k // 2
    out = np.empty_like(pts)
    for c in (0, 1):
        padded = np.pad(pts[:, c], pad, mode=mode)
        out[:, c] = np.convolve(padded, ker, mode="valid")[: len(pts)]
    return out


def _circ_smooth(v: np.ndarray, k: int) -> np.ndarray:
    ker = np.ones(k, np.float32) / k
    pad = k // 2
    return np.convolve(np.pad(v, pad, mode="wrap"), ker, mode="valid")[: len(v)]


def _circ_extrema(r: np.ndarray, min_gap: int, prominence: float = 0.15):
    """Peaks and valleys of a closed radius profile, pruned by prominence."""
    n = len(r)
    rs = _circ_smooth(r, max(3, n // 24 * 2 + 1))
    lo, hi = float(rs.min()), float(rs.max())
    span = max(hi - lo, 1e-6)
    mid = 0.5 * (lo + hi)

    peaks, valleys = [], []
    for i in range(n):
        a, b, c = rs[(i - 1) % n], rs[i], rs[(i + 1) % n]
        if b >= a and b > c and (b - mid) / span > prominence:
            peaks.append(i)
        if b <= a and b < c and (mid - b) / span > prominence:
            valleys.append(i)

    def merge(idx, prefer_max):
        if not idx:
            return []
        out = []
        group = [idx[0]]
        for i in idx[1:]:
            if i - group[-1] <= min_gap:
                group.append(i)
            else:
                out.append(max(group, key=lambda j: rs[j]) if prefer_max
                           else min(group, key=lambda j: rs[j]))
                group = [i]
        out.append(max(group, key=lambda j: rs[j]) if prefer_max
                   else min(group, key=lambda j: rs[j]))
        # wrap-around merge
        if len(out) > 1 and (out[0] + n - out[-1]) <= min_gap:
            a, b = out[0], out.pop()
            out[0] = (a if (rs[a] > rs[b]) == prefer_max else b)
        return out

    return merge(peaks, True), merge(valleys, False)


def _regular_polygon(center, radius, n, phase) -> np.ndarray:
    th = phase + np.arange(n, dtype=np.float32) * (2.0 * math.pi / n)
    return np.stack([center[0] + radius * np.cos(th),
                     center[1] + radius * np.sin(th)], 1).astype(np.float32)


def _ellipse_pts(center, a, b, angle_rad, n=96) -> np.ndarray:
    th = np.linspace(0.0, 2.0 * math.pi, n, endpoint=False, dtype=np.float32)
    ca, sa = math.cos(angle_rad), math.sin(angle_rad)
    x = a * np.cos(th)
    y = b * np.sin(th)
    return np.stack([center[0] + x * ca - y * sa,
                     center[1] + x * sa + y * ca], 1).astype(np.float32)


# --------------------------------------------------------------------------
# Shape recognition
# --------------------------------------------------------------------------
ALL_SHAPES_DEFAULT = ("line", "triangle", "square", "rectangle", "circle",
                      "ellipse", "pentagon", "hexagon", "star", "arrow")


@dataclass
class Shape:
    kind: str
    pts: np.ndarray          # (M,2) float32, absolute pixel coords
    closed: bool
    score: float


N_RS = 128          # resample resolution used through classification
TURN_K = 7          # half-window, in resampled points, for the turning angle


def _turning(P: np.ndarray, k: int) -> np.ndarray:
    """Turn angle at every point of a closed polyline.

    This is the discriminator that matters. A circle spreads its 360 degrees
    evenly, so every point turns by the same small amount. A polygon does
    almost nothing along its edges and dumps the whole turn at its corners.
    Counting sharp local maxima therefore separates "pentagon" from "circle"
    far more reliably than approxPolyDP vertex counting, which flip-flops with
    epsilon and happily reports 5 vertices for a wobbly circle.
    """
    a = P - np.roll(P, k, axis=0)
    b = np.roll(P, -k, axis=0) - P
    cross = a[:, 0] * b[:, 1] - a[:, 1] * b[:, 0]
    dot = (a * b).sum(1)
    return np.abs(np.arctan2(cross, dot))


def _find_corners(turn: np.ndarray, min_gap: int, rel: float = 0.42,
                  floor: float = 0.30) -> List[int]:
    n = len(turn)
    thr = max(floor, rel * float(turn.max()))
    cand = [i for i in range(n)
            if turn[i] >= thr and turn[i] >= turn[(i - 1) % n]
            and turn[i] > turn[(i + 1) % n]]
    if not cand:
        return []
    out, group = [], [cand[0]]
    for i in cand[1:]:
        if i - group[-1] <= min_gap:
            group.append(i)
        else:
            out.append(max(group, key=lambda j: turn[j]))
            group = [i]
    out.append(max(group, key=lambda j: turn[j]))
    if len(out) > 1 and (out[0] + n - out[-1]) <= min_gap:
        a, b = out[0], out.pop()
        out[0] = a if turn[a] >= turn[b] else b
        out.sort()
    return out


def _dtheta(P: np.ndarray) -> np.ndarray:
    """Signed direction change at each vertex; sums to +-2*pi round a loop."""
    d = np.roll(P, -1, axis=0) - P
    ang = np.arctan2(d[:, 1], d[:, 0])
    dt = np.diff(np.concatenate([ang, ang[:1]]))
    return (dt + math.pi) % (2.0 * math.pi) - math.pi


EDGE_MARGIN = 3          # points skipped either side of a corner
MAX_EDGE_TURN = 0.80     # radians an edge may net before we try splitting it
SPLIT_MIN_REL = 0.55     # a split point must be this sharp, relative to the
                         # corners already found, before it counts as one
MAX_EDGE_SHARE = 0.34    # fraction of the full 2*pi allowed to happen off-corner


def _edge_spans(corners: List[int], n: int, margin: int = -1):
    """Index list for each edge, with the rounded corners trimmed off."""
    if margin < 0:
        margin = EDGE_MARGIN
    m = len(corners)
    out = []
    for i in range(m):
        a, b = corners[i], corners[(i + 1) % m]
        span = (b - a) % n
        out.append([] if span <= 2 * margin + 1
                   else [(a + j) % n for j in range(margin, span - margin)])
    return out


def _edge_share(dt: np.ndarray, corners: List[int]) -> float:
    """How much of the loop's turning happens along edges rather than at corners.

    A real polygon edge nets out near zero however much the hand shook, because
    wobble turns one way and then back. An arc of a circle nets the whole arc
    angle. Summing over all edges rather than capping each one keeps this
    robust when a corner is localised a few points off and leaks a little turn
    into its neighbours: a circle carved into five "edges" scores ~1.0 here,
    any real polygon scores well under 0.3.
    """
    total = 0.0
    for idx in _edge_spans(corners, len(dt)):
        if idx:
            total += abs(float(dt[idx].sum()))
    return total / (2.0 * math.pi)


def _refine_corners(turn: np.ndarray, dt: np.ndarray, corners: List[int],
                    limit: int = 7) -> List[int]:
    """Insert a corner where an "edge" hides a real bend - but only a real one.

    This rescues a hexagon whose sixth corner was too soft to trip the peak
    detector. The sharpness condition is what stops it running away: a wobbly
    edge also nets a lot of turning, but spread out, with no spike, so it is
    left alone. A circle does keep getting split, until the count runs past the
    limit and the shape correctly falls through to the ellipse test.
    """
    n = len(dt)
    for _ in range(4):
        if len(corners) < 3:
            return corners
        base = float(np.mean(turn[corners]))
        adds = []
        for idx in _edge_spans(corners, n):
            if len(idx) < 3 or abs(float(dt[idx].sum())) <= MAX_EDGE_TURN:
                continue
            j = idx[int(np.argmax(turn[idx]))]
            if turn[j] >= SPLIT_MIN_REL * base:
                adds.append(j)
        if not adds:
            return corners
        corners = sorted(set(corners) | set(adds))
        if len(corners) > limit:
            return corners
    return corners


def _line_fit(seg: np.ndarray):
    c = seg.mean(0)
    q = seg - c
    cxx = float((q[:, 0] * q[:, 0]).sum())
    cyy = float((q[:, 1] * q[:, 1]).sum())
    cxy = float((q[:, 0] * q[:, 1]).sum())
    th = 0.5 * math.atan2(2.0 * cxy, cxx - cyy)
    return c, np.array([math.cos(th), math.sin(th)], np.float32)


def _intersect(l1, l2):
    (c1, u1), (c2, u2) = l1, l2
    den = u1[0] * (-u2[1]) - u1[1] * (-u2[0])
    if abs(den) < 1e-6:
        return None
    d = c2 - c1
    t = (d[0] * (-u2[1]) - d[1] * (-u2[0])) / den
    return c1 + u1 * t


def _edge_fit(S: np.ndarray, corners: List[int]):
    """Fit a line to every edge between consecutive corners.

    Returns (lines, straightness_error). The error is the worst
    deviation-to-length ratio over all edges, and it is the test that finally
    separates a hexagon from a circle: chop a circle into six arcs and each
    "edge" bows away from its chord by ~13% of the chord, while a real
    hexagon's edges are straight to within the hand's own wobble.
    """
    n, m = len(S), len(corners)
    if m < 3:
        return None, 1.0
    lines, worst = [], 0.0
    for i in range(m):
        a, b = corners[i], corners[(i + 1) % m]
        span = (b - a) % n
        if span < 6:
            return None, 1.0
        seg = S[[(a + j) % n for j in range(1, span)]]
        trim = max(1, len(seg) // 5)                # drop the rounded corners
        core = seg[trim:len(seg) - trim] if len(seg) - 2 * trim >= 4 else seg
        c, u = _line_fit(core)
        nrm = np.array([-u[1], u[0]], np.float32)
        dev = float(np.abs((core - c) @ nrm).max())
        L = float(np.linalg.norm(core[-1] - core[0]))
        if L > 12.0:
            worst = max(worst, dev / L)
        lines.append((c, u))
    return lines, worst


def _corner_verts(S: np.ndarray, lines):
    """Recover sharp vertices by intersecting consecutive fitted edges.

    Taking the contour point at each corner under-shoots, because a hand-drawn
    corner is rounded off. Intersecting the two edges puts the vertex back
    where the person actually meant it.
    """
    m = len(lines)
    verts = []
    for i in range(m):
        p = _intersect(lines[i - 1], lines[i])
        if p is None:
            return None
        verts.append(p)
    V = np.array(verts, np.float32)
    lo, hi = S.min(0), S.max(0)
    pad = 0.35 * (hi - lo) + 8.0
    if (V < lo - pad).any() or (V > hi + pad).any():
        return None
    return V


LINE_MIN_CHORD = 0.82     # path length may exceed the chord by this much
LINE_MAX_DEV = 0.10       # max sideways wander, as a fraction of the chord


def _try_line(P, path_len, closure):
    d = P[-1] - P[0]
    chord = float(np.linalg.norm(d))
    if chord < 1e-3 or closure < 0.55:
        return None
    # A hand-drawn line wanders, but its path length stays close to the
    # straight-line distance between its endpoints.
    if chord < LINE_MIN_CHORD * path_len:
        return None
    u = d / chord
    nrm = np.array([-u[1], u[0]], np.float32)
    dev = float(np.abs((P - P[0]) @ nrm).max())
    ratio = dev / chord
    if ratio > LINE_MAX_DEV:
        return None
    proj = (P - P[0]) @ u
    a = P[0] + u * float(proj.min())
    b = P[0] + u * float(proj.max())
    ang = math.degrees(math.atan2(b[1] - a[1], b[0] - a[0]))
    for target in (-180.0, -90.0, 0.0, 90.0, 180.0):
        if abs(ang - target) <= 7.0:               # they meant it to be straight
            mid = (a + b) * 0.5
            half = float(np.linalg.norm(b - a)) * 0.5
            tr = math.radians(target)
            u2 = np.array([math.cos(tr), math.sin(tr)], np.float32)
            a, b = mid - u2 * half, mid + u2 * half
            break
    return Shape("line", np.stack([a, b]).astype(np.float32), False,
                 float(1.0 - ratio / LINE_MAX_DEV))


STAR_PROMINENCE = 0.10


def _try_star(S, r, cen, solidity):
    # Solidity is the giveaway: a five-pointed star fills barely half its own
    # convex hull, while every other shape here fills 90%+ of it.
    if solidity > 0.80:
        return None
    peaks, valleys = _circ_extrema(r, min_gap=max(3, len(r) // 14),
                                   prominence=STAR_PROMINENCE)
    if not (4 <= len(peaks) <= 7 and 4 <= len(valleys) <= 7):
        return None
    r_out = float(r[peaks].mean())
    r_in = float(r[valleys].mean())
    if not (0.20 <= r_in / max(r_out, 1e-6) <= 0.72):
        return None
    ang = np.arctan2(S[peaks, 1] - cen[1], S[peaks, 0] - cen[0])
    k = len(peaks)
    base = ang - np.arange(k) * (2 * math.pi / k)
    phase = math.atan2(float(np.sin(base).mean()), float(np.cos(base).mean()))
    outer = _regular_polygon(cen, r_out, 5, phase)
    inner = _regular_polygon(cen, r_out * 0.382, 5, phase + math.pi / 5)
    pts = np.empty((10, 2), np.float32)
    pts[0::2] = outer
    pts[1::2] = inner
    return Shape("star", pts, True, 0.85)


def _try_ellipse(S, cnt, circ):
    if len(S) < 5 or circ < 0.70:
        return None
    try:
        (ex, ey), (MA, ma), deg = cv2.fitEllipse(cnt)
    except cv2.error:
        return None
    a, b = MA * 0.5, ma * 0.5
    if a < 4.0 or b < 4.0:
        return None
    rad = math.radians(deg)
    ca, sa = math.cos(rad), math.sin(rad)
    q = S - np.array([ex, ey], np.float32)
    # In the ellipse's own frame a good fit has rho == 1 at every point.
    u = (q[:, 0] * ca + q[:, 1] * sa) / a
    v = (-q[:, 0] * sa + q[:, 1] * ca) / b
    resid = float(np.sqrt(u * u + v * v).std())
    if resid > 0.085:
        return None
    center = np.array([ex, ey], np.float32)
    score = float(1.0 - resid / 0.085)
    if min(a, b) / max(a, b) > 0.86:               # round enough - call it a circle
        rr = 0.5 * (a + b)
        return Shape("circle", _ellipse_pts(center, rr, rr, 0.0), True, score)
    return Shape("ellipse", _ellipse_pts(center, a, b, rad), True, score)


def _fit_triangle(verts):
    c = verts.mean(0)
    sides = np.linalg.norm(np.roll(verts, -1, 0) - verts, axis=1)
    if sides.max() / max(float(sides.min()), 1e-6) < 1.18:
        rad = float(np.linalg.norm(verts - c, axis=1).mean())
        ang = np.arctan2(verts[:, 1] - c[1], verts[:, 0] - c[0])
        base = ang - np.arange(3) * (2 * math.pi / 3)
        phase = math.atan2(float(np.sin(base).mean()), float(np.cos(base).mean()))
        return _regular_polygon(c, rad, 3, phase)
    return verts.astype(np.float32)


def _fit_quad(verts):
    (cx, cy), (w, h), deg = cv2.minAreaRect(verts.astype(np.float32))
    if w < 4.0 or h < 4.0:
        return None, None
    snapped = round(deg / 90.0) * 90.0
    if abs(deg - snapped) <= 12.0:
        deg = snapped
    kind = "rectangle"
    if abs(w - h) / max(w, h) < 0.18:
        w = h = 0.5 * (w + h)
        kind = "square"
    return cv2.boxPoints(((cx, cy), (w, h), deg)).astype(np.float32), kind


def _fit_regular(verts, n):
    c = verts.mean(0)
    rad = float(np.linalg.norm(verts - c, axis=1).mean())
    ang = np.arctan2(verts[:, 1] - c[1], verts[:, 0] - c[0])
    base = ang - np.arange(n) * (2 * math.pi / n)
    phase = math.atan2(float(np.sin(base).mean()), float(np.cos(base).mean()))
    return _regular_polygon(c, rad, n, phase)


def _try_arrow(P, diag):
    """One-stroke arrow: shaft out, barb, back over the tip, barb.

    The give-away is a vertex the path visits twice - the arrowhead tip.
    """
    cnt = P.reshape(-1, 1, 2)
    perim = cv2.arcLength(cnt, False)
    best = None
    for eps in (0.030, 0.040, 0.055):
        ap = cv2.approxPolyDP(cnt, eps * perim, False).reshape(-1, 2)
        if len(ap) in (4, 5):
            best = ap.astype(np.float32)
            break
    if best is None:
        return None
    tail = best[0]
    seg = np.linalg.norm(np.diff(best, axis=0), axis=1)
    if seg[0] < 0.45 * seg.sum():
        return None
    tip_idx = None
    for i in range(1, len(best)):
        if any(np.linalg.norm(best[i] - best[j]) < 0.13 * diag
               for j in range(i + 2, len(best))):
            tip_idx = i
            break
    if tip_idx is None:
        return None
    tip = best[tip_idx]
    shaft = tip - tail
    L = float(np.linalg.norm(shaft))
    if L < 0.5 * diag:
        return None
    u = shaft / L
    nrm = np.array([-u[1], u[0]], np.float32)
    head = min(0.30 * L, 0.34 * diag)
    pts = np.stack([tail, tip, tip - u * head + nrm * head * 0.5, tip,
                    tip - u * head - nrm * head * 0.5]).astype(np.float32)
    return Shape("arrow", pts, False, 0.7)


EDGE_STRAIGHT_MAX = 0.16      # worst deviation/length a real edge may have
CONTOUR_SMOOTH = 5            # box-filter width applied before corner finding


def _classify_closed(P, allow):
    S = _smooth(P, CONTOUR_SMOOTH, True)
    cnt = S.reshape(-1, 1, 2)
    area = abs(float(cv2.contourArea(cnt)))
    perim = float(cv2.arcLength(cnt, True))
    if area < 400.0 or perim < 1.0:
        return None
    circ = 4.0 * math.pi * area / (perim * perim)
    cen = S.mean(0)
    r = np.linalg.norm(S - cen, axis=1)
    solidity = area / max(abs(float(cv2.contourArea(cv2.convexHull(cnt)))), 1e-6)

    if "star" in allow:
        s = _try_star(S, r, cen, solidity)
        if s is not None:
            return s

    turn = _turning(S, TURN_K)
    dt = _dtheta(S)
    c0 = _find_corners(turn, min_gap=max(3, len(S) // 16))
    c1 = _refine_corners(turn, dt, c0)
    # Refined set first, raw set second. Refinement recovers a missed corner;
    # falling back recovers an edge that got split for no good reason.
    for corners in ([c1, c0] if c1 != c0 else [c0]):
        if not (3 <= len(corners) <= 6):
            continue
        if _edge_share(dt, corners) >= MAX_EDGE_SHARE:
            continue
        lines_, err = _edge_fit(S, corners)
        if lines_ is None or err >= EDGE_STRAIGHT_MAX:
            continue
        verts = _corner_verts(S, lines_)
        if verts is None:
            continue
        n = len(verts)
        score = float(1.0 - err / EDGE_STRAIGHT_MAX)
        if n == 3 and "triangle" in allow:
            return Shape("triangle", _fit_triangle(verts), True, score)
        if n == 4:
            p, kind = _fit_quad(verts)
            if p is not None and kind in allow:
                return Shape(kind, p, True, score)
        if n in (5, 6):
            kind = "pentagon" if n == 5 else "hexagon"
            if kind in allow:
                return Shape(kind, _fit_regular(verts, n), True, score)

    s = _try_ellipse(S, cnt, circ)
    if s is not None and s.kind in allow:
        return s
    if circ > 0.74 and "circle" in allow:
        rr = float(r.mean())
        return Shape("circle", _ellipse_pts(cen, rr, rr, 0.0), True, 0.6)
    return None


def recognise(raw: np.ndarray, min_len: float = 80.0, min_diag: float = 45.0,
              allow=ALL_SHAPES_DEFAULT) -> Optional[Shape]:
    """Classify a finished stroke and return a perfected version of it."""
    pts = np.asarray(raw, np.float32)
    if len(pts) < 8:
        return None
    pts = _dedup(pts, 1.0)
    if len(pts) < 8:
        return None
    P, path_len = _resample(pts, N_RS)
    if P is None or path_len < min_len:
        return None
    span = P.max(0) - P.min(0)
    diag = float(math.hypot(span[0], span[1]))
    if diag < min_diag:
        return None
    closure = float(np.linalg.norm(P[0] - P[-1]) / diag)

    if "line" in allow:
        s = _try_line(P, path_len, closure)
        if s is not None:
            return s
    if closure < 0.40:
        return _classify_closed(P, allow)
    if "arrow" in allow:
        return _try_arrow(P, diag)
    return None


# --------------------------------------------------------------------------
# Layers and compositing
# --------------------------------------------------------------------------
def _union(a, b):
    if a is None:
        return b
    if b is None:
        return a
    return (min(a[0], b[0]), min(a[1], b[1]), max(a[2], b[2]), max(a[3], b[3]))


def _clip_rect(rect, w, h):
    if rect is None:
        return None
    x0 = max(0, int(rect[0])); y0 = max(0, int(rect[1]))
    x1 = min(w, int(rect[2])); y1 = min(h, int(rect[3]))
    if x1 <= x0 or y1 <= y0:
        return None
    return (x0, y0, x1, y1)


def _pts_rect(pts, pad, w, h):
    lo = pts.min(0) - pad
    hi = pts.max(0) + pad
    return _clip_rect((lo[0], lo[1], hi[0] + 1, hi[1] + 1), w, h)


# Measured, not assumed: on a 1280x720 frame np.maximum over a strided ROI
# runs ~0.10 ms while cv2.max over the same ROI runs ~3.8 ms, because the
# OpenCV binding copies non-contiguous input. So: numpy for ROIs.
def _max_into(dst_layer, src_layer, rect):
    x0, y0, x1, y1 = rect
    d = dst_layer[y0:y1, x0:x1]
    np.maximum(d, src_layer[y0:y1, x0:x1], out=d)


class Canvas:
    """Persistent ink layer + transient FX layer, both dirty-rect tracked."""

    MAX_JUMP_FRAC = 0.35        # segments longer than this are tracking glitches

    def __init__(self, w: int, h: int, glow: bool = True):
        self.w, self.h = w, h
        self.ink = np.zeros((h, w, 3), np.uint8)
        self.fx = np.zeros((h, w, 3), np.uint8)
        self.ink_rect = None
        self.fx_rect = None
        self.glow = glow
        self.strokes: List[Tuple[np.ndarray, tuple, int]] = []

    # -- ink ---------------------------------------------------------------
    GLOW_GAIN = 2.7

    def _glow_into(self, draw_fn, rect, thickness: int) -> None:
        """Blur a small ROI around what was just drawn and max it into the ink.

        Blurring the whole layer every frame would cost milliseconds; blurring
        a 60x60 patch costs microseconds, and because the blend is `max` the
        overlapping patches of a slow stroke saturate instead of blowing out.
        """
        if not self.glow or rect is None:
            return
        x0, y0, x1, y1 = rect
        if (x1 - x0) * (y1 - y0) > 400_000:      # guard against silly ROIs
            return
        tmp = np.zeros((y1 - y0, x1 - x0, 3), np.uint8)
        draw_fn(tmp, x0, y0)
        k = int(min(31, max(5, thickness * 2 + 9))) | 1
        cv2.GaussianBlur(tmp, (k, k), 0, dst=tmp)
        tmp = cv2.convertScaleAbs(tmp, alpha=self.GLOW_GAIN)
        dst = self.ink[y0:y1, x0:x1]
        np.maximum(dst, tmp, out=dst)

    def add_segment(self, p0, p1, color, thickness: int) -> bool:
        """Draw one new segment. Returns False if it looked like a glitch."""
        dx, dy = p1[0] - p0[0], p1[1] - p0[1]
        if math.hypot(dx, dy) > self.MAX_JUMP_FRAC * self.w:
            return False
        a = (int(round(p0[0])), int(round(p0[1])))
        b = (int(round(p1[0])), int(round(p1[1])))
        pad = thickness * 3 + 12
        rect = _clip_rect((min(a[0], b[0]) - pad, min(a[1], b[1]) - pad,
                           max(a[0], b[0]) + pad, max(a[1], b[1]) + pad),
                          self.w, self.h)
        if rect is None:
            return True
        self._glow_into(
            lambda t, ox, oy: cv2.line(t, (a[0] - ox, a[1] - oy),
                                       (b[0] - ox, b[1] - oy), color,
                                       thickness + 3, cv2.LINE_AA),
            rect, thickness)
        cv2.line(self.ink, a, b, color, thickness, cv2.LINE_AA)
        cv2.circle(self.ink, b, max(1, thickness // 2), color, -1, cv2.LINE_AA)
        self.ink_rect = _union(self.ink_rect, rect)
        return True

    def draw_polyline(self, pts: np.ndarray, color, thickness: int) -> None:
        """Whole-stroke draw - one blur pass instead of one per segment."""
        poly = pts.round().astype(np.int32).reshape(-1, 1, 2)
        pad = thickness * 3 + 12
        rect = _pts_rect(pts, pad, self.w, self.h)
        if rect is None:
            return
        self._glow_into(
            lambda t, ox, oy: cv2.polylines(
                t, [poly - np.array([[[ox, oy]]], np.int32)], False, color,
                thickness + 3, cv2.LINE_AA),
            rect, thickness)
        cv2.polylines(self.ink, [poly], False, color, thickness, cv2.LINE_AA)
        self.ink_rect = _union(self.ink_rect, rect)

    def commit(self, pts: np.ndarray, color, thickness: int) -> None:
        self.strokes.append((pts, color, thickness))

    def rebuild(self, rect=None) -> None:
        """Erase a region (or everything) and replay the committed strokes."""
        if rect is None:
            self.ink[:] = 0
            self.ink_rect = None
        else:
            x0, y0, x1, y1 = rect
            self.ink[y0:y1, x0:x1] = 0
        for pts, color, th in self.strokes:
            if rect is not None:
                r = _pts_rect(pts, th * 3 + 12, self.w, self.h)
                if r is None or r[0] >= rect[2] or r[2] <= rect[0] \
                        or r[1] >= rect[3] or r[3] <= rect[1]:
                    continue
            self.draw_polyline(pts, color, th)
    def clear(self) -> None:
        self.strokes.clear()
        self.ink[:] = 0
        self.ink_rect = None

    def undo(self) -> None:
        if not self.strokes:
            return
        pts, _, th = self.strokes.pop()
        rect = _pts_rect(pts, th * 3 + 16, self.w, self.h)
        if rect is not None:
            self.rebuild(rect)

    # -- fx ----------------------------------------------------------------
    def begin_fx(self) -> None:
        if self.fx_rect is not None:
            x0, y0, x1, y1 = self.fx_rect
            self.fx[y0:y1, x0:x1] = 0
            self.fx_rect = None

    def note_fx(self, rect) -> None:
        self.fx_rect = _union(self.fx_rect, rect)

    # -- output ------------------------------------------------------------
    def composite(self, out: np.ndarray) -> None:
        if self.ink_rect is not None:
            _max_into(out, self.ink, self.ink_rect)
        if self.fx_rect is not None:
            _max_into(out, self.fx, self.fx_rect)


# --------------------------------------------------------------------------
# The float-away
# --------------------------------------------------------------------------
def _smoothstep(x: float) -> float:
    x = min(1.0, max(0.0, x))
    return x * x * (3.0 - 2.0 * x)


class FloatShape:
    """A recognised shape: flashes, then drifts up and fades out.

    Every value below is derived from wall-clock elapsed time, never from a
    frame counter, so the motion looks the same at 20 fps on an iGPU as it
    does at 90 fps on a dGPU.
    """

    POP = 0.13          # fraction of life spent in the recognition flash
    HOLD = 0.24         # fraction held at full opacity before fading

    def __init__(self, shape: Shape, color, thickness: int, t0: float,
                 life: float = 2.6, rise: float = 190.0, rng=None):
        rng = rng or np.random
        self.center = shape.pts.mean(0).astype(np.float32)
        self.pts = (shape.pts - self.center).astype(np.float32)
        self.closed = shape.closed
        self.color = color
        self.thickness = thickness
        self.label = shape.kind.upper()
        self.t0 = t0
        self.life = life
        self.rise = rise
        self.sway_a = float(rng.uniform(9.0, 20.0))
        self.sway_f = float(rng.uniform(0.7, 1.2))
        self.spin = float(rng.uniform(-0.16, 0.16))
        self.phase = float(rng.uniform(0.0, 2.0 * math.pi))

    def render(self, layer: np.ndarray, now: float, show_label: bool = True):
        u = (now - self.t0) / self.life
        if u >= 1.0:
            return None
        u = max(0.0, u)

        pop = max(0.0, 1.0 - u / self.POP)
        pop = pop * pop
        # Gentle ease-IN, not ease-out: the shape lingers where it was drawn
        # for a beat, then drifts off as it lightens. Front-loading the rise
        # reads as a lurch rather than a float.
        rise = self.rise * (0.35 * u + 0.65 * u * u)
        sway = self.sway_a * math.sin(self.phase + u * self.sway_f * 2 * math.pi) \
            * min(1.0, u * 3.0)
        scale = 1.0 + 0.24 * u + 0.18 * pop
        spin = self.spin * u

        alpha = 1.0 if u < self.HOLD else \
            (1.0 - _smoothstep((u - self.HOLD) / (1.0 - self.HOLD))) ** 1.25
        bright = min(1.0, alpha * (1.0 + 1.2 * pop))
        if bright <= 0.01:
            return None

        ca, sa = math.cos(spin), math.sin(spin)
        p = self.pts * scale
        x = p[:, 0] * ca - p[:, 1] * sa + self.center[0] + sway
        y = p[:, 0] * sa + p[:, 1] * ca + self.center[1] - rise
        poly = np.stack([x, y], 1).round().astype(np.int32).reshape(-1, 1, 2)

        core = tuple(int(v * bright) for v in self.color)
        mid = tuple(int(v * bright * 0.55) for v in self.color)
        halo = tuple(int(v * bright * 0.22) for v in self.color)
        th = max(1, int(round(self.thickness * (1.0 + 1.1 * pop))))
        # Three-pass fake bloom. Blurring the FX layer would cost milliseconds
        # every frame; three widening strokes cost microseconds and, at this
        # size, look the same.
        cv2.polylines(layer, [poly], self.closed, halo, th + 12, cv2.LINE_AA)
        cv2.polylines(layer, [poly], self.closed, mid, th + 5, cv2.LINE_AA)
        cv2.polylines(layer, [poly], self.closed, core, th, cv2.LINE_AA)

        rect = _pts_rect(np.stack([x, y], 1), th + 18, layer.shape[1], layer.shape[0])

        if show_label and alpha > 0.05:
            txt = self.label
            (tw, thh), _ = cv2.getTextSize(txt, cv2.FONT_HERSHEY_DUPLEX, 0.52, 1)
            tx = int(x.mean() - tw / 2)
            ty = int(y.min()) - 14
            if 0 <= ty < layer.shape[0]:
                cv2.putText(layer, txt, (tx, ty), cv2.FONT_HERSHEY_DUPLEX, 0.52,
                            tuple(int(v * alpha * 0.9) for v in self.color), 1,
                            cv2.LINE_AA)
                rect = _union(rect, _clip_rect(
                    (tx - 4, ty - thh - 6, tx + tw + 4, ty + 6),
                    layer.shape[1], layer.shape[0]))
        return rect


class Pulse:
    """Expanding ring at the moment of recognition."""

    def __init__(self, center, color, t0: float, radius: float, life: float = 0.45):
        self.c = (int(center[0]), int(center[1]))
        self.color = color
        self.t0 = t0
        self.r0 = float(radius)
        self.life = life

    def render(self, layer: np.ndarray, now: float):
        u = (now - self.t0) / self.life
        if u >= 1.0 or u < 0.0:
            return None
        r = int(self.r0 * (0.65 + 0.75 * _smoothstep(u)))
        a = (1.0 - u) ** 2
        th = max(1, int(round(5 * (1.0 - u))) )
        cv2.circle(layer, self.c, r, tuple(int(v * a) for v in self.color), th,
                   cv2.LINE_AA)
        pad = th + 3
        return _clip_rect((self.c[0] - r - pad, self.c[1] - r - pad,
                           self.c[0] + r + pad, self.c[1] + r + pad),
                          layer.shape[1], layer.shape[0])


# --------------------------------------------------------------------------
# Application
# --------------------------------------------------------------------------
WINDOW = "Air-Draw"
FONT = cv2.FONT_HERSHEY_DUPLEX
ALL_SHAPES = ALL_SHAPES_DEFAULT


class AirDraw:
    def __init__(self, args):
        self.args = args
        self.allow = tuple(args.shapes.split(",")) if args.shapes else ALL_SHAPES

        self.cam = CameraStream(args.camera, args.width, args.height,
                                args.fps, mirror=not args.no_mirror).start()
        frame = self.cam.read()[0]
        self.h, self.w = frame.shape[:2]

        self.canvas = Canvas(self.w, self.h, glow=not args.no_glow)
        self.tracker = HandTracker(args.infer_width, args.complexity,
                                   args.det_conf, args.trk_conf,
                                   threaded=not args.sync)
        self.pose = HandPose()
        self.fsm = GestureFSM(clear_hold=args.clear_hold)
        self.filt = OneEuroPoint(args.mincutoff, args.beta)
        self.rng = np.random.default_rng()

        self.stroke: List[Tuple[float, float]] = []
        self.floats: List[FloatShape] = []
        self.pulses: List[Pulse] = []
        self.color_idx = 0
        self.thickness = args.thickness
        self.show_hud = True
        self.show_skel = False
        self.base = None
        self.base_t = 0.0
        self.lost_since = None
        self.last_infer = 0.0
        self.help = self._make_help()
        self.dim_lut = (np.clip(np.arange(256) * args.dim, 0, 255)
                        .astype(np.uint8) if args.dim < 0.999 else None)

    # -- helpers -----------------------------------------------------------
    @property
    def color(self):
        return PALETTE[self.color_idx]

    def _make_help(self):
        sp = np.zeros((28, self.w, 3), np.uint8)
        cv2.putText(sp, "point = draw   |   two fingers = move   |   open palm = clear"
                    "   |   [ ] colour   - = size   |   z undo   s save   h hud"
                    "   l skeleton   ESC quit",
                    (14, 19), FONT, 0.44, (135, 135, 145), 1, cv2.LINE_AA)
        return sp

    def _cursor(self, now):
        """Filtered fingertip, nudged forward to cancel inference latency."""
        if self.base is None:
            return None
        lag = min(max(now - self.base_t, 0.0), 0.06)
        vx, vy = self.filt.velocity
        dx, dy = vx * lag * self.args.predict, vy * lag * self.args.predict
        m = math.hypot(dx, dy)
        if m > 30.0:
            dx, dy = dx * 30.0 / m, dy * 30.0 / m
        return self.base[0] + dx, self.base[1] + dy

    # -- stroke lifecycle --------------------------------------------------
    def _finalise(self, now):
        if len(self.stroke) < 2:
            self.stroke = []
            return
        pts = np.asarray(self.stroke, np.float32)
        self.stroke = []
        shape = recognise(pts, allow=self.allow)
        if shape is None:
            self.canvas.commit(pts, self.color, self.thickness)
            return
        # Recognised: wipe the messy original, then let the clean one drift off.
        rect = _pts_rect(pts, self.thickness * 3 + 18, self.w, self.h)
        if rect is not None:
            self.canvas.rebuild(rect)
        self.floats.append(FloatShape(shape, self.color, self.thickness, now,
                                      life=self.args.life, rise=self.args.rise,
                                      rng=self.rng))
        c = shape.pts.mean(0)
        r = float(np.linalg.norm(shape.pts - c, axis=1).max())
        self.pulses.append(Pulse(c, self.color, now, r * 1.05))

    def _clear(self):
        self.canvas.clear()
        self.stroke = []

    def _on_landmarks(self, lm, t_lm, now):
        if lm is None:
            if self.stroke and self.lost_since is None:
                self.lost_since = now
            if self.lost_since is not None and now - self.lost_since > self.args.grace:
                self._finalise(now)
                self.base = None
                self.lost_since = None
                self.filt.reset()
                self.fsm.update(IDLE, now)
            return
        self.lost_since = None
        lm_px = lm[:, :2] * np.array([self.w, self.h], np.float32)
        self.pose.update(lm_px)
        state, cleared = self.fsm.update(self.pose.gesture(), now)
        self.base = self.filt(float(lm_px[8, 0]), float(lm_px[8, 1]), t_lm)
        self.base_t = t_lm
        if cleared:
            self._clear()
            return
        cur = self._cursor(now)
        if state == DRAW and cur is not None:
            if not self.stroke:
                self.stroke.append(cur)
            else:
                last = self.stroke[-1]
                if math.hypot(cur[0] - last[0], cur[1] - last[1]) >= 1.4:
                    if self.canvas.add_segment(last, cur, self.color, self.thickness):
                        self.stroke.append(cur)
                    else:
                        self._finalise(now)
        elif self.stroke:
            self._finalise(now)

    # -- drawing -----------------------------------------------------------
    def _draw_fx(self, now):
        self.canvas.begin_fx()
        alive = []
        for f in self.floats:
            r = f.render(self.canvas.fx, now, show_label=not self.args.no_labels)
            if r is not None:
                self.canvas.note_fx(r)
                alive.append(f)
        self.floats = alive
        alive_p = []
        for p in self.pulses:
            r = p.render(self.canvas.fx, now)
            if r is not None:
                self.canvas.note_fx(r)
                alive_p.append(p)
        self.pulses = alive_p

    def _draw_overlay(self, out, lm, now, fps):
        if self.show_skel and lm is not None:
            pts = (lm[:, :2] * np.array([self.w, self.h], np.float32)) \
                .round().astype(np.int32)
            cv2.polylines(out, [pts[i].reshape(-1, 1, 2) for i in SKELETON],
                          False, (70, 70, 78), 1, cv2.LINE_AA)

        cur = self._cursor(now)
        if cur is not None:
            c = (int(cur[0]), int(cur[1]))
            r = self.thickness + 8
            if self.fsm.state == DRAW:
                cv2.circle(out, c, r, self.color, 2, cv2.LINE_AA)
                cv2.circle(out, c, max(2, self.thickness // 2), self.color, -1,
                           cv2.LINE_AA)
            else:
                cv2.circle(out, c, r, (185, 185, 195), 1, cv2.LINE_AA)
            prog = self.fsm.clear_progress(now)
            if prog > 0.0:
                cv2.ellipse(out, c, (r + 9, r + 9), -90, 0, int(360 * prog),
                            (90, 120, 255), 3, cv2.LINE_AA)

        if not self.show_hud:
            return
        cv2.putText(out, f"{fps:5.1f} fps   infer {self.tracker.infer_ms:4.1f} ms"
                         f"   {self.fsm.state}", (14, 30), FONT, 0.55,
                    (225, 225, 235), 1, cv2.LINE_AA)
        cv2.rectangle(out, (14, 42), (44, 58), self.color, -1)
        cv2.rectangle(out, (14, 42), (44, 58), (40, 40, 40), 1)
        cv2.putText(out, f"{self.thickness}px", (52, 56), FONT, 0.46,
                    (170, 170, 180), 1, cv2.LINE_AA)
        band = out[self.h - 28:self.h]
        np.maximum(band, self.help, out=band)

    # -- main loop ---------------------------------------------------------
    def run(self):
        cv2.namedWindow(WINDOW, cv2.WINDOW_AUTOSIZE)
        out = np.empty((self.h, self.w, 3), np.uint8)
        last_cam, last_lm = -1, -1
        lm_cur = None
        fps, t_prev = 0.0, time.perf_counter()
        period = 1.0 / self.args.max_fps if self.args.max_fps > 0 else 0.0
        deadline = time.perf_counter()
        frame = None

        try:
            while True:
                now = time.perf_counter()
                got = self.cam.read()
                if got is None:
                    break
                if got[1] != last_cam:
                    last_cam = got[1]
                    frame = got[0]
                    if self.args.infer_fps <= 0 or \
                            now - self.last_infer >= 1.0 / self.args.infer_fps:
                        self.last_infer = now
                        self.tracker.submit(frame, got[2])
                if frame is None:
                    continue

                res = self.tracker.latest()
                if res is not None and res[2] != last_lm:
                    last_lm = res[2]
                    lm_cur = res[0]
                    self._on_landmarks(res[0], res[1], now)
                elif self.lost_since is not None:
                    self._on_landmarks(None, 0.0, now)

                if self.dim_lut is not None:
                    cv2.LUT(frame, self.dim_lut, out)
                else:
                    np.copyto(out, frame)

                self._draw_fx(now)
                self.canvas.composite(out)
                self._draw_overlay(out, lm_cur, now, fps)
                cv2.imshow(WINDOW, out)

                dt = now - t_prev
                t_prev = now
                if dt > 0:
                    fps += 0.12 * (1.0 / dt - fps)

                if period:
                    slack = deadline - time.perf_counter()
                    if slack > 0.0005:
                        time.sleep(slack)
                    deadline = max(time.perf_counter(), deadline + period)

                k = cv2.waitKey(1) & 0xFF
                if k in (27, ord("q")):
                    break
                elif k == ord("c"):
                    self._clear()
                elif k == ord("z"):
                    self.canvas.undo()
                elif k == ord("["):
                    self.color_idx = (self.color_idx - 1) % len(PALETTE)
                elif k == ord("]"):
                    self.color_idx = (self.color_idx + 1) % len(PALETTE)
                elif k in (ord("-"), ord("_")):
                    self.thickness = max(2, self.thickness - 1)
                elif k in (ord("="), ord("+")):
                    self.thickness = min(24, self.thickness + 1)
                elif k == ord("s"):
                    name = f"airdraw_{int(time.time())}.png"
                    cv2.imwrite(name, out)
                    print(f"saved {name}")
                elif k == ord("h"):
                    self.show_hud = not self.show_hud
                elif k == ord("l"):
                    self.show_skel = not self.show_skel
                elif k == ord("g"):
                    self.canvas.glow = not self.canvas.glow

                try:
                    if cv2.getWindowProperty(WINDOW, cv2.WND_PROP_VISIBLE) < 1:
                        break
                except cv2.error:
                    break
        finally:
            self.tracker.close()
            self.cam.release()
            cv2.destroyAllWindows()


# --------------------------------------------------------------------------
# Synthetic strokes - used by --selftest and --demo (no webcam required)
# --------------------------------------------------------------------------
def _densify(verts: np.ndarray, closed: bool, n: int = 400) -> np.ndarray:
    v = np.vstack([verts, verts[:1]]) if closed else verts
    seg = np.linalg.norm(np.diff(v, axis=0), axis=1)
    total = seg.sum()
    out = []
    for i in range(len(v) - 1):
        k = max(2, int(round(n * seg[i] / total)))
        t = np.linspace(0, 1, k, endpoint=False)[:, None]
        out.append(v[i] + (v[i + 1] - v[i]) * t)
    return np.vstack(out).astype(np.float32)


def ideal_shape(kind: str, rng, w=960, h=720) -> Tuple[np.ndarray, bool]:
    cx = rng.uniform(0.33, 0.67) * w
    cy = rng.uniform(0.35, 0.65) * h
    R = rng.uniform(0.13, 0.21) * min(w, h)
    rot = rng.uniform(0, 2 * math.pi)
    c = np.array([cx, cy], np.float32)

    if kind == "line":
        ang = rng.uniform(0, 2 * math.pi)
        u = np.array([math.cos(ang), math.sin(ang)], np.float32)
        return _densify(np.stack([c - u * R * 1.4, c + u * R * 1.4]), False), False
    if kind == "arrow":
        ang = rng.uniform(0, 2 * math.pi)
        u = np.array([math.cos(ang), math.sin(ang)], np.float32)
        n = np.array([-u[1], u[0]], np.float32)
        tail, tip = c - u * R * 1.3, c + u * R * 1.3
        head = R * 0.55
        b1 = tip - u * head + n * head * 0.55
        b2 = tip - u * head - n * head * 0.55
        return _densify(np.stack([tail, tip, b1, tip, b2]), False), False
    if kind == "star":
        th = rot + np.arange(10) * (math.pi / 5)
        rad = np.where(np.arange(10) % 2 == 0, R, R * rng.uniform(0.36, 0.44))
        v = np.stack([cx + rad * np.cos(th), cy + rad * np.sin(th)], 1)
        return _densify(v.astype(np.float32), True), True
    if kind in ("circle", "ellipse"):
        a = R
        b = R if kind == "circle" else R * rng.uniform(0.42, 0.66)
        return _ellipse_pts(c, a, b, rot, 400), True
    n = {"triangle": 3, "square": 4, "pentagon": 5, "hexagon": 6}.get(kind)
    if n:
        v = _regular_polygon(c, R, n, rot)
        return _densify(v, True), True
    if kind == "rectangle":
        rw, rh = R, R * rng.uniform(0.40, 0.62)
        deg = math.degrees(rot) % 180
        v = cv2.boxPoints(((cx, cy), (rw * 2, rh * 2), deg)).astype(np.float32)
        return _densify(v, True), True
    raise ValueError(kind)


def humanise(base: np.ndarray, closed: bool, rng, wobble=0.020, noise=0.55,
             gap=(0.02, 0.10)) -> np.ndarray:
    """Turn a perfect outline into something a shaky finger would produce."""
    P, _ = _resample(base, 400)
    span = P.max(0) - P.min(0)
    diag = float(math.hypot(span[0], span[1]))
    t = np.linspace(0.0, 1.0, len(P), dtype=np.float32)
    off = np.zeros_like(P)
    for _ in range(3):
        # Integer frequencies for closed shapes so the wobble has no seam.
        f = float(rng.integers(1, 5)) if closed else rng.uniform(0.7, 3.5)
        ph = rng.uniform(0, 2 * math.pi)
        amp = wobble * diag * rng.uniform(0.5, 1.15)
        off[:, 0] += amp * np.sin(2 * math.pi * f * t + ph)
        off[:, 1] += amp * np.cos(2 * math.pi * f * t + ph * 1.31)
    P = P + off + rng.normal(0.0, noise, P.shape).astype(np.float32)

    if closed:                                    # people rarely close the loop
        P = P[: int(len(P) * (1.0 - rng.uniform(*gap)))]
    n = int(rng.integers(45, 150))                # uneven sampling / hand speed
    idx = np.sort(rng.choice(len(P), min(n, len(P)), replace=False))
    idx[0] = 0
    idx[-1] = len(P) - 1
    return P[idx].astype(np.float32)


ALIAS = {
    # A hand-drawn square is a rectangle and vice versa; an ellipse rounder
    # than ~0.86 is deliberately reported as a circle. Score those as correct.
    "square": {"square", "rectangle"},
    "rectangle": {"rectangle", "square"},
    "circle": {"circle", "ellipse"},
    "ellipse": {"ellipse", "circle"},
}


def _run_trials(n, seed, wobble):
    rng = np.random.default_rng(seed)
    kinds = list(ALL_SHAPES)
    tally = {k: [0, 0] for k in kinds}
    misses = {}
    for i in range(n):
        kind = kinds[i % len(kinds)]
        base, closed = ideal_shape(kind, rng)
        got = recognise(humanise(base, closed, rng, wobble=wobble))
        name = got.kind if got else "none"
        tally[kind][1] += 1
        if name in ALIAS.get(kind, {kind}):
            tally[kind][0] += 1
        else:
            misses[(kind, name)] = misses.get((kind, name), 0) + 1
    return tally, misses


def selftest(trials: int = 400, seeds: int = 6) -> int:
    """Accuracy on synthetic shaky strokes, averaged over several seeds.

    Averaging matters: single-seed accuracy on 400 strokes swings by six points
    either way, which is more than most of the tuning decisions here are worth.
    """
    kinds = list(ALL_SHAPES)
    per = {k: [] for k in kinds}
    overall = []
    misses = {}
    t0 = time.perf_counter()
    for s in range(seeds):
        tally, m = _run_trials(trials, 7 + s, 0.020)
        for k, (ok, n) in tally.items():
            per[k].append(ok / max(n, 1))
        overall.append(sum(v[0] for v in tally.values()) / max(trials, 1))
        for key, c in m.items():
            misses[key] = misses.get(key, 0) + c
    dt = (time.perf_counter() - t0) / (trials * seeds) * 1000.0

    print(f"\n  shape recognition - {trials} strokes x {seeds} seeds, "
          f"{dt:.2f} ms per classification\n")
    for k in kinds:
        v = np.array(per[k])
        bar = "#" * int(round(v.mean() * 22))
        print(f"   {k:<10} {100*v.mean():5.1f}%  "
              f"(min {100*v.min():4.1f}  max {100*v.max():5.1f})  {bar}")
    o = np.array(overall)
    print(f"\n   overall    {100*o.mean():5.1f}%  "
          f"(min {100*o.min():4.1f}  max {100*o.max():5.1f}, sd {100*o.std():.1f})")

    print("\n   accuracy vs how much the hand shakes "
          "(wobble as a fraction of shape size):")
    for w in (0.008, 0.014, 0.020, 0.028, 0.036):
        accs = [sum(v[0] for v in _run_trials(trials // 2, 400 + s, w)[0].values())
                / max(trials // 2, 1) for s in range(2)]
        a = float(np.mean(accs))
        print(f"     {w*100:4.1f}%   {100*a:5.1f}%  {'#' * int(round(a * 22))}")

    if misses:
        print("\n   most common misses:")
        for (a, b), c in sorted(misses.items(), key=lambda kv: -kv[1])[:8]:
            print(f"     {a:<10} -> {b:<10} x{c}")
    print()
    return 0 if np.mean(overall) >= 0.88 else 1


# --------------------------------------------------------------------------
# Headless animation demo - proves the float-away without a camera
# --------------------------------------------------------------------------
def demo(path: str, seed: int = 3, fps: int = 30) -> int:
    W, H = 960, 720
    rng = np.random.default_rng(seed)
    canvas = Canvas(W, H, glow=True)
    bg = np.full((H, W, 3), 16, np.uint8)
    cv2.rectangle(bg, (0, 0), (W, H), (26, 22, 20), -1)

    kinds = ["circle", "square", "triangle", "star", "line", "pentagon"]
    frames: List[np.ndarray] = []
    t = 0.0
    dt = 1.0 / fps
    floats: List[FloatShape] = []
    pulses: List[Pulse] = []

    for si, kind in enumerate(kinds):
        color = PALETTE[si % len(PALETTE)]
        base, closed = ideal_shape(kind, rng, W, H)
        stroke = humanise(base, closed, rng)
        stroke, _ = _resample(stroke, 70)
        th = 6

        for i in range(1, len(stroke)):          # draw it, one segment a frame
            canvas.add_segment(stroke[i - 1], stroke[i], color, th)
            if i % 2 == 0:
                frames.append(_demo_frame(bg, canvas, floats, pulses, t))
                t += dt

        shape = recognise(stroke)
        if shape is not None:
            rect = _pts_rect(stroke, th * 2 + 14, W, H)
            if rect:
                canvas.rebuild(rect)
            floats.append(FloatShape(shape, color, th, t, life=2.4, rise=210, rng=rng))
            c = shape.pts.mean(0)
            r = float(np.linalg.norm(shape.pts - c, axis=1).max())
            pulses.append(Pulse(c, color, t, r * 1.05))
        else:
            canvas.commit(stroke, color, th)
        for _ in range(int(1.1 * fps)):
            frames.append(_demo_frame(bg, canvas, floats, pulses, t))
            t += dt
    for _ in range(int(1.6 * fps)):
        frames.append(_demo_frame(bg, canvas, floats, pulses, t))
        t += dt

    if path.lower().endswith(".png"):
        picks = np.linspace(0, len(frames) - 1, 12).round().astype(int)
        sel = [cv2.resize(frames[i], (W // 2, H // 2)) for i in picks]
        rows = [np.hstack(sel[i:i + 4]) for i in range(0, 12, 4)]
        sheet = np.vstack(rows)
        cv2.imwrite(path, sheet)
    else:
        vw = cv2.VideoWriter(path, cv2.VideoWriter_fourcc(*"mp4v"), fps, (W, H))
        for f in frames:
            vw.write(f)
        vw.release()
    print(f"wrote {path}  ({len(frames)} frames)")
    return 0


def _demo_frame(bg, canvas, floats, pulses, t):
    out = bg.copy()
    canvas.begin_fx()
    for f in list(floats):
        r = f.render(canvas.fx, t)
        if r is None:
            floats.remove(f)
        else:
            canvas.note_fx(r)
    for p in list(pulses):
        r = p.render(canvas.fx, t)
        if r is None:
            pulses.remove(p)
        else:
            canvas.note_fx(r)
    canvas.composite(out)
    return out


def bench() -> int:
    """Time the render pipeline on this machine - no camera needed."""
    W, H = 1280, 720
    canvas = Canvas(W, H, glow=True)
    frame = np.random.randint(0, 255, (H, W, 3), np.uint8)
    out = np.empty_like(frame)
    lut = (np.arange(256) * 0.82).astype(np.uint8)
    rng = np.random.default_rng(0)

    def timed(fn, n=120):
        fn()
        t0 = time.perf_counter()
        for _ in range(n):
            fn()
        return (time.perf_counter() - t0) / n * 1000.0

    pts = rng.uniform([0, 0], [W, H], (200, 2)).astype(np.float32)
    i = [0]

    def seg():
        i[0] = (i[0] + 1) % (len(pts) - 1)
        a = pts[i[0]]
        b = a + rng.uniform(-14, 14, 2).astype(np.float32)
        canvas.add_segment(a, b, (255, 220, 60), 6)

    print("\n  render pipeline, 1280x720")
    print(f"   dim (LUT)            {timed(lambda: cv2.LUT(frame, lut, out)):6.3f} ms")
    print(f"   stroke segment+glow  {timed(seg):6.3f} ms")
    canvas.ink_rect = (0, 0, W, H)
    print(f"   composite full frame {timed(lambda: canvas.composite(out)):6.3f} ms")
    canvas.ink_rect = (300, 200, 900, 560)
    print(f"   composite dirty rect {timed(lambda: canvas.composite(out)):6.3f} ms")
    print(f"\n   OpenCV {cv2.__version__}   threads={cv2.getNumThreads()}   "
          f"OpenCL={'yes' if cv2.ocl.haveOpenCL() else 'no'}")
    print("   (MediaPipe inference is not measured here - run the app and read"
          "\n    the 'infer' number in the HUD.)\n")
    return 0


# --------------------------------------------------------------------------
def main(argv=None) -> int:
    p = argparse.ArgumentParser(
        description="Air-Draw: hand drawing with shape snapping and float-away.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    p.add_argument("--camera", type=int, default=0)
    p.add_argument("--width", type=int, default=1280)
    p.add_argument("--height", type=int, default=720)
    p.add_argument("--fps", type=int, default=60, help="requested camera fps")
    p.add_argument("--max-fps", type=int, default=60,
                   help="render cap; keeps a laptop from spinning its fans for nothing")
    p.add_argument("--no-mirror", action="store_true")

    g = p.add_argument_group("performance")
    g.add_argument("--infer-width", type=int, default=320,
                   help="width the hand model actually sees; the big lever")
    g.add_argument("--complexity", type=int, default=0, choices=(0, 1),
                   help="0 = lite hand model (~2x faster), 1 = full")
    g.add_argument("--infer-fps", type=float, default=0,
                   help="cap inference rate to save battery; 0 = uncapped")
    g.add_argument("--sync", action="store_true",
                   help="run inference on the render thread (debugging)")
    g.add_argument("--cv-threads", type=int, default=0,
                   help="OpenCV thread cap; 0 = auto (half the cores), which "
                        "leaves room for the inference thread")

    g = p.add_argument_group("feel")
    g.add_argument("--mincutoff", type=float, default=1.3,
                   help="One Euro: lower = steadier when still")
    g.add_argument("--beta", type=float, default=0.012,
                   help="One Euro: higher = less lag when moving fast")
    g.add_argument("--predict", type=float, default=0.35,
                   help="0..1 lag compensation; 0 disables")
    g.add_argument("--grace", type=float, default=0.16,
                   help="seconds of lost tracking tolerated mid-stroke")
    g.add_argument("--clear-hold", type=float, default=0.5)
    g.add_argument("--det-conf", type=float, default=0.6)
    g.add_argument("--trk-conf", type=float, default=0.5)

    g = p.add_argument_group("look")
    g.add_argument("--thickness", type=int, default=6)
    g.add_argument("--dim", type=float, default=0.82,
                   help="camera dim factor so the neon reads; 1.0 = off")
    g.add_argument("--no-glow", action="store_true")
    g.add_argument("--no-labels", action="store_true",
                   help="hide the shape name on the float-away")
    g.add_argument("--life", type=float, default=2.6, help="float-away seconds")
    g.add_argument("--rise", type=float, default=190.0, help="float-away pixels")
    g.add_argument("--shapes", type=str, default="",
                   help="comma list to restrict recognition, e.g. "
                        "'line,circle,square,triangle'")

    g = p.add_argument_group("headless")
    g.add_argument("--selftest", action="store_true")
    g.add_argument("--trials", type=int, default=400)
    g.add_argument("--seeds", type=int, default=6,
                   help="selftest: random seeds to average over")
    g.add_argument("--demo", type=str, default="",
                   help="render the animation to .mp4 or a .png contact sheet")
    g.add_argument("--bench", action="store_true")

    args = p.parse_args(argv)

    cv2.setNumThreads(args.cv_threads if args.cv_threads > 0
                      else max(2, (os.cpu_count() or 4) // 2))
    if args.selftest:
        return selftest(args.trials, args.seeds)
    if args.demo:
        return demo(args.demo)
    if args.bench:
        return bench()

    if args.shapes:
        bad = set(args.shapes.split(",")) - set(ALL_SHAPES)
        if bad:
            p.error(f"unknown shapes: {', '.join(sorted(bad))}")
    try:
        AirDraw(args).run()
    except RuntimeError as e:
        print(f"error: {e}", file=sys.stderr)
        return 2
    except KeyboardInterrupt:
        pass
    return 0


if __name__ == "__main__":
    sys.exit(main())
