"use client";

import { useEffect, useRef, useState } from "react";

type Point = { x: number; y: number };
type Stroke = { points: Point[]; color: string; width: number };
type CameraState = "idle" | "loading" | "ready" | "error";
type SnappedShape = { name: string; points: Point[] };
type FloatingShape = SnappedShape & {
  color: string;
  width: number;
  startedAt: number;
};

const COLORS = ["#69f7ff", "#9b8cff", "#ff5ccf", "#ffda5c", "#73ff9d", "#ffffff"];
const CONNECTIONS = [
  [0, 1], [1, 2], [2, 3], [3, 4], [0, 5], [5, 6], [6, 7], [7, 8],
  [5, 9], [9, 10], [10, 11], [11, 12], [9, 13], [13, 14], [14, 15],
  [15, 16], [13, 17], [17, 18], [18, 19], [19, 20], [0, 17],
];

function fingerUp(landmarks: Array<{ x: number; y: number }>, tip: number, pip: number) {
  return landmarks[tip].y < landmarks[pip].y - 0.015;
}

function pointLineDistance(point: Point, start: Point, end: Point) {
  const dx = end.x - start.x;
  const dy = end.y - start.y;
  const denominator = dx * dx + dy * dy;
  if (!denominator) return Math.hypot(point.x - start.x, point.y - start.y);
  const t = Math.max(0, Math.min(1, ((point.x - start.x) * dx + (point.y - start.y) * dy) / denominator));
  return Math.hypot(point.x - (start.x + t * dx), point.y - (start.y + t * dy));
}

function simplify(points: Point[], epsilon: number): Point[] {
  if (points.length < 3) return points;
  let farthest = 0;
  let index = 0;
  for (let i = 1; i < points.length - 1; i += 1) {
    const distance = pointLineDistance(points[i], points[0], points[points.length - 1]);
    if (distance > farthest) { farthest = distance; index = i; }
  }
  if (farthest <= epsilon) return [points[0], points[points.length - 1]];
  const left = simplify(points.slice(0, index + 1), epsilon);
  const right = simplify(points.slice(index), epsilon);
  return [...left.slice(0, -1), ...right];
}

function recognizeStroke(stroke: Stroke): SnappedShape | null {
  const points = stroke.points;
  if (points.length < 10) return null;
  const xs = points.map((point) => point.x);
  const ys = points.map((point) => point.y);
  const minX = Math.min(...xs); const maxX = Math.max(...xs);
  const minY = Math.min(...ys); const maxY = Math.max(...ys);
  const width = maxX - minX; const height = maxY - minY;
  const diagonal = Math.hypot(width, height);
  if (diagonal < 36) return null;
  const start = points[0]; const end = points[points.length - 1];
  const closed = Math.hypot(start.x - end.x, start.y - end.y) < diagonal * .2;

  if (!closed) {
    const deviation = Math.max(...points.map((point) => pointLineDistance(point, start, end)));
    if (deviation < diagonal * .055) return { name: "Line", points: [start, end] };
    return null;
  }

  const center = { x: (minX + maxX) / 2, y: (minY + maxY) / 2 };
  const radii = points.map((point) => Math.hypot((point.x - center.x) / Math.max(width, 1), (point.y - center.y) / Math.max(height, 1)));
  const mean = radii.reduce((total, radius) => total + radius, 0) / radii.length;
  const spread = Math.sqrt(radii.reduce((total, radius) => total + (radius - mean) ** 2, 0) / radii.length) / Math.max(mean, .001);
  if (spread < .17 && Math.min(width, height) / Math.max(width, height) > .34) {
    const ellipse = Array.from({ length: 73 }, (_, index) => {
      const angle = (index / 72) * Math.PI * 2;
      return { x: center.x + Math.cos(angle) * width / 2, y: center.y + Math.sin(angle) * height / 2 };
    });
    const ratio = Math.min(width, height) / Math.max(width, height);
    return { name: ratio > .82 ? "Circle" : "Ellipse", points: ellipse };
  }

  for (const factor of [.025, .04, .06, .08]) {
    const candidate = simplify(points, diagonal * factor);
    const vertices = [...candidate];
    if (vertices.length > 2 && Math.hypot(vertices[0].x - vertices[vertices.length - 1].x, vertices[0].y - vertices[vertices.length - 1].y) < diagonal * .22) vertices.pop();
    const count = vertices.length;
    if (count >= 3 && count <= 6) {
      const names: Record<number, string> = { 3: "Triangle", 5: "Pentagon", 6: "Hexagon" };
      const name = count === 4 ? (Math.min(width, height) / Math.max(width, height) > .78 ? "Square" : "Rectangle") : names[count];
      return { name, points: [...vertices, vertices[0]] };
    }
    if (count >= 8 && count <= 12) return { name: "Star", points: [...vertices, vertices[0]] };
  }
  return null;
}

export default function Home() {
  const videoRef = useRef<HTMLVideoElement>(null);
  const inkRef = useRef<HTMLCanvasElement>(null);
  const effectsRef = useRef<HTMLCanvasElement>(null);
  const overlayRef = useRef<HTMLCanvasElement>(null);
  const streamRef = useRef<MediaStream | null>(null);
  const landmarkerRef = useRef<any>(null);
  const rafRef = useRef<number | null>(null);
  const runningRef = useRef(false);
  const strokesRef = useRef<Stroke[]>([]);
  const currentStrokeRef = useRef<Stroke | null>(null);
  const lastPointRef = useRef<Point | null>(null);
  const lastDetectRef = useRef(0);
  const lastFrameRef = useRef(performance.now());
  const frameCountRef = useRef(0);
  const palmStartRef = useRef<number | null>(null);
  const clearCooldownRef = useRef(0);
  const pointerDrawingRef = useRef(false);
  const colorRef = useRef(COLORS[0]);
  const brushRef = useRef(7);
  const glowRef = useRef(true);
  const skeletonRef = useRef(true);
  const hudRef = useRef(true);
  const snapRef = useRef(true);
  const gestureRef = useRef("Waiting");
  const toastTimerRef = useRef<ReturnType<typeof setTimeout> | null>(null);
  const floatingShapesRef = useRef<FloatingShape[]>([]);
  const effectsRafRef = useRef<number | null>(null);

  const [camera, setCamera] = useState<CameraState>("idle");
  const [error, setError] = useState("");
  const [gesture, setGesture] = useState("Waiting");
  const [fps, setFps] = useState(0);
  const [color, setColor] = useState(COLORS[0]);
  const [brush, setBrush] = useState(7);
  const [glow, setGlow] = useState(true);
  const [skeleton, setSkeleton] = useState(true);
  const [hud, setHud] = useState(true);
  const [snap, setSnap] = useState(true);
  const [shapeToast, setShapeToast] = useState("");
  const [helpOpen, setHelpOpen] = useState(false);
  const [hasInk, setHasInk] = useState(false);

  useEffect(() => { colorRef.current = color; }, [color]);
  useEffect(() => { brushRef.current = brush; }, [brush]);
  useEffect(() => { glowRef.current = glow; redrawInk(); }, [glow]);
  useEffect(() => { skeletonRef.current = skeleton; }, [skeleton]);
  useEffect(() => { hudRef.current = hud; }, [hud]);
  useEffect(() => { snapRef.current = snap; }, [snap]);

  useEffect(() => {
    prepareCanvases();
    const resize = () => prepareCanvases();
    window.addEventListener("resize", resize);
    return () => window.removeEventListener("resize", resize);
  }, []);

  useEffect(() => {
    const onKey = (event: KeyboardEvent) => {
      if (event.key.toLowerCase() === "c") clearCanvas();
      if (event.key.toLowerCase() === "z") undo();
      if (event.key.toLowerCase() === "s") savePng();
      if (event.key.toLowerCase() === "g") setGlow((value) => !value);
      if (event.key.toLowerCase() === "l") setSkeleton((value) => !value);
      if (event.key.toLowerCase() === "h") setHud((value) => !value);
      if (event.key === "[") cycleColor(-1);
      if (event.key === "]") cycleColor(1);
      if (event.key === "-") setBrush((value) => Math.max(2, value - 1));
      if (event.key === "=" || event.key === "+") setBrush((value) => Math.min(24, value + 1));
    };
    window.addEventListener("keydown", onKey);
    return () => window.removeEventListener("keydown", onKey);
  }, []);

  useEffect(() => () => {
    stopCamera();
    if (effectsRafRef.current) cancelAnimationFrame(effectsRafRef.current);
    if (toastTimerRef.current) clearTimeout(toastTimerRef.current);
  }, []);

  function setGestureSafe(next: string) {
    if (gestureRef.current !== next) {
      gestureRef.current = next;
      setGesture(next);
    }
  }

  function prepareCanvases() {
    const video = videoRef.current;
    const ink = inkRef.current;
    const effects = effectsRef.current;
    const overlay = overlayRef.current;
    if (!video || !ink || !effects || !overlay) return;
    const width = video.videoWidth || 1280;
    const height = video.videoHeight || 720;
    if (ink.width !== width || ink.height !== height) {
      ink.width = width;
      ink.height = height;
      effects.width = width;
      effects.height = height;
      overlay.width = width;
      overlay.height = height;
      redrawInk();
    }
  }

  function styleInk(context: CanvasRenderingContext2D, strokeColor: string, width: number) {
    context.lineCap = "round";
    context.lineJoin = "round";
    context.strokeStyle = strokeColor;
    context.lineWidth = width;
    context.shadowColor = strokeColor;
    context.shadowBlur = glowRef.current ? width * 2.3 : 0;
  }

  function drawSegment(from: Point, to: Point, strokeColor: string, width: number) {
    const context = inkRef.current?.getContext("2d");
    if (!context) return;
    context.save();
    styleInk(context, strokeColor, width);
    context.beginPath();
    context.moveTo(from.x, from.y);
    context.quadraticCurveTo(from.x, from.y, to.x, to.y);
    context.stroke();
    context.restore();
  }

  function drawStroke(context: CanvasRenderingContext2D, stroke: Stroke) {
    if (stroke.points.length < 2) return;
    context.save();
    styleInk(context, stroke.color, stroke.width);
    context.beginPath();
    context.moveTo(stroke.points[0].x, stroke.points[0].y);
    for (let i = 1; i < stroke.points.length - 1; i += 1) {
      const current = stroke.points[i];
      const next = stroke.points[i + 1];
      context.quadraticCurveTo(current.x, current.y, (current.x + next.x) / 2, (current.y + next.y) / 2);
    }
    const last = stroke.points[stroke.points.length - 1];
    context.lineTo(last.x, last.y);
    context.stroke();
    context.restore();
  }

  function redrawInk() {
    const canvas = inkRef.current;
    const context = canvas?.getContext("2d");
    if (!canvas || !context) return;
    context.clearRect(0, 0, canvas.width, canvas.height);
    strokesRef.current.forEach((stroke) => drawStroke(context, stroke));
  }

  function beginStroke(point: Point) {
    const stroke = { points: [point], color: colorRef.current, width: brushRef.current };
    currentStrokeRef.current = stroke;
    strokesRef.current.push(stroke);
    lastPointRef.current = point;
    setHasInk(true);
  }

  function appendPoint(point: Point) {
    if (!currentStrokeRef.current) beginStroke(point);
    const stroke = currentStrokeRef.current;
    const last = lastPointRef.current;
    if (!stroke || !last) return;
    const smoothed = { x: last.x * 0.58 + point.x * 0.42, y: last.y * 0.58 + point.y * 0.42 };
    const distance = Math.hypot(smoothed.x - last.x, smoothed.y - last.y);
    if (distance < 1.5) return;
    stroke.points.push(smoothed);
    drawSegment(last, smoothed, stroke.color, stroke.width);
    lastPointRef.current = smoothed;
  }

  function animateFloatingShapes(now: number) {
    const canvas = effectsRef.current;
    const context = canvas?.getContext("2d");
    if (!canvas || !context) {
      effectsRafRef.current = null;
      return;
    }

    context.clearRect(0, 0, canvas.width, canvas.height);
    floatingShapesRef.current = floatingShapesRef.current.filter((shape) => now - shape.startedAt < 1850);

    floatingShapesRef.current.forEach((shape) => {
      const progress = Math.min(1, (now - shape.startedAt) / 1850);
      const eased = 1 - Math.pow(1 - progress, 3);
      const opacity = Math.pow(1 - progress, 1.35);
      const xs = shape.points.map((point) => point.x);
      const ys = shape.points.map((point) => point.y);
      const center = {
        x: (Math.min(...xs) + Math.max(...xs)) / 2,
        y: (Math.min(...ys) + Math.max(...ys)) / 2,
      };
      const top = Math.min(...ys);
      const lift = eased * Math.max(80, canvas.height * .14);
      const scale = 1 + Math.sin(Math.min(progress * 2.2, 1) * Math.PI) * .045;

      context.save();
      context.globalAlpha = opacity;
      context.translate(center.x, center.y - lift);
      context.scale(scale, scale);
      context.translate(-center.x, -center.y);
      styleInk(context, shape.color, shape.width * (1 + progress * .25));
      context.beginPath();
      context.moveTo(shape.points[0].x, shape.points[0].y);
      shape.points.slice(1).forEach((point) => context.lineTo(point.x, point.y));
      context.stroke();

      context.textAlign = "center";
      context.textBaseline = "bottom";
      context.fillStyle = shape.color;
      context.font = `800 ${Math.max(17, canvas.width / 58)}px Inter, ui-sans-serif, system-ui`;
      context.shadowColor = shape.color;
      context.shadowBlur = 18;
      context.fillText(shape.name.toUpperCase(), center.x, top - Math.max(20, canvas.height * .025));
      context.restore();
    });

    if (floatingShapesRef.current.length) {
      effectsRafRef.current = requestAnimationFrame(animateFloatingShapes);
    } else {
      context.clearRect(0, 0, canvas.width, canvas.height);
      effectsRafRef.current = null;
    }
  }

  function floatRecognizedShape(shape: SnappedShape, stroke: Stroke) {
    floatingShapesRef.current.push({
      ...shape,
      points: shape.points.map((point) => ({ ...point })),
      color: stroke.color,
      width: stroke.width,
      startedAt: performance.now(),
    });
    if (!effectsRafRef.current) effectsRafRef.current = requestAnimationFrame(animateFloatingShapes);
  }

  function clearFloatingShapes() {
    floatingShapesRef.current = [];
    if (effectsRafRef.current) cancelAnimationFrame(effectsRafRef.current);
    effectsRafRef.current = null;
    const canvas = effectsRef.current;
    if (canvas) canvas.getContext("2d")?.clearRect(0, 0, canvas.width, canvas.height);
  }

  function endStroke() {
    const stroke = currentStrokeRef.current;
    if (stroke && snapRef.current) {
      const shape = recognizeStroke(stroke);
      if (shape) {
        const strokeIndex = strokesRef.current.indexOf(stroke);
        if (strokeIndex >= 0) strokesRef.current.splice(strokeIndex, 1);
        redrawInk();
        floatRecognizedShape(shape, stroke);
        setShapeToast(`${shape.name} detected`);
        setHasInk(strokesRef.current.length > 0);
        if (toastTimerRef.current) clearTimeout(toastTimerRef.current);
        toastTimerRef.current = setTimeout(() => setShapeToast(""), 1500);
      }
    }
    currentStrokeRef.current = null;
    lastPointRef.current = null;
  }

  function clearCanvas() {
    strokesRef.current = [];
    endStroke();
    clearFloatingShapes();
    redrawInk();
    setHasInk(false);
  }

  function undo() {
    endStroke();
    strokesRef.current.pop();
    redrawInk();
    setHasInk(strokesRef.current.length > 0);
  }

  function cycleColor(direction: number) {
    setColor((current) => {
      const index = COLORS.indexOf(current);
      return COLORS[(index + direction + COLORS.length) % COLORS.length];
    });
  }

  function savePng() {
    const ink = inkRef.current;
    if (!ink) return;
    const output = document.createElement("canvas");
    output.width = ink.width || 1280;
    output.height = ink.height || 720;
    const context = output.getContext("2d");
    if (!context) return;
    const gradient = context.createLinearGradient(0, 0, output.width, output.height);
    gradient.addColorStop(0, "#060a13");
    gradient.addColorStop(1, "#0d1024");
    context.fillStyle = gradient;
    context.fillRect(0, 0, output.width, output.height);
    context.drawImage(ink, 0, 0);
    const link = document.createElement("a");
    link.download = `airdraw-${new Date().toISOString().slice(0, 10)}.png`;
    link.href = output.toDataURL("image/png");
    link.click();
  }

  function drawHand(landmarks: Array<{ x: number; y: number }>, width: number, height: number, palmProgress: number) {
    const context = overlayRef.current?.getContext("2d");
    if (!context) return;
    const points = landmarks.map((landmark) => ({ x: (1 - landmark.x) * width, y: landmark.y * height }));
    if (skeletonRef.current) {
      context.save();
      context.strokeStyle = "rgba(105, 247, 255, .55)";
      context.lineWidth = Math.max(2, width / 520);
      context.shadowColor = "#69f7ff";
      context.shadowBlur = 8;
      for (const [a, b] of CONNECTIONS) {
        context.beginPath();
        context.moveTo(points[a].x, points[a].y);
        context.lineTo(points[b].x, points[b].y);
        context.stroke();
      }
      context.fillStyle = "#e7fdff";
      points.forEach((point, index) => {
        context.beginPath();
        context.arc(point.x, point.y, index === 8 ? 5 : 3, 0, Math.PI * 2);
        context.fill();
      });
      context.restore();
    }
    const tip = points[8];
    context.save();
    context.strokeStyle = colorRef.current;
    context.lineWidth = 3;
    context.shadowColor = colorRef.current;
    context.shadowBlur = 14;
    context.beginPath();
    context.arc(tip.x, tip.y, 12, 0, Math.PI * 2);
    context.stroke();
    if (palmProgress > 0) {
      context.strokeStyle = "#ffda5c";
      context.lineWidth = 6;
      context.beginPath();
      context.arc(points[9].x, points[9].y, 34, -Math.PI / 2, -Math.PI / 2 + Math.PI * 2 * palmProgress);
      context.stroke();
    }
    context.restore();
  }

  async function startCamera() {
    if (runningRef.current) {
      stopCamera();
      return;
    }
    if (!navigator.mediaDevices?.getUserMedia) {
      setError("This browser cannot access a camera. Try the latest Chrome or Edge.");
      setCamera("error");
      return;
    }
    setCamera("loading");
    setError("");
    try {
      const stream = await navigator.mediaDevices.getUserMedia({
        video: { facingMode: "user", width: { ideal: 1280 }, height: { ideal: 720 } },
        audio: false,
      });
      streamRef.current = stream;
      const video = videoRef.current;
      if (!video) return;
      video.srcObject = stream;
      await video.play();
      prepareCanvases();

      const { FilesetResolver, HandLandmarker } = await import("@mediapipe/tasks-vision");
      const vision = await FilesetResolver.forVisionTasks(
        "https://cdn.jsdelivr.net/npm/@mediapipe/tasks-vision@1.0.1/wasm",
      );
      landmarkerRef.current = await HandLandmarker.createFromOptions(vision, {
        baseOptions: {
          modelAssetPath: "https://storage.googleapis.com/mediapipe-models/hand_landmarker/hand_landmarker/float16/1/hand_landmarker.task",
          delegate: "GPU",
        },
        runningMode: "VIDEO",
        numHands: 1,
        minHandDetectionConfidence: 0.55,
        minHandPresenceConfidence: 0.5,
        minTrackingConfidence: 0.5,
      });

      runningRef.current = true;
      setCamera("ready");
      setGestureSafe("Show your hand");
      lastFrameRef.current = performance.now();
      rafRef.current = requestAnimationFrame(processFrame);
    } catch (reason) {
      const message = reason instanceof DOMException && reason.name === "NotAllowedError"
        ? "Camera permission was blocked. Allow camera access, then try again."
        : "The camera or hand tracker could not start. Refresh and try again.";
      setError(message);
      setCamera("error");
      stopCamera(false);
    }
  }

  function stopCamera(resetState = true) {
    runningRef.current = false;
    if (rafRef.current) cancelAnimationFrame(rafRef.current);
    streamRef.current?.getTracks().forEach((track) => track.stop());
    streamRef.current = null;
    landmarkerRef.current?.close?.();
    landmarkerRef.current = null;
    const overlay = overlayRef.current;
    if (overlay) overlay.getContext("2d")?.clearRect(0, 0, overlay.width, overlay.height);
    endStroke();
    if (resetState) {
      setCamera("idle");
      setGestureSafe("Waiting");
      setFps(0);
    }
  }

  function processFrame(now: number) {
    if (!runningRef.current) return;
    const video = videoRef.current;
    const overlay = overlayRef.current;
    const landmarker = landmarkerRef.current;
    if (!video || !overlay || !landmarker || video.readyState < 2) {
      rafRef.current = requestAnimationFrame(processFrame);
      return;
    }
    prepareCanvases();
    const context = overlay.getContext("2d");
    context?.clearRect(0, 0, overlay.width, overlay.height);

    if (now - lastDetectRef.current >= 30) {
      lastDetectRef.current = now;
      const result = landmarker.detectForVideo(video, now);
      const landmarks = result.landmarks?.[0];
      if (landmarks) {
        const index = fingerUp(landmarks, 8, 6);
        const middle = fingerUp(landmarks, 12, 10);
        const ring = fingerUp(landmarks, 16, 14);
        const pinky = fingerUp(landmarks, 20, 18);
        const openPalm = index && middle && ring && pinky;
        const point = { x: (1 - landmarks[8].x) * overlay.width, y: landmarks[8].y * overlay.height };
        let palmProgress = 0;

        if (openPalm && now > clearCooldownRef.current) {
          palmStartRef.current ??= now;
          palmProgress = Math.min(1, (now - palmStartRef.current) / 700);
          setGestureSafe(palmProgress >= 1 ? "Canvas cleared" : "Hold to clear");
          endStroke();
          if (palmProgress >= 1) {
            clearCanvas();
            clearCooldownRef.current = now + 1400;
            palmStartRef.current = null;
          }
        } else {
          palmStartRef.current = null;
          if (index && !middle) {
            setGestureSafe("Drawing");
            appendPoint(point);
          } else {
            if (currentStrokeRef.current) endStroke();
            setGestureSafe(index && middle ? "Hover / move" : "Hand ready");
          }
        }
        drawHand(landmarks, overlay.width, overlay.height, palmProgress);
      } else {
        palmStartRef.current = null;
        if (currentStrokeRef.current) endStroke();
        setGestureSafe("No hand detected");
      }
    }

    frameCountRef.current += 1;
    if (now - lastFrameRef.current >= 700) {
      setFps(Math.round((frameCountRef.current * 1000) / (now - lastFrameRef.current)));
      frameCountRef.current = 0;
      lastFrameRef.current = now;
    }
    rafRef.current = requestAnimationFrame(processFrame);
  }

  function canvasPoint(event: React.PointerEvent<HTMLCanvasElement>) {
    const canvas = overlayRef.current!;
    const box = canvas.getBoundingClientRect();
    return {
      x: ((event.clientX - box.left) / box.width) * canvas.width,
      y: ((event.clientY - box.top) / box.height) * canvas.height,
    };
  }

  function pointerDown(event: React.PointerEvent<HTMLCanvasElement>) {
    event.currentTarget.setPointerCapture(event.pointerId);
    pointerDrawingRef.current = true;
    beginStroke(canvasPoint(event));
    setGestureSafe("Pointer drawing");
  }

  function pointerMove(event: React.PointerEvent<HTMLCanvasElement>) {
    if (pointerDrawingRef.current) appendPoint(canvasPoint(event));
  }

  function pointerUp() {
    pointerDrawingRef.current = false;
    endStroke();
    setGestureSafe(camera === "ready" ? "Hand ready" : "Waiting");
  }

  return (
    <main className="app-shell">
      <header className="topbar">
        <a className="brand" href="#studio" aria-label="AirDraw Studio home">
          <span className="brand-mark" aria-hidden="true"><i /><i /><i /></span>
          <span><strong>AirDraw</strong><small>STUDIO</small></span>
        </a>
        <div className="top-meta">
          <span className="privacy"><span>●</span> Camera stays in your browser</span>
          <button className="icon-button" onClick={() => setHelpOpen(true)} aria-label="Open gesture guide">?</button>
        </div>
      </header>

      <section className="workspace" id="studio">
        <div className="intro-row">
          <div><p className="eyebrow">GESTURE-POWERED CANVAS</p><h1>Your hand is the brush.</h1></div>
          <p className="intro-copy">Raise one finger to draw. Raise two to move. Hold an open palm to clear.</p>
        </div>

        <div className="stage-card">
          <div className="stage" data-camera={camera}>
            <video ref={videoRef} playsInline muted className="camera-feed" />
            <div className="stage-grid" aria-hidden="true" />
            <canvas ref={inkRef} className="ink-canvas" />
            <canvas ref={effectsRef} className="effects-canvas" />
            <canvas ref={overlayRef} className="overlay-canvas" onPointerDown={pointerDown} onPointerMove={pointerMove} onPointerUp={pointerUp} onPointerCancel={pointerUp} />

            {camera !== "ready" && (
              <div className="camera-gate">
                <div className="hand-orbit" aria-hidden="true"><span className="orbit-ring" /><span className="cursor-dot" /><span className="air-stroke" /></div>
                <p className="gate-kicker">NO INSTALLS • NO UPLOADS</p>
                <h2>{camera === "loading" ? "Warming up the hand tracker…" : "Draw without touching the screen"}</h2>
                <p>{camera === "error" ? error : "Allow camera access once. Your video never leaves this device."}</p>
                <button className="primary-button" onClick={startCamera} disabled={camera === "loading"}>
                  <span className="camera-icon" aria-hidden="true" />
                  {camera === "loading" ? "Starting…" : camera === "error" ? "Try camera again" : "Start camera"}
                </button>
                <span className="pointer-note">You can also draw with a mouse or touch.</span>
              </div>
            )}

            {hud && <><div className="status-chip"><span className={camera === "ready" ? "live-dot" : "idle-dot"} />{gesture}</div><div className="fps-chip">{camera === "ready" ? `${fps} FPS` : "READY"}</div></>}
            {shapeToast && <div className="shape-toast">✦ {shapeToast}</div>}
            <div className="corner corner-tl" aria-hidden="true" /><div className="corner corner-tr" aria-hidden="true" />
            <div className="corner corner-bl" aria-hidden="true" /><div className="corner corner-br" aria-hidden="true" />
          </div>

          <div className="toolbar" aria-label="Drawing tools">
            <button className={`camera-toggle ${camera === "ready" ? "active" : ""}`} onClick={startCamera}><span className="camera-icon" aria-hidden="true" /> {camera === "ready" ? "Stop" : "Camera"}</button>
            <span className="divider" />
            <div className="tool-group color-group" aria-label="Ink color">
              <span className="tool-label">INK</span>
              {COLORS.map((swatch) => <button key={swatch} className={`swatch ${color === swatch ? "selected" : ""}`} style={{ "--swatch": swatch } as React.CSSProperties} onClick={() => setColor(swatch)} aria-label={`Use ${swatch} ink`} />)}
            </div>
            <span className="divider responsive-hide" />
            <label className="brush-control"><span className="tool-label">BRUSH <b>{brush}</b></span><input type="range" min="2" max="24" value={brush} onChange={(event) => setBrush(Number(event.target.value))} /></label>
            <span className="divider" />
            <button className={`tool-button ${glow ? "active" : ""}`} onClick={() => setGlow((value) => !value)} title="Glow (G)">✦ <span>Glow</span></button>
            <button className={`tool-button ${skeleton ? "active" : ""}`} onClick={() => setSkeleton((value) => !value)} title="Hand skeleton (L)">◇ <span>Hand</span></button>
            <button className={`tool-button ${snap ? "active" : ""}`} onClick={() => setSnap((value) => !value)} title="Smart shape snapping">⬡ <span>Snap</span></button>
            <button className="tool-button" onClick={undo} disabled={!hasInk} title="Undo (Z)">↶ <span>Undo</span></button>
            <button className="tool-button danger" onClick={clearCanvas} disabled={!hasInk} title="Clear (C)">× <span>Clear</span></button>
            <button className="save-button" onClick={savePng} disabled={!hasInk}>↓ <span>Save PNG</span></button>
          </div>
        </div>

        <div className="gesture-strip">
          <div><span className="gesture-number">01</span><span className="gesture-symbol one">│</span><p><strong>One finger</strong><small>Draw</small></p></div>
          <div><span className="gesture-number">02</span><span className="gesture-symbol two">Ⅱ</span><p><strong>Two fingers</strong><small>Move without drawing</small></p></div>
          <div><span className="gesture-number">03</span><span className="gesture-symbol palm">✋</span><p><strong>Open palm</strong><small>Hold to clear</small></p></div>
          <button onClick={() => setHelpOpen(true)}>View all controls <span>→</span></button>
        </div>
      </section>

      <footer><span>AirDraw Studio</span><p>Private by design. Powered on-device.</p><button onClick={() => setHud((value) => !value)}>HUD {hud ? "ON" : "OFF"}</button></footer>

      {helpOpen && (
        <div className="modal-backdrop" role="presentation" onMouseDown={() => setHelpOpen(false)}>
          <section className="help-modal" role="dialog" aria-modal="true" aria-labelledby="help-title" onMouseDown={(event) => event.stopPropagation()}>
            <button className="modal-close" onClick={() => setHelpOpen(false)} aria-label="Close">×</button>
            <p className="eyebrow">QUICK GUIDE</p><h2 id="help-title">Draw naturally in the air.</h2>
            <div className="help-grid">
              <article><b>☝</b><strong>Draw</strong><p>Point your index finger up and fold the others.</p></article>
              <article><b>✌</b><strong>Hover</strong><p>Raise index and middle fingers to move safely.</p></article>
              <article><b>✋</b><strong>Clear</strong><p>Hold an open palm until the gold ring completes.</p></article>
            </div>
            <div className="shortcuts"><span><kbd>C</kbd> Clear</span><span><kbd>Z</kbd> Undo</span><span><kbd>S</kbd> Save</span><span><kbd>G</kbd> Glow</span><span><kbd>L</kbd> Hand</span><span><kbd>[ ]</kbd> Colour</span></div>
            <button className="primary-button" onClick={() => setHelpOpen(false)}>Got it</button>
          </section>
        </div>
      )}
    </main>
  );
}
