# Air-Draw

Air-Draw turns your webcam into a virtual drawing canvas. Move your index
finger through the air to draw neon strokes. Recognised shapes snap into clean
geometry, flash, float upward and fade away.

## Features

- Real-time hand tracking through MediaPipe
- Smooth fingertip motion using a One Euro filter
- Threaded webcam capture and hand inference
- Neon drawing, configurable colours, brush sizes and glow
- Undo, clear and PNG export
- Automatic recognition of lines, triangles, squares, rectangles, circles,
  ellipses, pentagons, hexagons, stars and arrows
- Built-in self-test, animation demo and performance benchmark

## Requirements

- Windows 10 or Windows 11
- Python 3.11 recommended
- A working webcam

The project intentionally uses `mediapipe==0.10.21`. Newer MediaPipe releases
removed the legacy `mp.solutions.hands` API used by this program.

## Installation on Windows

Put these files in the same folder:

```text
hand_draw.py
requirements.txt
README.md
```

Open that folder in File Explorer, click the address bar, type `cmd`, and press
Enter.

### 1. Create a separate environment

Using a separate environment prevents this project's MediaPipe and OpenCV
versions from interfering with other computer-vision projects.

```powershell
py -3.11 -m venv .venv
```

If `py -3.11` is unavailable but your installed Python is version 3.11, use:

```powershell
py -m venv .venv
```

### 2. Activate it

In Command Prompt:

```powershell
.venv\Scripts\activate
```

The command line should now begin with `(.venv)`.

### 3. Install the packages

```powershell
python -m pip install --upgrade pip
python -m pip install -r requirements.txt
```

### 4. Start Air-Draw

```powershell
python hand_draw.py
```

Allow camera access if Windows asks for permission.

## Hand gestures

| Gesture | Action |
| --- | --- |
| Index finger up, middle finger down | Draw |
| Index and middle fingers up | Move without drawing |
| Open palm held for about 0.5 seconds | Clear the canvas |

## Keyboard controls

| Key | Action |
| --- | --- |
| `Q` or `Esc` | Quit |
| `C` | Clear the canvas |
| `Z` | Undo the last stroke |
| `[` / `]` | Previous / next colour |
| `-` / `=` | Decrease / increase brush size |
| `S` | Save the canvas as a PNG |
| `H` | Show or hide the HUD |
| `L` | Show or hide the hand skeleton |
| `G` | Enable or disable glow |

## Performance presets

The default settings are suitable for an ordinary laptop:

```powershell
python hand_draw.py
```

For a thin or slower laptop:

```powershell
python hand_draw.py --infer-width 256 --max-fps 30 --infer-fps 20
```

For better tracking accuracy when plugged in:

```powershell
python hand_draw.py --infer-width 384 --complexity 1 --max-fps 0
```

If glow reduces performance:

```powershell
python hand_draw.py --no-glow
```

If the wrong camera opens, try another camera index:

```powershell
python hand_draw.py --camera 1
```

## Restrict shape recognition

To recognise only selected shapes:

```powershell
python hand_draw.py --shapes line,circle,square,triangle
```

Supported names are:

```text
line, triangle, square, rectangle, circle, ellipse,
pentagon, hexagon, star, arrow
```

## Testing without a webcam

Test the shape recogniser:

```powershell
python hand_draw.py --selftest
```

Generate an animation demo:

```powershell
python hand_draw.py --demo demo.mp4
```

Generate a PNG contact sheet:

```powershell
python hand_draw.py --demo demo.png
```

Benchmark the rendering pipeline:

```powershell
python hand_draw.py --bench
```

## Troubleshooting

### `mediapipe` has no attribute `solutions`

First confirm that `(.venv)` appears at the beginning of the command line.
Then reinstall the compatible version inside that environment:

```powershell
python -m pip uninstall mediapipe -y
python -m pip install --force-reinstall mediapipe==0.10.21
```

Verify the installation:

```powershell
python -c "import mediapipe as mp; print(mp.__version__); print(hasattr(mp, 'solutions'))"
```

The result should end with:

```text
0.10.21
True
```

### No compatible MediaPipe version is found

Check the Python version:

```powershell
py --version
```

Install 64-bit Python 3.11, recreate `.venv`, activate it, and install the
requirements again.

### The camera does not open

1. Close applications that may already be using the webcam.
2. Enable camera permission under Windows Settings.
3. Try `python hand_draw.py --camera 1`.

### Tracking is slow

Use the slower-laptop preset:

```powershell
python hand_draw.py --infer-width 256 --max-fps 30 --infer-fps 20 --no-glow
```

### Exit the environment

When finished, close Air-Draw and run:

```powershell
deactivate
```
