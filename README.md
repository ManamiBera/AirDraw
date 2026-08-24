# AirDraw Studio

AirDraw Studio is a browser-based version of the original desktop hand-drawing program. It uses the webcam and MediaPipe hand landmarks directly in the browser, so the camera stream is not uploaded to a server.

## Run it on Windows

1. Install Node.js 22 LTS from <https://nodejs.org/>.
2. Extract this project and open its folder in PowerShell or Command Prompt.
3. Install the packages:

   ```bash
   npm install
   ```

4. Start the local website:

   ```bash
   npm run dev
   ```

5. Open the address shown in the terminal, then click **Start camera** and allow camera access.

## Deploy on Vercel

1. Upload this folder to a GitHub repository, or import the folder with the Vercel CLI.
2. In Vercel, click **Add New → Project** and select the repository.
3. Keep **Framework Preset: Next.js**, leave **Root Directory** at the repository root, and click **Deploy**.

The included `package.json`, `app/page.tsx`, and `vercel.json` give Vercel a web entrypoint. Do not upload `hand_draw.py` by itself; it is a desktop OpenCV program and cannot access a visitor's browser camera on Vercel.

## Gestures and controls

- Index finger up: draw
- Index and middle fingers up: hover without drawing
- Open palm held for about 0.7 seconds: clear
- Mouse or touch: draw without starting the camera
- `C`: clear, `Z`: undo, `S`: save PNG
- `G`: glow, `L`: hand skeleton, `H`: HUD
- `[` / `]`: change colour, `-` / `+`: change brush size

Smart Snap cleans up lines, circles, ellipses, triangles, quadrilaterals, pentagons, hexagons, and stars. A recognized shape displays its name, floats upward, and fades away. Turn Smart Snap off from the toolbar for fully freehand drawing that remains on the canvas.

## Notes

- Use HTTPS in production. Browsers generally require a secure page before allowing webcam access.
- Chrome or Edge is recommended for the best webcam and WebGL performance.
- This web version does not use Python or `requirements.txt`; dependencies are listed in `package.json`.
