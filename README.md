# StepUp

StepUp is a Doodle‑Jump‑style platformer built with Pygame that blends classic jump‑up gameplay with health‑themed metrics. It supports:

- Multiple platform types (normal, moving, fragile).
- Coins, score, and height tracking.
- Per‑user accounts with persistent stats (highscore, lifetime calories, minutes played).
- A full menu system with sign‑in and health info screens.
- CV‑driven input and avatar capture using YOLO + SAM.

---

## Quick Start

### 1) Create the Python environment

If you only want to run the game (no CV capture):

```bash
python -m venv .venv
source .venv/bin/activate
pip install pygame
```

If you want the CV pipeline (YOLO + SAM avatar capture):

```bash
conda env create -f environment.yaml
conda activate yolo-sam
pip install pygame
```

### 2) Run the game

```bash
python game.py
```

---

## Controls

### Main Menu
- Arrow keys: select menu options
- Enter/Space: activate selection
- Mouse: hover/click to select

### In‑Game
- CV input (default): movement and jump driven by camera
- Settings menu: change intensity, music/sfx, etc.

### Pause Menu
- Arrow keys or mouse: change selections
- Enter/Space: activate
- Esc: toggle pause

---

## Gameplay Notes

- **Score** = `height * 2 + coins * 100`
- **Calories** are based on intensity:
  - Low: 5 / min
  - Medium: 8 / min
  - High: 10 / min
- Calories only count while the game is active (player in control).
- Platform fading speed depends on intensity.

---

## Account & Health Metrics

Each user account stores:
- Highscore
- Lifetime calories
- Minutes played
- Balance ability
- Jump height metrics
- Shuffle speed metrics

Sign in on the main menu, then view **Health Info** for full stats.

---

## CV / Avatar Capture (YOLO + SAM)

The capture pipeline uses `yolo_sam_segment.py` to generate sprite images from a webcam.

### Run capture directly

```bash
python yolo_sam_segment.py --capture-poses --webcam-index 2
```

### Notes
- Set your webcam index with `--webcam-index`.
- The CV capture runs fullscreen and may pause game input while active.

---

## Project Structure

```
.
├─ game.py                  # Entry point
├─ game_core.py             # Main game loop and rendering
├─ entities.py              # Player + platform classes
├─ collisions.py            # Collision handling
├─ coins.py                 # Coin spawning and animation
├─ platforms.py             # Platform generation logic
├─ config.py                # Game settings/constants
├─ yolo_sam_segment.py      # Avatar capture pipeline
├─ cv_tester.py             # CV debug/test harness
├─ assets/                  # Art, audio, fonts
└─ users.json               # Persistent user data
```

---

## Troubleshooting

**No music or audio**
- Ensure `assets/Sounds` exists and files are present.
- Verify your audio device isn’t muted.

**Webcam not detected**
- Use `--webcam-index` to select the correct device.
- On Linux, check `/dev/video*` devices.

**CV capture fails**
- Make sure CUDA / torch / torchvision versions match your environment.

---

## Credits

- Pygame
- Ultralytics (YOLO + SAM)
- Asset packs in `assets/`

