# WeatherSense
 
**Stress-testing object detection models under adverse driving conditions.**
 
WeatherSense synthesizes rain, fog, low-light, and motion blur on clean dashcam footage, then measures how much each condition degrades YOLO's per-class detection accuracy. The goal is a deployable tool that quantifies — not just visualizes — where a perception model breaks.
 
![All four augmentations applied to a US arterial dashcam frame](docs/images/clip_2_all_augmentations.jpg)
*Clean baseline (top-left) with rain, fog, night, and motion blur applied at medium intensity. Same scene; same objects; four failure modes.*
 
---
 
## Status
 
**Phase 1 complete.** Augmentation pipelines built, tuned, and visually validated against real-world reference clips.
 
| Phase | Focus | Status |
|---|---|---|
| 1 | Data pipeline + 4 weather augmentations | ✅ Complete |
| 2 | YOLOv8 inference + BDD100K mAP benchmarking | In progress |
| 3 | FastAPI service + degradation report visualization | Planned |
| 4 | MLflow tracking + Docker + AWS deployment | Planned |
| 5 | CI/CD + documentation polish | Planned |
 
---
 
## Why this exists
 
YOLO benchmarks are reported on clean, well-lit, daytime images. In production — autonomous vehicles, traffic cameras, dashcam ADAS — those conditions don't hold. A model that gets 92% mAP on COCO might drop to 60% in fog and *nobody knows by how much until something goes wrong on the road*.
 
WeatherSense closes that gap by:
 
1. **Synthesizing controlled adverse conditions** on clean footage where ground truth is known.
2. **Running detection inference** on clean and degraded frames side-by-side.
3. **Quantifying per-class accuracy loss** — so you know whether your model fails first on pedestrians, traffic signs, or distant vehicles.
---
 
## Weather augmentations
 
Four conditions, each at three intensity levels (light / medium / heavy).
 
| Condition | Approach | Models |
|---|---|---|
| **Rain** | Albumentations `RandomRain` with tuned drop geometry | Streaks, blur, brightness reduction |
| **Fog** | Custom 3-stage pipeline: patchy fog + uniform haze + contrast reduction + desaturation | Atmospheric scattering, mid-distance visibility loss |
| **Night** | Brightness scaling + gamma correction + blue shift + Gaussian noise | Low-light / civil twilight (see Limitations) |
| **Motion Blur** | Directional kernel convolution via `cv2.filter2D` | Camera shake, lateral motion |
 
### Real vs. synthetic validation
 
Each augmentation was validated against a real-world Pexels reference clip with a categorically matching scene. Synthetic intensities were tuned to match the severity of the real reference.
 
![Real vs. synthetic comparison across all four weather conditions](docs/images/all_conditions_comparison.jpg)
*Left column: real-world Pexels reference clips. Right column: WeatherSense synthetic augmentations applied to scene-matched clean dashcam frames. Fog and rain achieve strong perceptual realism; night and motion blur are first-order approximations with documented limitations below.*
 
---
 
## Project structure
 
~~~
WeatherSense/
├── data/
│   ├── raw/                    # Clean dashcam clips (5 scenes, mixed resolution)
│   └── real_weather/           # Real-world reference clips for validation
├── notebooks/
│   ├── data_check.py           # Frame extraction sanity check
│   ├── test_rain.py            # Per-condition 2×2 intensity grids
│   ├── test_fog.py
│   ├── test_night.py
│   ├── test_motion_blur.py
│   ├── build_unified_grid.py   # All-augmentations grid per clip
│   └── build_real_vs_synthetic.py
├── src/
│   ├── augmentations/
│   │   ├── rain.py             # Albumentations-based
│   │   ├── fog.py              # Custom multi-stage pipeline
│   │   ├── night.py            # Brightness + gamma + chromatic shift
│   │   └── motion_blur.py      # Directional kernel
│   └── utils/
│       └── frame_io.py         # Frame loading, resize to 1280×720
├── outputs/
│   ├── aug_rain/, aug_fog/, aug_night/, aug_motion_blur/
│   ├── aug_unified/            # Per-clip combined grids
│   └── real_vs_synthetic/      # Side-by-side validation
├── docs/images/                # README assets
├── requirements.txt
└── README.md
~~~
 
---
 
## Setup
 
Python 3.12 recommended.
 
~~~bash
git clone https://github.com/21Mandar/WeatherSense.git
cd WeatherSense
 
python3.12 -m venv venv
source venv/bin/activate          # macOS / Linux
# .\venv\Scripts\activate         # Windows
 
pip install -r requirements.txt
~~~
 
### Reproduce the visuals
 
~~~bash
# Validate each augmentation individually (produces 2×2 intensity grids)
python notebooks/test_rain.py
python notebooks/test_fog.py
python notebooks/test_night.py
python notebooks/test_motion_blur.py
 
# Combined grid: all four augmentations on each clean clip
python notebooks/build_unified_grid.py
 
# Real-vs-synthetic comparison against Pexels reference clips
python notebooks/build_real_vs_synthetic.py
~~~
 
All outputs land in `outputs/`.
 
---
 
## Dataset
 
**Clean source clips** — 5 Pexels CC0 dashcam clips spanning US arterials, US suburbs, Indian highways, and varied lighting. Mixed resolutions (720p–4K) downsampled on-the-fly to 1280×720; originals untouched.
 
**Real-weather reference clips** — 4 Pexels CC0 clips selected as visual baselines for synthesis validation (not for quantitative benchmarking).
 
**Benchmark dataset (Phase 2)** — Will use the BDD100K validation split (10K labeled frames with ground-truth bounding boxes) via Hugging Face `dgural/bdd100k`, since computing detection mAP requires labeled ground truth that Pexels clips do not provide.
 
---
 
## Limitations
 
Documented honestly rather than buried. WeatherSense is a perception-degradation testing tool, not a photorealistic weather generator. Specifically:
 
- **Night augmentation models civil twilight / low-light**, not deep urban night with active artificial lighting. Generating realistic streetlights, taillight glare, and headlight cones from daytime source frames would require depth-aware light placement or generative models, both out of scope for v1.
- **Fog is depth-uniform**, not depth-dependent. Real fog density scales with distance; WeatherSense applies a uniform haze layer. A depth-aware version would require a monocular depth model in the pipeline.
- **Motion blur is uniform-directional**, not optical-flow-derived. Real motion blur varies by image region (foreground/background, center/edge). The current kernel applies the same blur everywhere.
- **Rain models streaks but not windshield droplets or wet-pavement reflections.** Streaks alone are sufficient to produce detection degradation in published studies, but the visual gap to real heavy rain is noticeable.
These limitations are deliberate scope choices. Phase 2's quantitative results will determine whether these first-order approximations are sufficient to surface meaningful YOLO failure modes — which is the actual research question.
 
---
 
## Tech stack
 
OpenCV 4.10 · Albumentations 1.4 · Ultralytics YOLOv8 · NumPy · Matplotlib · Python 3.12
 
Later phases will add: FastAPI · MLflow · Docker · AWS (EC2/S3) · GitHub Actions
 
---
 
## License
 
MIT — see `LICENSE`.
 
Source dashcam footage and real-weather reference clips are from Pexels under the Pexels License (free commercial use, no attribution required; attribution provided in `data/SOURCES.md` for traceability).
