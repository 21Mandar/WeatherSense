# Data Sources & Attribution

All video footage used in WeatherSense is sourced from [Pexels](https://www.pexels.com/) under the [Pexels License](https://www.pexels.com/license/) (free for personal and commercial use, attribution not required).

Attribution is documented here voluntarily for **traceability, reproducibility, and to credit the original creators** whose work made this project possible.

---

## Clean source clips (`data/raw/`)

Used as clean baselines for synthetic weather augmentation.

| File | Scene | Resolution | Contributor | Source |
|---|---|---|---|---|
| `clip_1.mp4` | US arterial road, daytime, light traffic | 4K | Jabriel | [TODO: Pexels URL] |
| `clip_2.mp4` | US arterial intersection with overpass, multiple vehicles | 4K | Jabriel | [TODO: Pexels URL] |
| `clip_3.mp4` | Indian highway with truck, rural, daytime | 1080p | K | [TODO: Pexels URL] |
| `clip_4.mp4` | US suburban road with STOP sign | 720p | K | [TODO: Pexels URL] |
| `clip_5.mp4` | Portland highway, bright cloudy sky, urban infrastructure | 1080p | Altaf Shah | [TODO: Pexels URL] |

---

## Real-weather reference clips (`data/real_weather/`)

Used as visual baselines for validating synthetic augmentations. Not used for quantitative benchmarking.

| File | Scene | Contributor | Source |
|---|---|---|---|
| `real_rain.mp4` | Highway with visible rain streaks, wet pavement | Kim Dodge | [TODO: Pexels URL] |
| `real_fog.mp4` | Curved highway with autumn trees, fading into white | Talha Uğuz | [TODO: Pexels URL] |
| `real_night.mp4` | London highway, streetlights, taillights, sensor grain | Altaf Shah | [TODO: Pexels URL] |
| `real_motion_blur.mp4` | Country highway, stylized motion blur (center sharp, sides streaking) | Dimitar Germanov | [TODO: Pexels URL] |

---

## Benchmark dataset (Phase 2)

For quantitative mAP benchmarking, this project will use the **BDD100K validation split** (10,000 labeled frames with ground-truth bounding boxes), accessed via Hugging Face:

- **Dataset**: `dgural/bdd100k`
- **License**: BSD 3-Clause (research use, see [BDD100K license](https://bdd-data.berkeley.edu/))
- **Citation**: Yu, F., Chen, H., Wang, X., Xian, W., Chen, Y., Liu, F., Madhavan, V., & Darrell, T. (2020). *BDD100K: A Diverse Driving Dataset for Heterogeneous Multitask Learning*. CVPR.

---

## How to fill in the remaining TODOs

For each Pexels video, only the source URL is still missing:

1. Go to the original Pexels page where you downloaded the clip
2. Copy the URL (format: `https://www.pexels.com/video/some-description-12345678/`)
3. Replace the corresponding `[TODO: Pexels URL]` placeholder above

This file is not blocking for the README or the repo — URLs can be filled in via a follow-up commit at any time.
