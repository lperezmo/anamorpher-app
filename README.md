<div align="center">
  <h1>Anamorpher Studio</h1>
  <p>Generate adversarial images that reveal hidden text when downscaled 4x.<br>Exploits how different interpolation algorithms sample pixels during downscaling.</p>

  <a href="https://www.python.org/"><img src="https://img.shields.io/badge/python-%E2%89%A53.11-blue" alt="Python >= 3.11"></a>
  <a href="https://github.com/lperezmo/anamorpher-app/blob/main/LICENSE"><img src="https://img.shields.io/github/license/lperezmo/anamorpher-app" alt="License"></a>
  <br>
  <a href="https://anamorpher.streamlit.app/"><img src="https://static.streamlit.io/badges/streamlit_badge_black_white.svg" alt="Open in Streamlit"></a>
</div>

---

<div align="center">
  <img src="images/example.png" alt="Gemini detecting hidden text in adversarial image" width="400">
  <p><em>Gemini detected the hidden prompt injection embedded in the cat photo.</em></p>
</div>

## How it works

The app creates full-resolution images that look normal to the eye but reveal hidden text when downscaled by a factor of 4. Three different interpolation methods are supported, each exploiting a different downscaling kernel:

| Method | Kernel | Samples per 4x4 block | Best for |
|--------|--------|----------------------|----------|
| **Bicubic** | Catmull-Rom cubic (4x4) | 16 | Natural photos |
| **Bilinear** | Linear triangle (2x2 center) | 4 | Balanced quality |
| **Nearest Neighbor** | Single pixel | 1 | Sharp text edges |

The generators modify only the **red channel** of the decoy image, solving a least-squares optimization per 4x4 block so that the specific downscaling kernel produces the target text pattern. A multi-library comparison view shows how PIL, OpenCV, and other libraries each interpret the adversarial image differently.

## Installation

```bash
uv sync
```

## Usage

```bash
streamlit run app.py
```

Then in the browser:

1. **Select or upload** a decoy image (sample images are included)
2. **Enter hidden text** to embed
3. **Choose an interpolation method** (bicubic, bilinear, or nearest)
4. **Tweak parameters** in the advanced popover (lambda, epsilon, gamma, dark fraction)
5. **Generate** and check the "Hidden Text Revealed" tab to verify

## Generator parameters

| Parameter | Effect |
|-----------|--------|
| **Lambda** | Mean-preservation weight. Higher = less visible artifacts but weaker embedding |
| **Epsilon** | Null-space dither magnitude for robustness to noise |
| **Gamma** | Gamma correction applied to target before fitting (1.0 = none) |
| **Dark Fraction** | Fraction of luma range where edits are allowed (bicubic/bilinear only) |
| **Offset** | Which pixel in 4x4 block to sample, 0-3 (nearest-neighbor only) |
| **Font Size** | Auto-sizes by default; can be set manually up to 500px |

## Project structure

```
anamorpher-app/
├── app.py                          # Streamlit UI + orchestration
├── adversarial_generators/
│   ├── bicubic_gen_payload.py      # Catmull-Rom cubic solver
│   ├── bilinear_gen_payload.py     # 2x2 bilinear solver (OpenCV)
│   ├── nearest_gen_payload.py      # Nearest-neighbor closed-form solver
│   └── decoy_images/               # Sample decoy images
├── .streamlit/
│   └── config.toml                 # Light/dark theme (indigo/slate)
├── pyproject.toml
└── uv.lock
```

## References

- Based on the [anamorpher](https://github.com/lperezmo/anamorpher) project
- Research: *Weaponizing image scaling against production AI systems* (Trail of Bits, 2025)
- Related: USENIX Security papers on image scaling attacks (2019-2020)
