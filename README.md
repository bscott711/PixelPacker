# PixelPacker  
[![Python Version](https://img.shields.io/pypi/pyversions/pixelpacker)](https://pypi.org/project/pixelpacker/) [![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)

**Automate your 3D TIFF→WebP pipeline** for fast, web‑friendly volumetric viewing. 

**Achieve ~100X file‑size reduction (e.g. 75 MB → 600 KB).**

---

## 🚀 Quick Start

1. **Clone & enter repo**  
   ```bash
   git clone https://github.com/bscott711/PixelPacker.git
   cd PixelPacker
   ```
2. **Create & activate virtualenv**  
   ```bash
   uv venv
   source .venv/bin/activate
   ```
3. **Install in editable mode**  
   ```bash
   uv pip install -e .
   ```

---

## 🎨 Features

- **TIFF Volume Extraction**  
  Handles 2D, 3D, 4D, 5D TIFF stacks (with singleton dims).
- **Flexible Contrast Stretching**  
  Choose your stretch mode:  
  - `smart` (dynamic histogram, default)  
  - `smart-late` (alternative dynamic)  
  - `imagej-auto` (ImageJ‑style percentiles)  
  - `max` (min→max linear)

  **Example Debug Histogram (`smart-late` mode):**
        ![Example Debug Histogram Plot](./debug_hist_example.png "PixelPacker Debug Histogram")

- **Global Contrast**  
  `--global-contrast` for consistent brightness/contrast across all timepoints.
- **2D Atlas Tiling**  
  Packs Z‑slices into an optimal grid layout.
- **WebP Compression**  
  Achieve ~×100 file‑size reduction (e.g. 75 MB → 0.75 MB).
- **Metadata Manifest**  
  Auto‑generates `manifest.json` with dimensions, tile layout, channels, timepoints, and contrast bounds.
- **Parallel Processing**  
  `--threads` to speed up batch jobs.
- **Debug Mode**  
  `--debug` saves histograms & preview slices.
- **Easy CLI**  
  Clear flags and built‑in `--help`.

---

## 💻 Usage

```bash
python -m pixelpacker.cli   --input  <input_tiff_folder>   --output <output_volume_folder>   [OPTIONS]
```

### Examples

- **Default (smart contrast)**  
  ```bash
  python -m pixelpacker.cli     --input ./Input_TIFFs     --output ./volumes
  ```
- **ImageJ contrast + 8 threads + debug**  
  ```bash
  python -m pixelpacker.cli     --input /path/to/tiffs     --output /path/to/web_volumes     --stretch imagej-auto     --threads 8     --debug
  ```
- **Global contrast across timepoints**  
  ```bash
  python -m pixelpacker.cli     --input ./Input_TIFFs     --output ./volumes_global     --global-contrast
  ```

---

## ⚙️ CLI Options

| Flag                   | Description                                                                                          | Default        |
|------------------------|------------------------------------------------------------------------------------------------------|----------------|
| `--input <folder>`     | Input directory of TIFF stacks                                                                      | `./Input_TIFFs`|
| `--output <folder>`    | Output directory for WebP atlases + manifest                                                        | `./volumes`    |
| `--stretch <mode>`     | Contrast mode: `smart`│`smart-late`│`imagej-auto`│`max`                                                | `smart`        |
| `--global-contrast`    | Two‑pass analysis for consistent contrast across timepoints                                          | off            |
| `--threads <n>`        | Number of worker threads                                                                            | `8`            |
| `--dry-run`            | Simulate without reading/writing image files                                                         | off            |
| `--debug`              | Save intermediate histograms & preview slices                                                        | off            |
| `-h`, `--help`         | Show help message                                                                                   | —              |
| `--version`            | Show installed version                                                                              | —              |

> **Tip:** If you see module name issues, run  
> `python -m tiff_preprocessor.cli …`

---

## 📂 Input / Output Formats

**Input filenames** must match:  
```
[prefix]_ch[channel]_stack[timepoint][suffix].tif
```
- e.g. `exp1_runA_ch0_stack0005_decon.tif`

**Output** (`<output_volume_folder>`):  
- **WebP atlases**  
  ```
  volume_[timepoint]_c[channel].webp
  ```
  A 2D grid of Z‑slices in 8‑bit WebP format.  
- **Manifest** `manifest.json`  
  ```json
  {
    "tile_layout": { "cols": X, "rows": Y },
    "volume_size": { "width": W, "height": H, "depth": Z },
    "channels": C,
    "timepoints": [
      {
        "time": "stack0005",
        "files": {
          "c0": {
            "file": "volume_stack0005_c0.webp",
            "p_low": 0.01,
            "p_high": 0.99
          }
        }
      }
    ],
    "global_intensity": {
      "c0": { "p_low": 0.01, "p_high": 0.99 }
    }
  }
  ```
---

## 🔗 Dependencies

- Python ≥ 3.10  
- numpy  
- tifffile  
- Pillow  
- scipy  
- matplotlib  
- docopt  
- tqdm  

See [pyproject.toml](pyproject.toml) for versions.

---

## 📄 License

MIT — see [LICENSE](LICENSE)

---

## 🐛 Issues

Please report bugs & feature requests on [GitHub Issues](https://github.com/bscott711/PixelPacker/issues).
