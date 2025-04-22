![Banner](Banner.png)
---

[![PixelPacker CI](https://github.com/bscott711/PixelPacker/actions/workflows/ci.yml/badge.svg)](https://github.com/bscott711/PixelPacker/actions/workflows/ci.yml)
[![PyPI version](https://img.shields.io/pypi/v/pixelpacker.svg)](https://pypi.org/project/pixelpacker/)
[![Python Version](https://img.shields.io/pypi/pyversions/pixelpacker)](https://pypi.org/project/pixelpacker/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)
[![Code style: ruff](https://img.shields.io/endpoint?url=https://raw.githubusercontent.com/astral-sh/ruff/main/assets/badge/v2.json)](https://github.com/astral-sh/ruff)

## Automate your 3D TIFF→WebP pipeline for fast, web‑friendly volumetric viewing.  
### Achieve >100X file‑size reduction (e.g. 75 MB → 750 KB).  

## 🚀 **Quick Start**

### Recommended Installation (PyPI)

Install from PyPI:  
```bash
pip install pixelpacker
```  
*(Ensure you have Python ≥ 3.10 and pip installed.)*

### Development Installation (from Source)

Clone & enter repo:  
```bash
git clone https://github.com/bscott711/PixelPacker.git
cd PixelPacker
```

Create & activate virtualenv (Requires uv or use your preferred tool like venv or conda):  
```bash
uv venv
source .venv/bin/activate
# Or: python -m venv .venv && source .venv/bin/activate
```

Install in editable mode:  
```bash
uv pip install -e .
# Or: pip install -e .
```

🎨 **Features**

- **TIFF Volume Extraction**: Handles 2D, 3D, 4D, 5D TIFF stacks (with singleton dimensions automatically squeezed)  
- **Flexible Contrast Stretching**: Choose your stretch mode:  
  - `smart` (dynamic histogram, default)  
  - `smart-late` (alternative dynamic)  
  - `imagej-auto` (ImageJ‑style percentiles)  
  - `max` (min→max linear)  
- **Global vs. Per-Image Contrast**: Use `--global-contrast` (default) for consistent brightness across timepoints, or `--per-image-contrast` for individually optimized frames  
- **Automatic Z-Cropping**: Removes empty Z‑slices using slope analysis (default) or a simple threshold method  
- **2D Atlas Tiling**: Packs Z‑slices into an optimal 2D grid layout  
- **WebP Compression**: Achieves significant file-size reduction using lossy WebP format  
- **Metadata Manifest**: Auto‑generates `manifest.json` with dimensions, tile layout, channels, timepoints, and contrast bounds  
- **Parallel Processing**: Uses multiple processes (`--executor process`, default) or threads (`--executor thread`) via the `--threads` flag to speed up batch jobs  
- **Configuration Files**: Manage settings using YAML or JSON config files via `--config`  
- **Flexible Input**: Customize the input file search using `--input-pattern`  
- **Debugging Tools**: Use `--debug` to save intermediate histograms, preview slices, and enable verbose timing logs; use `--dry-run` to simulate without modifying files  
- **Easy CLI**: Clear flags and built‑in `--help` powered by Typer  

💻 **Usage**

```bash
pixelpacker --input <input_tiff_folder> --output <output_volume_folder> [OPTIONS]
```

#### Basic Examples

- Default settings (smart contrast, global timepoint contrast, slope Z-crop, process executor)  
  ```bash
  pixelpacker --input ./Input_TIFFS --output ./volumes
  ```
- ImageJ contrast, 8 threads, per-image contrast optimization, debug output  
  ```bash
  pixelpacker \
    --input /path/to/tiffs \
    --output /path/to/web_volumes \
    --stretch imagej-auto \
    --threads 8 \
    --per-image-contrast \
    --debug
  ```
- Using a Configuration File  
  ```bash
  # Create config.yaml (see example below)
  pixelpacker --config config.yaml
  ```

#### Example `config.yaml`

```yaml
input_folder: "./Input_TIFFS"
output_folder: "./volumes_config_run"
stretch_mode: "max"
max_threads: 4
debug: true
use_global_contrast: true  # Equivalent to --global-contrast flag
```

✨ **Advanced Usage**

- `--debug`: Enables detailed logging (including stage timings) and saves intermediate files to the output directory:  
  - `debug_hist_T<time>_C<channel>.png`: Histogram showing pixel distribution and calculated contrast limits  
  - `preview_T<time>_C<channel>.png`: PNG preview of the middle Z-slice after contrast stretching (useful for channel 0)  
  - If using slope Z-crop: `T<time>_C<channel>_debug_mip_yz.png`, etc.  
- `--dry-run`: Performs all calculations and logs intended actions but skips reading pixel data and writing output files  
- `--executor [process|thread]`: Choose the concurrency model  
  - `process` (default): Multiple processes (better for CPU-bound tasks)  
  - `thread`: Multiple threads (lighter overhead, may suit I/O-bound tasks)  
- `--z-crop-method [slope|threshold]`: Select Z-crop algorithm  
  - `slope` (default): Detects content boundaries via max-intensity projection slope  
  - `threshold`: Keeps slices above the specified intensity threshold  
- `--input-pattern <pattern>`: Glob pattern for input filenames (default `*_ch*_stack*.tif*`; must contain `_ch[channel]_stack[timepoint]`)  
- `--config <path>`: Load settings from a YAML or JSON file (CLI args override config)  

⚙️ **CLI Options Reference**

| Flag                                | Description                                                      | Default            |
|-------------------------------------|------------------------------------------------------------------|--------------------|
| `--input <folder>`                  | Input directory of TIFF stacks                                   | *Required*         |
| `--output <folder>`                 | Output directory for WebP atlases and manifest                   | *Required*         |
| `--config <file>`                   | Path to YAML or JSON configuration file                          | None               |
| `--input-pattern <pat>`             | Glob pattern for input TIFFs                                     | `*_ch*_stack*.tif*` |
| `--stretch <mode>`                  | Contrast mode: `smart` \| `smart-late` \| `imagej-auto` \| `max` | `smart`            |
| `--z-crop-method <meth>`            | Z-crop method: `slope` \| `threshold`                             | `slope`            |
| `--z-crop-threshold <int>`          | Intensity threshold for threshold Z-crop mode                    | `0`                |
| `--per-image-contrast`              | Per-image contrast optimization (use instead of `--global-contrast`) | `--global-contrast` |
| `--executor <exec>`                 | Concurrency model: `process` \| `thread`                          | `process`          |
| `--threads <n>`                     | Number of worker threads or processes                            | `8`                |
| `--dry-run`                         | Simulate processing without reading/writing files                | `false`            |
| `--debug`                           | Enable debug logging and save intermediate files                 | `false`            |
| `--profile`                         | Enable cProfile for performance analysis (adds overhead)         | `false`            |
| `--version`                         | Show installed version and exit                                  | N/A                |
| `-h, --help`                        | Show this help message and exit                                  | N/A                |

## 📂 Input / Output Formats

**Input Filenames**  
Must contain channel and timepoint info matching `[prefix]_ch[channel]_stack[timepoint].tif(f)`.  
- `[channel]`: digits (e.g., `ch0`, `ch1`)  
- `[timepoint]`: exactly four digits (e.g., `stack0000`)  
- Example: `experiment1_runA_GFP_ch0_stack0005_D3D.tif`

**Output Files**  
- **WebP Atlases**: `volume_[timepoint]_c[channel].webp` (8-bit grayscale)  
- **Manifest File**: `manifest.json` describing dataset structure and parameters  

```json
{
  "tile_layout": { "cols": 12, "rows": 11 },
  "volume_size": { "width": 790, "height": 766, "depth": 127 },
  "channels": 1,
  "global_z_crop_range": [22, 148],
  "timepoints": [
    {
      "time": "stack0000",
      "files": {
        "c0": { "file": "volume_stack0000_c0.webp", "p_low": 1.0, "p_high": 1107.88 }
      }
    }
  ],
  "global_intensity": {
    "c0": { "p_low": 1.0, "p_high": 1107.88 }
  }
}
```

🔧 **Troubleshooting**

- **Configuration Error**: `PreprocessingConfig.__init__()` missing arguments. Ensure required fields are in defaults, config, or CLI flags.  
- **No files found**: Check `--input` path and `--input-pattern`. Default is `*_ch*_stack*.tif*`.  
- **Skipping <filename>**: Filenames must contain `_ch[digit]_stack[4 digit]`.  
- **MemoryError**: Reduce `--threads` or switch executors; processes copy data between processes.  
- **Permission Denied**: Ensure `--output` directory is writable.  

🔗 **Dependencies**

- Python ≥ 3.10  
- `typer>=0.9.0`  
- `PyYAML>=6.0.1`  
- `numpy>=1.21,<3.0`  
- `tifffile>=2023.1.1`  
- `Pillow>=10.0,<12.0`  
- `scipy>=1.7,<2.0`  
- `matplotlib>=3.7,<4.0`  
- `tqdm>=4.64`  
- `python-json-logger>=2.0,<3.0`

📄 **License**

MIT — see `LICENSE`

🐛 **Issues**

Report bugs & feature requests on GitHub Issues.  
