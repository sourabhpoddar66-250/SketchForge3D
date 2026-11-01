# SketchForge3D

**Sketch-guided 3D asset generation using ControlNet and TripoSR**  
TechZephyr 4.0 · IIT Bhubaneswar · Sketch2Shape AI Challenge  
**Team BSP:** Saurabh Poddar and Biswasmita Rout

## Overview and objective

Traditional 3D modeling requires specialized tools and skills. **SketchForge3D** makes early-stage 3D prototyping more accessible by turning one hand-drawn or uploaded **2D sketch into actual 3D geometry**. A text description can guide the object's appearance, but is **not required**. The result is an inspectable, downloadable **GLB mesh** (optional OBJ and UV texture).

## Demo, report and downloads

| Resource | Link |
|---|---|
| **Demonstration video** | [Watch the end-to-end demo](PASTE_VIDEO_DRIVE_LINK_HERE) |
| **Project report and screenshots** | [Architecture, evaluation, comparisons and screenshots](PASTE_REPORT_DRIVE_LINK_HERE) |
| **Run in Colab** | [SketchForge3D.ipynb](SketchForge3D.ipynb) |
| **Sample 3D outputs** | [GLB example 1](runs/20261101_003249_49f2f14a/model.glb) · [GLB example 2](runs/20261101_003401_e6434789/model.glb) |
| **Machine-readable evaluation** | [evaluation.csv](evaluation.csv) |

> The report contains the detailed architecture diagram, equations, result images, UV comparisons, mesh-quality inspection and limitations. This README focuses on running and verifying the project.

## Setup and execution — Google Colab

1. Open [SketchForge3D.ipynb](SketchForge3D.ipynb) in [Google Colab](https://colab.research.google.com/) (or download and upload the notebook).
2. Select **Runtime → Change runtime type → GPU**. Development and measured runs used an **NVIDIA Tesla T4**.
3. Execute the notebook's setup cells **in order**. They prepare an isolated Python 3.11 environment; pretrained weights download on first use.
4. Launch the interface from the notebook, then **upload a sketch or draw one on the canvas**.
5. Optionally describe the object. For a first run choose **Preview**, **Seed 42**, and **1 candidate**; press **Generate 3D model**.
6. Inspect the 3D result and download the **GLB/result ZIP**. Try strict UV baking only after a normal run succeeds.

**Inputs:** A sketch image or in-app drawing, with optional text. **Outputs:** `model.glb`, intermediate images, masks, logs, configuration and a run report. Each inference uses a fresh output directory.

## Architecture and model choices

**Sketch/drawing → OpenCV conditioning → Stable Diffusion 1.5 + ControlNet Scribble → rembg/IS-Net foreground isolation → TripoSR reconstruction → mesh validation, optional UV baking and GLB export.**

| Component | Why it is used |
|---|---|
| Stable Diffusion 1.5 + ControlNet Scribble | Generate an object image while conditioning on sketch contours |
| OpenCV | Normalize, frame and prepare sketch guidance |
| rembg / IS-Net | Separate the object from its background |
| TripoSR (float32) | Reconstruct real 3D geometry from the prepared image |
| trimesh, PyMCubes, xatlas | Extract, validate, export and optionally UV-texture the mesh |

The pipeline integrates **pretrained models**; it does not claim custom model training. Image generation and reconstruction run in separate workers to release GPU allocations between stages.

### Model and data attribution

- **Models:** [Stable Diffusion 1.5](https://huggingface.co/stable-diffusion-v1-5/stable-diffusion-v1-5), [ControlNet Scribble](https://huggingface.co/lllyasviel/control_v11p_sd15_scribble), [TripoSR](https://huggingface.co/stabilityai/TripoSR), and rembg's IS-Net segmentation model. Their pretrained weights are obtained from the respective upstream sources; model training was **not** performed in this project.
- **Input/evaluation data:** User-supplied or hand-drawn single-object sketches; the documented mug sketch is one evaluated example. **No custom training dataset, data collection pipeline, or train/test split** is claimed. Detailed example inputs and screenshots are linked in the project report.
- **Licenses:** Follow the original model cards and library license terms. Pretrained model weights are **not redistributed** in this repository.

## Reproducible CLI run

On Linux with an NVIDIA GPU, use the environment created by `python bootstrap.py`:

```bash
.venv/bin/python engine.py run \
  --input sketch.png \
  --run-dir runs/my_first_run \
  --triposr-dir TripoSR \
  --quality preview \
  --seed 42 \
  --candidates 1
```

Provide `--prompt "a ceramic mug"` to guide appearance. For a **strict UV test**, also pass `--texture --require-texture`. Use a **new** `--run-dir` each time; run `engine.py run --help` for all options. Dependencies are pinned in [`requirements.lock`](requirements.lock); original source is in [`engine.py`](engine.py). The separately preserved [`engine_uv_fixed.py`](engine_uv_fixed.py) contains the tested UV material-color adjustment.

## Sample results and evaluation

**Hardware:** Tesla T4 · **Preview:** seed 42, one candidate · **Test object:** mug sketch.

| Test | Total time | Observed result |
|---|---:|---|
| Mug with prompt (batch evaluation) | **50.618 s** | Valid GLB; vertex colors |
| Mug without prompt (batch evaluation) | **46.661 s** | Valid GLB; vertex colors |
| Mug with detailed prompt (mesh inspection) | **51.759 s** | Valid GLB; handle connected, but surface banding visible |
| Strict UV bake | **99.566 s** | GLB with verified embedded 512×512 texture; material appearance needed adjustment |

**Representative stage times (52.254 s mug run):** conditioning **0.094 s** · image generation **25.544 s** · foreground preparation **6.288 s** · reconstruction/export **20.324 s**.

Peak **PyTorch-allocated** GPU memory observed: approximately **3.04 GiB** during image generation and **1.82 GiB** during reconstruction. These are allocation peaks, **not** total device VRAM consumption. The first-run model-download duration was **not independently measured**; downloads add substantial time and require internet access. See the linked report for visual comparisons and the evaluation methodology.

**Validation:** GLBs were reopened and checked for nonempty finite triangle geometry. Texture tests checked embedded image data. A valid export does **not** guarantee a watertight, artifact-free or manufacturing-ready mesh.

## Reproducibility

- **Environment:** Python 3.11, PyTorch 2.6.0, Diffusers 0.30.3, Transformers 4.44.2 and pinned Hugging Face Hub 0.25.2. Use the lockfile; incompatible upgrades can break imports.
- **Weights:** Model weights are fetched at runtime and are **not** stored in this repository. GPU availability and runtime vary in Colab. Fixed seeds improve comparability but do not guarantee cross-environment identical results.
- **Evidence:** Each run saves its configuration, input/intermediate images, logs and validation report. Use a new run directory, seed 42 and one candidate for repeatable comparison.

## Limitations

- **Geometry:** A single view cannot reveal all hidden surfaces. Some mug reconstructions showed disconnected handles, surface banding or unwanted background-derived geometry.
- **Texture:** Strict UV baking produced an embedded texture, but the original export appeared too dark. The material adjustment was tested in a **separate backend copy**; the original engine remains unchanged.
- **Resources:** Colab GPU availability varies. Model downloads add time, and exact first-download duration was not recorded. Export validity does not imply manufacturing readiness.

## Troubleshooting

| Problem | What to try |
|---|---|
| No GPU / out of memory | Enable a Colab GPU, use **Preview** and **1 candidate**, and free other GPU workloads. |
| Dependency import error | Recreate the isolated environment using `bootstrap.py` and `requirements.lock`; avoid upgrading the pinned Hugging Face packages. |
| Wrong object / missing parts | Compare the saved control map, generated image and foreground mask to locate the failing stage. |
| Dark or missing UV texture | Check strict UV logs and the separate `engine_uv_fixed.py` appearance experiment. |
| Interrupted Colab session | Download the result ZIP promptly; `/content` data may disappear when the runtime ends. |

## Project files

| File | Purpose |
|---|---|
| [`SketchForge3D.ipynb`](SketchForge3D.ipynb) | Colab workflow and notebook interface |
| [`engine.py`](engine.py) / [`engine_uv_fixed.py`](engine_uv_fixed.py) | Original pipeline / separate UV appearance correction |
| [`bootstrap.py`](bootstrap.py) / [`requirements.lock`](requirements.lock) | Isolated setup and pinned dependencies |
| [`notebook_app.py`](notebook_app.py) | Notebook-native input and preview interface |
| [`test_engine.py`](test_engine.py) | CPU regression tests |
| [`evaluation.csv`](evaluation.csv) / `runs/` | Recorded evaluation and example GLBs |

## Credits and responsible use

**Team BSP:** Saurabh Poddar · Biswasmita Rout.  
Built with pretrained [Stable Diffusion 1.5](https://huggingface.co/stable-diffusion-v1-5/stable-diffusion-v1-5), [ControlNet Scribble](https://huggingface.co/lllyasviel/control_v11p_sd15_scribble), [TripoSR](https://github.com/VAST-AI-Research/TripoSR), [rembg](https://github.com/danielgatis/rembg), [Hugging Face Diffusers](https://github.com/huggingface/diffusers), [PyTorch](https://pytorch.org/), [trimesh](https://github.com/mikedh/trimesh), [PyMCubes](https://github.com/pmneila/PyMCubes) and [xatlas](https://github.com/mworchel/xatlas-python). Respective upstream licenses and model usage restrictions apply.

**Purpose:** Rapid concept exploration, educational demonstrations and early-stage asset prototyping—not guaranteed manufacturing-ready geometry.
