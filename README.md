# SketchForge 3D — complete Colab implementation

Convert a single object sketch into a generated object image, then a genuine TripoSR mesh, with GLB export and optional UV texture baking.

## Start in Google Colab

1. Open https://colab.research.google.com/ and choose **File → Upload notebook**.
2. Upload `SketchForge3D_Colab_Complete.ipynb`.
3. Select **Runtime → Change runtime type → GPU**. Free GPU access is subject to availability.
4. Run the notebook cells in order, or choose **Run all**. Setup creates an isolated Python environment; it does not replace the Colab kernel's packages. Several GB of downloads and several minutes of setup may be needed.
5. In the interface, upload a sketch or draw on the canvas and press **Use drawing**.
6. Optionally describe the object. Start with **Preview** quality and press **Generate 3D model**.
7. Inspect the exported model in the viewer, then download the result ZIP. Try **Balanced** quality and **Bake UV texture** after your first successful run.

Running all cells prepares the interface and runs small CPU checks. It does **not** start a GPU generation job, mount Google Drive, or launch a dataset batch. Those actions need their corresponding controls or explicitly enabled cells.

## Included implementation

- Stable Diffusion 1.5 + ControlNet Scribble through Diffusers; the model's default safety checker stays enabled.
- Upload and drawing inputs with an explicit input selector, optional text, seed and quality controls.
- Sketch framing without stretching; support for black-on-white and white-on-black drawings.
- CPU background removal with IS-Net through rembg. Small object parts are retained.
- TripoSR reconstruction in float32 with a CPU PyMCubes extraction adapter.
- Process separation between neural stages: worker exit releases GPU allocations.
- Optional CPU UV rasterization through xatlas and chunked GPU color queries. OpenGL/EGL is not needed.
- Conservative cleanup, GLB export/reload validation, vertex colors or verified embedded texture.
- Optional OBJ export with material files.
- An interactive viewer that loads the final exported GLB, rather than the neural field.
- Per-stage logs, immutable model revisions, a full Linux dependency lock, seeds, timings, GPU statistics, checksums, failure reports and downloadable bundles.
- Optional small evaluation and Google Drive saving, both initiated explicitly.
- A command-line interface and CPU regression checks.

## Files and architecture

The notebook embeds the complete project source. It materializes it under `/content/sketchforge3d` when run. The companion source ZIP is provided for inspection and command-line use on Linux/NVIDIA hardware.

`engine.py` is the CLI and inference pipeline. `bootstrap.py` creates the isolated environment and checks imports and a real CUDA operation. `notebook_app.py` contains the notebook-native interface. `test_engine.py` contains CPU checks. `requirements.lock` pins the Linux/Python 3.11 dependency set; `requirements.in` records the selected primary versions.

The image-generation worker exits before the reconstruction worker starts. Intermediate images on disk connect the stages. Repeated runs reload models from the local download cache; this trades some loading time for predictable GPU memory use.

## Command-line use

On Linux x86_64 with an NVIDIA GPU, unpack the source ZIP, run `python bootstrap.py`, then use the generated environment:

```bash
.venv/bin/python engine.py run --input sketch.png --run-dir runs/example1 --triposr-dir TripoSR --prompt "blue ceramic mug" --quality preview
```

Use a new output directory for each run. Add `--texture` for UV baking, `--require-texture` to fail if the requested bake fails, and `--obj` for an additional OBJ bundle. Run `engine.py run --help` for other options.

For an explicit image-to-3D diagnostic, supply a product image and `--product-image`. This is recorded as skipped diffusion, not counted as a full sketch-to-3D run.

## Result bundle

Each run contains original/framed sketches, the control map, generated image candidates, selected object image, foreground/mask, reconstruction input, `model.glb`, optional texture and OBJ files, configuration, stage reports, logs and `report.json`. The notebook adds `viewer.html`. The ZIP records failures as failures and includes diagnostic files.

The GLB is the authoritative validated output. A successful reload confirms actual triangle geometry and the expected appearance representation. Watertightness is reported, but manufacturing readiness is not claimed. Inspect the silhouette, important parts and unseen surfaces yourself.

## Reproducibility and resources

The setup uses managed Python 3.11.13, PyTorch 2.6.0 and a locked package set in a separate environment. It checks out TripoSR commit `107cefdc244c39106fa830359024f6a2f1c78871`. The source model, ControlNet, TripoSR checkpoint and DINO configuration have fixed Hugging Face revisions recorded in `engine.py` and every run report.

Pretrained model weights are downloaded at runtime, not bundled. The rembg library version fixes its checkpoint configuration and checksum expectations. Record `environment-resolved.txt` and reports with a submission. The same seed does not imply identical results across different GPU/software environments.

A GPU runtime and sufficient free disk space are required. Preview settings are designed as a starting point for a free Colab GPU, not a guarantee of runtime or availability. See https://research.google.com/colaboratory/faq.html for resource and notebook-interface restrictions.

## Validation status

The notebook format, Python cell syntax, preprocessing, UV rasterization/baking, mesh checks, GLB/texture round trips, and failure reports are tested locally. The full Linux dependency set has been resolved. Full neural inference and Colab's interactive UI still require verification in a GPU Colab session; no GPU-generated results or benchmark timings are fabricated.

## Troubleshooting

- **No GPU:** attach a GPU runtime; free availability varies. Setup stops before downloading model weights if none is attached.
- **Setup download error:** rerun setup; the environment is separate from Colab's built-in packages. Do not randomly upgrade the locked packages.
- **Out of GPU memory:** use Preview, one candidate and no texture. Interrupt any other GPU work. Restart if an earlier external notebook retained GPU memory.
- **Background removal error:** inspect the selected object image. Try a complete object with a plain background and a more specific optional description.
- **Wrong object or missing parts:** inspect the control map, then the generated image and mask before diagnosing the 3D model. Text is optional, but ambiguous drawings remain difficult.
- **Texture baking failure:** a non-strict run keeps vertex colors and records the failure. Enable the strict texture option when evaluating actual texture generation.
- **Viewer unavailable:** download `model.glb` and open it in Blender. The embedded viewer needs internet access to load its pinned model-viewer script.
- **Interrupted session:** saved files under `/content` can disappear when the runtime is replaced. Download result ZIPs or explicitly save them to Drive.

## Models, libraries and licenses

- [Stable Diffusion 1.5 mirror](https://huggingface.co/stable-diffusion-v1-5/stable-diffusion-v1-5): CreativeML OpenRAIL-M, with use restrictions.
- [ControlNet Scribble](https://huggingface.co/lllyasviel/control_v11p_sd15_scribble): see its OpenRAIL license and model card.
- [TripoSR](https://github.com/VAST-AI-Research/TripoSR), [weights](https://huggingface.co/stabilityai/TripoSR): MIT; retain upstream attribution when distributing upstream code.
- [rembg](https://github.com/danielgatis/rembg): MIT library; inspect the linked IS-Net checkpoint/project terms separately.
- [DINO](https://huggingface.co/facebook/dino-vitb16): configuration used by TripoSR's image tokenizer.
- [PyMCubes](https://github.com/pmneila/PyMCubes), [xatlas bindings](https://github.com/mworchel/xatlas-python), [trimesh](https://github.com/mikedh/trimesh), [Diffusers](https://github.com/huggingface/diffusers), [PyTorch](https://pytorch.org/) and [model-viewer](https://modelviewer.dev/): retain their respective notices when redistributing components.

New integration source is supplied in full. This does not replace or override any pretrained model or third-party library license.
