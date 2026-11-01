"""Notebook-native UI. Run after ROOT, PYTHON, and RUN_ENV have been configured."""
import base64
import csv
import datetime
import html
import io
import json
import os
from pathlib import Path
import signal
import subprocess
import uuid
import zipfile

import ipywidgets as widgets
from IPython.display import HTML, JSON, display, clear_output
from PIL import Image, ImageDraw
from google.colab import output, files

output.enable_custom_widget_manager()
state = {"drawing": None, "last_run": None, "busy": False}
input_source = widgets.ToggleButtons(options=["Upload", "Drawing"], description="Input:")
upload = widgets.FileUpload(accept="image/png,image/jpeg,image/webp", multiple=False, description="Upload sketch")
description = widgets.Textarea(placeholder="Optional: e.g. ceramic mug with a large blue handle", description="Describe:", layout=widgets.Layout(width="95%", height="70px"))
quality = widgets.Dropdown(options=[("Preview: start here", "preview"), ("Balanced", "balanced"), ("Detailed: more memory", "detailed")], value="preview", description="Quality:")
seed = widgets.IntText(value=42, description="Seed:")
strength = widgets.FloatSlider(value=1.1, min=0.5, max=1.5, step=0.1, description="Sketch guide:", readout_format=".1f")
candidates = widgets.IntSlider(value=1, min=1, max=2, description="Candidates:")
texture = widgets.Checkbox(value=False, description="Bake UV texture (slower)")
require_texture = widgets.Checkbox(value=False, description="Fail if requested texture cannot be baked")
obj_export = widgets.Checkbox(value=False, description="Also export OBJ + material files")
generate = widgets.Button(description="Generate 3D model", button_style="primary", icon="cube", layout=widgets.Layout(width="220px"))
download = widgets.Button(description="Download last result ZIP", icon="download", disabled=True)
save_drive = widgets.Button(description="Save last result to Drive", icon="save", disabled=True)
message = widgets.HTML("Upload a sketch or use the drawing canvas below. Generation starts only when you press the button.")
results_output = widgets.Output()


def current_upload_path():
    value = upload.value
    items = list(value.values()) if isinstance(value, dict) else list(value)
    if not items:
        raise ValueError("Upload a sketch first, or select Drawing.")
    data = bytes(items[0]["content"])
    if len(data) > 20 * 1024 ** 2:
        raise ValueError("Use an image smaller than 20 MB.")
    destination = ROOT / "inputs" / (uuid.uuid4().hex + ".png")
    destination.parent.mkdir(exist_ok=True)
    with Image.open(io.BytesIO(data)) as image:
        if image.width * image.height > 25_000_000:
            raise ValueError("Resize this image to fewer than 25 million pixels.")
        image.save(destination, format="PNG")
    return destination


def save_drawing(data_url):
    if not isinstance(data_url, str) or not data_url.startswith("data:image/png;base64,") or len(data_url) > 4_000_000:
        raise ValueError("The drawing data is invalid.")
    destination = ROOT / "inputs" / ("drawing_" + uuid.uuid4().hex + ".png")
    destination.parent.mkdir(exist_ok=True)
    data = base64.b64decode(data_url.split(",", 1)[1], validate=True)
    with Image.open(io.BytesIO(data)) as image:
        image.convert("RGB").save(destination)
    state["drawing"] = destination
    input_source.value = "Drawing"
    message.value = "Drawing saved. You can now generate the 3D model."
    return JSON({"saved": True})


output.register_callback("sketchforge.save_drawing", save_drawing)


def viewer_html(glb_path):
    payload = base64.b64encode(Path(glb_path).read_bytes()).decode("ascii")
    return '''<!doctype html><html><head><meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1">
<script type="module" src="https://unpkg.com/@google/model-viewer@4.0.0/dist/model-viewer.min.js"></script>
<style>body{margin:0;background:#f5f6f8;font-family:system-ui}model-viewer{width:100%;height:460px}p{padding:0 16px;color:#334155}</style></head>
<body><model-viewer src="data:model/gltf-binary;base64,''' + payload + '''" camera-controls auto-rotate shadow-intensity="0.4" exposure="1" alt="The final exported GLB model"></model-viewer>
<p>Drag to rotate; scroll to zoom. This viewer loads the exported GLB. Its viewer script requires internet access.</p></body></html>'''


def rebuild_result_bundle(directory):
    destination = directory / "result_bundle.zip"
    with zipfile.ZipFile(destination, "w", zipfile.ZIP_DEFLATED) as archive:
        for path in sorted(directory.iterdir()):
            if path.is_file() and path != destination:
                archive.write(path, arcname=path.name)
    return destination


def show_result(directory):
    report = json.loads((directory / "report.json").read_text())
    print("Status:", report["status"])
    if report["status"] != "success":
        print("Failed stage:", report.get("stage"), "\n", report.get("error", ""))
        print("The diagnostic ZIP includes the report, intermediate images, and logs.")
        return
    reconstruction = report["reconstruction"]
    print("Backend: TripoSR | Appearance:", reconstruction["appearance"], "| Time:", report["total_time_s"], "seconds")
    for warning in reconstruction["warnings"]:
        print("Note:", warning)
    for name, label in [("00_input.png", "Input"), ("02_control.png", "Sketch guidance"), ("03_product.png", "Generated object"), ("04_foreground.png", "Isolated object")]:
        if not (directory / name).is_file():
            continue
        print(label)
        with Image.open(directory / name) as image:
            display(image.copy().resize((280, 280)))
    viewer = viewer_html(directory / "model.glb")
    (directory / "viewer.html").write_text(viewer)
    display(HTML('<iframe title="Exported 3D model" style="width:100%;height:530px;border:1px solid #ddd;border-radius:8px" srcdoc="' + html.escape(viewer, quote=True) + '"></iframe>'))
    print("Export checks:")
    display(JSON(reconstruction["export_validation"]))
    rebuild_result_bundle(directory)


def run_case(image_path, prompt="", quality_name="preview", seed_value=42, bake=False, strict_texture=False, number_candidates=1, sketch_strength=1.1, export_obj=False, show=True, product_image=False):
    stamp = datetime.datetime.now(datetime.timezone.utc).strftime("%Y%m%d_%H%M%S")
    directory = ROOT / "runs" / (stamp + "_" + uuid.uuid4().hex[:8])
    directory.parent.mkdir(exist_ok=True)
    command = [str(PYTHON), str(ROOT / "engine.py"), "run", "--input", str(image_path), "--run-dir", str(directory), "--triposr-dir", str(ROOT / "TripoSR"), "--prompt", str(prompt), "--seed", str(seed_value), "--quality", quality_name, "--candidates", str(number_candidates), "--control-strength", str(sketch_strength)]
    for enabled, flag in [(bake, "--texture"), (strict_texture, "--require-texture"), (export_obj, "--obj"), (product_image, "--product-image")]:
        if enabled:
            command.append(flag)
    process = subprocess.Popen(command, stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True, bufsize=1, env=RUN_ENV, start_new_session=True)
    captured = []
    try:
        for line in process.stdout:
            print(line, end="")
            captured.append(line)
        process.wait()
    except BaseException:
        if process.poll() is None:
            os.killpg(process.pid, signal.SIGTERM)
            try:
                process.wait(timeout=10)
            except subprocess.TimeoutExpired:
                os.killpg(process.pid, signal.SIGKILL)
                process.wait()
        raise
    if directory.is_dir():
        (directory / "notebook_execution.log").write_text("".join(captured))
    if not (directory / "report.json").is_file():
        raise RuntimeError("The pipeline could not create a report. Check the output above for an environment error.")
    state["last_run"] = directory
    download.disabled = False
    save_drive.disabled = False
    rebuild_result_bundle(directory)
    if show:
        show_result(directory)
    return directory


def on_generate(_):
    if state["busy"]:
        return
    state["busy"] = True
    generate.disabled = True
    with results_output:
        clear_output(wait=True)
        try:
            if require_texture.value and not texture.value:
                raise ValueError("Enable UV texture baking before requiring a texture.")
            path = current_upload_path() if input_source.value == "Upload" else state["drawing"]
            if path is None:
                raise ValueError("Press Use drawing on the canvas first.")
            directory = run_case(path, prompt=description.value, quality_name=quality.value, seed_value=seed.value, bake=texture.value, strict_texture=require_texture.value, number_candidates=candidates.value, sketch_strength=strength.value, export_obj=obj_export.value)
            status = json.loads((directory / "report.json").read_text())["status"]
            message.value = "Finished. Download the result ZIP below." if status == "success" else "This run failed. The saved report explains which stage needs attention."
        except Exception as error:
            print(type(error).__name__ + ":", error)
        finally:
            state["busy"] = False
            generate.disabled = False


def on_download(_):
    if state["last_run"]:
        files.download(str(rebuild_result_bundle(state["last_run"])))


def on_save_drive(_):
    import shutil
    from google.colab import drive
    with results_output:
        drive.mount("/content/drive")
        destination = Path("/content/drive/MyDrive/SketchForge3D_results")
        destination.mkdir(exist_ok=True)
        archive = rebuild_result_bundle(state["last_run"])
        output_path = destination / (state["last_run"].name + ".zip")
        shutil.copy2(archive, output_path)
        print("Saved:", output_path)


generate.on_click(on_generate)
download.on_click(on_download)
save_drive.on_click(on_save_drive)
advanced = widgets.Accordion(children=[widgets.VBox([seed, strength, candidates, texture, require_texture, obj_export])])
advanced.set_title(0, "Optional settings")
advanced.selected_index = None
display(widgets.VBox([input_source, upload, description, quality, advanced, generate, message, widgets.HBox([download, save_drive]), results_output]))

DRAWING_HTML = r'''<div id="sketchforge-drawing" style="font:14px system-ui;max-width:540px;margin:16px 0">
<p><strong>Drawing canvas</strong> — draw one complete object.</p>
<canvas width="512" height="512" style="border:1px solid #9ca3af;border-radius:6px;width:100%;touch-action:none"></canvas>
<div style="display:flex;gap:8px;align-items:center;margin-top:8px"><button data-action="pen">Pen</button><button data-action="erase">Eraser</button><button data-action="clear">Clear</button><label>Size <input type="range" min="2" max="24" value="5"></label><button data-action="save">Use drawing</button></div><p data-status></p>
<script>(function(){const root=document.getElementById('sketchforge-drawing');const c=root.querySelector('canvas');const ctx=c.getContext('2d');let drawing=false;let erase=false;ctx.fillStyle='white';ctx.fillRect(0,0,512,512);ctx.lineCap='round';ctx.lineJoin='round';
function pos(e){const r=c.getBoundingClientRect();return [(e.clientX-r.left)*512/r.width,(e.clientY-r.top)*512/r.height]}
c.onpointerdown=e=>{drawing=true;c.setPointerCapture(e.pointerId);const p=pos(e);ctx.strokeStyle=erase?'white':'black';ctx.lineWidth=Number(root.querySelector('input').value)*(erase?3:1);ctx.beginPath();ctx.moveTo(p[0],p[1]);ctx.lineTo(p[0]+0.1,p[1]+0.1);ctx.stroke()};
c.onpointermove=e=>{if(!drawing)return;const p=pos(e);ctx.lineTo(p[0],p[1]);ctx.stroke()};c.onpointerup=()=>{drawing=false};c.onpointercancel=()=>{drawing=false};
root.querySelector('[data-action="pen"]').onclick=()=>{erase=false};root.querySelector('[data-action="erase"]').onclick=()=>{erase=true};root.querySelector('[data-action="clear"]').onclick=()=>{ctx.fillStyle='white';ctx.fillRect(0,0,512,512)};
root.querySelector('[data-action="save"]').onclick=async()=>{const status=root.querySelector('[data-status]');try{await google.colab.kernel.invokeFunction('sketchforge.save_drawing',[c.toDataURL('image/png')],{});status.textContent='Drawing saved and selected. Press Generate 3D model above.'}catch(e){status.textContent='Could not save: '+e.message}};
})();</script></div>'''
display(HTML(DRAWING_HTML))


def evaluate_cases(cases, quality_name="preview"):
    """Explicit, small evaluation: cases = [(image_path, optional_prompt), ...]."""
    if not cases or len(cases) > 12:
        raise ValueError("Evaluate between 1 and 12 examples at a time on free Colab.")
    rows = []
    for number, (path, prompt) in enumerate(cases, 1):
        print(f"\nEvaluation example {number}/{len(cases)}: {Path(path).name}")
        directory = run_case(path, prompt=prompt, quality_name=quality_name, show=False)
        report = json.loads((directory / "report.json").read_text())
        reconstruction = report.get("reconstruction", {})
        rows.append({"input": Path(path).name, "prompt_provided": bool(prompt.strip()), "status": report["status"], "backend": reconstruction.get("backend", ""), "glb_passed": reconstruction.get("export_validation", {}).get("passed", False), "appearance": reconstruction.get("appearance", ""), "total_time_s": report["total_time_s"], "run_dir": str(directory), "error": report.get("error", "")})
    destination = ROOT / "evaluation.csv"
    with destination.open("w", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    print("Saved:", destination)
    return rows
