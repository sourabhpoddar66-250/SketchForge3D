"""SketchForge: isolated sketch-to-image and TripoSR workers, plus a CLI.

Original integration code. Upstream models retain their respective licenses.
This module does not import torch until an inference worker is started.
"""
from __future__ import annotations

import argparse
import contextlib
import hashlib
import importlib.metadata
import json
import math
import os
from pathlib import Path
import platform
import subprocess
import sys
import time
import traceback
import types
import uuid
import zipfile

import cv2
import numpy as np
from PIL import Image, ImageOps
import trimesh

VERSION = "1.0.0"
PINS = {
    "triposr_code": "107cefdc244c39106fa830359024f6a2f1c78871",
    "base": {"id": "stable-diffusion-v1-5/stable-diffusion-v1-5", "revision": "451f4fe16113bff5a5d2269ed5ad43b0592e9a14"},
    "controlnet": {"id": "lllyasviel/control_v11p_sd15_scribble", "revision": "3564ec7b87dd706d5fc760d21c9e90bb160d1e9f"},
    "triposr": {"id": "stabilityai/TripoSR", "revision": "5b521936b01fbe1890f6f9baed0254ab6351c04a"},
    "dino_config": {"id": "facebook/dino-vitb16", "revision": "f205d5d8e640a89a2b8ef0369670dfc37cc07fc2"},
}
PRESETS = {
    "preview": {"size": 512, "steps": 20, "mesh_resolution": 128, "texture_size": 512, "chunk_size": 4096},
    "balanced": {"size": 512, "steps": 25, "mesh_resolution": 192, "texture_size": 1024, "chunk_size": 4096},
    "detailed": {"size": 512, "steps": 30, "mesh_resolution": 256, "texture_size": 1024, "chunk_size": 4096},
}


def save_json(path, value):
    path = Path(path)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(value, indent=2, allow_nan=False, default=str), encoding="utf-8")
    temporary.replace(path)


def load_json(path):
    return json.loads(Path(path).read_text(encoding="utf-8"))


def sha256(path):
    result = hashlib.sha256()
    with open(path, "rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            result.update(block)
    return result.hexdigest()


def read_rgb(source):
    with Image.open(source) as opened:
        if opened.width * opened.height > 25_000_000:
            raise ValueError("Please resize the input to fewer than 25 million pixels.")
        image = ImageOps.exif_transpose(opened).convert("RGBA")
    white = Image.new("RGBA", image.size, "white")
    white.alpha_composite(image)
    return white.convert("RGB")


def square_frame(array, box, margin=0.12, fill=255):
    x0, y0, x1, y1 = box
    side = max(8, int(math.ceil(max(x1 - x0, y1 - y0) * (1 + 2 * margin))))
    left = int(math.floor((x0 + x1 - side) / 2))
    top = int(math.floor((y0 + y1 - side) / 2))
    framed = np.full((side, side) + array.shape[2:], fill, dtype=array.dtype)
    xa, ya = max(left, 0), max(top, 0)
    xb, yb = min(left + side, array.shape[1]), min(top + side, array.shape[0])
    if xa < xb and ya < yb:
        framed[ya - top:yb - top, xa - left:xb - left] = array[ya:yb, xa:xb]
    return framed


def bounding_box(mask):
    ys, xs = np.nonzero(mask)
    if len(xs) == 0:
        raise ValueError("The input contains no visible sketch strokes.")
    return int(xs.min()), int(ys.min()), int(xs.max()) + 1, int(ys.max()) + 1


def condition_sketch(source, directory, size=512):
    directory = Path(directory)
    image = read_rgb(source)
    image.save(directory / "00_input.png")
    array = np.asarray(image)
    gray = cv2.cvtColor(array, cv2.COLOR_RGB2GRAY)
    # Inverted sketches are accepted explicitly by inspecting the background.
    border = np.concatenate((gray[0], gray[-1], gray[:, 0], gray[:, -1]))
    inverted = bool(np.median(border) < 128)
    if inverted:
        gray = 255 - gray
        array = 255 - array
    if float(gray.std()) < 2.0:
        raise ValueError("The input is blank. Draw or upload dark strokes on a plain background.")
    threshold, _ = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    threshold = float(np.clip(threshold, 24, 224))
    mask = gray <= threshold
    if mask.sum() < 8 or mask.mean() > 0.65:
        raise ValueError("The sketch is empty or the background is too dark. Use a clear line drawing.")
    box = bounding_box(mask)
    framed_mask = square_frame(mask.astype(np.uint8) * 255, box, fill=0)
    control = cv2.resize(framed_mask, (size, size), interpolation=cv2.INTER_AREA)
    control = np.where(control > 60, 255, 0).astype(np.uint8)
    control = cv2.dilate(control, np.ones((2, 2), np.uint8), iterations=1)
    source_frame = square_frame(array, box)
    Image.fromarray(source_frame).resize((size, size), Image.Resampling.LANCZOS).save(directory / "01_framed_sketch.png")
    Image.fromarray(control).convert("RGB").save(directory / "02_control.png")
    return {"inverted_input": inverted, "stroke_fraction": float((control > 0).mean()), "crop_box": list(box)}


def image_metrics(image, control):
    structure = np.asarray(control.convert("L")) > 0
    gray = cv2.cvtColor(np.asarray(image.convert("RGB")), cv2.COLOR_RGB2GRAY)
    edges = cv2.Canny(cv2.GaussianBlur(gray, (3, 3), 0), 60, 160) > 0
    kernel = np.ones((9, 9), np.uint8)
    near_structure = cv2.dilate(structure.astype(np.uint8), kernel) > 0
    near_edges = cv2.dilate(edges.astype(np.uint8), kernel) > 0
    recall = float((structure & near_edges).sum() / max(int(structure.sum()), 1))
    precision = float((edges & near_structure).sum() / max(int(edges.sum()), 1))
    f1 = 2 * precision * recall / max(precision + recall, 1e-8)
    a = np.asarray(image.convert("RGB"))
    border = np.concatenate((a[:20].reshape(-1, 3), a[-20:].reshape(-1, 3), a[:, :20].reshape(-1, 3), a[:, -20:].reshape(-1, 3)))
    white = float((border.mean(axis=1) > 220).mean())
    return {"edge_f1_heuristic": round(f1, 5), "white_border_fraction": round(white, 5), "ranking_score": round(0.85 * f1 + 0.15 * white, 5)}


def prepare_foreground(product_path, directory, foreground_ratio=0.85):
    from rembg import new_session, remove
    image = read_rgb(product_path)
    session = new_session("isnet-general-use", providers=["CPUExecutionProvider"])
    rgba = np.asarray(remove(image, session=session).convert("RGBA"))
    alpha = rgba[..., 3]
    coverage = float((alpha > 127).mean())
    if not 0.01 < coverage < 0.99:
        raise ValueError("Background removal did not isolate an object. Inspect the generated image and try a clearer sketch or description.")
    # Retain small components and soft alpha: they may be handles or legs.
    margin = (1 / foreground_ratio - 1) / 2
    framed = square_frame(rgba, bounding_box(alpha > 127), margin=margin, fill=0)
    rgba_image = Image.fromarray(framed, "RGBA").resize((512, 512), Image.Resampling.LANCZOS)
    directory = Path(directory)
    rgba_image.save(directory / "04_foreground.png")
    rgba_image.getchannel("A").save(directory / "04_mask.png")
    f = np.asarray(rgba_image).astype(np.float32) / 255
    composite = f[..., :3] * f[..., 3:4] + 0.5 * (1 - f[..., 3:4])
    Image.fromarray(np.round(composite * 255).astype(np.uint8)).save(directory / "05_triposr_input.png")
    return {"method": "rembg/isnet-general-use", "provider": "CPUExecutionProvider", "original_coverage": coverage, "foreground_ratio": foreground_ratio, "small_components_removed": False}


def require_gpu():
    import torch
    if not torch.cuda.is_available():
        raise RuntimeError("No NVIDIA GPU is available. In Colab select Runtime > Change runtime type > GPU, then rerun setup.")
    torch.cuda.reset_peak_memory_stats()
    props = torch.cuda.get_device_properties(0)
    return {"name": props.name, "total_gib": round(props.total_memory / 1024 ** 3, 2), "torch": torch.__version__, "cuda_build": torch.version.cuda}


def gpu_usage():
    import torch
    torch.cuda.synchronize()
    return {"peak_allocated_gib": round(torch.cuda.max_memory_allocated() / 1024 ** 3, 3), "peak_reserved_gib": round(torch.cuda.max_memory_reserved() / 1024 ** 3, 3)}


def generate_worker(directory):
    import torch
    from diffusers import ControlNetModel, StableDiffusionControlNetPipeline, UniPCMultistepScheduler
    directory = Path(directory)
    cfg = load_json(directory / "config.json")
    t0 = time.perf_counter()
    hardware = require_gpu()
    print("Loading image generation models; the first run downloads weights.", flush=True)
    controlnet = ControlNetModel.from_pretrained(**{"pretrained_model_name_or_path": PINS["controlnet"]["id"], "revision": PINS["controlnet"]["revision"], "torch_dtype": torch.float16, "variant": "fp16", "use_safetensors": True})
    pipe = StableDiffusionControlNetPipeline.from_pretrained(PINS["base"]["id"], revision=PINS["base"]["revision"], controlnet=controlnet, torch_dtype=torch.float16, variant="fp16", use_safetensors=True)
    # Keep the model's default safety checker enabled.
    pipe.scheduler = UniPCMultistepScheduler.from_config(pipe.scheduler.config)
    pipe.enable_model_cpu_offload()
    pipe.enable_vae_slicing()
    control = Image.open(directory / "02_control.png").convert("RGB")
    subject = cfg["prompt"].strip() or "a single solid physical object matching the supplied sketch"
    prompt = f"studio product photograph of {subject}, preserve the outline and proportions of the supplied drawing, same viewpoint, one complete object, plain white background, soft even lighting, realistic surfaces"
    negative = "multiple objects, duplicate object, room, scenery, cropped object, text, watermark, blurry, out of frame"
    candidates = []
    for number in range(cfg["candidates"]):
        seed = (cfg["seed"] + number) % (2 ** 31)
        print(f"Generating candidate {number + 1}/{cfg['candidates']} (seed {seed}).", flush=True)
        with torch.inference_mode():
            result = pipe(prompt=prompt, negative_prompt=negative, image=control, height=cfg["size"], width=cfg["size"], num_inference_steps=cfg["steps"], guidance_scale=7.5, controlnet_conditioning_scale=cfg["control_strength"], generator=torch.Generator(device="cpu").manual_seed(seed), num_images_per_prompt=1)
        flags = result.nsfw_content_detected
        if flags is not None and bool(flags[0]):
            print("This candidate was filtered by the image model; it will not be reconstructed.", flush=True)
            continue
        image = result.images[0]
        name = f"03_candidate_{number + 1}.png"
        image.save(directory / name)
        candidates.append({"file": name, "seed": seed, **image_metrics(image, control)})
    if not candidates:
        raise RuntimeError("No usable image was generated. Try a clearer object sketch or a different description.")
    candidates.sort(key=lambda item: item["ranking_score"], reverse=True)
    with Image.open(directory / candidates[0]["file"]) as chosen:
        chosen.save(directory / "03_product.png")
    save_json(directory / "generation.json", {"hardware": hardware, "memory": gpu_usage(), "duration_s": round(time.perf_counter() - t0, 3), "full_prompt": prompt, "models": {key: PINS[key] for key in ("base", "controlnet")}, "selected_seed": candidates[0]["seed"], "candidates": candidates, "note": "Edge agreement is a 2D ranking heuristic, not a measure of 3D correctness."})


def marching_cubes_numpy(volume, threshold):
    """Match torchmcubes axis ordering; TripoSR itself swaps it back."""
    import mcubes
    vertices, faces = mcubes.marching_cubes(np.asarray(volume, dtype=np.float32), float(threshold))
    return np.ascontiguousarray(vertices[:, ::-1], dtype=np.float32), np.ascontiguousarray(faces, dtype=np.int64)


def install_cpu_marching_cubes():
    import torch
    module = types.ModuleType("torchmcubes")

    def marching_cubes(volume, threshold):
        vertices, faces = marching_cubes_numpy(volume.detach().float().cpu().numpy(), threshold)
        return torch.from_numpy(vertices), torch.from_numpy(faces)

    module.marching_cubes = marching_cubes
    sys.modules["torchmcubes"] = module


def clean_mesh(mesh, target_faces=0):
    if len(mesh.vertices) == 0 or len(mesh.faces) == 0:
        raise ValueError("TripoSR returned empty geometry.")
    if not np.isfinite(mesh.vertices).all():
        raise ValueError("TripoSR returned invalid vertex coordinates.")
    mesh = mesh.copy()
    mesh.update_faces(mesh.nondegenerate_faces())
    mesh.update_faces(mesh.unique_faces())
    mesh.remove_unreferenced_vertices()
    # Deliberately retain disconnected components; geometry is not repaired by deleting them.
    if target_faces and len(mesh.faces) > target_faces:
        original = mesh
        mesh = mesh.simplify_quadric_decimation(face_count=int(target_faces))
        if original.visual.kind == "vertex":
            from scipy.spatial import cKDTree
            nearest = cKDTree(original.vertices).query(mesh.vertices)[1]
            mesh.visual = trimesh.visual.ColorVisuals(mesh, vertex_colors=original.visual.vertex_colors[nearest])
    trimesh.repair.fix_normals(mesh, multibody=True)
    return mesh


def geometry_report(mesh):
    nv, nf = len(mesh.vertices), len(mesh.faces)
    finite = bool(nv and np.isfinite(mesh.vertices).all())
    extents = np.asarray(mesh.extents if finite else [0, 0, 0], dtype=float)
    depth_present = bool(extents.max() > 0 and extents.min() / extents.max() > 0.005)
    checks = {"vertices_present": nv > 0, "faces_present": nf > 0, "finite_vertices": finite, "depth_present": depth_present, "positive_area": bool(nf and np.isfinite(mesh.area) and mesh.area > 0)}
    return {"passed": all(checks.values()), "checks": checks, "vertices": nv, "faces": nf, "extents": extents.tolist(), "watertight": bool(nf and mesh.is_watertight), "winding_consistent": bool(nf and mesh.is_winding_consistent), "components": len(mesh.split(only_watertight=False)) if nf else 0, "note": "Geometry checks detect invalid files and flat output; visual fidelity must also be inspected."}


def rasterize_uv_positions(vertices, faces, uv, resolution):
    """Rasterize positions into a PIL-oriented UV atlas without OpenGL/EGL."""
    positions = np.zeros((resolution, resolution, 3), dtype=np.float32)
    covered = np.zeros((resolution, resolution), dtype=bool)
    pixels = np.column_stack((uv[:, 0] * resolution, (1 - uv[:, 1]) * resolution))
    for face in faces:
        tri = pixels[face]
        lower = np.maximum(np.floor(tri.min(axis=0)).astype(int), 0)
        upper = np.minimum(np.ceil(tri.max(axis=0)).astype(int), resolution - 1)
        if (upper < lower).any():
            continue
        xx, yy = np.meshgrid(np.arange(lower[0], upper[0] + 1) + 0.5, np.arange(lower[1], upper[1] + 1) + 0.5)
        a, b, c = tri
        denom = (b[1] - c[1]) * (a[0] - c[0]) + (c[0] - b[0]) * (a[1] - c[1])
        if abs(denom) < 1e-10:
            continue
        w0 = ((b[1] - c[1]) * (xx - c[0]) + (c[0] - b[0]) * (yy - c[1])) / denom
        w1 = ((c[1] - a[1]) * (xx - c[0]) + (a[0] - c[0]) * (yy - c[1])) / denom
        w2 = 1 - w0 - w1
        inside = (w0 >= -1e-6) & (w1 >= -1e-6) & (w2 >= -1e-6)
        ys, xs = np.nonzero(inside)
        if not len(xs):
            continue
        weights = np.column_stack((w0[inside], w1[inside], w2[inside]))
        target_y, target_x = ys + lower[1], xs + lower[0]
        positions[target_y, target_x] = weights @ vertices[face]
        covered[target_y, target_x] = True
    if not covered.any():
        raise ValueError("UV unwrapping produced no rasterizable triangles.")
    return positions, covered


def bake_texture_cpu_atlas(mesh, resolution, color_query):
    import xatlas
    from scipy.ndimage import distance_transform_edt
    atlas = xatlas.Atlas()
    atlas.add_mesh(np.asarray(mesh.vertices, dtype=np.float32), np.asarray(mesh.faces, dtype=np.uint32))
    packing = xatlas.PackOptions()
    packing.resolution = int(resolution)
    packing.padding = 4
    atlas.generate(pack_options=packing)
    mapping, indices, uv = atlas[0]
    verts = np.asarray(mesh.vertices)[mapping]
    positions, covered = rasterize_uv_positions(verts, indices, uv, resolution)
    colors = np.asarray(color_query(positions[covered]), dtype=np.float32)
    if colors.shape != (int(covered.sum()), 3) or not np.isfinite(colors).all():
        raise ValueError("Texture color queries returned invalid data.")
    texture = np.zeros((resolution, resolution, 3), dtype=np.uint8)
    texture[covered] = np.round(np.clip(colors, 0, 1) * 255).astype(np.uint8)
    nearest = distance_transform_edt(~covered, return_distances=False, return_indices=True)
    texture[~covered] = texture[nearest[0][~covered], nearest[1][~covered]]
    texture_image = Image.fromarray(texture, "RGB")
    material = trimesh.visual.material.PBRMaterial(baseColorTexture=texture_image, baseColorFactor=[255, 255, 255, 255], metallicFactor=0.0, roughnessFactor=0.75)
    output = trimesh.Trimesh(vertices=verts, faces=indices, process=False)
    output.visual = trimesh.visual.TextureVisuals(uv=uv, material=material)
    return output, texture_image


def normalize_mesh(mesh):
    # Same two rotations used by the upstream TripoSR viewer utility.
    mesh.apply_transform(trimesh.transformations.rotation_matrix(-math.pi / 2, [1, 0, 0]))
    mesh.apply_transform(trimesh.transformations.rotation_matrix(math.pi / 2, [0, 1, 0]))
    mesh.apply_translation(-mesh.bounds.mean(axis=0))
    mesh.apply_scale(1 / max(float(mesh.extents.max()), 1e-9))
    return mesh


def inspect_glb(path, expect_texture=False, expect_vertex_colors=False):
    scene = trimesh.load(path, force="scene", process=False)
    meshes = [g for g in scene.geometry.values() if isinstance(g, trimesh.Trimesh)]
    if not meshes:
        raise ValueError("The exported GLB contains no triangle mesh.")
    reports = [geometry_report(mesh) for mesh in meshes]
    has_texture = False
    has_vertex_colors = False
    for mesh in meshes:
        if mesh.visual.kind == "texture":
            material = mesh.visual.material
            texture = getattr(material, "baseColorTexture", None)
            if texture is None:
                texture = getattr(material, "image", None)
            has_texture = has_texture or bool(texture is not None and getattr(mesh.visual, "uv", None) is not None)
        if mesh.visual.kind == "vertex":
            has_vertex_colors = has_vertex_colors or len(mesh.visual.vertex_colors) == len(mesh.vertices)
    appearance_ok = (has_texture if expect_texture else True) and (has_vertex_colors if expect_vertex_colors else True)
    result = {"passed": all(r["passed"] for r in reports) and appearance_ok, "meshes": reports, "has_texture": has_texture, "has_vertex_colors": has_vertex_colors, "appearance_preserved": appearance_ok, "bytes": Path(path).stat().st_size, "sha256": sha256(path)}
    return result


def write_obj(mesh, directory):
    from trimesh.exchange.obj import export_obj
    # SimpleMaterial gives OBJ an MTL + PNG rather than assuming PBR support.
    obj_mesh = mesh.copy()
    if obj_mesh.visual.kind == "texture":
        mat = obj_mesh.visual.material
        texture = getattr(mat, "baseColorTexture", None)
        if texture is None:
            texture = getattr(mat, "image", None)
        obj_mesh.visual.material = trimesh.visual.material.SimpleMaterial(image=texture)
    text, files = export_obj(obj_mesh, return_texture=True, include_texture=True, mtl_name="model.mtl")
    (Path(directory) / "model.obj").write_text(text, encoding="utf-8")
    for filename, content in (files or {}).items():
        if Path(filename).name != filename:
            raise ValueError("Unexpected relative export path.")
        (Path(directory) / filename).write_bytes(content)


def reconstruct_worker(directory):
    import torch
    from huggingface_hub import hf_hub_download, snapshot_download
    directory = Path(directory)
    cfg = load_json(directory / "config.json")
    hardware = require_gpu()
    t0 = time.perf_counter()
    install_cpu_marching_cubes()
    sys.path.insert(0, cfg["triposr_dir"])
    import tsr.models.tokenizers.image as dino_module
    from tsr.system import TSR

    def download_dino(**kwargs):
        if kwargs.get("repo_id") == PINS["dino_config"]["id"]:
            kwargs["revision"] = PINS["dino_config"]["revision"]
        return hf_hub_download(**kwargs)

    dino_module.hf_hub_download = download_dino
    weights = snapshot_download(PINS["triposr"]["id"], revision=PINS["triposr"]["revision"], allow_patterns=["config.yaml", "model.ckpt"])
    print("Loading TripoSR (float32); no substitute reconstruction backend is used.", flush=True)
    model = TSR.from_pretrained(weights, config_name="config.yaml", weight_name="model.ckpt").to("cuda").eval()
    model.renderer.set_chunk_size(cfg["chunk_size"])
    with torch.inference_mode():
        scene_codes = model([Image.open(directory / "05_triposr_input.png").convert("RGB")], device="cuda")
        mesh = model.extract_mesh(scene_codes, True, resolution=cfg["mesh_resolution"])[0]
    raw_report = geometry_report(mesh)
    mesh = clean_mesh(mesh, target_faces=cfg["target_faces"])
    if not geometry_report(mesh)["passed"]:
        raise ValueError("The reconstructed geometry failed validation. Inspect the foreground image and try another input.")
    warnings = []
    appearance = "vertex_colors"
    if cfg["bake_texture"]:
        print(f"Baking a {cfg['texture_size']}px texture with CPU UV rasterization.", flush=True)

        def query(points):
            colors = []
            with torch.inference_mode():
                for offset in range(0, len(points), cfg["chunk_size"]):
                    batch = torch.from_numpy(points[offset:offset + cfg["chunk_size"]].astype(np.float32)).to("cuda")
                    colors.append(model.renderer.query_triplane(model.decoder, batch, scene_codes[0])["color"].detach().cpu().numpy())
            return np.concatenate(colors)

        try:
            textured, texture = bake_texture_cpu_atlas(mesh, cfg["texture_size"], query)
            mesh = textured
            texture.save(directory / "texture.png")
            appearance = "uv_texture"
        except torch.cuda.OutOfMemoryError:
            raise
        except Exception as error:
            if cfg["require_texture"]:
                raise RuntimeError(f"Required texture baking failed: {error}") from error
            warnings.append(f"Texture baking failed; the exported model uses vertex colors: {type(error).__name__}: {error}")
            print(warnings[-1], flush=True)
    final_mesh = normalize_mesh(mesh)
    if not geometry_report(final_mesh)["passed"]:
        raise ValueError("The final mesh failed geometry validation.")
    glb = directory / "model.glb"
    final_mesh.export(glb)
    check = inspect_glb(glb, expect_texture=appearance == "uv_texture", expect_vertex_colors=appearance == "vertex_colors")
    if not check["passed"]:
        raise ValueError(f"Export validation failed: {check}")
    obj_status = "not_requested"
    if cfg["export_obj"]:
        try:
            write_obj(final_mesh, directory)
            obj_status = "exported; GLB is the authoritative validated output"
        except Exception as error:
            warnings.append(f"Optional OBJ export failed: {error}")
            obj_status = "failed"
    save_json(directory / "reconstruction.json", {"backend": "TripoSR", "hardware": hardware, "memory": gpu_usage(), "duration_s": round(time.perf_counter() - t0, 3), "raw_geometry": raw_report, "final_geometry": geometry_report(final_mesh), "export_validation": check, "appearance": appearance, "obj": obj_status, "warnings": warnings, "models": {"triposr": PINS["triposr"], "dino_config": PINS["dino_config"], "code_commit": PINS["triposr_code"]}, "adapters": ["CPU PyMCubes adapter with TripoSR axis convention", "DINO config revision pin", "CPU UV rasterization with GPU color queries"]})


def package_versions():
    names = ["torch", "torchvision", "diffusers", "transformers", "accelerate", "huggingface-hub", "numpy", "Pillow", "opencv-python-headless", "rembg", "onnxruntime", "trimesh", "PyMCubes", "xatlas", "scipy", "fast-simplification"]
    versions = {}
    for name in names:
        with contextlib.suppress(importlib.metadata.PackageNotFoundError):
            versions[name] = importlib.metadata.version(name)
    return {"python": sys.version, "platform": platform.platform(), "packages": versions}


def worker(stage, directory):
    command = [sys.executable, str(Path(__file__).resolve()), stage, "--job", str(directory)]
    with open(Path(directory) / f"{stage}.log", "w", encoding="utf-8") as log:
        process = subprocess.Popen(command, stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True, bufsize=1)
        try:
            for line in process.stdout:
                print(line, end="", flush=True)
                log.write(line)
                log.flush()
            code = process.wait()
        except BaseException:
            process.terminate()
            with contextlib.suppress(subprocess.TimeoutExpired):
                process.wait(timeout=10)
            if process.poll() is None:
                process.kill()
                process.wait()
            raise
    if code:
        tail = (Path(directory) / f"{stage}.log").read_text(encoding="utf-8")[-1800:]
        raise RuntimeError(f"The {stage} stage failed. See {stage}.log.\n{tail}")


def bundle_run(directory):
    directory = Path(directory)
    destination = directory / "result_bundle.zip"
    with zipfile.ZipFile(destination, "w", zipfile.ZIP_DEFLATED) as archive:
        for file in sorted(directory.iterdir()):
            if file.is_file() and file != destination:
                archive.write(file, arcname=file.name)
    return destination


def run_pipeline(args):
    directory = Path(args.run_dir).resolve()
    directory.mkdir(parents=True, exist_ok=False)
    cfg = {**PRESETS[args.quality], "prompt": args.prompt, "seed": int(args.seed) if args.seed >= 0 else int.from_bytes(os.urandom(4), "little") % (2 ** 31), "quality": args.quality, "candidates": args.candidates, "control_strength": args.control_strength, "bake_texture": args.texture, "require_texture": args.require_texture, "target_faces": args.target_faces, "export_obj": args.obj, "triposr_dir": str(Path(args.triposr_dir).resolve())}
    save_json(directory / "config.json", cfg)
    report = {"version": VERSION, "status": "running", "stage": "conditioning", "model_pins": PINS, "config": cfg, "environment": package_versions(), "input_sha256": None, "timings_s": {}, "limitations": ["Single-view hidden surfaces are inferred.", "An empty text prompt is accepted, but ambiguous sketches can be misinterpreted.", "Mesh checks do not establish semantic or manufacturing accuracy."]}
    start = time.perf_counter()
    report_path = directory / "report.json"

    def stage(name, action):
        report["stage"] = name
        save_json(report_path, report)
        print(f"\n--- {name} ---", flush=True)
        t0 = time.perf_counter()
        result = action()
        report["timings_s"][name] = round(time.perf_counter() - t0, 3)
        return result

    try:
        report["input_sha256"] = sha256(args.input)
        if args.product_image:
            # Explicit reconstruction-only diagnostic, never substituted silently.
            image = read_rgb(args.input)
            image.save(directory / "00_input.png")
            image.save(directory / "03_product.png")
            report["conditioning"] = {"mode": "product_image_diagnostic", "sketch_conditioning_skipped": True}
            report["generation"] = {"explicitly_skipped": True, "reason": "User selected product-image diagnostic mode."}
        else:
            report["conditioning"] = stage("conditioning", lambda: condition_sketch(args.input, directory, cfg["size"]))
            stage("image_generation", lambda: worker("generate", directory))
            report["generation"] = load_json(directory / "generation.json")
        report["preparation"] = stage("foreground_preparation", lambda: prepare_foreground(directory / "03_product.png", directory))
        stage("3d_reconstruction_and_export", lambda: worker("reconstruct", directory))
        report["reconstruction"] = load_json(directory / "reconstruction.json")
        report.update(status="success", stage="complete", total_time_s=round(time.perf_counter() - start, 3), glb="model.glb")
        save_json(report_path, report)
        bundle_run(directory)
        print(f"\nSUCCESS: {directory / 'model.glb'}", flush=True)
        return 0
    except BaseException as error:
        report.update(status="failed", error=f"{type(error).__name__}: {error}", total_time_s=round(time.perf_counter() - start, 3))
        save_json(report_path, report)
        (directory / "failure_traceback.txt").write_text(traceback.format_exc(), encoding="utf-8")
        bundle_run(directory)
        print(f"\nFAILED during {report['stage']}: {error}\nReport: {report_path}", flush=True)
        return 1


def parser():
    p = argparse.ArgumentParser(description=__doc__)
    commands = p.add_subparsers(dest="command", required=True)
    run = commands.add_parser("run", help="Generate a 3D asset from a sketch.")
    run.add_argument("--input", required=True)
    run.add_argument("--run-dir", required=True, help="A new, unused output directory.")
    run.add_argument("--triposr-dir", required=True)
    run.add_argument("--prompt", default="")
    run.add_argument("--seed", type=int, default=42)
    run.add_argument("--quality", choices=PRESETS, default="preview")
    run.add_argument("--candidates", type=int, choices=[1, 2], default=1)
    run.add_argument("--control-strength", type=float, default=1.1)
    run.add_argument("--texture", action="store_true")
    run.add_argument("--require-texture", action="store_true")
    run.add_argument("--obj", action="store_true")
    run.add_argument("--target-faces", type=int, default=0)
    run.add_argument("--product-image", action="store_true", help="Explicit image-to-3D diagnostic; skips diffusion.")
    for name in ("generate", "reconstruct"):
        sub = commands.add_parser(name, help=argparse.SUPPRESS)
        sub.add_argument("--job", required=True)
    return p


def main():
    args = parser().parse_args()
    if args.command == "run":
        if args.require_texture and not args.texture:
            raise SystemExit("--require-texture also needs --texture.")
        if args.target_faces < 0 or not 0.3 <= args.control_strength <= 2.0:
            raise SystemExit("Use nonnegative target faces and sketch strength between 0.3 and 2.0.")
        raise SystemExit(run_pipeline(args))
    if args.command == "generate":
        generate_worker(args.job)
    elif args.command == "reconstruct":
        reconstruct_worker(args.job)


if __name__ == "__main__":
    main()
