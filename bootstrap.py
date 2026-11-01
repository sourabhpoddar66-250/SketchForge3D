"""Prepare an isolated Linux/NVIDIA environment without changing Colab's kernel."""
import hashlib
import json
import os
from pathlib import Path
import platform
import shutil
import subprocess
import sys

ROOT = Path(__file__).resolve().parent
VENV = ROOT / ".venv"
PYTHON = VENV / "bin" / "python"
UV_VERSION = "0.8.22"
PYTHON_VERSION = "3.11.13"
COMMIT = "107cefdc244c39106fa830359024f6a2f1c78871"


def run(command, **kwargs):
    print("Running:", " ".join(map(str, command)), flush=True)
    return subprocess.run(list(map(str, command)), check=True, **kwargs)


def main():
    if platform.system() != "Linux" or platform.machine() not in ("x86_64", "AMD64"):
        raise RuntimeError("This setup targets Linux x86_64 with an NVIDIA GPU, including Google Colab.")
    try:
        run(["nvidia-smi", "--query-gpu=name,memory.total,driver_version", "--format=csv,noheader"])
    except (FileNotFoundError, subprocess.CalledProcessError) as error:
        raise RuntimeError("No NVIDIA GPU is attached. Select a GPU runtime in Colab, then rerun this cell.") from error
    if shutil.disk_usage(ROOT).free < 10 * 1024 ** 3:
        raise RuntimeError("Free at least 10 GiB of local disk space for the environment and model downloads.")
    environment = dict(os.environ)
    environment.pop("PYTHONPATH", None)
    environment.update(UV_CACHE_DIR=str(ROOT / ".uv-cache"), UV_PYTHON_INSTALL_DIR=str(ROOT / ".managed-python"), HF_HOME=str(ROOT / "cache" / "huggingface"), U2NET_HOME=str(ROOT / "cache" / "rembg"), TOKENIZERS_PARALLELISM="false", HF_HUB_DISABLE_TELEMETRY="1")
    tools_dir = ROOT / ".bootstrap"
    if not (tools_dir / "uv").is_dir():
        run([sys.executable, "-m", "pip", "install", "--quiet", "--target", tools_dir, f"uv=={UV_VERSION}"])
    uv_environment = dict(environment, PYTHONPATH=str(tools_dir))
    uv = [sys.executable, "-m", "uv"]
    if not PYTHON.is_file():
        run(uv + ["venv", "--python", PYTHON_VERSION, VENV], env=uv_environment)
    lock = ROOT / "requirements.lock"
    fingerprint = hashlib.sha256(lock.read_bytes()).hexdigest()
    marker = ROOT / "setup_state.json"
    previous = json.loads(marker.read_text()) if marker.is_file() else {}
    if previous.get("requirements_sha256") != fingerprint:
        run(uv + ["pip", "sync", "--python", PYTHON, lock], env=uv_environment)
    run(uv + ["pip", "check", "--python", PYTHON], env=uv_environment)
    upstream = ROOT / "TripoSR"
    if not (upstream / ".git").is_dir():
        upstream.mkdir(exist_ok=True)
        run(["git", "init", "--quiet", upstream])
        run(["git", "-C", upstream, "remote", "add", "origin", "https://github.com/VAST-AI-Research/TripoSR.git"])
    revision = subprocess.check_output(["git", "-C", str(upstream), "rev-parse", "--verify", "HEAD"], text=True, stderr=subprocess.DEVNULL).strip() if (upstream / "tsr").is_dir() else ""
    if revision != COMMIT:
        run(["git", "-C", upstream, "fetch", "--depth", "1", "origin", COMMIT])
        run(["git", "-C", upstream, "checkout", "--quiet", "--detach", "FETCH_HEAD"])
    smoke = """
import sys, torch, cv2, numpy, rembg, xatlas, mcubes
from diffusers import StableDiffusionControlNetPipeline
from transformers import CLIPTextModel
import engine
engine.install_cpu_marching_cubes()
sys.path.insert(0, 'TripoSR')
from tsr.system import TSR
from tsr.models.tokenizers.image import DINOSingleImageTokenizer
from tsr.models.tokenizers.triplane import Triplane1DTokenizer
from tsr.models.transformer.transformer_1d import Transformer1D
from tsr.models.network_utils import TriplaneUpsampleNetwork, NeRFMLP
from tsr.models.nerf_renderer import TriplaneNeRFRenderer
assert torch.cuda.is_available(), 'CUDA is not available in the isolated environment'
assert float(torch.ones(8, device='cuda').sum()) == 8.0
print('Imports and an actual CUDA operation passed on', torch.cuda.get_device_name(0))
"""
    run([PYTHON, "-c", smoke], cwd=ROOT, env=environment)
    with open(ROOT / "environment-resolved.txt", "w") as stream:
        run(uv + ["pip", "freeze", "--python", PYTHON], env=uv_environment, stdout=stream)
    marker.write_text(json.dumps({"requirements_sha256": fingerprint, "python": PYTHON_VERSION, "triposr_commit": COMMIT}, indent=2))
    print("Setup passed. Model weights will download when you first press Generate.")


if __name__ == "__main__":
    main()
