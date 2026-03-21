import os
import shutil
import sys
import urllib.error
import urllib.request

os.environ["HF_ENDPOINT"] = "https://hf-mirror.com"
from transformers import AutoImageProcessor, ViTForImageClassification


RETINAFACE_FILENAME = "retinaface.h5"
RETINAFACE_DEFAULT_URLS = [
    "https://github.com/serengil/deepface_models/releases/download/v1.0/retinaface.h5",
    "https://ghproxy.com/https://github.com/serengil/deepface_models/releases/download/v1.0/retinaface.h5",
]


def _project_root():
    script_dir = os.path.dirname(os.path.abspath(__file__))
    return os.path.abspath(os.path.join(script_dir, ".."))


def _download_file(url, out_path, chunk_size=1024 * 1024):
    req = urllib.request.Request(url, headers={"User-Agent": "Mozilla/5.0"})
    with urllib.request.urlopen(req, timeout=60) as response:
        total = response.getheader("Content-Length")
        total = int(total) if total is not None else None

        os.makedirs(os.path.dirname(out_path), exist_ok=True)
        downloaded = 0
        with open(out_path, "wb") as f:
            while True:
                chunk = response.read(chunk_size)
                if not chunk:
                    break
                f.write(chunk)
                downloaded += len(chunk)
                if total:
                    pct = downloaded * 100.0 / total
                    print(f"\rDownloading: {pct:6.2f}%", end="", flush=True)
                else:
                    print(f"\rDownloaded: {downloaded / (1024 * 1024):.1f} MB", end="", flush=True)
        print()


def download_retinaface_weights(force=False):
    project_root = _project_root()
    project_weight_path = os.path.join(project_root, "models", "retinaface", RETINAFACE_FILENAME)

    deepface_home = os.getenv("DEEPFACE_HOME", project_root)
    if os.path.basename(os.path.normpath(deepface_home)) == ".deepface":
        deepface_home = os.path.dirname(os.path.normpath(deepface_home))
    runtime_weight_path = os.path.join(deepface_home, ".deepface", "weights", RETINAFACE_FILENAME)

    os.environ["DEEPFACE_HOME"] = deepface_home

    if os.path.exists(project_weight_path) and not force:
        print(f"RetinaFace weights already exist at: {project_weight_path}")
    else:
        urls = []
        custom_url = os.getenv("RETINAFACE_WEIGHTS_URL")
        if custom_url:
            urls.append(custom_url)
        urls.extend(RETINAFACE_DEFAULT_URLS)

        last_error = None
        for url in urls:
            try:
                print(f"Trying RetinaFace weight URL: {url}")
                _download_file(url, project_weight_path)
                print(f"RetinaFace weights downloaded to: {project_weight_path}")
                last_error = None
                break
            except (urllib.error.URLError, urllib.error.HTTPError, TimeoutError) as exc:
                last_error = exc
                print(f"Failed: {exc}")

        if last_error is not None:
            raise RuntimeError(
                "Unable to download RetinaFace weights. "
                "Set RETINAFACE_WEIGHTS_URL to an accessible mirror and retry."
            ) from last_error

    os.makedirs(os.path.dirname(runtime_weight_path), exist_ok=True)
    shutil.copy2(project_weight_path, runtime_weight_path)
    print(f"RetinaFace runtime weight staged at: {runtime_weight_path}")

def download_model():
    model_name = "dima806/facial_emotions_image_detection"
    
    # Resolve absolute path to models directory relative to this script
    script_dir = os.path.dirname(os.path.abspath(__file__))
    local_path = os.path.join(script_dir, "..", "models", "emotion_vit")
    
    print(f"Downloading model '{model_name}' from Hugging Face...")
    print(f"This might take a moment. The model will be saved to: {local_path}")
    
    # Download processor and model
    processor = AutoImageProcessor.from_pretrained(model_name)
    model = ViTForImageClassification.from_pretrained(model_name)
    
    # Save them locally
    os.makedirs(local_path, exist_ok=True)
    processor.save_pretrained(local_path)
    model.save_pretrained(local_path)
    
    print()
    print(f"✅ Download completed successfully! Model cached at '{local_path}'.")
    print("You can now run inferences locally without needing an internet connection.")


def main():
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--retinaface-only",
        action="store_true",
        help="Only download RetinaFace weights and skip emotion ViT model download.",
    )
    parser.add_argument(
        "--skip-retinaface",
        action="store_true",
        help="Skip RetinaFace weight download.",
    )
    parser.add_argument(
        "--force",
        action="store_true",
        help="Force re-download existing RetinaFace weights.",
    )
    args = parser.parse_args()

    if not args.retinaface_only:
        download_model()

    if not args.skip_retinaface:
        try:
            download_retinaface_weights(force=args.force)
        except Exception as exc:
            print(f"RetinaFace weight setup failed: {exc}")
            print(
                "You can manually place retinaface.h5 at: "
                f"{os.path.join(_project_root(), 'models', 'retinaface', RETINAFACE_FILENAME)}"
            )
            sys.exit(1)

if __name__ == "__main__":
    main()
