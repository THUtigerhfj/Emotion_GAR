# Emotion Prediction & Vision Transformer Explainability

This project uses a CLIP ViT-L/14 based pipeline to predict facial emotions from images and adds a **Gradient Attention Rollout** mechanism to visualize *where* the model is looking in order to make that prediction.

The attention rollout produces a heatmap overlay showing the facial regions (e.g., eyes, mouth) that most strongly contributed to the predicted emotion.

## Requirements

The project uses Python. Ensure you have installed the required dependencies:

```bash
# Linux only: install OpenCV runtime dependency once
sudo apt-get update && sudo apt-get install -y libgl1

# Activate your conda env first
conda activate emotion

# Install Python packages (example uses Tsinghua mirror)
python -m pip install -r requirements.txt -i https://pypi.tuna.tsinghua.edu.cn/simple
```

If you already have `tensorflow==2.21.x`, make sure `tf-keras` is installed (it is included in `requirements.txt`):

```bash
python -m pip install tf-keras -i https://pypi.tuna.tsinghua.edu.cn/simple
```

*(Note: As specified, `torch` and `numpy` should already be available in the environment).*

If you re-run dependency installation later, use the same command above. Do not install `cv2` directly (there is no package named `cv2` on PyPI).

## Setup

First, download local model assets once:

- Vision encoder: `tanganke/clip-vit-large-patch14_fer2013` (fine-tuned on FER2013)
- Text branch + processor: `openai/clip-vit-large-patch14`
- RetinaFace detector weights: `retinaface.h5`

```bash
python src/download_model.py
```

The script merges the fine-tuned vision encoder into the base CLIP model and stores everything in `models/clip_fer2013/` so you can run inferences entirely offline in the future.

If your network cannot access GitHub release URLs, you can download RetinaFace weights only and provide a mirror URL:

```bash
export RETINAFACE_WEIGHTS_URL="https://your-mirror.example.com/retinaface.h5"
python src/download_model.py --retinaface-only
```

After download, the script stores RetinaFace weights in `models/retinaface/retinaface.h5` and stages them to local DeepFace runtime cache, so Gradio does not need online download during inference.

## Usage

Place any facial images you want to evaluate inside the `data/` folder, then run the inference script. The script will automatically utilize CUDA/GPU acceleration if available.

```bash
python src/inference.py --image_path data/your_test_image.jpg
```

For example:

```bash
python src/inference.py --image_path data/images/validation/happy/8.jpg
```

### Options

- `--image_path`: Path to an input facial expression image.
- `--discard_ratio`: (Optional) Ratio of lowest attention heads to discard to reduce noise. Default is `0.9` (keeps the top 10% highest attention signals). You can experiment with lower values (e.g., `0.5`) if the heatmap is too constrained.
- `--head_fusion`: (Optional) How to fuse attention heads. Options are `mean` (default) or `max` or `min`.

### Gradio UI

Set up UI by running:

```bash
python src/gradio_app.py
```

### Expose Gradio with Cloudflare Tunnel

If you want others to access your local Gradio service without renting a cloud server, use Cloudflare Tunnel.

1. Start Gradio locally (this project machine):

```bash
# Linux/macOS
export GRADIO_SERVER_NAME=0.0.0.0
export GRADIO_SERVER_PORT=7860
export GRADIO_QUEUE_MAX_SIZE=10
export GRADIO_QUEUE_WORKERS=1
export GRADIO_MAX_THREADS=16
export GRADIO_SHARE=false
python src/gradio_app.py
```

```powershell
# Windows PowerShell
$env:GRADIO_SERVER_NAME="0.0.0.0"
$env:GRADIO_SERVER_PORT="7860"
$env:GRADIO_QUEUE_MAX_SIZE="10"
$env:GRADIO_QUEUE_WORKERS="1"
$env:GRADIO_MAX_THREADS="16"
$env:GRADIO_SHARE="false"
python src/gradio_app.py
```

1. Install `cloudflared` on the same machine where Gradio is running.

- Linux (recommended quick install):

```bash
wget -q https://github.com/cloudflare/cloudflared/releases/latest/download/cloudflared-linux-amd64.deb
sudo dpkg -i cloudflared-linux-amd64.deb
cloudflared --version
```

- Windows:
: download from the Cloudflare official releases page, install it, then verify in PowerShell:

```powershell
cloudflared --version
```

1. Start a quick tunnel to your local Gradio port (run this in another terminal):

```bash
cloudflared tunnel --url http://localhost:7860
```

You will get a public URL like:

```text
https://xxxx.trycloudflare.com
```

1. Share that URL with users.

1. Stability tips (important):

- Prefer HTTP/2 for better cross-region stability:

```bash
cloudflared tunnel --url http://localhost:7860 --protocol http2
```

- Keep Gradio queue enabled to absorb burst requests.
- Keep `GRADIO_QUEUE_WORKERS=2` (or reduce to `1` if GPU memory is tight).

1. Use two terminal sessions for long-running service:

- Session 1: run Gradio (`python src/gradio_app.py`)
- Session 2: run Cloudflare Tunnel (`cloudflared tunnel --url http://localhost:7860 --protocol http2`)

On Linux servers, you can use `tmux` and keep one process per tmux pane/window.

`conda info --base` gives you the base path, and before activate conda in tmux, run `source <conda_base_path>/etc/profile.d/conda.sh` to enable `conda activate` in tmux. e.g., `source /base/mambaforge/etc/profile.d/conda.sh`

### Mass Inference

To run inference on a batch of images, run:

```bash
python src/observe_batch_inference.py --input_dir data/ --output_dir outputs/
```

### Output

The output will be saved into the `outputs/` directory.

## Project Structure

```text
├── data/                  # Place your input images here 
├── models/                # Local cache for the downloaded classifier model
├── outputs/               # Generated heatmaps and visualizations land here
├── src/                   # Source code
│   ├── download_model.py  # Utility to pull HF models locally
│   ├── inference.py       # Core prediction and gradient rollout script
|   ├── gradio_app.py      # Gradio UI for interactive inference
│   └── observe_batch_inference.py  # Script for batch inference on image directories
├── requirements.txt       # Python dependencies
└── README.md
```

## Credits

- [FER2013 Vision Encoder](https://huggingface.co/tanganke/clip-vit-large-patch14_fer2013) - By `tanganke`
- [Base CLIP Model](https://huggingface.co/openai/clip-vit-large-patch14) - By `openai`
- [vit-explain](https://github.com/jacobgil/vit-explain) - Original gradient rollout explanation method for PyTorch by Jacob Gildenblat. Modified for Hugging Face transformer layers and CUDA usage here.
- [RetinaFace](https://github.com/serengil/retinaface) - Face detection model by Serengil.

## Problems & Future Work

We find that the Gradient Attention Rollout (by Jacob Gildenblat) works well for vanilla ViT on imagenet classification. However, it suffers from poor visualization quality when applied to the facial emotion recognition task using finetuned CLIP ViT-L/14. Specifically, there are often meaningless areas with extremely high importance (even off the face). We use a trick to discard the extremely large portion of mask which seems to lie out of a resonable distribution (see detail in `src/inference.py`'s `enhance_rollout_mask()` function). After this, the visualization quality is improved greatly and the true facial features (e.g., eyes, mouth) are highlighted more clearly.

Actually, this phenomenon does not only happen to finetuned models. For more details, please refer to the paper "VISION TRANSFORMERS NEED REGISTERS".

```bibtex
@misc{darcet2024visiontransformersneedregisters,
      title={Vision Transformers Need Registers}, 
      author={Timothée Darcet and Maxime Oquab and Julien Mairal and Piotr Bojanowski},
      year={2024},
      eprint={2309.16588},
      archivePrefix={arXiv},
      primaryClass={cs.CV},
      url={https://arxiv.org/abs/2309.16588}, 
}
```
