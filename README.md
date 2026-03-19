# Emotion Prediction & Vision Transformer Explainability

This project uses a CLIP ViT-L/14 based pipeline to predict facial emotions from images and adds a **Gradient Attention Rollout** mechanism to visualize *where* the model is looking in order to make that prediction.

The attention rollout produces a heatmap overlay showing the facial regions (e.g., eyes, mouth) that most strongly contributed to the predicted emotion.

## Requirements

The project uses Python. Ensure you have installed the required dependencies:

```bash
pip install -r requirements.txt
```

*(Note: As specified, `torch` and `numpy` should already be available in the environment).*

## Setup

First, download and assemble the local CLIP runtime once:

- Vision encoder: `tanganke/clip-vit-large-patch14_fer2013` (fine-tuned on FER2013)
- Text branch + processor: `openai/clip-vit-large-patch14`

```bash
python src/download_model.py
```

The script merges the fine-tuned vision encoder into the base CLIP model and stores everything in `models/clip_fer2013/` so you can run inferences entirely offline in the future.

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

## Problems & Future Work

We find that the Gradient Attention Rollout (by Jacob Gildenblat) works well for vanilla ViT on imagenet classification. However, it suffers from poor visualization quality when applied to the facial emotion recognition task using finetuned CLIP ViT-L/14. Specifically, there are often meaningless areas with extremely high importance (even off the face). We use a trick to discard the extremely large portion of mask which seems to lie out of a resonable distribution (see detail in `src/inference.py`'s `enhance_rollout_mask()` function). After this, the visualization quality is improved greatly and the true facial features (e.g., eyes, mouth) are highlighted more clearly.

Actually, this phenomenon does not only happen to finetuned models. For more details, please refer to the paper "VISION TRANSFORMERS NEED REGISTERS".

```
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