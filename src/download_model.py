import os
from transformers import AutoProcessor, CLIPModel, CLIPVisionModel

def download_model():
    vision_model_name = "tanganke/clip-vit-large-patch14_fer2013"
    base_clip_name = "openai/clip-vit-large-patch14"
    
    # Resolve absolute path to models directory relative to this script
    script_dir = os.path.dirname(os.path.abspath(__file__))
    local_path = os.path.join(script_dir, "..", "models", "clip_fer2013")
    
    print(f"Downloading model '{vision_model_name}' from Hugging Face...")
    print(f"This might take a moment. The model will be saved to: {local_path}")

    # Build a full CLIP model by combining:
    # - fine-tuned vision encoder (FER2013)
    # - base CLIP text branch and processor
    processor = AutoProcessor.from_pretrained(base_clip_name)
    clip_model = CLIPModel.from_pretrained(base_clip_name)
    vision_model = CLIPVisionModel.from_pretrained(vision_model_name)

    # CLIPVisionModel wraps parameters under "vision_model.*".
    # CLIPModel.vision_model expects the inner transformer keys directly.
    clip_model.vision_model.load_state_dict(vision_model.vision_model.state_dict(), strict=True)
    
    # Save them locally
    os.makedirs(local_path, exist_ok=True)
    processor.save_pretrained(local_path)
    clip_model.save_pretrained(local_path)
    
    print()
    print(f"✅ Download completed successfully! Model cached at '{local_path}'.")
    print("You can now run inferences locally without needing an internet connection.")

if __name__ == "__main__":
    download_model()
