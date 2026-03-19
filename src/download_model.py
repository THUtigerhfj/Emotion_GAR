import os
from transformers import AutoImageProcessor, ViTForImageClassification

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

if __name__ == "__main__":
    download_model()
