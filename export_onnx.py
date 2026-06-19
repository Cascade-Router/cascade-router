from optimum.onnxruntime import ORTModelForFeatureExtraction
from transformers import AutoTokenizer

print("Downloading and exporting model to ONNX...")
model_id = "sentence-transformers/all-MiniLM-L6-v2"

# Load tokenizer and export the model
tokenizer = AutoTokenizer.from_pretrained(model_id)
model = ORTModelForFeatureExtraction.from_pretrained(model_id, export=True)

# Save to our models directory
save_dir = "models/all-MiniLM-L6-v2-onnx"
tokenizer.save_pretrained(save_dir)
model.save_pretrained(save_dir)

print(f"Success! Model saved to {save_dir}")