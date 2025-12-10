from transformers import Qwen2VLForConditionalGeneration, AutoProcessor
import torch

print("Downloading Qwen2-VL-7B to local cache...")
model = Qwen2VLForConditionalGeneration.from_pretrained(
    "Qwen/Qwen2-VL-7B-Instruct",
    torch_dtype=torch.bfloat16,
    device_map="auto",
)
processor = AutoProcessor.from_pretrained("Qwen/Qwen2-VL-7B-Instruct")
print("Download complete.")