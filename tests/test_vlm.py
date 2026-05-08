from src.vision.vlm import QwenVL


qwen = QwenVL(use_flash_attention=False) 

# Text only
response = qwen.generate("What is the capital of France?")
print(response)

# Local image
response = qwen.generate("Describe this image.", image="test_model/test_data/screen/skin_customization_last_of_us.jpg")
print(response)
