from PIL import Image
import requests
from transformers import AutoProcessor, CLIPVisionModel

model = CLIPVisionModel.from_pretrained("openai/clip-vit-base-patch32")
processor = AutoProcessor.from_pretrained("openai/clip-vit-base-patch32")

url = "http://images.cocodataset.org/val2017/000000039769.jpg"
image = Image.open(requests.get(url, stream=True).raw)

inputs = processor(images=image, return_tensors="pt").pixel_values

# model.forward returns an object that contains "last_hidden_state" which is the feature map we need
outputs = model(inputs).last_hidden_state
print(outputs.shape)  # 1*50*768