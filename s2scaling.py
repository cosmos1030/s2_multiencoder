from PIL import Image
import requests
from transformers import AutoProcessor, CLIPVisionModel
from s2wrapper import forward as multiscale_forward


model = CLIPVisionModel.from_pretrained("openai/clip-vit-base-patch32")
processor = AutoProcessor.from_pretrained("openai/clip-vit-base-patch32")

url = "http://images.cocodataset.org/val2017/000000039769.jpg"
image = Image.open(requests.get(url, stream=True).raw)

inputs = processor(images=image, return_tensors="pt").pixel_values

# wrap the feature extraction process into a single function that
# takes image tensor as input and outputs feature tensor
def forward_features(inputs):
    return model(inputs).last_hidden_state

# extracting features with scales=[1, 2]. Note the output has one [CLS] token
# so setting num_prefix_token=1.
outputs = multiscale_forward(forward_features, inputs, scales=[1, 2], num_prefix_token=1)
print(outputs.shape)  # 1*50*1536