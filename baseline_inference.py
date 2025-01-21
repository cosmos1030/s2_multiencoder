import json
import os

import pandas as pd
import torch

from open_clip import create_model_from_pretrained, get_tokenizer
from PIL import Image
from sklearn.metrics import accuracy_score, classification_report

data_path = '/home/pv/Project/s2/imagenet100'

with open(os.path.join(data_path, 'Labels.json'), 'r') as f:
    labels = json.load(f)

images = []

for label in os.listdir(os.path.join(data_path, 'val.X')):
    for filename in os.listdir(os.path.join(data_path, f'val.X/{label}')):
        images.append({
            'path': os.path.join(data_path, f'val.X/{label}/{filename}'),
            'label': labels[label],
        })

df = pd.DataFrame(images)

model_name = 'UCSC-VLAA/ViT-H-14-CLIPA-336-laion2B'

model, preprocess = create_model_from_pretrained(f'hf-hub:{model_name}')
tokenizer = get_tokenizer(f'hf-hub:{model_name}')


model = model.to('cuda')

# print(preprocess)

# print(model)

labels = list(labels.values())

text_descriptions = [f"This is a photo of a {label}" for label in labels]
text_tokens = tokenizer(labels, context_length=model.context_length).to('cuda')

with torch.no_grad():
    text_features = model.encode_text(text_tokens).float()
    text_features /= text_features.norm(dim=-1, keepdim=True)

predictions = []

for index, row in df.iterrows():
    image = Image.open(row['path'])
    inputs = preprocess(image).unsqueeze(0).to('cuda')

    with torch.no_grad():
        image_features = model.encode_image(inputs)
        image_features /= image_features.norm(dim=-1, keepdim=True)

        # text_probs = (100.0 * image_features @ text_features.T).softmax(dim=-1)
        text_probs = (100.0 * image_features.matmul(text_features.T)).softmax(dim=-1)

    predicted_label = text_probs.argmax(-1).item()
    predictions.append(labels[predicted_label])

    print(f'{index + 1}/{len(df)}', end='\r')

df['prediction'] = predictions

print(accuracy_score(df['label'], df['prediction']))