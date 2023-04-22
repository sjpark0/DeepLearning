from torchvision.io import read_image
from torchvision.models import vit_b_16, vit_b_32, vit_h_14, vit_l_16, vit_l_32, ViT_B_16_Weights, ViT_B_32_Weights, ViT_H_14_Weights, ViT_L_16_Weights, ViT_L_32_Weights
import torch

img = read_image("./data/Cat03.jpg")

# Step 1: Initialize model with the best available weights
weights = ViT_B_16_Weights.DEFAULT
model = vit_b_16(weights=weights)
model.eval()

# Step 2: Initialize the inference transforms
preprocess = weights.transforms()

# Step 3: Apply inference preprocessing transforms
batch = preprocess(img).unsqueeze(0)

# Step 4: Use the model and print the predicted category
#prediction = model(batch).squeeze(0).softmax(0)
tt = model(batch).squeeze(0)
print(tt.shape)
prediction = tt.softmax(0)
class_id = prediction.argmax().item()
score = prediction[class_id].item()
category_name = weights.meta["categories"][class_id]
print(f"{category_name}: {100 * score:.1f}%")
