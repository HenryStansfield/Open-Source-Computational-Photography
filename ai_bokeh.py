import os
import torch
from torch.utils.data import DataLoader, Dataset
from torchvision import transforms, datasets, models
from PIL import Image

from ai_classes import ImprovedUNet, BokehDataset, PerceptualLoss  # Assuming ai_classes.py contains the SimpleUNet and BokehDataset definitions

import torch.nn as nn
import numpy as np

original_dir = 'original'
bokeh_dir = 'bokeh'
train_dir = 'train'
val_dir = 'validation'

transform = transforms.Compose([
    transforms.Resize((256, 256)),
    transforms.ToTensor(),
])

train_dataset = BokehDataset(
    orig_folder=os.path.join(train_dir, original_dir),
    bokeh_folder=os.path.join(train_dir, bokeh_dir),
    transform=transform
)
val_dataset = BokehDataset(
    orig_folder=os.path.join(val_dir, original_dir),
    bokeh_folder=os.path.join(val_dir, bokeh_dir),
    transform=transform
)
train_loader = DataLoader(train_dataset, batch_size=8, shuffle=True)
val_loader = DataLoader(val_dataset, batch_size=8, shuffle=False)


device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
model = ImprovedUNet().to(device)
criterion = PerceptualLoss().to(device)
optimizer = torch.optim.Adam(model.parameters(), lr=1e-4)

# Overtraining on the first image to sanity check
model.train()
orig, bokeh = train_dataset[2] 
orig = orig.unsqueeze(0).to(device)
bokeh = bokeh.unsqueeze(0).to(device)

for i in range(1000):
    output = model(orig)
    loss = criterion(output, bokeh)
    optimizer.zero_grad()
    loss.backward()
    optimizer.step()
    
    if i % 50 == 0:
        print(f"Step {i}, Loss: {loss.item():.6f}")

# use the model to generate bokeh effect on the original image
output = model(orig)
def tensor_to_image(tensor):
    tensor = tensor.squeeze(0)
    tensor = tensor.cpu().detach().numpy()
    tensor = np.transpose(tensor, (1, 2, 0))
    return np.clip(tensor, 0, 1)
output_img = tensor_to_image(output)

import numpy as np
import matplotlib.pyplot as plt
def display_images(orig_img, bokeh_img, output_img):
    fig, axes = plt.subplots(1, 3, figsize=(15, 5))
    axes[0].imshow(orig_img)
    axes[0].set_title('Original Image')
    axes[0].axis('off')

    axes[1].imshow(bokeh_img)
    axes[1].set_title('Bokeh Image')
    axes[1].axis('off')

    axes[2].imshow(output_img)
    axes[2].set_title('Output Image')
    axes[2].axis('off')

    plt.show()
orig_img = tensor_to_image(orig)
bokeh_img = tensor_to_image(bokeh)
display_images(orig_img, bokeh_img, output_img)

"""
for epoch in range(10):
    model.train()
    for orig, bokeh in train_loader:
        orig, bokeh = orig.to(device), bokeh.to(device)
        optimizer.zero_grad()
        output = model(orig)
        loss = criterion(output, bokeh)
        loss.backward()
        optimizer.step()
        if (epoch + 1) % 2 == 0:
            print(f"Batch Loss: {loss.item():.4f}")
    print(f"Epoch {epoch+1}, Loss: {loss.item():.4f}")

    # Validation
    model.eval()
    val_loss = 0
    with torch.no_grad():
        for orig, bokeh in val_loader:
            orig, bokeh = orig.to(device), bokeh.to(device)
            output = model(orig)
            val_loss += criterion(output, bokeh).item()
    print(f"Validation Loss: {val_loss / len(val_loader):.4f}")

# Save model
torch.save(model.state_dict(), 'bokeh_effect_model.pth')
"""