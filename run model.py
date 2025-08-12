import os
import torch
from torch.utils.data import DataLoader, Dataset
from torchvision import transforms
from PIL import Image
import matplotlib.pyplot as plt
import numpy as np


from ai_classes import ImprovedUNet, BokehDataset
import torch.nn as nn



# Paths
model_path = 'bokeh_effect_model.pth'
val_dir = 'train'
original_dir = 'original'
bokeh_dir = 'bokeh'

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')


transform = transforms.Compose([
    transforms.Resize((256, 256)),
    transforms.ToTensor(),
])


# Load dataset
val_dataset = BokehDataset(
    orig_folder=os.path.join(val_dir, original_dir),
    bokeh_folder=os.path.join(val_dir, bokeh_dir),
    transform=transform
)
val_loader = DataLoader(val_dataset, batch_size=1, shuffle=True)

# Load model
model = ImprovedUNet().to(device)
model.load_state_dict(torch.load(model_path, map_location=device))
model.eval()

def tensor_to_image(tensor):
    tensor = tensor.squeeze().cpu().numpy()
    tensor = np.transpose(tensor, (1, 2, 0))
    return np.clip(tensor, 0, 1)

with torch.no_grad():
    for i, (orig, bokeh) in enumerate(val_loader):
        orig, bokeh = orig.to(device), bokeh.to(device)
        output = model(orig)

        orig_img = tensor_to_image(orig)
        pred_img = tensor_to_image(output)
        bokeh_img = tensor_to_image(bokeh)

        fig, axes = plt.subplots(1, 3, figsize=(12, 4))
        axes[0].imshow(orig_img)
        axes[0].set_title('Original')
        axes[1].imshow(pred_img)
        axes[1].set_title('Predicted Bokeh')
        axes[2].imshow(bokeh_img)
        axes[2].set_title('Ground Truth Bokeh')

        for ax in axes:
            ax.axis('off')
        plt.tight_layout()
        plt.show()

        # the actual dataset contains thousands of images, for now we only want a few
        if i >= 4:
            break
