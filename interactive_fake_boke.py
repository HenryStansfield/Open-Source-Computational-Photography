import cv2
import torch
import numpy as np
from PIL import Image
from torchvision.transforms import ToTensor, Normalize, Resize, Compose
from fake_bokeh import get_depth_map, blur_based_on_depth_smooth_inpaint

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
midas = torch.hub.load("intel-isl/MiDaS", "MiDaS_small").to(device)
midas.eval()

def get_depth_map(image):
    transform = Compose([
        Resize((384, 384)),
        ToTensor(),
        Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])
    image_tensor = transform(image).unsqueeze(0).to(device)
    with torch.no_grad():
        depth_map = midas(image_tensor)
    depth_map = depth_map.squeeze().cpu().numpy()
    return cv2.resize(depth_map, image.size)

def select_focus_point(image_np, display_max_size=800):
    clicked = {"coords": None}

    h, w = image_np.shape[:2]
    scale = min(display_max_size / max(h, w), 1.0)
    display_size = (int(w * scale), int(h * scale))
    image_resized = cv2.resize(image_np, display_size)

    def on_mouse(event, x, y, flags, param):
        if event == cv2.EVENT_LBUTTONDOWN:
            orig_x = int(x / scale)
            orig_y = int(y / scale)
            clicked["coords"] = (orig_x, orig_y)
            cv2.destroyAllWindows()

    cv2.imshow("Click to select focus point", cv2.cvtColor(image_resized, cv2.COLOR_RGB2BGR))
    cv2.setMouseCallback("Click to select focus point", on_mouse)
    cv2.waitKey(0)
    return clicked["coords"]


def process_image(image_path):
    image = Image.open(image_path).convert("RGB")
    image_np = np.array(image)

    depth_map = get_depth_map(image)
    focus_coords = select_focus_point(image_np)

    if focus_coords is None:
        print("No focus point selected.")
        return image

    x, y = focus_coords
    depth_norm = cv2.normalize(depth_map, None, 0.0, 1.0, cv2.NORM_MINMAX)
    depth_norm = 1.0 - depth_norm
    focus_depth = depth_norm[y, x]  

    return blur_based_on_depth_smooth_inpaint(image, depth_map, focus_depth)

if __name__ == "__main__":
    image_path = "flowernobokhe.jpg" #this uses the flower image from the EBB dataset
    final_image = process_image(image_path)
    final_image.show()
    final_image.save("focused_output_interactive.jpg")
