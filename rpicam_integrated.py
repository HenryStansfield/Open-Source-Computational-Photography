import subprocess
import os
import piexif
from PIL import Image
import numpy as np


from fake_bokeh import get_depth_map, sharpen_depth_map, blur_based_on_depth_realistic_lens

photo_path = "/home/hstans/Documents/bokeh-experiment-mybranch/photo.jpg"
blurred_path = photo_path.replace(".jpg", "_blurred.jpg")

print("Capturing image using rpicam-still")
try:
    subprocess.run(["rpicam-still", "-o", photo_path, "--autofocus-on-capture"], check=True)
    print("Image captured:", photo_path)
except subprocess.CalledProcessError as e:
    print("Failed to capture image:", e)
    exit(1)

image = Image.open(photo_path).convert("RGB")

exif_data = piexif.load(photo_path)
subject_distance = None
try:
    distance_rational = exif_data["Exif"][piexif.ExifIFD.SubjectDistance]
    subject_distance = distance_rational[0] / distance_rational[1]
    print(f"EXIF Subject Distance: {subject_distance:.4f} meters")
except KeyError:
    print("No EXIF Subject Distance found. Using default focus.")
    subject_distance = 1.0  # Assume 1 meter if missing

def map_subject_distance_to_focus_depth(subject_distance_meters, depth_map):
    if subject_distance_meters <= 0:
        raise ValueError("Subject distance must be positive.")
    estimated_midas_depth = subject_distance_meters
    dmin = np.min(depth_map)
    dmax = np.max(depth_map)
    focus_depth = (estimated_midas_depth - dmin) / (dmax - dmin)
    return np.clip(focus_depth, 0.0, 1.0)

print("Estimating depth...")
depth_map = get_depth_map(image)
sharpened_depth = sharpen_depth_map(depth_map, image)
focus_depth = map_subject_distance_to_focus_depth(subject_distance, sharpened_depth)

print(f"Applying blur with focus depth: {focus_depth:.3f}")
blurred_image = blur_based_on_depth_realistic_lens(image, sharpened_depth, focus_depth=focus_depth)

blurred_image.save(blurred_path)
print("Blurred image saved:", blurred_path)
