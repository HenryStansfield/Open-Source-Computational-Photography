import cv2
import torch
import numpy as np
from PIL import Image
from torchvision.transforms import ToTensor, Normalize, Resize, Compose

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
midas = torch.hub.load("intel-isl/MiDaS", "MiDaS_small").to(device)
#midas = torch.hub.load("intel-isl/MiDaS", "DPT_Large").to(device)
#midas = torch.hub.load("intel-isl/MiDaS", "DPT_Hybrid").to(device)
midas.eval()
def get_depth_map(image):
    """
    Get depth map from an image using MiDaS model.
    """
    transform = Compose([
        Resize((384, 384)),
        ToTensor(),
        Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])
    
    image_tensor = transform(image).unsqueeze(0).to(device)
    with torch.no_grad():
        depth_map = midas(image_tensor)
    
    depth_map = depth_map.squeeze().cpu().numpy()
    depth_map = cv2.resize(depth_map, (image.width, image.height))
    
    return depth_map

def get_multiple_depth_maps(image):
    """
    This function is used for testing different MiDaS models.
    """
    midas_small = torch.hub.load("intel-isl/MiDaS", "MiDaS_small").to(device)
    transform_small = Compose([
        Resize((384, 384)),
        ToTensor(),
        Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])
    midas_large = torch.hub.load("intel-isl/MiDaS", "DPT_Large").to(device)
    transform_large = Compose([
        ToTensor(),
        Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])
    midas_hybrid = torch.hub.load("intel-isl/MiDaS", "DPT_Hybrid").to(device)
    transform_hybrid = Compose([
        Resize((384, 384)),
        ToTensor(),
        Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])
    import time
    start_time = time.time()
    image_tensor_small = transform_small(image).unsqueeze(0).to(device)
    with torch.no_grad():
        depth_map_small = midas_small(image_tensor_small)
    depth_map_small = depth_map_small.squeeze().cpu().numpy()
    depth_map_small = cv2.resize(depth_map_small, (image.width, image.height))
    print("Small model time:", time.time() - start_time)
    # show depth_map_small
    start_time = time.time()
    image_tensor_large = transform_large(image).unsqueeze(0).to(device)
    with torch.no_grad():
        depth_map_large = midas_large(image_tensor_large)
    depth_map_large = depth_map_large.squeeze().cpu().numpy()
    depth_map_large = cv2.resize(depth_map_large, (image.width, image.height))
    print("Large model time:", time.time() - start_time)
    # show depth_map_large
    start_time = time.time()
    image_tensor_hybrid = transform_hybrid(image).unsqueeze(0).to(device)
    with torch.no_grad():
        depth_map_hybrid = midas_hybrid(image_tensor_hybrid)
    depth_map_hybrid = depth_map_hybrid.squeeze().cpu().numpy()
    depth_map_hybrid = cv2.resize(depth_map_hybrid, (image.width, image.height))
    print("Hybrid model time:", time.time() - start_time)
    # show depth_map_hybrid
    return depth_map_small, depth_map_large, depth_map_hybrid

def get_depth_map_low_res(image):
    transform = Compose([
        Resize((384, 384)),
        ToTensor(),
        Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])
    
    image_tensor = transform(image).unsqueeze(0).to(device)
    with torch.no_grad():
        depth_map = midas(image_tensor)
    
    depth_map = depth_map.squeeze().cpu().numpy()
    
    return depth_map

def blur_based_on_depth(image, depth_map, blur_strength=35, focus_depth=0.9, threshold=0.4):
    """
    Applies a depth-based blur to the image, using a Gaussian blur for areas outside the focus depth.
    Starting point for the project.
    """
    print("Depth min:", depth_map.min(), "max:", depth_map.max())
    depth_map_normalized = cv2.normalize(depth_map, None, 0.0, 1.0, cv2.NORM_MINMAX)
    print("Normalized depth min:", depth_map_normalized.min(), "max:", depth_map_normalized.max())
    
    image_np = np.array(image)
    
    depth_norm = cv2.normalize(depth_map, None, 0.0, 1.0, cv2.NORM_MINMAX)

    mask = np.abs(depth_norm - focus_depth) < threshold
    mask = mask.astype(np.float32)

    mask = cv2.GaussianBlur(mask, (15, 15), 0)
    mask_3ch = np.repeat(mask[:, :, np.newaxis], 3, axis=2)
    blurred = cv2.GaussianBlur(image_np, (blur_strength, blur_strength), 0)

    composite = (image_np * mask_3ch + blurred * (1 - mask_3ch)).astype(np.uint8)

    return Image.fromarray(composite)

def blur_based_on_depth_variable(image, depth_map, blur_levels=5, max_blur=80):
    """
    Simple variable depth-based blur that uses a linear scale for blur strength.
    """
    image_np = np.array(image)
    h, w, _ = image_np.shape

    depth_norm = cv2.normalize(depth_map, None, 0.0, 1.0, cv2.NORM_MINMAX)
    depth_norm = 1.0 - depth_norm

    blurred_versions = []
    for i in range(blur_levels):
        blur_amount = 1 + 2 * int((i / (blur_levels - 1)) * (max_blur // 2))
        blurred = cv2.GaussianBlur(image_np, (blur_amount, blur_amount), 0)
        blurred_versions.append(blurred)

    depth_scaled = (depth_norm * (blur_levels - 1)).astype(np.uint8)
    output_np = np.zeros_like(image_np)

    for i in range(blur_levels):
        mask = (depth_scaled == i).astype(np.uint8)
        mask_3ch = cv2.merge([mask]*3)
        output_np += blurred_versions[i] * mask_3ch

    return Image.fromarray(output_np.astype(np.uint8))

import matplotlib.pyplot as plt
def blur_based_on_depth_smooth(image, depth_map, focus_depth, max_blur=160, blur_levels=15):
    """
    Smooth depth-based blur that uses a quadratic curve for blur strength.
    """
    image_np = np.array(image)
    h, w, _ = image_np.shape

    depth_norm = cv2.normalize(depth_map, None, 0.0, 1.0, cv2.NORM_MINMAX)
    depth_norm = 1.0 - depth_norm

    depth_diff = np.abs(depth_norm - focus_depth)

    max_diff = np.max(depth_diff)
    depth_weight = depth_diff / (max_diff + 1e-6)
    depth_weight = depth_weight ** 2

    depth_bins = (depth_weight * (blur_levels - 1)).astype(np.uint8)

    blurred_versions = []
    for i in range(blur_levels):
        curve_pos = i / (blur_levels - 1)
        kernel_scale = curve_pos ** 2
        k = 1 + 2 * int(kernel_scale * (max_blur // 2))
        k = max(k, 1)
        if k % 2 == 0:
            k += 1
        blurred = cv2.GaussianBlur(image_np, (k, k), 0)
        #if k is over a threshold apply lens blur over the top
        #if k > 25:
        #blurred = lens_blur(blurred, blur_size=k, aperture_shape=6)
        #if blurred.shape[0] != h or blurred.shape[1] != w:
        #    blurred = cv2.resize(blurred, (w, h), interpolation=cv2.INTER_LINEAR)
        blurred_versions.append(blurred)

    output = np.zeros_like(image_np, dtype=np.float32)
    for i in range(blur_levels):
        mask = (depth_bins == i).astype(np.float32)
        mask = cv2.GaussianBlur(mask, (15, 15), 0)
        mask_3ch = np.repeat(mask[:, :, np.newaxis], 3, axis=2)
        output += blurred_versions[i].astype(np.float32) * mask_3ch

    foreground_mask = (depth_diff < 0.02).astype(np.float32)
    #foreground_mask = cv2.GaussianBlur(foreground_mask, (9, 9), 0)
    foreground_mask_3ch = np.repeat(foreground_mask[:, :, np.newaxis], 3, axis=2)

    output = output * (1 - foreground_mask_3ch) + image_np.astype(np.float32) * foreground_mask_3ch

    return Image.fromarray(np.clip(output, 0, 255).astype(np.uint8))

def blur_based_on_depth_smooth_inpaint(image, depth_map, focus_depth, max_blur=160, blur_levels=15, debug=False):
    """
    This is the final version of the depth-based blur function that uses inpainting to fill in the background.
    Based on blur_based_on_depth_smooth but with inpainting for the background.
    Also uses a more realistic lens blur for the background, combined with a Gaussian blur to look nicer.
    """
    image_np = np.array(image)
    h, w, _ = image_np.shape

    depth_norm = cv2.normalize(depth_map, None, 0.0, 1.0, cv2.NORM_MINMAX)
    depth_norm = 1.0 - depth_norm # Invert (plant darker, background brighter)

    depth_diff = np.abs(depth_norm - focus_depth)


    foreground_threshold = 0.25

    true_subject_mask_binary = (depth_diff < foreground_threshold).astype(np.uint8) 

    kernel_small = np.ones((3,3), np.uint8)
    true_subject_mask_binary = cv2.morphologyEx(true_subject_mask_binary, cv2.MORPH_OPEN, kernel_small, iterations=1) 
    true_subject_mask_binary = cv2.morphologyEx(true_subject_mask_binary, cv2.MORPH_CLOSE, kernel_small, iterations=1) 
    true_subject_mask_binary = cv2.dilate(true_subject_mask_binary, np.ones((3,3), np.uint8), iterations=1)

    if debug:
        plt.figure(figsize=(15, 5))

        plt.subplot(1, 2, 1)
        plt.title('DEBUG: Depth Map (Normalized, Inverted)')
        plt.imshow(depth_norm, cmap='gray')
        plt.axis('off')

        plt.subplot(1, 2, 2)
        plt.title('DEBUG: TRUE Subject Mask (White for Subject)')
        plt.imshow(true_subject_mask_binary, cmap='gray')
        plt.axis('off')
        
        print(f"DEBUG: Min depth_norm: {np.min(depth_norm):.4f}, Max depth_norm: {np.max(depth_norm):.4f}")
        print(f"DEBUG: Focus depth set to: {focus_depth:.4f}")
        print(f"DEBUG: Foreground threshold: {foreground_threshold:.4f}")
        print(f"DEBUG: Sum of TRUE Subject Mask: {np.sum(true_subject_mask_binary)} (should be > 0 and < total_pixels)")

        plt.show()

    # Convert true_subject_mask_binary to 3 channel
    true_subject_mask_3ch = np.repeat(true_subject_mask_binary[:, :, np.newaxis], 3, axis=2)



    inpainting_mask = true_subject_mask_binary * 255

    if debug:
        plt.figure(figsize=(15, 5))
        plt.subplot(1, 2, 1)
        plt.title('DEBUG: Inpainting Mask (255=Hole)')
        plt.imshow(inpainting_mask, cmap='gray')
        plt.axis('off')

        # Checking validity of inpainting mask for debugging
        if np.sum(inpainting_mask) == 0:
            print("WARNING: Inpainting mask is empty (all black). No inpainting will occur. Adjust 'foreground_threshold' or 'focus_depth'.")
            background_inpainted = image_np.copy()
        elif np.sum(inpainting_mask) == inpainting_mask.size * 255:
            print("WARNING: Inpainting mask covers entire image (all white). No meaningful inpainting will occur. Adjust 'foreground_threshold' or 'focus_depth'.")
            background_inpainted = image_np.copy()
        else:
            background_inpainted = cv2.inpaint(
                image_np, 
                inpainting_mask,
                inpaintRadius=5, 
                flags=cv2.INPAINT_TELEA 
            )
            plt.subplot(1, 2, 2)
            plt.title('DEBUG: Inpainted Background')
            background_inpainted_rgb = cv2.cvtColor(background_inpainted, cv2.COLOR_BGR2RGB)
            plt.imshow(background_inpainted_rgb)
            plt.axis('off')

        plt.show()
    else:
        background_inpainted = image_np.copy()


    max_diff = np.max(depth_diff)
    depth_weight = depth_diff / (max_diff + 1e-6)
    depth_weight = depth_weight ** 2

    depth_bins = (depth_weight * (blur_levels - 1)).astype(np.uint8)

    blurred_versions = []
    for i in range(blur_levels):
        curve_pos = i / (blur_levels - 1)
        kernel_scale = curve_pos ** 2
        k = 1 + 2 * int(kernel_scale * (max_blur // 2))
        k = max(k, 1)
        
        if k % 2 == 0:
            k += 1
        
        blurred = cv2.GaussianBlur(background_inpainted, (k, k), 0) 
        
        if k > 25:
            blurred = lens_blur(blurred, blur_size=k, aperture_shape=6)
        
        blurred_versions.append(blurred)

    output = np.zeros_like(image_np, dtype=np.float32)
    for i in range(blur_levels):
        mask = (depth_bins == i).astype(np.float32)
        mask = cv2.GaussianBlur(mask, (25, 25), 0) 
        mask_3ch = np.repeat(mask[:, :, np.newaxis], 3, axis=2)
        output += blurred_versions[i].astype(np.float32) * mask_3ch

    final_combined_image = output * (1 - true_subject_mask_3ch) + image_np.astype(np.float32) * true_subject_mask_3ch
    
    return Image.fromarray(np.clip(final_combined_image, 0, 255).astype(np.uint8))



from skimage import color

import cv2
import numpy as np
from skimage import color
from PIL import Image # Assuming PIL is used for initial image loading
# from scipy.ndimage import binary_erosion # Not strictly needed if we iterate over pixels


def lens_blur(image, blur_size=25, aperture_shape=6):
    """
    Simulates a more realistic lens blur with a specified aperture shape.
    """
    if isinstance(image, Image.Image):
        img = np.array(image)
    else:
        img = image.copy()

    if blur_size % 2 == 0:
        blur_size += 1

    k = blur_size
    aperture = np.zeros((k, k), np.float32)
    center = k // 2
    radius = k // 2

    pts = []
    for i in range(aperture_shape):
        angle = 2 * np.pi * i / aperture_shape
        pts.append((center + radius * np.cos(angle), center + radius * np.sin(angle)))
    pts = np.array(pts, np.int32)

    cv2.fillConvexPoly(aperture, pts, 1)
    aperture /= np.sum(aperture)

    blurred = np.zeros_like(img, dtype=np.float32)
    for c in range(3):
        blurred[:, :, c] = cv2.filter2D(img[:, :, c].astype(np.float32), -1, aperture)

    return blurred.astype(img.dtype)



def blur_based_on_depth_aggressive(image, depth_map, focus_depth=0.01, max_blur=65, min_blur=1):
    """
    Aggressive depth-of-field blur using a quadratic curve for blur strength, to see if this would reduce haloing.
    """
    image_np = np.array(image).astype(np.float32)

    depth_norm = cv2.normalize(depth_map, None, 0.0, 1.0, cv2.NORM_MINMAX)
    depth_diff = np.abs(depth_norm - focus_depth)

    # x^2 curve to increase blur more aggressively with distance
    blur_strength_map = (depth_diff ** 2)

    blur_strength_map = np.clip(blur_strength_map, 0, 1)
    blur_kernels = (min_blur + (max_blur - min_blur) * blur_strength_map)
    blur_kernels = (2 * (blur_kernels // 2) + 1).astype(np.uint8)  # force odd kernel size

    output = np.zeros_like(image_np)

    unique_kernels = np.unique(blur_kernels)
    for k in unique_kernels:
        if k < 3:  # skip almost no-blur
            continue
        mask = (blur_kernels == k).astype(np.uint8)
        if np.count_nonzero(mask) == 0:
            continue
        mask_3ch = np.repeat(mask[:, :, np.newaxis], 3, axis=2)
        blurred = cv2.GaussianBlur(image_np, (int(k), int(k)), 0)
        output += blurred * mask_3ch

    sharp_mask = (depth_diff < 0.01).astype(np.uint8)
    sharp_mask_3ch = np.repeat(sharp_mask[:, :, np.newaxis], 3, axis=2)
    output = output * (1 - sharp_mask_3ch) + image_np * sharp_mask_3ch

    return Image.fromarray(np.clip(output, 0, 255).astype(np.uint8))

import numpy as np
import cv2
from PIL import Image

def add_bokeh_highlights(image_np, depth_diff, focus_depth=0.01, threshold=230, bokeh_size=25):
    """
    Adds bokeh highlights to out-of-focus bright spots in the image. Experimental.
    """
    gray = cv2.cvtColor(image_np.astype(np.uint8), cv2.COLOR_RGB2GRAY)
    bright_mask = (gray > threshold).astype(np.uint8) * 255

    highlights = cv2.bitwise_and(image_np.astype(np.uint8), image_np.astype(np.uint8), mask=bright_mask)

    k = bokeh_size | 1
    bokeh_kernel = np.zeros((k, k), dtype=np.float32)
    cv2.circle(bokeh_kernel, (k // 2, k // 2), k // 2, 1, -1)
    bokeh_kernel /= np.sum(bokeh_kernel)
    bokeh_layer = np.zeros_like(image_np)
    for c in range(3):
        bokeh_layer[:, :, c] = cv2.filter2D(highlights[:, :, c], -1, bokeh_kernel)

    depth_diff_abs = np.abs(depth_diff)
    dof_mask = (depth_diff_abs > 0.03).astype(np.float32)
    dof_mask = cv2.GaussianBlur(dof_mask, (15, 15), 0)
    dof_mask_3ch = np.repeat(dof_mask[:, :, np.newaxis], 3, axis=2)

    result = image_np + bokeh_layer.astype(np.float32) * dof_mask_3ch * 0.6  # scale for realism

    return np.clip(result, 0, 255)


def blur_based_on_depth_realistic_lens(image, depth_map, focus_depth=0.01,
                                       max_blur=65, min_blur=1, blur_levels=10, aperture_shape=6):
    """
    Depth-of-field blur using lens blur kernels instead of Gaussian.
    """
    image_np = np.array(image).astype(np.float32)
    h, w, _ = image_np.shape

    depth_norm = cv2.normalize(depth_map, None, 0.0, 1.0, cv2.NORM_MINMAX)
    depth_norm = 1.0 - depth_norm

    depth_diff = depth_norm - focus_depth

    blur_weight = np.zeros_like(depth_diff)
    blur_weight[depth_diff < 0] = np.abs(depth_diff[depth_diff < 0]) ** 1.2
    blur_weight[depth_diff > 0] = np.abs(depth_diff[depth_diff > 0]) ** 0.8

    blur_weight /= np.max(blur_weight + 1e-6)
    depth_bins = (blur_weight * (blur_levels - 1)).astype(np.uint8)

    blurred_versions = []
    for i in range(blur_levels):
        norm_pos = i / (blur_levels - 1)
        blur_amount = min_blur + norm_pos * (max_blur - min_blur)
        blurred = lens_blur(image_np, blur_size=int(blur_amount), aperture_shape=aperture_shape)
        blurred_versions.append(blurred)

    output = np.zeros_like(image_np)
    for i in range(blur_levels):
        mask = (depth_bins == i).astype(np.float32)
        mask = cv2.GaussianBlur(mask, (21, 21), 0)
        mask_3ch = np.repeat(mask[:, :, np.newaxis], 3, axis=2)
        output += blurred_versions[i] * mask_3ch

    in_focus_mask = (np.abs(depth_diff) < 0.015).astype(np.float32)
    in_focus_mask = cv2.GaussianBlur(in_focus_mask, (9, 9), 0)
    in_focus_mask_3ch = np.repeat(in_focus_mask[:, :, np.newaxis], 3, axis=2)

    output = output * (1 - in_focus_mask_3ch) + image_np * in_focus_mask_3ch

    return Image.fromarray(np.clip(output, 0, 255).astype(np.uint8))

def blur_based_on_depth_two_zone(image, depth_map, focus_depth=0.01,
                                 gaussian_zone=0.05, lens_zone=0.2,
                                 gaussian_strength=25, max_lens_blur=65,
                                 aperture_shape=6):
    """
    Depth-of-field blur with two zones:
    - Near focus plane: Gaussian blur (soft transition)
    - Farther from subject: Lens blur (bokeh)
    """
    img_np = np.array(image).astype(np.float32)
    h, w, _ = img_np.shape

    depth_norm = cv2.normalize(depth_map, None, 0.0, 1.0, cv2.NORM_MINMAX)
    depth_norm = 1.0 - depth_norm
    depth_diff = np.abs(depth_norm - focus_depth)
    gaussian_blur_img = cv2.GaussianBlur(img_np, (gaussian_strength, gaussian_strength), 0)
    lens_blur_img = lens_blur(img_np, blur_size=max_lens_blur, aperture_shape=aperture_shape)
    gaussian_weight = np.clip(1 - (depth_diff - gaussian_zone) / (lens_zone - gaussian_zone), 0, 1)
    lens_weight = np.clip((depth_diff - gaussian_zone) / (lens_zone - gaussian_zone), 0, 1)
    gaussian_weight = cv2.GaussianBlur(gaussian_weight, (31, 31), 0)
    lens_weight = cv2.GaussianBlur(lens_weight, (31, 31), 0)
    gaussian_w3 = np.repeat(gaussian_weight[:, :, None], 3, axis=2)
    lens_w3 = np.repeat(lens_weight[:, :, None], 3, axis=2)
    sharp_w3 = 1.0 - (gaussian_w3 + lens_w3)
    result = img_np * sharp_w3 + gaussian_blur_img * gaussian_w3 + lens_blur_img * lens_w3

    return Image.fromarray(np.clip(result, 0, 255).astype(np.uint8))




def distanceDisplayAsGrayscale(depth_map):
    # Normalize to 0–255 for colormap
    depth_map_normalized = cv2.normalize(depth_map, None, 0, 255, cv2.NORM_MINMAX)
    depth_map_normalized = depth_map_normalized.astype(np.uint8)
    
    depth_map_colored = cv2.applyColorMap(depth_map_normalized, cv2.COLORMAP_JET)
    depth_map_colored = cv2.cvtColor(depth_map_colored, cv2.COLOR_BGR2RGB)
    return Image.fromarray(depth_map_colored)

def sharpen_depth_map(depth_map, image, sigma=1.0, edge_weight=1.0):
    """
    Enhance depth map edges using colour lines from the original image
    """
    image_np = np.array(image).astype(np.float32) / 255.0
    depth_map = depth_map.astype(np.float32)

    guidance = cv2.cvtColor((image_np * 255).astype(np.uint8), cv2.COLOR_RGB2GRAY)
    guidance = guidance.astype(np.float32) / 255.0

    depth_norm = cv2.normalize(depth_map, None, 0, 1, cv2.NORM_MINMAX)

    edge_mask = cv2.Laplacian(guidance, cv2.CV_32F)
    edge_mask = np.abs(edge_mask)
    edge_mask = cv2.normalize(edge_mask, None, 0.0, 1.0, cv2.NORM_MINMAX)

    sharpened = depth_norm + edge_weight * edge_mask
    #sharpened = cv2.GaussianBlur(sharpened, (3, 3), sigmaX=sigma)
    sharpened = cv2.normalize(sharpened, None, 0.0, 1.0, cv2.NORM_MINMAX)

    return sharpened

def map_subject_distance_to_focus_depth(subject_distance_meters, depth_map):
    """
    Maps a real-world subject distance in meters to a normalized focus_depth value.
    """
    if subject_distance_meters <= 0:
        raise ValueError("Subject distance must be positive.")
    estimated_midas_depth = 1.0 / subject_distance_meters
    dmin = np.min(depth_map)
    dmax = np.max(depth_map)
    focus_depth = (estimated_midas_depth - dmin) / (dmax - dmin)
    focus_depth = np.clip(focus_depth, 0.0, 1.0)

    return focus_depth

def test_all_models(image):
    depth_small, depth_large, depth_map_hybrid = get_multiple_depth_maps(image) 
    depth_small_image = sharpen_depth_map(depth_small, image)
    #depth_small_image = distanceDisplayAsGrayscale(depth_small_image)
    #depth_small_image.show()
    blur_small = blur_based_on_depth_smooth(image, depth_small_image, focus_depth=0.3)
    blur_small.show()
    depth_large_image = sharpen_depth_map(depth_large, image)
    #depth_large_image = distanceDisplayAsGrayscale(depth_large_image)
    #depth_large_image.show()
    blur_large = blur_based_on_depth_smooth(image, depth_large_image, focus_depth=0.3)
    blur_large.show()
    depth_map_hybrid_image = sharpen_depth_map(depth_map_hybrid, image)
    #depth_map_hybrid_image = distanceDisplayAsGrayscale(depth_map_hybrid_image)
    #depth_map_hybrid_image.show()
    blur_hybrid = blur_based_on_depth_smooth(image, depth_map_hybrid_image, focus_depth=0.3)
    blur_hybrid.show()

def process_image(image_path):
    image = Image.open(image_path).convert("RGB")
    image = image.resize((1024, 1024), Image.LANCZOS)
    depth_map = get_depth_map(image)



    sharpened_depth = sharpen_depth_map(depth_map, image)

    
    depth_map_image = distanceDisplayAsGrayscale(sharpened_depth)
    depth_map_image.show()

    focus_depth = float(input("Enter focus depth (0.01 for close, 0.9 for far): ") or 0.01)
    max_blur = int(input("Enter max blur strength (default 160): ") or 160)
    blurred_image = blur_based_on_depth_smooth_inpaint(image, sharpened_depth, focus_depth=focus_depth, max_blur=max_blur, blur_levels=15, debug=True)
    return blurred_image



image_path = "flowernobokhe.jpg"

if __name__ == "__main__":
    final_image = process_image(image_path)
    final_image.show()
    final_image.save("67focused_output.jpg")
