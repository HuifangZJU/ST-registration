#!/usr/bin/env python3

import numpy as np
from skimage import io, color, filters, morphology
import sys
def separate_background_mode(input_path, output_path, delta=10, min_size=100, hole_size=100):
    """
    Reads an RGBA or RGB image from 'input_path'.
    1. Discards the alpha channel if present.
    2. Finds the mode of each channel (R, G, B).
    3. Builds a mask for background by including pixels whose R,G,B each lie
       in [mode - delta, mode + delta].
    4. Removes small objects and fills small holes in the background mask.
    5. Saves the resulting binary mask to 'output_path' (background=white, foreground=black).

    :param input_path:  Path to the input RGBA or RGB image
    :param output_path: Path to save the binary mask
    :param delta:       Half-range around each channel's mode for background
    :param min_size:    Minimum area of background objects to keep
    :param hole_size:   Maximum hole area to fill within background
    """
    # 1. Read the image
    rgba_image = io.imread(input_path)

    # If there's an alpha channel, discard it
    if rgba_image.shape[-1] == 4:
        rgb_image = rgba_image[..., :3]
    else:
        rgb_image = rgba_image

    # Ensure it's uint8 for easy mode calculation (0..255)
    if rgb_image.dtype != np.uint8:
        rgb_image = rgb_image.astype(np.uint8)

    # Extract channels
    r_vals = rgb_image[..., 0].ravel()
    g_vals = rgb_image[..., 1].ravel()
    b_vals = rgb_image[..., 2].ravel()

    # 2. Find the mode of each channel using np.bincount
    r_mode = np.argmax(np.bincount(r_vals))
    g_mode = np.argmax(np.bincount(g_vals))
    b_mode = np.argmax(np.bincount(b_vals))

    # 3. Define a helper to check if channel is in [mode - delta, mode + delta]
    def in_range(channel, ch_mode):
        low = max(ch_mode - delta, 0)
        high = min(ch_mode + delta, 255)
        return (channel >= low) & (channel <= high)

    # 4. Create a boolean background mask
    bg_mask = in_range(rgb_image[..., 0], r_mode)
    bg_mask &= in_range(rgb_image[..., 1], g_mode)
    bg_mask &= in_range(rgb_image[..., 2], b_mode)

    # 5. Remove small "background objects" & fill small holes in the background
    #    remove_small_objects: removes connected True regions < min_size
    #    remove_small_holes: fills connected False regions < hole_size (within True region)
    bg_mask = morphology.remove_small_objects(bg_mask, min_size=min_size)
    bg_mask = morphology.remove_small_holes(bg_mask, area_threshold=hole_size)

    # 6. Convert to a uint8 mask: white=255 for background, black=0 for foreground
    bg_binary = (bg_mask * 255).astype(np.uint8)

    # 7. Save the result
    io.imsave(output_path, bg_binary)
    print(f"Saved mask to {output_path}.")
    print(f"Background mode (R={r_mode}, G={g_mode}, B={b_mode}), delta={delta}, "
          f"min_size={min_size}, hole_size={hole_size}")



def segment_tissue(input_path, output_path):
    """
        Reads an RGBA image from input_path, discards the alpha channel,
        segments tissue using Otsu's thresholding, removes small specks,
        and writes out a binary image (white tissue, black background)
        to output_path.
        """
    # 1. Read the RGBA image
    rgba_image = io.imread(input_path)
    if rgba_image.shape[-1] == 4:
        # Discard the alpha channel, keep only RGB
        rgb_image = rgba_image[..., :3]
    else:
        # In case the image does not actually have an alpha channel
        rgb_image = rgba_image

    # 2. Convert to grayscale
    gray_image = color.rgb2gray(rgb_image)

    # 3. Compute threshold using Otsu's method
    threshold_value = filters.threshold_triangle(gray_image)

    # 4. Create a binary mask where tissue is 'True' (white in final) and background is 'False'
    tissue_mask = gray_image < threshold_value

    # 5. (Optional) Clean up the mask with morphological operations
    tissue_mask = morphology.remove_small_objects(tissue_mask, min_size=200)
    # tissue_mask = morphology.remove_small_holes(tissue_mask, area_threshold=200)

    # 6. Convert the boolean mask to an 8-bit format for saving
    #    White (255) for tissue, Black (0) for background
    tissue_binary = (tissue_mask * 255).astype(np.uint8)

    # 7. Save the result
    io.imsave(output_path, tissue_binary)
    print(f"Saved binary tissue mask to {output_path}.")

if __name__ == "__main__":

    for i in range(3):
        for j in range(4):
            input_image_path = "/home/huifang/workspace/code/registration/data/DLPFC/huifang/"+str(i)+"_"+str(j)+"_image.png"
            output_image_path = input_image_path[:-4]+'_out.png'

            # segment_tissue(input_image_path, output_image_path)
            separate_background_mode(input_image_path,output_image_path)
