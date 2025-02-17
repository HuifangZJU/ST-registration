import os.path

import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import numpy as np
import scanpy as sc
import paste as pst
import SimpleITK as sitk
import pandas as pd

# Normalize images to [0, 1] for overlay
def normalize(image):
    return (image - np.min(image)) / (np.max(image) - np.min(image))

# Create elegant overlays (Cyan for Moving, Magenta for Fixed)
def create_overlay(fixed, moving):
    overlay = np.zeros((fixed.shape[0], fixed.shape[1], 3))
    overlay[..., 0] = fixed  # Magenta - Red Channel
    overlay[..., 1] = moving  # Cyan - Green and Blue Channel
    overlay[..., 2] = moving
    return overlay

def get_paste_transformation(slice1,slice2):
    # Pairwise align the slices
    pi12 = pst.pairwise_align(slice1, slice2)
    # Stack slices after alignment
    slices, pis = [slice1, slice2], [pi12]
    new_slices, R, T = pst.stack_slices_pairwise(slices, pis, output_params=True,matrix=True)
    return new_slices,R,T
def apply_paste_transformation(R,T,slice1,slice2):
    tX = T[0]
    tY = T[1]
    # Translate and rotate Y manually
    Y_translated = slice2.obsm['spatial'] - tY  # Apply translation
    Y_aligned = np.squeeze((R @ Y_translated.T).T)  # Apply rotation

    X_translated = slice1.obsm['spatial'] - tX  # Apply translation
    X_aligned = np.squeeze((R @ X_translated.T).T)  # Apply rotation

    new_slice1 = slice1.copy()
    new_slice2 = slice2.copy()

    new_slices = [new_slice1, new_slice2]
    new_slices[1].obsm['spatial'] = Y_aligned
    new_slices[0].obsm['spatial'] = X_translated
    return new_slices

def run_simpleITK(fixed_image,moving_image):
    # Set up the B-spline transform
    transform = sitk.BSplineTransformInitializer(fixed_image, [3, 3], order=3)  # Control points grid size

    # Registration setup
    registration_method = sitk.ImageRegistrationMethod()
    registration_method.SetInitialTransform(transform)
    registration_method.SetMetricAsMeanSquares()
    registration_method.SetOptimizerAsLBFGSB(gradientConvergenceTolerance=1e-3, numberOfIterations=100)
    registration_method.SetInterpolator(sitk.sitkLinear)

    # Perform registration
    final_transform = registration_method.Execute(fixed_image, moving_image)

    return final_transform

def rgb_to_grayscale(rgb_array):
    # Apply luminance conversion: Y = 0.299R + 0.587G + 0.114B
    return np.dot(rgb_array[..., :3], [0.5, 0.5, 0.5]).astype(np.float32)

def get_simpleITK_transformation(slice1,slice2):
    fixed_image = sitk.ReadImage(slice1.image_path, sitk.sitkFloat32)
    moving_image = sitk.ReadImage(slice2.image_path, sitk.sitkFloat32)

    # Perform registration (B-spline or non-linear)
    itk_transform = run_simpleITK(fixed_image, moving_image)

    # Generate displacement field
    displacement_filter = sitk.TransformToDisplacementFieldFilter()
    displacement_filter.SetReferenceImage(fixed_image)
    displacement_field = displacement_filter.Execute(itk_transform)

    # Resample moving image with the transform
    resampler = sitk.ResampleImageFilter()
    resampler.SetReferenceImage(fixed_image)
    resampler.SetTransform(itk_transform)
    resampler.SetInterpolator(sitk.sitkLinear)
    registered_image = resampler.Execute(moving_image)

    # Convert images to numpy arrays for visualization
    fixed_array = sitk.GetArrayViewFromImage(fixed_image)
    moving_array = sitk.GetArrayViewFromImage(moving_image)
    registered_array = sitk.GetArrayViewFromImage(registered_image)


    fixed_norm = normalize(fixed_array)
    moving_norm = normalize(moving_array)
    registered_norm = normalize(registered_array)
    temp = create_overlay(fixed_norm, registered_norm)
    plt.imshow(temp)
    plt.show()

    return fixed_norm, moving_norm, registered_norm, displacement_field


# # Assume that the coordinates of slices are named slice_name + "_coor.csv"
def load_sample_slices(data_dir='./sample_data/', slice_names=["slice1", "slice2"]):
    slices = []
    for slice_name in slice_names:
        slice_i = sc.read_csv(data_dir + slice_name + ".csv")
        slice_i_coor = np.genfromtxt(data_dir + slice_name + "_coor.csv", delimiter = ',')
        slice_i.obsm['spatial'] = slice_i_coor
        # Preprocess slices
        sc.pp.filter_genes(slice_i, min_counts = 15)
        sc.pp.filter_cells(slice_i, min_counts = 100)
        slices.append(slice_i)
    return slices


def load_slices(data_dir='/media/huifang/data/registration/humanpilot', slice_names=["151508", "151509"]):
    slices = []
    for slice_name in slice_names:
        # adata: spots are observation, genes are variables
        adata = sc.read_visium(f"{data_dir}/{slice_name}",count_file='filtered_matrix.h5',position_file='spatial/tissue_positions_list.txt')
        adata.image_path = os.path.join(data_dir,slice_name,"spatial/tissue_hires_image_image_0.png")
        adata.var_names_make_unique()
        # adata.var["mt"] = adata.var_names.str.startswith("MT-")
        # sc.pp.calculate_qc_metrics(adata, qc_vars=["mt"], inplace=True)
        ##filter out data by number_of_counts
        sc.pp.filter_cells(adata, min_counts=100)
        # adata = adata[adata.obs["pct_counts_mt"] < 20]
        sc.pp.filter_genes(adata, min_cells=15)
        ##normalize reads
        # sc.pp.normalize_total(adata, inplace=True)
        slices.append(adata)

    return slices

def transform_uv_with_displacement(uv_coords, deformation_field):
    deformation_np = sitk.GetArrayFromImage(deformation_field)
    # plt.subplot(1,2,1)
    # plt.imshow(deformation_np[:,:,0])
    # plt.subplot(1, 2, 2)
    # plt.imshow(deformation_np[:,:,1])
    # plt.show()

    deformation_size = deformation_field.GetSize()

    transformed_coords = []

    for u, v in uv_coords:
        u_int, v_int = int(u), int(v)

        # Ensure UV coordinates are within bounds
        if 0 <= u_int < deformation_size[0] and 0 <= v_int < deformation_size[1]:
            # Sample displacement at (u, v)
            displacement = deformation_np[v_int,u_int]  # (v, u) - numpy row-major

            # Apply displacement directly to UV
            u_transformed = u - displacement[0]  # x-component
            v_transformed = v - displacement[1]  # y-component
            transformed_coords.append([u_transformed, v_transformed])
        else:
            # If out of bounds, keep original point
            transformed_coords.append([u, v])
    return np.array(transformed_coords)




slices = load_slices()
slice1, slice2 = slices

fixed,moving,registered,displacement_field = get_simpleITK_transformation(slice1,slice2)




# overlay_before = create_overlay(fixed, moving)
overlay_after = create_overlay(fixed, registered)



plt.figure(figsize=(12, 12))  # Adjust the figure size for side-by-side comparison

# plt.subplot(1, 2, 1)
# plt.imshow(overlay_before)

plt.subplot(1, 2, 2)
plt.imshow(overlay_after)

plt.show()



[library_id]=slice1.uns['spatial'].keys()
low_res_scale = slice1.uns['spatial'][library_id]['scalefactors']['tissue_lowres_scalef']
uv_coords_fixed= slice1.obsm['spatial'] * low_res_scale


[library_id]=slice2.uns['spatial'].keys()
low_res_scale = slice2.uns['spatial'][library_id]['scalefactors']['tissue_lowres_scalef']
uv_coords= slice2.obsm['spatial'] * low_res_scale

# Apply transformation to UV coordinates
# Transform UV points using displacement field
transformed_coords = transform_uv_with_displacement(uv_coords, displacement_field)





# # Visualization to verify alignment
plt.subplot(1,3,1)
plt.imshow(fixed,cmap='gray')
plt.scatter(uv_coords_fixed[:, 0], uv_coords_fixed[:, 1], color='red', label='Original spots',s=8)
plt.legend()

plt.subplot(1,3,2)
plt.imshow(moving,cmap='gray')
plt.scatter(uv_coords[:, 0], uv_coords[:, 1], color='blue', label='Original spots',s=8)
plt.legend()
plt.subplot(1,3,3)
plt.imshow(registered,cmap='gray')
plt.scatter(transformed_coords[:, 0], transformed_coords[:, 1], color='blue', label='Transformed spots',s=8)
# plt.scatter(uv_coords[:, 0], uv_coords[:, 1], color='red', label='Original UV')
plt.legend()
plt.show()


new_slice2 = slice2.copy()
new_slice2.obsm['spatial'] = transformed_coords / low_res_scale
new_slices=[slice1,new_slice2]

new_new_slices,R,T = get_paste_transformation(slices[0],slices[1])


slice_colors = ['#e41a1c', '#377eb8']
# Plot before registration
plt.figure(figsize=(15,5))
plt.subplot(1, 3, 1)
for i in range(len(slices)):
    pst.plot_slice(slices[i], slice_colors[i], s=100)
plt.title('Before Registration')
plt.legend(handles=[mpatches.Patch(color=slice_colors[0], label='Slice 1'),
                    mpatches.Patch(color=slice_colors[1], label='Slice 2')])
plt.gca().invert_yaxis()
plt.axis('off')

# Plot after registration
plt.subplot(1, 3, 2)
for i in range(len(new_slices)):
    pst.plot_slice(new_slices[i], slice_colors[i], s=100)
plt.title('After Image Registration')
plt.legend(handles=[mpatches.Patch(color=slice_colors[0], label='Slice 1'),
                    mpatches.Patch(color=slice_colors[1], label='Slice 2')])
plt.gca().invert_yaxis()
plt.axis('off')

# Plot after registration
plt.subplot(1, 3, 3)
for i in range(len(new_new_slices)):
    pst.plot_slice(new_new_slices[i], slice_colors[i], s=100)
plt.title('After Paste Registration')
plt.legend(handles=[mpatches.Patch(color=slice_colors[0], label='Slice 1'),
                    mpatches.Patch(color=slice_colors[1], label='Slice 2')])
plt.gca().invert_yaxis()
plt.axis('off')

plt.tight_layout()
plt.show()

