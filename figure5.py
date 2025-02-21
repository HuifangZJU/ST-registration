import math
import pandas as pd
import numpy as np
import scipy
import seaborn as sns
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from matplotlib import style
import matplotlib
import time
import json
from scipy.ndimage import gaussian_filter
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import numpy as np
import scanpy as sc
import paste as pst
import SimpleITK as sitk
from scipy.spatial import cKDTree
import numpy as np
from collections import Counter
from sklearn.decomposition import PCA
from sklearn.manifold import TSNE
import numpy as np
from functools import reduce
from scipy.spatial import cKDTree

style.use('seaborn-white')
import ot

# Create elegant overlays (Cyan for Moving, Magenta for Fixed)
def create_overlay(fixed, moving):
    overlay = np.zeros((fixed.shape[0], fixed.shape[1], 3))
    overlay[..., 0] = fixed  # Magenta - Red Channel
    overlay[..., 1] = moving  # Cyan - Green and Blue Channel
    overlay[..., 2] = moving
    return overlay
def normalize(image):
    return (image - np.min(image)) / (np.max(image) - np.min(image))

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


def knn_label_accuracy(points_x, labels_x, points_y, labels_y, k=1):
       # 1. Build a cKDTree for points_y
    labels_x = np.array(labels_x)
    labels_y = np.array(labels_y)
    labels_x = labels_x.ravel()
    labels_y = labels_y.ravel()
    tree_y = cKDTree(points_y)
    # 2. For each point in X, find the indices of the k nearest neighbors in Y
    distances, neighbor_indices = tree_y.query(points_x, k=k)

    # If k=1, neighbor_indices is shape (n,). Make it (n,k) for consistent indexing
    if k == 1:
        neighbor_indices = neighbor_indices[:, None]
        distances = distances[:, None]

    # 3. Predict label by majority vote among the k neighbors
    predicted_labels = np.empty(len(points_x), dtype=labels_y.dtype)

    for i in range(len(points_x)):
        # labels of the k nearest neighbors
        neighbor_labels = labels_y[neighbor_indices[i]]
        # majority vote
        label_counts = Counter(neighbor_labels)
        # break ties by choosing the smallest label (or define your tie-break rule)
        predicted_labels[i] = sorted(label_counts.items(), key=lambda x: (-x[1], x[0]))[0][0]

    # 4. Compute accuracy

    correct = (predicted_labels == labels_x)
    accuracy = np.mean(correct)
    return accuracy


import numpy as np
import matplotlib.pyplot as plt
from scipy.interpolate import griddata


def plot_dimensional_images_side_by_side(coords: np.ndarray,
                                         reduced_data: np.ndarray,
                                         grid_size=(2000, 2000)):
    if coords.shape[0] != reduced_data.shape[0]:
        raise ValueError("coords and reduced_data must have the same number of rows (spots).")

    # plt.scatter(coords[:,0],coords[:,1])
    # plt.show()

    n_spots, n_dims = reduced_data.shape
    x_min, x_max = 1, grid_size[1]
    y_min, y_max = 1, grid_size[0]

    # Create a regular grid to interpolate onto
    grid_x, grid_y = np.meshgrid(
        np.linspace(x_min, x_max, grid_size[1]),
        np.linspace(y_min, y_max, grid_size[0])
    )

    # Figure setup: for d=10, let’s do 2 rows x 5 columns
    ncols = 5
    nrows = int(np.ceil(n_dims / ncols))
    fig, axes = plt.subplots(nrows=nrows, ncols=ncols, figsize=(4 * ncols, 4 * nrows))
    axes = axes.flatten()  # Flatten so we can index easily

    for dim_idx in range(n_dims):
        # Interpolate the spot values of the current dimension onto the grid
        # You can try 'nearest', 'linear', or 'cubic'—whichever works best for your data
        grid_z = griddata(
            points=coords,
            values=reduced_data[:, dim_idx],
            xi=(grid_x, grid_y),
            method='linear'
        )
        # 3) Build a KD-Tree to quickly find nearest neighbor distances
        tree = cKDTree(coords)  # shape (N, 2)

        # Flatten grid_x, grid_y so we can query them in bulk
        query_points = np.column_stack((grid_x.ravel(), grid_y.ravel()))

        # 4) For each pixel center, find the distance to the nearest data point
        dists, _ = tree.query(query_points, k=1)  # returns (distances, indices)
        dists_2d = dists.reshape(grid_size)  # reshape back to image

        # 5) Mask out any pixel with distance > threshold
        mask_far = (dists_2d > 21)
        grid_z[mask_far] = np.nan

        ax = axes[dim_idx]
        im = ax.imshow(
            grid_z,
            origin='upper',
            extent=(x_min, x_max, y_min, y_max),
            aspect='auto',
            cmap='gray'
        )
        ax.set_title(f"Dimension {dim_idx + 1}")
        ax.set_xlabel("X coordinate")
        ax.set_ylabel("Y coordinate")
        plt.colorbar(im, ax=ax, fraction=0.046, pad=0.04)

        cmap = plt.cm.gray.copy()
        cmap.set_bad(color='white')  # or another color you prefer
        # Mask NaNs so that the colormap "bad" color is applied:
        masked_array = np.ma.masked_invalid(grid_z)
        # Save directly to PNG with the colormap
        plt.imsave("../data/DLPFC/pca_" + str(dim_idx) + ".png", masked_array, cmap=cmap)

    # Hide any unused subplot axes (in case n_dims < nrows*ncols)
    for i in range(n_dims, nrows * ncols):
        axes[i].axis("off")

    plt.tight_layout()
    plt.show()

def plot_dimensional_images_side_by_side_with_patch(coords: np.ndarray,
                                         reduced_data: np.ndarray,
                                         image_size=(2016, 2016),
                                         patch_size=32):
    """
    Given:
      - coords:        (N, 2) 2D positions (x, y) for each data point
      - reduced_data:  (N, D) data values at each coordinate (N points, D dims)
      - image_size:    (height, width) of the *original* large image
      - patch_size:    size of the patch to downsample into

    We'll create a patch grid of shape:
       out_height = image_size[0] // patch_size
       out_width  = image_size[1] // patch_size
      and accumulate data from reduced_data into that grid.

    Steps:
      1) For each point, compute which patch (px, py) it belongs to.
      2) Accumulate the reduced_data values into sum_array[py, px, :].
      3) Keep track of the number of points in each patch (count_array).
      4) patch_matrix = sum_array / count_array (elementwise), ignoring patches with zero count.
      5) Plot each dimension side by side using subplots.
    """
    if coords.shape[0] != reduced_data.shape[0]:
        raise ValueError("coords and reduced_data must have the same number of rows.")

    # Number of points (N) and number of dimensions (D)
    n_spots, n_dims = reduced_data.shape

    # Image size
    height, width = image_size

    # Compute the shape of the patch matrix
    out_height = height // patch_size
    out_width  = width  // patch_size

    # We'll accumulate sums in sum_array and the count of points in count_array
    sum_array = np.zeros((out_height, out_width, n_dims), dtype=float)
    count_array = np.zeros((out_height, out_width), dtype=int)

    # 1) Assign each data point to its corresponding patch
    for i in range(n_spots):
        x, y = coords[i]  # e.g., coords might be (x, y) in [0..width, 0..height]
        px = int(x) // patch_size
        py = int(y) // patch_size

        # Check if we're within valid patch bounds
        if 0 <= px < out_width and 0 <= py < out_height:
            sum_array[py, px, :] += reduced_data[i]  # Accumulate the data
            count_array[py, px] += 1

    # 2) Compute the average (or keep as sum if you prefer) for each patch
    #    We'll avoid division by zero by clipping count_array
    patch_matrix = np.zeros_like(sum_array)
    valid_mask = (count_array > 0)
    patch_matrix[valid_mask, :] = (
        sum_array[valid_mask, :] / count_array[valid_mask, np.newaxis]
    )
    print(patch_matrix.shape)
    # All patches with zero points remain 0 or could set them to np.nan
    # patch_matrix[~valid_mask, :] = np.nan

    # Now patch_matrix.shape = (out_height, out_width, n_dims).
    # We can visualize each dimension as a 2D image of shape (out_height, out_width).

    # 3) Plot each dimension in a subplot
    ncols = 5
    nrows = int(np.ceil(n_dims / ncols))
    fig, axes = plt.subplots(nrows, ncols, figsize=(4*ncols, 4*nrows), squeeze=False)

    # Flatten axes so we can index easily
    axes_flat = axes.flatten()

    for dim_idx in range(n_dims):
        ax = axes_flat[dim_idx]

        # Extract the 2D patch grid for this dimension
        patch_image = patch_matrix[:, :, dim_idx]

        im = ax.imshow(
            patch_image,           # shape (out_height, out_width)
            origin='upper',        # row=0 at the top
            cmap='viridis',        # or 'gray', etc.
            aspect='auto'
        )
        ax.set_title(f"Dimension {dim_idx + 1}")
        ax.set_xlabel("Patch (x)")
        ax.set_ylabel("Patch (y)")
        plt.colorbar(im, ax=ax, fraction=0.046, pad=0.04)

    # Hide any unused subplots if n_dims < nrows*ncols
    for dim_idx in range(n_dims, nrows*ncols):
        axes_flat[dim_idx].axis("off")

    plt.tight_layout()
    plt.show()

    # # 4) Return patch_matrix if you want to use it for other processing
    # return patch_matrix

def plot_dimensional_images_side_by_side_compact(coordinates, reduced_data,image_shape):
    height, width = image_shape
    n_spots, n_components = reduced_data.shape

    # Figure setup: for example, 2 rows x 5 columns if we have 10 components
    ncols = 5
    nrows = int(np.ceil(n_components / ncols))
    fig, axes = plt.subplots(nrows=nrows, ncols=ncols, figsize=(4 * ncols, 4 * nrows))
    axes = axes.flatten()

    for dim_idx in range(n_components):
        # Initialize an empty image with zeros
        image_array = np.zeros((height, width), dtype=float)

        # Fill the image with the dimension values at each spot
        for i in range(n_spots):
            x_coord = int(coordinates[i, 0])
            y_coord = int(coordinates[i, 1])
            if 0 <= x_coord < width and 0 <= y_coord < height:
                image_array[y_coord, x_coord] = reduced_data[i, dim_idx]

        # Plot
        # image_array[image_array == 0] = np.nan
        ax = axes[dim_idx]
        # If your data range is known, you can set vmin/vmax or leave them auto
        im = ax.imshow(image_array, cmap='viridis', origin='upper')

        # Force NaN/Mask to be white
        im.cmap.set_bad(color='white')

        ax.set_title(f"Dimension {dim_idx + 1}")
        ax.axis('off')

        # (Optional) add colorbar for each subplot
        plt.colorbar(im, ax=ax, fraction=0.046, pad=0.04)

        cmap = plt.cm.viridis.copy()
        cmap.set_bad(color='white')  # or another color you prefer
        # Mask NaNs so that the colormap "bad" color is applied:
        masked_array = np.ma.masked_invalid(image_array)
        # Save directly to PNG with the colormap
        plt.imsave("../data/DLPFC/image_output_" + str(dim_idx) + ".png", masked_array, cmap=cmap)

    # Hide extra subplots if n_components isn't a perfect multiple of ncols*nrows
    for extra_idx in range(dim_idx + 1, len(axes)):
        axes[extra_idx].axis('off')

    plt.tight_layout()
    plt.show()




def make_2d_gaussian_kernel(kernel_size=15, sigma=2.0):
    """
    Creates a 2D Gaussian kernel array of shape (kernel_size, kernel_size)
    with standard deviation 'sigma'.

    The kernel is normalized so that the sum of all values = 1.
    """
    # Ensure kernel_size is odd to have a well-defined center
    assert kernel_size % 2 == 1, "kernel_size should be odd for a centered kernel"

    # Coordinates to sample the Gaussian; center is (0, 0)
    center = kernel_size // 2
    x = np.arange(kernel_size) - center
    y = np.arange(kernel_size) - center
    xx, yy = np.meshgrid(x, y)

    # 2D Gaussian formula
    gaussian = np.exp(-(xx ** 2 + yy ** 2) / (2 * sigma ** 2))

    # Normalize so the kernel sum is 1
    # gaussian /= gaussian.sum()
    return gaussian


def plot_dimensional_images_gaussian_smoothing(coordinates, reduced_data, image_shape,
                                               kernel_size=21, sigma=15.0):
    """
    Plots each dimension in 'reduced_data' by placing a Gaussian 'stamp'
    around each spot. This reduces discretization effects.

    Parameters
    ----------
    coordinates : np.ndarray
        Shape (n_spots, 2). Each row is (x, y) for a spot (float or int).
    reduced_data : np.ndarray
        Shape (n_spots, n_components). Each column is a dimension (e.g., PCA).
    image_shape : tuple
        (height, width) of the output image.
    kernel_size : int
        Size of the Gaussian kernel. Must be an odd number.
    sigma : float
        Standard deviation for the Gaussian kernel.
    """
    height, width = image_shape
    n_spots, n_components = reduced_data.shape

    # Create a 2D Gaussian kernel (e.g., 15x15)
    kernel = make_2d_gaussian_kernel(kernel_size, sigma)
    half_k = kernel_size // 2  # Radius from center

    # Figure setup
    ncols = 5
    nrows = int(np.ceil(n_components / ncols))
    fig, axes = plt.subplots(nrows=nrows, ncols=ncols, figsize=(4 * ncols, 4 * nrows))
    # If there's only 1 row or 1 column, axes might not be a 2D array
    axes = np.atleast_1d(axes).flatten()

    # Round coordinates to nearest integers
    coords_int = np.rint(coordinates).astype(int)
    x_coords = coords_int[:, 0]
    y_coords = coords_int[:, 1]

    for dim_idx in range(n_components):
        # Initialize an empty image
        image_array = np.zeros((height, width), dtype=float)


        # For each spot, add a Gaussian "bump"
        for i in range(n_spots):
            val = reduced_data[i, dim_idx]  # The value for this spot/dimension

            x_c = x_coords[i]
            y_c = y_coords[i]

            # Compute bounding box of where the kernel will be added
            x_start = x_c - half_k
            x_end = x_c + half_k + 1
            y_start = y_c - half_k
            y_end = y_c + half_k + 1

            # Check overlap with the image boundaries
            # (some spots might be near edges)
            kx_start = 0
            kx_end = kernel_size
            ky_start = 0
            ky_end = kernel_size

            # If x_start < 0, then we need to shift the kernel
            if x_start < 0:
                kx_start = -x_start
                x_start = 0
            if y_start < 0:
                ky_start = -y_start
                y_start = 0
            if x_end > width:
                kx_end = kernel_size - (x_end - width)
                x_end = width
            if y_end > height:
                ky_end = kernel_size - (y_end - height)
                y_end = height

            # Add the kernel to the image
            # scaled by the spot's value 'val'
            image_array[y_start:y_end, x_start:x_end] += val * kernel[ky_start:ky_end, kx_start:kx_end]

        # Plot
        image_array[image_array == 0] = np.nan
        ax = axes[dim_idx]
        # If your data range is known, you can set vmin/vmax or leave them auto
        im = ax.imshow(image_array, cmap='viridis', origin='upper')
        ax.set_title(f"Dimension {dim_idx + 1}")
        ax.axis('off')
        plt.colorbar(im, ax=ax, fraction=0.046, pad=0.04)

        cmap = plt.cm.viridis.copy()
        cmap.set_bad(color='white')  # or another color you prefer
        # Mask NaNs so that the colormap "bad" color is applied:
        masked_array = np.ma.masked_invalid(image_array)
        # Save directly to PNG with the colormap
        plt.imsave("../data/DLPFC/pca_output_"+str(dim_idx)+".png", masked_array, cmap=cmap)

    # Hide any extra subplots if n_components < ncols * nrows
    for extra_idx in range(dim_idx + 1, len(axes)):
        axes[extra_idx].axis('off')

    plt.tight_layout()
    plt.show()

def reduce_gene_reads(gene_reads: np.ndarray, method: str = 'pca', n_components: int = 10) -> np.ndarray:

    if not isinstance(gene_reads, np.ndarray):
        raise ValueError("gene_reads must be a NumPy array of shape (n, m).")

    if method.lower() == 'pca':
        # Principal Component Analysis
        pca = PCA(n_components=n_components)
        reduced_data = pca.fit_transform(gene_reads)

    elif method.lower() == 'umap':
        # UMAP
        import umap.umap_ as umap
        reducer = umap.UMAP(n_components=n_components)
        reduced_data = reducer.fit_transform(gene_reads)
    else:
        raise ValueError("method must be one of ['pca', 'umap'].")

    return reduced_data

def get_uv_coordinates(slice):
    scale_path = slice.image_scale_path
    image = plt.imread(slice.image_path)
    with open(scale_path, 'r') as f:
        data = json.load(f)
        low_res_scale = data['tissue_hires_scalef']

    position_prefix = slice.spatial_prefix
    try:
        # Try reading as CSV
        positions = pd.read_csv(position_prefix + '.csv', header=None, sep=',')
    except (FileNotFoundError, pd.errors.EmptyDataError, pd.errors.ParserError):
        positions = pd.read_csv(position_prefix + '.txt', header=None, sep=',')
    slice.obs['position'].index = (
        slice.obs['position'].index
        .str.replace(r"\.\d+$", "", regex=True)
    )
    positions.columns = [
        "barcode",
        "in_tissue",
        "array_row",
        "array_col",
        "pxl_col_in_fullres",
        "pxl_row_in_fullres",
    ]
    # 1) Get the barcodes from AnnData that are in `positions`
    positions.index = positions["barcode"]
    adata_barcodes = slice.obs['position'].index
    common_barcodes = adata_barcodes[adata_barcodes.isin(positions.index)]

    # 2) Now reindex `positions` in the exact order of `common_barcodes`
    positions_filtered = positions.reindex(common_barcodes)

    spatial_locations = positions_filtered[["pxl_row_in_fullres", "pxl_col_in_fullres"]].to_numpy()

    # spatial_locations = slice.image_coor
    uv_coords = spatial_locations * low_res_scale
    return uv_coords, image


def channelwise_min_max_normalize(data):
    """
    data: shape (N, C)
      N = number of samples (e.g., 17952)
      C = number of channels/features (e.g., 10)

    Returns:
      normalized_data of the same shape, where each channel is mapped to [0, 1]
      across the N samples.
    """
    mins = data.min(axis=0)   # shape (C,)
    maxs = data.max(axis=0)   # shape (C,)
    ranges = maxs - mins

    # Avoid division by zero (when all values in a channel are the same)
    ranges[ranges == 0] = 1e-8

    normalized_data = (data - mins) / ranges
    return normalized_data

def get_image_gene_transformation(slices):

    for sl in slices:
        sc.pp.filter_genes(sl, min_counts=3)  # example threshold
        sc.pp.filter_cells(sl, min_genes=200)  # example threshold
        sc.pp.normalize_total(sl, target_sum=1e4)
        sc.pp.log1p(sl)
        sc.pp.scale(sl, max_value=10)

    all_gene_lists = [sl.var.index for sl in slices]
    common_genes = reduce(np.intersect1d, all_gene_lists)
    # 2. Subset each slice to the common genes, gather coordinates & data
    gene_data_list = []
    coords_list = []
    for sl in slices:
        # Focus on common genes only
        sl_sub = sl[:, common_genes]
        # Convert to a NumPy array
        gene_data = np.array(sl_sub.X.toarray())  # shape: num_spots x num_genes
        gene_data_list.append(gene_data)
        # Extract coordinates from the slice
        sl_sub.image_scale_path = sl.image_scale_path

        sl_sub.image_coor = sl.image_coor
        sl_sub.image_path = sl.image_path
        sl_sub.spatial_prefix = sl.spatial_prefix
        coords, _ = get_uv_coordinates(sl_sub)  # your custom function
        coords_list.append(coords)

    # 3. Concatenate all gene data
    combined_data = np.vstack(gene_data_list)  # shape: (sum_of_all_spots, num_genes)


    # 4. Reduce dimensionality (e.g., PCA)
    reduced_data = reduce_gene_reads(
        combined_data,
        method='pca',
        n_components=10
    )  # shape: (sum_of_all_spots, 15)
    reduced_data = channelwise_min_max_normalize(reduced_data)
    # 5. Split the reduced data back for each slice and plot
    index_start = 0
    for i, data_slice in enumerate(gene_data_list):
        num_spots = data_slice.shape[0]
        index_end = index_start + num_spots

        # Slice out the portion that belongs to this slice
        reduced_slice_data = reduced_data[index_start:index_end, :]
        index_start = index_end
        # Get the corresponding coordinates
        coords = coords_list[i]

        # np.savez("../data/DLPFC/" + str(i)+"_pca_out.npz", img_size=(2016,2016), spot_coords=coords, pca_data=reduced_slice_data)

        # Plot
        # plot_dimensional_images_side_by_side(coords, reduced_slice_data)
        # plot_dimensional_images_side_by_side_compact(coords, reduced_slice_data, (2016,2016))
        plot_dimensional_images_gaussian_smoothing(coords, reduced_slice_data, (2016,2016))
        # plot_dimensional_images_side_by_side_with_patch(coords,reduced_slice_data,(2016, 2016))

    return None


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


    # temp = create_overlay(fixed_norm,registered_norm)
    # plt.imshow(temp)
    # plt.show()
    #
    # scale_path = slice1.image_scale_path
    # with open(scale_path, 'r') as f:
    #     data = json.load(f)
    #     low_res_scale = data['tissue_hires_scalef']
    # plt.subplot(1, 3, 1)
    # plt.imshow(fixed_norm, cmap='gray')
    # plt.scatter(slice1.image_coor[:, 1]*low_res_scale, slice1.image_coor[:, 0]*low_res_scale, color='red', label='Original spots', s=8)
    # plt.legend()
    #
    # scale_path = slice2.image_scale_path
    # with open(scale_path, 'r') as f:
    #     data = json.load(f)
    #     low_res_scale = data['tissue_hires_scalef']
    #
    # plt.subplot(1, 3, 2)
    # plt.imshow(moving_norm, cmap='gray')
    # plt.scatter(slice2.image_coor[:, 1]*low_res_scale, slice2.image_coor[:, 0]*low_res_scale, color='blue', label='Original spots', s=8)
    # plt.legend()
    #
    # spatial_locations = slice2.image_coor
    # spatial_locations = spatial_locations[:, [1, 0]]
    # uv_coords = spatial_locations * low_res_scale
    # transformed_coords = transform_uv_with_displacement(uv_coords, displacement_field)
    #
    # plt.subplot(1, 3, 3)
    # plt.scatter(slice1.image_coor[:, 1] * low_res_scale, slice1.image_coor[:, 0] * low_res_scale, color='red',
    #             label='Original spots', s=8)
    # plt.scatter(transformed_coords[:, 0] , transformed_coords[:, 1], color='blue',
    #             label='Original spots', s=8)
    # plt.legend()
    # plt.show()

    # return fixed_norm, moving_norm, registered_norm, displacement_field
    return displacement_field

def transform_uv_with_displacement(uv_coords, deformation_field):
    deformation_np = sitk.GetArrayFromImage(deformation_field)
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

def intersect(lst1, lst2):
    """
    Gets and returns intersection of two lists.

    Args:
        lst1: List
        lst2: List

    Returns:
        lst3: List of common elements.
    """

    temp = set(lst2)
    lst3 = [value for value in lst1 if value in temp]
    return lst3

def stack_slices_pairwise(slices,displacement_field):
    slice0 = slices[0]
    new_slice0 = slice0.copy()
    new_slice0.obsm['spatial']=slice0.image_coor
    new_slices = [new_slice0]
    accuracies = []
    for i in range(1,len(slices)):
        slice2 = slices[i]
        scale_path = slice2.image_scale_path
        with open(scale_path, 'r') as f:
            data = json.load(f)
            low_res_scale = data['tissue_hires_scalef']
        # # read coordinates
        # position_prefix = slice2.spatial_prefix
        # try:
        #     # Try reading as CSV
        #     positions = pd.read_csv(position_prefix + '.csv', header=None, sep=',', usecols=[4, 5])
        # except (FileNotFoundError, pd.errors.EmptyDataError, pd.errors.ParserError):
        #     positions = pd.read_csv(position_prefix + '.txt', header=None, sep=',', usecols=[4, 5])
        # spatial_locations = positions.to_numpy()
        spatial_locations = slice2.image_coor
        uv_coords = spatial_locations * low_res_scale
        transformed_coords = transform_uv_with_displacement(uv_coords, displacement_field[i-1])/ low_res_scale
        new_slice2 = slice2.copy()
        new_slice2.obsm['spatial'] = transformed_coords
        new_slices.append(new_slice2)

        mapping_dict = {'Layer1': 1, 'Layer2': 2, 'Layer3': 3, 'Layer4': 4, 'Layer5': 5, 'Layer6': 6, 'WM': 7}
        label1 = slice0.obs['layer_guess_reordered']
        label2 = new_slice2.obs['layer_guess_reordered']
        acc = knn_label_accuracy(new_slice0.obsm['spatial'], np.matrix(label1.map(mapping_dict)), new_slice2.obsm['spatial'], np.matrix(label2.map(mapping_dict)), k=5)
        print(acc)
        accuracies.append(acc)
    return new_slices

def largest_indices(ary, n):
    """Returns the n largest indices from a numpy array."""
    flat = ary.flatten()
    indices = np.argpartition(flat, -n)[-n:]
    indices = indices[np.argsort(-flat[indices])]
    return np.unravel_index(indices, ary.shape)


def plot2D_samples_mat(xs, xt, G, thr=1e-8, alpha=0.2, top=1000, weight_alpha=False, **kwargs):
    if ('color' not in kwargs) and ('c' not in kwargs):
        kwargs['color'] = 'k'
    mx = G.max()
    idx = largest_indices(G, top)
    for l in range(len(idx[0])):
        plt.plot([xs[idx[0][l], 0], xt[idx[1][l], 0]], [xs[idx[0][l], 1], xt[idx[1][l], 1]],
                 alpha=alpha * (1 - weight_alpha) + (weight_alpha * G[idx[0][l], idx[1][l]] / mx), c='k')


def plot_slice_pairwise_alignment(slice1, slice2, pi, thr=1 - 1e-8, alpha=0.05, top=1000, name='', save=False,
                                  weight_alpha=False):
    coordinates1, coordinates2 = slice1.obsm['spatial'], slice2.obsm['spatial']
    offset = (coordinates1[:, 0].max() - coordinates2[:, 0].min()) * 1.1
    temp = np.zeros(coordinates2.shape)
    temp[:, 0] = offset
    plt.figure(figsize=(20, 10))
    plot2D_samples_mat(coordinates1, coordinates2 + temp, pi, thr=thr, c='k', alpha=alpha, top=top,
                       weight_alpha=weight_alpha)
    plt.scatter(coordinates1[:, 0], coordinates1[:, 1], linewidth=0, s=100, marker=".", color=list(
        slice1.obs['layer_guess_reordered'].map(
            dict(zip(slice1.obs['layer_guess_reordered'].cat.categories, slice1.uns['layer_guess_reordered_colors'])))))
    plt.scatter(coordinates2[:, 0] + offset, coordinates2[:, 1], linewidth=0, s=100, marker=".", color=list(
        slice2.obs['layer_guess_reordered'].map(
            dict(zip(slice2.obs['layer_guess_reordered'].cat.categories, slice2.uns['layer_guess_reordered_colors'])))))
    plt.gca().invert_yaxis()
    plt.axis('off')
    plt.show()

def max_accuracy(labels1,labels2):
    w = min(1/len(labels1),1/len(labels2))
    cats = set(pd.unique(labels1)).union(set(pd.unique(labels1)))
    return sum([w * min(sum(labels1==c),sum(labels2==c)) for c in cats])
def mapping_accuracy(labels1,labels2,pi):
    mapping_dict = {'Layer1':1, 'Layer2':2, 'Layer3':3, 'Layer4':4, 'Layer5':5, 'Layer6':6, 'WM':7}
    return np.sum(pi*(scipy.spatial.distance_matrix(np.matrix(labels1.map(mapping_dict) ).T,np.matrix(labels2.map(mapping_dict)).T)==0))

def max_accuracy_mapping(labels1,labels2):
    n1,n2=len(labels1),len(labels2)
    mapping_dict = {'Layer1':1, 'Layer2':2, 'Layer3':3, 'Layer4':4, 'Layer5':5, 'Layer6':6, 'WM':7}
    dist = np.array(scipy.spatial.distance_matrix(np.matrix(labels1.map(mapping_dict)).T,np.matrix(labels2.map(mapping_dict)).T)!=0,dtype=float)
    pi = ot.emd(np.ones(n1)/n1, np.ones(n2)/n2, dist)
    return pi

def plot_slices_overlap(groups, adatas, sample_list, layer_to_color_map):
    # Suppose you have exactly 3 groups
    fig, axs = plt.subplots( nrows=1, ncols=len(groups),figsize=(20, 5))
    for j in range(len(groups)):
        ax = axs[j]  # select the subplot for group j
        # Plot each sample in group j on the same subplot
        for i in range(len(groups[j])):
            adata = adatas[sample_list[j * 4 + i]]
            # Map each layer label to its color
            colors = list(adata.obs['layer_guess_reordered'].astype(str).map(layer_to_color_map))
            # Scatter plot on the j-th subplot
            ax.scatter(groups[j][i].obsm['spatial'][:, 0],groups[j][i].obsm['spatial'][:, 1],linewidth=0,s=100,marker=".",color=colors)
        # Build legend for this group's subplot
        handles = [mpatches.Patch(color=layer_to_color_map[label],label=label) for label in adata.obs['layer_guess_reordered'].cat.categories]
        ax.legend(handles=handles,fontsize=10,title='Cortex layer',title_fontsize=15,bbox_to_anchor=(1, 1))
        ax.invert_yaxis()
        ax.axis('off')
        ax.set_title(f"Group {j + 1}")
    plt.tight_layout()
    plt.show()

sample_list = ["151507", "151508", "151509","151510", "151669", "151670","151671", "151672", "151673","151674", "151675", "151676"]
# sample_list = ["151507", "151508", "151509","151510", "151669", "151670"]

adatas = {sample:sc.read_h5ad('../data/DLPFC/{0}_preprocessed.h5'.format(sample)) for sample in sample_list}



for id in sample_list:
    adatas[id].image_path = '/media/huifang/data/registration/humanpilot/{0}/spatial/tissue_hires_image.png'.format(id)
    adatas[id].image_scale_path = '/media/huifang/data/registration/humanpilot/{0}/spatial/scalefactors_json.json'.format(id)
    adatas[id].spatial_prefix = '/media/huifang/data/registration/humanpilot/{0}/spatial/tissue_positions_list'.format(id)
    adatas[id].obs['position'].index = (
        adatas[id].obs['position'].index
        .str.replace(r"\.\d+$", "", regex=True)
    )
    position_prefix = adatas[id].spatial_prefix
    try:
        # Try reading as CSV
        positions = pd.read_csv(position_prefix + '.csv', header=None, sep=',')
    except (FileNotFoundError, pd.errors.EmptyDataError, pd.errors.ParserError):
        positions = pd.read_csv(position_prefix + '.txt', header=None, sep=',')

    positions.columns = [
        "barcode",
        "in_tissue",
        "array_row",
        "array_col",
        "pxl_col_in_fullres",
        "pxl_row_in_fullres",
    ]
    # 1) Get the barcodes from AnnData that are in `positions`
    positions.index = positions["barcode"]
    adata_barcodes = adatas[id].obs['position'].index
    common_barcodes = adata_barcodes[adata_barcodes.isin(positions.index)]


    # 2) Now reindex `positions` in the exact order of `common_barcodes`
    positions_filtered = positions.reindex(common_barcodes)

    spatial_locations = positions_filtered[["pxl_row_in_fullres", "pxl_col_in_fullres"]].to_numpy()
    adatas[id].image_coor = spatial_locations


sample_groups = [["151507", "151508", "151509","151510"],[ "151669", "151670","151671", "151672"],[ "151673","151674", "151675", "151676"]]
# sample_groups = [[ "151669", "151670","151671", "151672"],[ "151673","151674", "151675", "151676"]]

layer_groups = [[adatas[sample_groups[j][i]] for i in range(len(sample_groups[j]))] for j in range(len(sample_groups))]



layer_to_color_map = {'Layer{0}'.format(i+1):sns.color_palette()[i] for i in range(6)}
layer_to_color_map['WM'] = sns.color_palette()[6]

# slice_map = {0:'A',1:'B',2:'C',3:'D'}
# fig, axs = plt.subplots(3, 4,figsize=(15,11.5))
# for j in range(len(layer_groups)):
#     axs[j,0].text(-0.1, 0.5, 'Sample '+slice_map[j],fontsize=12,rotation='vertical',transform = axs[j,0].transAxes,verticalalignment='center')
#     for i in range(len(layer_groups[j])):
#         adata = adatas[sample_list[j*4+i]]
#         colors = list(adata.obs['layer_guess_reordered'].astype('str').map(layer_to_color_map))
#         axs[j,i].scatter(layer_groups[j][i].obsm['spatial'][:,0],layer_groups[j][i].obsm['spatial'][:,1],linewidth=0,s=20, marker=".",color=colors)
#         axs[j,i].set_title('Slice '+ slice_map[i],size=12)
#         axs[j,i].invert_yaxis()
#         axs[j,i].axis('off')
#         if i<3:
#             s = '300$\mu$m' if i==1 else "10$\mu$m"
#             delta = 0.05 if i==1 else 0
#             axs[j,i].annotate('',xy=(1-delta, 0.5), xytext=(1.2+delta, 0.5),xycoords=axs[j,i].transAxes,textcoords=axs[j,i].transAxes,arrowprops=dict(arrowstyle='<->',lw=1))
#             axs[j,0].text(1.1, 0.55, s,fontsize=9,transform = axs[j,i].transAxes,horizontalalignment='center')
#     axs[j,3].legend(handles=[mpatches.Patch(color=layer_to_color_map[adata.obs['layer_guess_reordered'].cat.categories[i]], label=adata.obs['layer_guess_reordered'].cat.categories[i]) for i in range(len(adata.obs['layer_guess_reordered'].cat.categories))],fontsize=10,title='Cortex layer',title_fontsize=12,bbox_to_anchor=(1, 1))
#
# plt.show()

# alpha = 0.1
# res_df = pd.DataFrame(columns=['Sample','Pair','Kind','Time','Accuracy'])
# pis = [[None for i in range(len(layer_groups[j])-1)] for j in range(len(layer_groups))]
# for j in range(len(layer_groups)):
#    for i in range(len(layer_groups[j])-1):
#        pi0 = pst.match_spots_using_spatial_heuristic(layer_groups[j][i].obsm['spatial'],layer_groups[j][i+1].obsm['spatial'],use_ot=True)
#        start = time.time()
#        pis[j][i] = pst.pairwise_align(layer_groups[j][i], layer_groups[j][i+1],alpha=alpha,G_init=pi0,norm=True,verbose=False,backend=ot.backend.TorchBackend(),use_gpu=True)
#        tt = time.time()-start
#        acc = mapping_accuracy(layer_groups[j][i].obs['layer_guess_reordered'],layer_groups[j][i+1].obs['layer_guess_reordered'],pis[j][i])
#        print(j,i,'Accuracy',acc,'time',tt)
#        np.savetxt('../data/DLPFC/saved_results/init_{0}_{1}_{2}.gz'.format(j,i,'ot'), pis[j][i], delimiter=',')
#        res_df.loc[len(res_df)] = [j,i,'PASTE',tt,acc]
# pis = [[None for i in range(len(layer_groups[j])-1)] for j in range(len(layer_groups))]
# for j in range(len(layer_groups)):
#     for i in range(len(layer_groups[j])-1):
#         pis[j][i]=np.loadtxt('../data/DLPFC/saved_results/init_{0}_{1}_{2}.gz'.format(j,i,'ot'), delimiter=',')
#         start = time.time()
#         acc = mapping_accuracy(layer_groups[j][i].obs['layer_guess_reordered'],layer_groups[j][i+1].obs['layer_guess_reordered'],pis[j][i])
#         tt = time.time()-start
#         print(j,i,'Accuracy',acc,'time',tt)
#         res_df.loc[len(res_df)] = [j,i,'PASTE',tt,acc]
# paste_layer_groups = [pst.stack_slices_pairwise(layer_groups[j], pis[j]) for j in range(len(layer_groups)) ]
# plot_slices_overlap(layer_groups, adatas, sample_list, layer_to_color_map)
# plot_slices_overlap(paste_layer_groups, adatas, sample_list, layer_to_color_map)

pis_sitk = [[None for i in range(len(layer_groups[j])-1)] for j in range(len(layer_groups))]
for j in range(len(layer_groups)):
   # for i in range(len(layer_groups[j])-1):
   start = time.time()

   test = get_image_gene_transformation(layer_groups[j])

   # pis_sitk[j][i] = get_simpleITK_transformation(layer_groups[j][0], layer_groups[j][i+1])

   tt = time.time()-start
   print(tt)
sitk_layer_groups = [stack_slices_pairwise(layer_groups[j], pis_sitk[j]) for j in range(len(layer_groups)) ]
plot_slices_overlap(sitk_layer_groups, adatas, sample_list, layer_to_color_map)
# Plot Stacking of Four slices with PASTE alignment
