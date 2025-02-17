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
from PIL import Image
style.use('seaborn-white')
import numpy as np
import matplotlib.pyplot as plt


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

def normalize(image):
    return (image - np.min(image)) / (np.max(image) - np.min(image))
def create_overlay(fixed, moving):
    fixed = normalize(fixed)
    moving = normalize(moving)
    overlay = np.zeros((fixed.shape[0], fixed.shape[1], 3))
    overlay[..., 0] = fixed  # Magenta - Red Channel
    overlay[..., 1] = moving  # Cyan - Green and Blue Channel
    overlay[..., 2] = moving
    return overlay

def get_simpleITK_transformation(image1,image2):
    image1 = image1.convert('L')
    image2 = image2.convert('L')

    fixed_image = sitk.GetImageFromArray(np.array(image1).astype(np.float32))

    moving_image = sitk.GetImageFromArray(np.array(image2).astype(np.float32))

    # Perform registration (B-spline or non-linear)
    itk_transform = run_simpleITK(fixed_image, moving_image)

    # Generate displacement field
    displacement_filter = sitk.TransformToDisplacementFieldFilter()
    displacement_filter.SetReferenceImage(moving_image)
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

    fixed_array = normalize(fixed_array)
    moving_array = normalize(moving_array)
    registered_array = normalize(registered_array)

    # f,a = plt.subplots(1,2)
    # a[0].imshow(create_overlay(fixed_array,moving_array))
    # a[1].imshow(create_overlay(fixed_array,registered_array))
    #
    # plt.show()

    return fixed_image, moving_image,fixed_array, moving_array, registered_array, displacement_field

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


def warp_coords_moving_to_fixed(uv_coords, moving_image, fixed_image, itk_transform):
    """
    uv_coords: array/list of (u,v) pixel coordinates in the moving image domain
    moving_image, fixed_image: SimpleITK images
    itk_transform: transform from moving->fixed
    returns: array of (u_fixed, v_fixed) in fixed pixel space
    """
    warped_coords = []
    for (u, v) in uv_coords:
        # 1) moving pixel -> physical
        phys_moving = moving_image.TransformIndexToPhysicalPoint([int(u), int(v)])

        # 2) apply transform (moving->fixed)
        phys_fixed = itk_transform.TransformPoint(phys_moving)

        # 3) physical -> fixed pixel index
        uv_fixed = fixed_image.TransformPhysicalPointToIndex(phys_fixed)

        warped_coords.append([uv_fixed[0], uv_fixed[1]])

    return np.array(warped_coords)


def simpleITK_align_to_center(image_list,coords_list):
    image_0 = image_list[0]
    registered_image_list=[np.asarray(image_0.convert('L'))]
    registered_coords_list=[coords_list[0]]
    for i in range(1,len(image_list)):
    # for i in range(1, 2):
        fixed_itk,moving_itk,_,_,new_image,transform = get_simpleITK_transformation(image_0,image_list[i])
        transformed_coords = transform_uv_with_displacement(coords_list[i],transform)
        # transformed_coords = warp_coords_moving_to_fixed(coords_list[i], moving_itk, fixed_itk, transform)
        registered_image_list.append(new_image)
        registered_coords_list.append(transformed_coords)

    return registered_image_list,registered_coords_list





def get_gene_feature_matrix(coords: np.ndarray,
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
    return patch_matrix

def plot_dimensional_images_side_by_side(patch_matrix: np.ndarray):
    # 3) Plot each dimension in a subplot
    n_dims = patch_matrix.shape[-1]
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
    image = Image.open(slice.image_path)
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

def crop_square_then_resize_square(
    img: Image.Image,
    original_uv: np.ndarray,
    crop_para
):
    # 1) Crop the square from the original image
    #    crop box = (left, top, right, bottom)
    left,top,side_length,final_size = crop_para
    right  = left + side_length
    bottom = top  + side_length
    crop_box = (left, top, right, bottom)

    cropped_img = img.crop(crop_box)

    # 2) Resize the cropped square to (final_size, final_size)
    final_img = cropped_img.resize((final_size, final_size), Image.LANCZOS)

    # 3) Transform the coordinates
    # Step A: shift by subtracting (left, top)
    shifted_uv = original_uv - np.array([left, top])  # shape (N, 2)

    # Step B: scale factor for both x and y
    scale = final_size / side_length
    final_uv = shifted_uv * scale

    return final_img, final_uv


def get_DLPFC_data():
    sample_list = ["151507", "151508", "151509", "151510", "151669", "151670", "151671", "151672", "151673", "151674",
                   "151675", "151676"]
    adatas = {sample: sc.read_h5ad('../data/DLPFC/{0}_preprocessed.h5'.format(sample)) for sample in sample_list}

    for id in sample_list:
        adatas[id].image_path = '/media/huifang/data/registration/humanpilot/{0}/spatial/tissue_hires_image_image_0.png'.format(id)
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

    sample_groups = [["151507", "151508", "151509", "151510"], ["151669", "151670", "151671", "151672"],
                     ["151673", "151674", "151675", "151676"]]
    layer_groups = [[adatas[sample_groups[j][i]] for i in range(len(sample_groups[j]))] for j in
                    range(len(sample_groups))]
    return layer_groups

def show_clusters(coors,labels):
    unique_labels = np.unique(labels)  # sorted unique label values
    palette = sns.color_palette("deep", len(unique_labels))  # e.g. 'deep', 'tab10', etc.
    color_index = np.searchsorted(unique_labels, labels)
    colors = np.array(palette)[color_index]

    plt.scatter(coors[:, 0], coors[:, 1], s=10, color=colors)
    plt.show()

layer_groups = get_DLPFC_data()
crop_paras = [[(50,0,1850,1024),(50,50,1900,1024),(50,120,1896,1024),(50,155,1861,1024)],
              [(300,200,1650,1024), (350,220,1550,1024),(350,290,1600,1024),(360,230,1600,1024)],
              [(160,10,1770,1024),(160,50,1770,1024),(160,120,1750,1024),(180,20,1770,1024)]]

layer_to_color_map = {'Layer{0}'.format(i+1):i for i in range(6)}
layer_to_color_map['WM'] = 6

for k,slices in enumerate(layer_groups):

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
    image_list = []
    label_list=[]
    for i, sl in enumerate(slices):

        # Focus on common genes only
        sl_sub = sl[:, common_genes]
        # Convert to a NumPy array
        gene_data = np.array(sl_sub.X.toarray())  # shape: num_spots x num_genes
        gene_data_list.append(gene_data)
        # Extract coordinates from the slice

        labels = list(sl.obs['layer_guess_reordered'].astype(str).map(layer_to_color_map))
        labels = np.asarray(labels)
        label_list.append(labels)

        sl_sub.image_scale_path = sl.image_scale_path
        sl_sub.image_coor = sl.image_coor
        sl_sub.image_path = sl.image_path
        sl_sub.spatial_prefix = sl.spatial_prefix
        coords, image = get_uv_coordinates(sl_sub)  # your custom function
        cropped_image, cropped_coor = crop_square_then_resize_square(image, coords,crop_paras[k][i])

        # show_clusters(cropped_coor, labels)
        # plt.imshow(cropped_image)
        # plt.scatter(cropped_coor[:,0],cropped_coor[:,1])
        # plt.show()
        image_list.append(cropped_image)
        coords_list.append(cropped_coor)

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

    # f, a = plt.subplots(1, 4)
    # id = [0, 1, 2, 3]
    # for i, img, coors in zip(id, image_list, coords_list):
    #     a[i].imshow(img)
    #     a[i].scatter(coors[:, 0], coors[:, 1])
    # plt.show()
    #
    #
    # image_list, coords_list = simpleITK_align_to_center(image_list,coords_list)
    #
    # f,a = plt.subplots(1,4)
    # id = [0,1,2,3]
    # for i,img, coors in zip(id,image_list,coords_list):
    #     a[i].imshow(img)
    #     a[i].scatter(coors[:, 0], coors[:, 1])
    # plt.show()






    for i, data_slice in enumerate(gene_data_list):
        num_spots = data_slice.shape[0]
        index_end = index_start + num_spots

        # Slice out the portion that belongs to this slice
        reduced_slice_data = reduced_data[index_start:index_end, :]
        index_start = index_end
        # Get the corresponding coordinates
        coords = coords_list[i]
        image = np.asarray(image_list[i])
        labels = label_list[i]

        feature_matrix = get_gene_feature_matrix(coords, reduced_slice_data, (1024, 1024), patch_size=16)
        plot_dimensional_images_side_by_side(feature_matrix)

        # plt.imsave("../data/DLPFC/huifang/" + str(k) + "_" + str(i) + "_image.png", image)
        # np.save("../data/DLPFC/huifang/" + str(k) + "_" + str(i) + "_pca_out.npy", feature_matrix)
        # np.savez("../data/DLPFC/huifang/" + str(k) + "_" + str(i) + "_validation", coord=coords,label = labels)
