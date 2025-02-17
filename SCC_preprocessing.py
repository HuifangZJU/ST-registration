import pandas as pd
from PIL import Image
Image.MAX_IMAGE_PIXELS = None
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
from matplotlib import style
import matplotlib as mpl
import paste as pst
import scanpy as sc
import anndata
from paste.helper import intersect
from sklearn.decomposition import NMF
import os
from functools import reduce
from sklearn.decomposition import PCA
style.use('seaborn-dark')
mpl.rc('xtick', labelsize=14)
mpl.rc('ytick', labelsize=14)

path_to_output_dir = '../data/SCC/cached-results/'

def plot_clusters(adata, col, title = ''):
    """
    Plots spatial data with cluster labels stored in adata.obs[col]
    """
    x = adata.obsm['spatial'][:, 0]
    y = adata.obsm['spatial'][:, 1]
    label = adata.obs[col]
    n_colors = len(np.unique(label))
    palette = sns.color_palette("Paired", n_colors)
    plt.figure(figsize= (5, 5))
    ax = sns.scatterplot(x=x, y=y, hue=label, legend="full", palette = palette)
    plt.legend(bbox_to_anchor=(1.05, 1), loc=2, borderaxespad=0.)
    ax.set_facecolor('white')
    plt.axis('off')
    plt.title(title)
    plt.show()


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



def load_layer(patient, sample, metadata,crop_para):
    """
    Return Layer object of Patient, Sample
    """
    layer_path = f"../data/SCC/scc_p{patient}_layer{sample}"
    layer = layer_path + ".tsv"
    coor_path = layer_path + "_coordinates.tsv"
    adata = anndata.read_csv(layer, delimiter="\t")

    # Data pre-processing
    coor = pd.read_csv(coor_path, sep="\t").iloc[:, :2]
    original_coor = pd.read_csv(coor_path, sep="\t").iloc[:, 4:]
    coor_index = []
    for pair in coor.values:
        coor_index.append('x'.join(str(e) for e in pair))
    coor.index = coor_index
    original_coor.index = coor_index
    # The metadata, coordinates, and gene expression might have missing cells between them
    idx = intersect(coor_index, adata.obs.index)

    df = metadata[metadata['patient'] == patient]
    df = df[df['sample'] == sample]

    meta_idx = []
    for i in df.index:
        meta_idx.append(i.split('_')[1])
    idx = intersect(idx, meta_idx)

    adata = adata[idx, :]
    adata.obsm['spatial'] = np.array(coor.loc[idx, :])
    image = Image.open(f"../data/SCC/scc_p{patient}_layer{sample}.jpg")
    image_coor = np.array(original_coor.loc[idx,:])
    # plt.imshow(image)
    # plt.scatter(new_spatial[:, 0], new_spatial[:, 1])
    # plt.show()

    cropped_image,cropped_coor = crop_square_then_resize_square(image,image_coor,crop_para)

    # plt.imshow(cropped_image)
    # plt.scatter(cropped_coor[:,0],cropped_coor[:,1])
    # plt.show()
    adata.obsm['spatial_image_coor'] = cropped_coor
    adata.uns["image_array"] = np.asarray(cropped_image)
    metadata_idx = ['P' + str(patient) + '_' + i + '_' + str(sample) for i in idx]
    adata.obs['original_clusters'] = [str(x) for x in list(metadata.loc[metadata_idx, 'SCT_snn_res.0.8'])]
    return adata

def gen_h5file():
    metadata_path = "../data/SCC/ST_all_metadata.txt"
    metadata = pd.read_csv(metadata_path, sep="\t", index_col=0)

    adata_2_1 = load_layer(2, 1, metadata, (2000,2100,10300,1024))
    adata_2_2 = load_layer(2, 2, metadata, (1800,1200,10400,1024))
    adata_2_3 = load_layer(2, 3, metadata, (2000,2800,10300,1024))
    patient_2 = [adata_2_1, adata_2_2, adata_2_3]

    adata_5_1 = load_layer(5, 1, metadata, (1500,1000,13000,1024))
    adata_5_2 = load_layer(5, 2, metadata, (1000,2000,13500,1024))
    adata_5_3 = load_layer(5, 3, metadata, (1100,2000,13000,1024))
    patient_5 = [adata_5_1, adata_5_2, adata_5_3]

    adata_9_1 = load_layer(9, 1, metadata, (1800,500,13000,1024))
    adata_9_2 = load_layer(9, 2, metadata, (1800,1500,13000,1024))
    adata_9_3 = load_layer(9, 3, metadata, (1800,1800,13000,1024))
    patient_9 = [adata_9_1, adata_9_2, adata_9_3]

    adata_10_1 = load_layer(10, 1, metadata, (2700,3000,10000,1024))
    adata_10_2 = load_layer(10, 2, metadata, (2700,3300,10000,1024))
    adata_10_3 = load_layer(10, 3, metadata, (4000,3600,9000,1024))
    patient_10 = [adata_10_1, adata_10_2, adata_10_3]

    patients = {
        "patient_2": patient_2,
        "patient_5": patient_5,
        "patient_9": patient_9,
        "patient_10": patient_10,
    }

    for p in patients.values():
        for adata in p:
            sc.pp.filter_genes(adata, min_cells=15, inplace=True)
            sc.pp.filter_cells(adata, min_genes=100, inplace=True)
    H5ADs_dir = path_to_output_dir + 'H5ADs/'

    if not os.path.exists(H5ADs_dir):
        os.makedirs(H5ADs_dir)

    for k, p in patients.items():
        for i in range(len(p)):
            p[i].write(H5ADs_dir + k + '_slice_' + str(i) + '.h5ad')

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



# gen_h5file()


path_to_h5ads = path_to_output_dir + 'H5ADs/'

patient_2 = []
patient_5 = []
patient_9 = []
patient_10 = []

patients = {
    "patient_2" : patient_2,
    "patient_5" : patient_5,
    "patient_9" : patient_9,
    "patient_10" : patient_10,
}
for k in patients.keys():
    for i in range(3):
        patients[k].append(sc.read_h5ad(path_to_h5ads + k + '_slice_' + str(i) + '.h5ad'))
for k, slices in patients.items():
    for adata, s in zip(slices, ['Slice A', 'Slice B', 'Slice C']):
        plot_clusters(adata, 'original_clusters', title = s + ' (' + k + ')')



for k, slices in patients.items():
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
        coords=sl.obsm['spatial_image_coor']  # your custom function
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

    index_start = 0
    for i, data_slice in enumerate(gene_data_list):
        sl = slices[i]
        image = sl.uns["image_array"]

        num_spots = data_slice.shape[0]
        index_end = index_start + num_spots

        # Slice out the portion that belongs to this slice
        reduced_slice_data = reduced_data[index_start:index_end, :]
        index_start = index_end
        # Get the corresponding coordinates
        coords = coords_list[i]

        feature_matrix = get_gene_feature_matrix(coords, reduced_slice_data, (1024, 1024), patch_size=16)

        plt.imsave("../data/SCC/" + str(k) + "_" + str(i) + "_image.png", image)
        np.save("../data/SCC/" + str(k) + "_" + str(i) + "_pca_out.npy", feature_matrix)
        # plot_dimensional_images_side_by_side(feature_matrix)










