import matplotlib.pyplot as plt
import numpy as np


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


loaded = np.load("../data/DLPFC/0_pca_out.npz")
imagesize = loaded["img_size"]
coords = loaded["spot_coords"]
reduced_slice_data = loaded["pca_data"]
plot_dimensional_images_side_by_side_with_patch(coords,reduced_slice_data,imagesize,patch_size=32)