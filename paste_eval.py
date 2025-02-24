import pandas as pd
import scipy
import seaborn as sns
from matplotlib import style
import time
import scanpy as sc
import paste as pst
from scipy.spatial import cKDTree
import numpy as np
import matplotlib.pyplot as plt
import cv2

style.use('seaborn-white')

def max_accuracy(labels1,labels2):
    w = min(1/len(labels1),1/len(labels2))
    cats = set(pd.unique(labels1)).union(set(pd.unique(labels1)))
    return sum([w * min(sum(labels1==c),sum(labels2==c)) for c in cats])
def mapping_accuracy(labels1,labels2,pi):
    mapping_dict = {'Layer1': 1, 'Layer2': 2, 'Layer3': 3, 'Layer4': 4, 'Layer5': 5, 'Layer6': 6, 'WM': 7}
    labels1 = np.matrix(labels1.map(mapping_dict))
    labels2 = np.matrix(labels2.map(mapping_dict))
    return np.sum(pi*(scipy.spatial.distance_matrix(labels1.T,labels2.T)==0))

def find_transformed_image_limit(image,R,T):
    h, w = image.shape[:2]

    # Compute the transformed corner positions
    corners = np.array([
        [0, 0], [w, 0], [0, h], [w, h]
    ])
    # transformed_corners = np.dot(corners, R.T) + T  # Apply the same transformation as points
    transformed_corners = R.dot((corners+T).T).T
    # Compute the new bounding box for the transformed image
    x_min, y_min = np.min(transformed_corners, axis=0)
    x_max, y_max = np.max(transformed_corners, axis=0)
    return x_min,y_min,x_max,y_max

def get_adjusted_affine_matrix(T,R,image_shift):
    translate_to_origin = np.array([
        [1, 0, T[0]],  # Move centroid to (0,0)
        [0, 1, T[1]]
    ])

    # Step 2: Apply Rotation
    affine_matrix = np.hstack([R, np.zeros((2, 1))])

    translate_to_overlay_center = np.array([
        [1, 0, -image_shift[0]],  # Move centroid to (0,0)
        [0, 1, -image_shift[1]]
    ])

    final_matrix = translate_to_overlay_center @ np.vstack([affine_matrix, [0, 0, 1]]) @ np.vstack([translate_to_origin, [0, 0, 1]])
    final_matrix = final_matrix[:2, :]
    return final_matrix


def apply_affine_transformation(image1, R1, T1,pts1, image2,R2,T2,pts2):
    """
    Applies rotation R and translation T to an image.

    Args:
        image: Input image (numpy array).
        R: 2×2 rotation matrix.
        T: 2×1 translation vector.

    Returns:
        Warped image with correct alignment.
    """
    x_min1, y_min1, x_max1, y_max1 = find_transformed_image_limit(image1,R1,T1)
    x_min2, y_min2, x_max2, y_max2 = find_transformed_image_limit(image2, R2, T2)
    x_min = min(x_min1,x_min2)
    y_min = min(y_min1, y_min2)

    x_max = max(x_max1, x_max2)
    y_max = max(y_max1, y_max2)

    # Define the required canvas size
    canvas_width = int(x_max - x_min)
    canvas_height = int(y_max - y_min)

    final_matrix1 = get_adjusted_affine_matrix(T1,R1,[x_min, y_min])
    final_matrix2 = get_adjusted_affine_matrix(T2, R2, [x_min, y_min])
    # Warp the image with the corrected translation
    transformed_image1 = cv2.warpAffine(image1, final_matrix1, (canvas_width, canvas_height), flags=cv2.INTER_LINEAR)
    transformed_image2 = cv2.warpAffine(image2, final_matrix2, (canvas_width, canvas_height), flags=cv2.INTER_LINEAR)

    return transformed_image1,transformed_image2, np.array([x_min, y_min])

def find_scale_translation(fixed_points, moving_points):
    """
    Computes the scale (s) and translation (T) between two sets of 2D points.

    Args:
        fixed_points: (N, 2) numpy array of fixed reference points.
        moving_points: (N, 2) numpy array of moving points before alignment.

    Returns:
        s: Scale factor.
        T: 2×1 translation vector.
    """
    # Compute centroids
    centroid_fixed = np.mean(fixed_points, axis=0)
    centroid_moving = np.mean(moving_points, axis=0)

    # Compute scale factor
    dist_fixed = np.linalg.norm(fixed_points - centroid_fixed, axis=1)
    dist_moving = np.linalg.norm(moving_points - centroid_moving, axis=1)
    s = np.mean(dist_fixed) / np.mean(dist_moving)  # Scale factor

    # Compute translation
    T = centroid_fixed - s * centroid_moving

    return s, T

def compute_transformed_bounding_box(image_shape, s, T):
    """
    Computes the bounding box of the transformed image based on its four corners.

    Args:
        image_shape: Tuple (height, width) of the image.
        s: Scale factor.
        T: Translation vector.

    Returns:
        (x_min, y_min, x_max, y_max): Cropped region coordinates.
    """
    h, w = image_shape[:2]

    # Define the four corners of the original image
    corners = np.array([
        [0, 0], [w, 0], [0, h], [w, h]
    ], dtype=np.float32)

    # Apply scale and translation to the corners
    transformed_corners = s * corners + T

    # Compute new bounding box
    x_min, y_min = np.min(transformed_corners, axis=0)
    x_max, y_max = np.max(transformed_corners, axis=0)

    return int(x_min), int(y_min), int(x_max), int(y_max)

def warp_image_resize_points(image, moving_points, fixed_points, output_size=(1024, 1024)):
    """
    Warps the moving image based on the transformation from moving_points to fixed_points.
    Crops the image to meaningful content, resizes it to `output_size`, and scales fixed points accordingly.

    Args:
        image: Moving image (numpy array).
        moving_points: (N,2) array of points in the moving image.
        fixed_points: (N,2) array of corresponding points in the fixed coordinate space.
        output_size: Tuple (width, height) defining the final image size.

    Returns:
        Resized warped image, adjusted fixed points.
    """
    h, w = image.shape[:2]

    # Compute transformation parameters
    s, T = find_scale_translation(fixed_points, moving_points)

    # Construct affine transformation matrix
    affine_matrix = np.array([
        [s, 0, T[0]],
        [0, s, T[1]]
    ], dtype=np.float32)

    # Compute the bounding box of transformed image
    x_min, y_min, x_max, y_max = compute_transformed_bounding_box(image.shape, s, T)

    # Apply transformation to the image
    transformed_image = cv2.warpAffine(image, affine_matrix, (w, h), flags=cv2.INTER_LINEAR)

    # Crop to the computed bounding box
    cropped_image = transformed_image[y_min:y_max, x_min:x_max]

    # Compute scale factors for resizing
    cropped_h, cropped_w = cropped_image.shape[:2]
    scale_x = output_size[0] / cropped_w
    scale_y = output_size[1] / cropped_h

    # Resize image to fixed size
    resized_image = cv2.resize(cropped_image, output_size, interpolation=cv2.INTER_LINEAR)

    # Adjust fixed points to align with the resized image
    adjusted_fixed_points = (fixed_points - np.array([x_min, y_min])) * np.array([scale_x, scale_y])

    return resized_image, adjusted_fixed_points




sample_list = ["151507", "151508", "151509","151510", "151669", "151670","151671", "151672", "151673","151674", "151675", "151676"]
processed_id = ["0_0","0_1","0_2","0_3","1_0","1_1","1_2","1_3","2_0","2_1","2_2","2_3"]

adatas = {sample:sc.read_h5ad('../data/DLPFC/{0}_preprocessed.h5'.format(sample)) for sample in sample_list}


for id,pid in zip(sample_list,processed_id):
    adatas[id].image_path = '/home/huifang/workspace/code/registration/data/DLPFC/huifang/{0}_image.png'.format(pid)
    info = np.load('/home/huifang/workspace/code/registration/data/DLPFC/huifang/{0}_validation.npz'.format(pid))
    adatas[id].label = info["label"]
    adatas[id].obsm['spatial'] = info["coord"]

sample_groups = [["151507", "151508", "151509","151510"],[ "151669", "151670","151671", "151672"],[ "151673","151674", "151675", "151676"]]
layer_groups = [[adatas[sample_groups[j][i]] for i in range(len(sample_groups[j]))] for j in range(len(sample_groups))]


layer_to_color_map = {'Layer{0}'.format(i+1):sns.color_palette()[i] for i in range(6)}
layer_to_color_map['WM'] = sns.color_palette()[6]

# alpha = 0.1
# res_df = pd.DataFrame(columns=['Sample','Pair','Kind','Time','Accuracy'])
# pis = [[None for i in range(len(layer_groups[j])-1)] for j in range(len(layer_groups))]
# for j in range(len(layer_groups)):
#    for i in range(len(layer_groups[j])-1):
#        pi0 = pst.match_spots_using_spatial_heuristic(layer_groups[j][i].obsm['spatial'],layer_groups[j][i+1].obsm['spatial'],use_ot=True)
#        start = time.time()
#        # pis[j][i] = pst.pairwise_align(layer_groups[j][i], layer_groups[j][i+1],alpha=alpha,G_init=pi0,norm=True,verbose=False,backend=ot.backend.TorchBackend(),use_gpu=True)
#        pis[j][i] = pst.pairwise_align(layer_groups[j][i], layer_groups[j][i + 1], alpha=alpha, G_init=pi0, norm=True,verbose=False)
#        tt = time.time()-start
#        acc = mapping_accuracy(layer_groups[j][i].obs['layer_guess_reordered'],layer_groups[j][i+1].obs['layer_guess_reordered'],pis[j][i])
#        print(j,i,'Accuracy',acc,'time',tt)
#        np.savetxt('../data/DLPFC/saved_results/transformed_init_{0}_{1}_{2}.gz'.format(j,i,'ot'), pis[j][i], delimiter=',')
#        res_df.loc[len(res_df)] = [j,i,'PASTE',tt,acc]
res_df = pd.DataFrame(columns=['Sample', 'Pair', 'Kind', 'Time', 'Accuracy'])
pis = [[None for i in range(len(layer_groups[j])-1)] for j in range(len(layer_groups))]
for j in range(len(layer_groups)):
    for i in range(len(layer_groups[j])-1):
        pis[j][i]=np.loadtxt('../data/DLPFC/saved_results/transformed_init_{0}_{1}_{2}.gz'.format(j,i,'ot'), delimiter=',')
        start = time.time()
        acc = mapping_accuracy(layer_groups[j][i].obs['layer_guess_reordered'],layer_groups[j][i+1].obs['layer_guess_reordered'],pis[j][i])
        tt = time.time()-start
        # print(j,i,'Accuracy',acc,'time',tt)
        res_df.loc[len(res_df)] = [j,i,'PASTE',tt,acc]

paste_layer_groups = []
transformations = []
visualization = True

mapping_dict = {'Layer1': 1, 'Layer2': 2, 'Layer3': 3, 'Layer4': 4, 'Layer5': 5, 'Layer6': 6, 'WM': 7}


for j in range(len(layer_groups)):
    slices = layer_groups[j]
    for i in range(len(slices)-1):
        pts_slice1 = slices[i].obsm['spatial']
        pts_slice2 = slices[i + 1].obsm['spatial']

        labels1 = np.matrix(slices[i].obs['layer_guess_reordered'].map(mapping_dict))
        labels2 = np.matrix(slices[i + 1].obs['layer_guess_reordered'].map(mapping_dict))

        warped_slice1, warped_slice2, rY, tX, tY = pst.visualization.generalized_procrustes_analysis(pts_slice1, pts_slice2, pis[j][i],
                                                          output_params=True, matrix=True)


        # plt.scatter(warped_slice1[:,0],warped_slice1[:,1],s=10)
        # test = pts_slice1-tX
        # plt.scatter(test[:,0],test[:,1],s=5)
        # plt.show()

        # plt.scatter(warped_slice2[:,0],warped_slice2[:,1],s=10)
        # test = R.dot((pts_slice2-tY).T).T
        # plt.scatter(test[:,0],test[:,1],s=5)
        # plt.show()

        # Load images
        img_slice1 = cv2.imread(slices[i].image_path)  # Moving image
        img_slice2 = cv2.imread(slices[i + 1].image_path)  # Fixed reference image

        # Convert to RGB for visualization
        img_slice1 = cv2.cvtColor(img_slice1, cv2.COLOR_BGR2RGB)
        img_slice2 = cv2.cvtColor(img_slice2, cv2.COLOR_BGR2RGB)
        # Blend images

        warped_img1,warped_img2, image_shift = apply_affine_transformation(img_slice1, np.eye(2), -tX,warped_slice1, img_slice2, rY,-tY,warped_slice2)
        # Step 3: Adjust the transformed points by the same global shift
        adjusted_warped_slice1 = warped_slice1 - image_shift
        adjusted_warped_slice2 = warped_slice2 - image_shift



        # np.savez("../result/PASTE/DLPFC/" + str(j) + "_" + str(i) + "_result", pts1=adjusted_warped_slice1,pts2=adjusted_warped_slice2,img1=warped_img1,img2=warped_img2,label1=labels1,label2=labels2)
        # np.savez("../result/original/DLPFC/" + str(j) + "_" + str(i) + "_result", pts1=pts_slice1,pts2=pts_slice2, img1=img_slice1, img2=img_slice2,label1=labels1,label2=labels2)



        if visualization:
            colors1 = list(slices[i].obs['layer_guess_reordered'].astype(str).map(layer_to_color_map))
            colors2 = list(slices[i+1].obs['layer_guess_reordered'].astype(str).map(layer_to_color_map))
            f, a = plt.subplots(2, 2, figsize=(10, 5))

            # Show warped image 1 with transformed points
            a[0, 0].imshow(img_slice1)

            a[0, 0].scatter(pts_slice1[:, 0], pts_slice1[:, 1], color=colors1, s=5,
                            label="Slice 1")
            a[0, 0].set_title("Image 1 with Original Points")
            a[0, 0].legend()

            # Show warped image 2 with transformed points
            a[0, 1].imshow(img_slice2)
            a[0, 1].scatter(pts_slice2[:, 0], pts_slice2[:, 1],color=colors2, s=5,
                            label="Slice 2")
            a[0, 1].set_title("Image 2 with Original Points")
            a[0, 1].legend()


            # Show warped image 1 with transformed points
            a[1,0].imshow(warped_img1)
            a[1,0].scatter(adjusted_warped_slice1[:, 0], adjusted_warped_slice1[:, 1], color=colors1, s=5,
                         label="Adjusted Warped Slice 1")
            a[1,0].set_title("Warped Image 1 with Transformed Points")
            a[1,0].legend()

            # Show warped image 2 with transformed points
            a[1,1].imshow(warped_img2)
            a[1,1].scatter(adjusted_warped_slice2[:, 0], adjusted_warped_slice2[:, 1], color=colors2, s=5,
                         label="Adjusted Warped Slice 2")
            a[1,1].set_title("Warped Image 2 with Transformed Points")
            a[1,1].legend()

            plt.show()





            # # Overlay images
            img_before_registration = cv2.addWeighted(img_slice1, 0.5, img_slice2, 0.5, 0)
            blended_image = cv2.addWeighted(warped_img1, 0.5, warped_img2, 0.5, 0)

            # Create subplots
            f, a = plt.subplots(2, 2, figsize=(10, 10))

            # Scatter plot of original points
            a[0, 0].scatter(pts_slice1[:, 0], pts_slice1[:, 1], color=colors1,label="Slice 1", alpha=0.6)
            a[0, 0].scatter(pts_slice2[:, 0], pts_slice2[:, 1], color=colors2,label="Slice 2", alpha=0.6)
            a[0, 0].invert_yaxis()  # Match image coordinate system
            a[0, 0].set_title("Original Points")
            a[0, 0].set_aspect('equal')
            a[0, 0].legend()

            # Scatter plot of warped (transformed) points
            a[0, 1].scatter(adjusted_warped_slice1[:, 0], adjusted_warped_slice1[:, 1],color=colors1, label="Warped Slice 1", alpha=0.6)
            a[0, 1].scatter(adjusted_warped_slice2[:, 0], adjusted_warped_slice2[:, 1],color=colors2, label="Warped Slice 2", alpha=0.6)
            a[0, 1].invert_yaxis()
            a[0, 1].set_title("Warped Points")
            a[0, 1].set_aspect('equal')
            a[0, 1].legend()
            # Show overlayed images
            a[1, 0].imshow(img_before_registration)
            a[1, 0].set_title("Before Registration")
            a[1, 0].axis("off")  # Hide axes

            # Show overlayed images
            a[1, 1].imshow(blended_image)
            a[1, 1].set_title("After Registration")
            a[1, 1].axis("off")  # Hide axes


            plt.tight_layout()
            plt.show()