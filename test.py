import matplotlib.pyplot as plt
import numpy as np

# 1. Load the image using matplotlib
#    (Be sure "my_image.jpg" is replaced with your actual image file name or path)
img = plt.imread("my_image.jpg")  # This returns a NumPy array

# 2. Inspect the image as a matrix
print("Image shape (Height, Width, Channels):", img.shape)

# 3. Print out a small region of pixels to illustrate the matrix values
#    Here, we take the top-left corner of 5 rows x 5 columns
if img.ndim == 3:  # Color image
    print("Top-left 5x5 region of pixel values (RGB):\n", img[:5, :5, :])
else:  # Grayscale image
    print("Top-left 5x5 region of pixel values (grayscale):\n", img[:5, :5])

# 4. Display the image
#    Matplotlib expects RGB format for display
plt.imshow(img)  # If the image is grayscale, it will show a colormap
plt.title("Example Image as a Matrix")
plt.axis('off')  # Hide axis ticks
plt.show()