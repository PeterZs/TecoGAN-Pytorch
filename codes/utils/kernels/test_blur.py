import scipy.io
import numpy as np
import cv2
import glob

mat_files = glob.glob("./*.mat")
img = cv2.imread("./test_image.jpg")

for mat_file in mat_files:
    mat_name = mat_file.strip("./").split(".")[0]
    blur_kernel = scipy.io.loadmat(mat_file)["kernel"]
    blurred_img = cv2.filter2D(img, -1, blur_kernel)
    cv2.imwrite("./{}_blurred_image.png".format(mat_name), blurred_img)
    blur_kernel_normalized = ((blur_kernel / blur_kernel.max()) * 255.0).astype(np.uint8)
    cv2.imwrite("./{}_vis_normalized.png".format(mat_name), blur_kernel_normalized)
