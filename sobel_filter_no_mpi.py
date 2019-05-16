import numpy as np
import math
import cv2

# Read image 
image = cv2.imread('test_image.jpg')

# Resize image, if need
#(row_num_im, col_num_im, chan) = image.shape
#image = cv2.resize(image,(int(col_num_im/2), int(row_num_im/2)))

A = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY).astype('float64')

# Matrix for output
(row_num, col_num) = A.shape
B = np.zeros((row_num, col_num))

# padding for convinience
A_padding = np.pad(A, pad_width=1, mode='constant', constant_values=0)

# Sobel operator
filt_size = 3
Sx = np.array([[-1, 0, 1], [-2, 0, 2], [-1, 0, 1]])
Sy = np.array([[1, 2, 1], [0, 0, 0], [-1, -2, -1]])
for i in range(row_num):
    for j in range(col_num):
        # because we do padding, index changed
        idx_i_A = i + 1
        idx_j_A = j + 1
        A_block = A_padding[idx_i_A-1:idx_i_A+2, idx_j_A-1:idx_j_A+2]
        B[i, j] = np.sum(np.multiply(A_block, Sx)) + np.sum(np.multiply(A_block, Sy))

B_min = np.amin(B)
B_max = np.amax(B)
print(B_min, B_max)
B = 255.0*((B - B_min)/B_max)

image = cv2.resize(B.astype('uint8'),(int(col_num/4), int(row_num/4)))
cv2.imshow('test', image)
cv2.waitKey(0)

