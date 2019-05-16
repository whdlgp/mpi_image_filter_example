import numpy as np
import math
from mpi4py import MPI
import cv2

# MPI communication object, get information
comm = MPI.COMM_WORLD
size = comm.Get_size()
rank = comm.Get_rank()

# Read image 
image = cv2.imread('test_image.jpg')

# Resize image, if need
#(row_num_im, col_num_im, chan) = image.shape
#image = cv2.resize(image,(int(col_num_im/2), int(row_num_im/2)))

A = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY).astype('float64')

# Matrix for output
(row_num, col_num) = A.shape
B = np.zeros((row_num, col_num))

print('Num of process', size, ', current rank', rank)

# for each process, allocate A matrix divided by size
if rank == (size-1):
    local_row_num = int(row_num / size) + (row_num % size)
else:
    local_row_num = int(row_num / size)

# allocate local memory for each process
# range = 'local_first_row' to 'local_last_row' - 1
local_first_row = rank * int(row_num / size)
local_last_row = local_first_row + local_row_num 
A_local = A[local_first_row:local_last_row, :]
B_local = B[local_first_row:local_last_row, :]

# padding for convinience
A_local_padding = np.pad(A_local, pad_width=1, mode='constant', constant_values=0)

# Send and Recv border region of A
if rank > 0:
    comm.Send(A_local[0, :], dest = rank-1, tag = 11)

    tmp = np.empty(col_num, dtype = A_local_padding.dtype)
    comm.Recv(tmp, source = rank - 1, tag = 22)
    A_local_padding[0, 1:col_num+1] = tmp
if rank < (size - 1):
    comm.Send(A_local[local_row_num - 1, :], dest = rank + 1, tag = 22)

    tmp = np.empty(col_num, dtype = A_local_padding.dtype)
    comm.Recv(tmp, source = rank + 1, tag = 11)
    A_local_padding[local_row_num +1, 1:col_num+1] = tmp

# Sobel operator
filt_size = 3
Sx = np.array([[-1, 0, 1], [-2, 0, 2], [-1, 0, 1]])
Sy = np.array([[1, 2, 1], [0, 0, 0], [-1, -2, -1]])
(row_num_local, col_num_local) = B_local.shape
for i in range(row_num_local):
    for j in range(col_num_local):
        # because we do padding, index changed
        idx_i_A = i + 1
        idx_j_A = j + 1
        A_block = A_local_padding[idx_i_A-1:idx_i_A+2, idx_j_A-1:idx_j_A+2]
        B_local[i, j] = np.sum(np.multiply(A_block, Sx)) + np.sum(np.multiply(A_block, Sy))

B_min = np.amin(B_local)
B_max = np.amax(B_local)
B_min_global = min(comm.allgather(B_min))
B_max_global = max(comm.allgather(B_max))
print(B_min_global, B_max_global)

B_local = 255.0*((B_local - B_min_global)/B_max_global)
comm.Gatherv(B_local, [B, MPI.DOUBLE], root=0)

if rank == 0:
    image = cv2.resize(B.astype('uint8'),(int(col_num/4), int(row_num/4)))
    cv2.imshow('test', image)
    cv2.waitKey(0)

