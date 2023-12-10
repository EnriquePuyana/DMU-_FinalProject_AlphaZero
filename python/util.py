import numpy as np

# def dense_to_sparse(dense_array):
#     """
#     Convert a dense 3D NumPy array to a sparse CSR matrix.
#     """
#     # Reshape the 3D array to 2D before converting to CSR matrix
#     dense_array_2d = dense_array.reshape((dense_array.shape[0], -1))
#     sparse_array = csr_matrix(dense_array_2d)
#     return sparse_array

# def sparse_to_dense(sparse_array, original_shape=(8, 8, 64)):
#     """
#     Convert a sparse CSR matrix to a dense 3D NumPy array with the original shape.
#     """
#     dense_array_2d = sparse_array.toarray()
#     dense_array = dense_array_2d.reshape(original_shape)
#     return dense_array

def flip_action_array(action_array):
    n = len(action_array) // 8
    return np.vstack((
        action_array[n:2*n, :],
        action_array[:n, :],
        action_array[2*n:4*n, :],
        action_array[5*n:6*n, :],
        action_array[4*n:5*n, :],
        action_array[7*n:8*n, :],
        action_array[6*n:7*n, :]
    ))
    
def append_and_assert_length(some_list: list, some_data: list, max_length: int) -> list:
    some_list.extend(some_data)
    list_length = len(some_list)
    if list_length>max_length:
        some_list = some_list[(list_length-max_length):]
        
    return some_list