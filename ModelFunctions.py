import numpy as np

# Updated function to include configurable padding and step size
def apply_kernel_with_padding_and_step(matrix, kernel, padding=0, step=1):
    # Pad the input matrix
    padded_matrix = np.pad(matrix, pad_width=padding, mode='constant', constant_values=0)

    kernel_flatten = kernel.flatten()
    # Adjust the output size calculation to account for padding
    output_size = ((np.array(padded_matrix.shape) - np.array(kernel.shape)) // step) + 1
    output = np.zeros(output_size)

    for i in range(0, output.shape[0]):
        for j in range(0, output.shape[1]):
            # Calculate current position in the padded matrix
            row_pos = i * step
            col_pos = j * step
            # Extract the current window from the padded matrix based on the step size
            window = padded_matrix[row_pos:row_pos + kernel.shape[0], col_pos:col_pos + kernel.shape[1]]
            # Flatten the window and compute the dot product with the flattened kernel
            window_flatten = window.flatten()
            output[i, j] = np.dot(window_flatten, kernel_flatten)

    return output


def apply_3d_kernel_with_padding_and_step(feature_maps, kernel_3d, padding=0, step=1):
    # Pad each feature map
    padded_maps = np.pad(feature_maps, pad_width=((0, 0), (padding, padding), (padding, padding)), mode='constant',
                         constant_values=0)

    # Output size calculation needs to adjust for 3D, focusing on spatial dimensions
    output_size = ((np.array(padded_maps.shape[1:3]) - np.array(kernel_3d.shape[1:3])) // step) + 1
    output_feature_map = np.zeros(output_size)

    for i in range(0, output_feature_map.shape[0]):
        for j in range(0, output_feature_map.shape[1]):
            # Calculate current position in the padded feature maps
            row_pos = i * step
            col_pos = j * step
            # Extract the current 12x5x5 window from the padded feature maps
            window = padded_maps[:, row_pos:row_pos + kernel_3d.shape[1], col_pos:col_pos + kernel_3d.shape[2]]
            # Element-wise multiply and sum to get a single value
            output_feature_map[i, j] = np.sum(window * kernel_3d)

    return output_feature_map

def windowExpander(j, k, inputMatrix, kSize, stepSize, dim3=False):
    startPosJ = j * stepSize
    startPosK = k * stepSize
    endPosJ = startPosJ + kSize
    endPosK = startPosK + kSize

    if dim3 == True:
        slice3d = inputMatrix[:, startPosJ:endPosJ, startPosK:endPosK]
        return slice3d

    else:
        slice2d = inputMatrix[startPosJ:endPosJ, startPosK:endPosK]
        return slice2d




def activate(x):
    return np.tanh(x)

def inverseActivate(x):
    return 1 - np.tanh(x)**2

def softmax(logits):
    exps = np.exp(logits)
    return exps / np.sum(exps)

def cross_entropy_loss(predictions, targets):
    return -np.sum(targets * np.log(predictions))


