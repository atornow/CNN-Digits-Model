import numpy as np
import matplotlib.pyplot as plt

# Updated function to include configurable padding and step size
def apply_kernel_with_padding_and_step(matrix, kernel, b, padding=0, step=1):
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
            output[i, j] = np.dot(window_flatten, kernel_flatten) + b[i][j]

    return output


def apply_3d_kernel_with_padding_and_step(feature_maps, kernel_3d, b, padding=0, step=1):
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
            output_feature_map[i, j] = np.sum(window * kernel_3d) + b[0][i][j]

    return output_feature_map

def windowExpander(j, k, inputMatrix, kSize, sSize, dim3=False):
    startPosJ = j * sSize
    startPosK = k * sSize
    endPosJ = startPosJ + kSize
    endPosK = startPosK + kSize

    if dim3 == True:
        slice3d = inputMatrix[:, startPosJ:endPosJ, startPosK:endPosK]
        return slice3d

    else:
        slice2d = inputMatrix[startPosJ:endPosJ, startPosK:endPosK]
        return slice2d

def overlappedSummer(dH2, K2, sSize, kSize):
    oSize = (len(dH2[1]) - 1) * sSize + kSize
    outputMatrix = np.zeros((len(K2), oSize, oSize))
    for k in range(len(K2)):
        for i in range(len(dH2[1])):
            for j in range(len(dH2[1])):
                startPosI = i * sSize
                endPosI = startPosI + kSize
                startPosJ = j * sSize
                endPosJ = startPosJ + kSize

                outputMatrix[:, startPosI:endPosI, startPosJ:endPosJ] += K2[k] * dH2[k][i][j]

    return outputMatrix



def activate(x):
    return np.tanh(x)

def inverseActivate(x):
    return 1 - np.tanh(x)**2

def softmax(logits):
    exps = np.exp(logits)
    return exps / np.sum(exps)

def cross_entropy_loss(predictions, targets):
    return -np.sum(targets * np.log(predictions))

def readImageData(filePath):
    with open(filePath, 'r') as file:
        for line in file:
            parts = line.strip().split(" ")
            # The first 256 values are the pixel values for the 16x16 image
            image_data = np.array([float(val) for val in parts[:256]])
            image = image_data.reshape(16, 16)

            # The last 10 values are the binary class labels
            label_data = np.array([int(float(val)) for val in parts[256:]])
            label = np.argmax(label_data)  # The index of the '1' is the class label

            yield image, label


def plot_error_set(ErrorSet):
    # Determine the size of ErrorSet
    size = len(ErrorSet)

    # Create a figure and axis
    fig, ax = plt.subplots(figsize=(10, 6))

    # Plotting
    ax.plot(ErrorSet, label='Error')

    # Scaling the x-axis based on the size of ErrorSet
    if size > 10000000:
        ax.set_xscale('log')
        ax.set_xlabel('Epoch (log scale)')
    else:
        ax.set_xlabel('Epoch')

    # Enhancements for readability
    ax.set_ylabel('Error')
    ax.set_title('Averaged error over all epochs')
    ax.grid(True)
    ax.legend()

    # Show the plot
    plt.show()


