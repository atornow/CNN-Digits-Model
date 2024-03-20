import numpy as np
import ModelFunctions as mf

# Hyperparamaters of the model
epochs = 2
filePath = 'semeion.data'
epsilon = 1e-9
learningRate = 0.001

# Initialize weights and biases
K1 = np.random.uniform(low=-0.001, high=0.001, size=(12, 5, 5))
b1 = np.random.uniform(low=-1.0, high=1.0, size=(12, 8, 8))  # Bias for conv layer 1
K2 = np.random.uniform(low=-0.001, high=0.001, size=(12, 12, 5, 5))
b2 = np.random.uniform(low=-1.0, high=1.0, size=(12, 4, 4))  # Bias for conv layer 2
W3 = np.random.uniform(low=-0.001, high=0.0001, size=(192, 30))
b3 = np.random.uniform(low=-1.0, high=1.0, size=(1, 30))  # Bias for fully connected layer 1
W4 = np.random.uniform(low=-0.001, high=0.001, size=(30, 10))
b4 = np.random.uniform(low=-1.0, high=1.0, size=(1, 10))  # Bias for fully connected layer 2
H1 = np.random.uniform(low=-1.0, high=1.0, size=(12, 8, 8))
H2 = np.random.uniform(low=-1.0, high=1.0, size=(12, 4, 4))
H3 = np.random.uniform(low=-1.0, high=1.0, size=(1, 30))
image = np.random.uniform(low=-3.0, high=2.0, size=(16, 16))
imagePadded = np.pad(image, pad_width=2, mode='constant', constant_values=0)
loss = []
for epoch in range(epochs):

    imageCount = 0

    for image, target in mf.readImageData(filePath):


        # Forward pass bulk, saving unactivated and activated layer values along the way
        for i in range(len(H1)):
            H1[i] = mf.apply_kernel_with_padding_and_step(image, K1[i], b1[i], padding=2, step=2)

        H1a = H1
        H1 = mf.activate(H1)

        for i in range(len(H2)):
            H2[i] = mf.apply_3d_kernel_with_padding_and_step(H1, K2[i], b2)

        H2a = H2
        H2 = mf.activate(H2a)
        H2F = H2a.flatten()
        H2F = np.reshape(H2F, (1, 192))

        H3a = np.dot(H2F, W3) + b3
        H3 = mf.activate(H3a)

        H4a = np.dot(H3, W4) + b4
        H4 = mf.activate(H4a)
        savedProb = mf.softmax(H4)

        loss.append(-np.log(savedProb[0][target] + epsilon))

        # Backpropagation bulk, updating weights per image currently
        dH4 = np.copy(savedProb)
        dH4[0][target] -= 1
        dH4 = mf.inverseActivate(H4a) * dH4
        db4 = dH4

        dW4 = np.dot(H3.T, dH4)
        dH3 = np.dot(dH4, W4.T)
        dH3 = dH3 * mf.inverseActivate(H3a)
        db3 = dH3

        dW3 = np.dot(H2F.T, dH3)
        dH2F = np.dot(dH3, W3.T)
        dH2 = H2F.reshape((12, 4, 4))
        dH2 = dH2 * mf.inverseActivate(H2a)
        db2 = dH2

        dK2 = np.zeros_like(K2)

        for i in range(len(dK2)):
            for j in range(len(dH2[i])):
                for k in range(len(H2[i][j])):
                    dK2[i] += mf.windowExpander(j, k, H1, len(K2[1][1]), 1, dim3=True) * dH2[i][j][k]

        dH1 = mf.overlappedSummer(dH2, K2, 1, len(K2[1][1]))
        dH1 = dH1 * mf.inverseActivate(H1a)
        db1 = dH1

        dK1 = np.zeros_like(K1)
        for i in range(len(dK1)):
            for j in range(len(dH1[i])):
                for k in range(len(H1[i][j])):
                    dK1[i] += mf.windowExpander(j, k, imagePadded, len(K2[1][1]), 2) * dH1[i][j][k]

        K1 -= learningRate * dK1
        K2 -= learningRate * dK2
        W3 -= learningRate * dW3
        W4 -= learningRate * dW4
        b1 -= learningRate * db1
        b2 -= learningRate * db2
        b3 -= learningRate * db3
        b4 -= learningRate * db4

        if imageCount % 2 == 0:
            print(loss[imageCount])

        imageCount += 1

mf.plot_error_set(loss)

