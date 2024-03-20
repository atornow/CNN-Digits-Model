import numpy as np
import ModelFunctions as mf

# Hyperparameters of the model
epochs = 1000
K1 = np.random.uniform(low=-1, high=1, size=(12, 5, 5))
K2 = np.random.uniform(low=-0.01, high=0.01, size=(12, 12, 5, 5))
image = np.random.uniform(low=-3.0, high=2.0, size=(16, 16))
H1 = np.random.uniform(low=-1.0, high=1.0, size=(12, 8, 8))
H2 = np.random.uniform(low=-1.0, high=1.0, size=(12, 4, 4))
H3 = np.random.uniform(low=-1.0, high=1.0, size=(1, 30))
W3 = np.random.uniform(low=-1.0, high=1.0, size=(192, 30))
W4 = np.random.uniform(low=-1.0, high=1.0, size=(30, 9))
target = 6

for i in range(len(H1)):
    H1[i] = mf.apply_kernel_with_padding_and_step(image, K1[i], padding=2, step=2)

H1a = H1
H1 = mf.activate(H1)

for i in range(len(H2)):
    H2[i] = mf.apply_3d_kernel_with_padding_and_step(H1, K2[i])

H2a = H2
H2 = mf.activate(H2a)
H2F = H2a.flatten()
H2F = np.reshape(H2F, (1, 192))

H3a = np.dot(H2F, W3)
H3 = mf.activate(H3a)

H4a = np.dot(H3, W4)
H4 = mf.activate(H4a)

savedProb = mf.softmax(H4)
dH4 = np.copy(savedProb)
dH4[0][target] -= 1
dH4 = mf.inverseActivate(H4a) * dH4

dW4 = np.dot(H3.T, dH4)
dH3 = np.dot(dH4, W4.T)
dH3 = dH3 * mf.inverseActivate(H3a)

dW3 = np.dot(H2F.T, dH3)
dH2F = np.dot(dH3, W3.T)
dH2 = H2F.reshape((12, 4, 4))
dH2 = dH2 * mf.inverseActivate(H2a)

dK2 = np.zeros_like(K2)

slice3d = H1[:, 1:6, 1:6]
print(slice3d.shape)

for i in range(len(dK2)):
    for j in range(len(dH2[i])):
        for k in range(len(H2[i][j])):
            dK2[i] += mf.windowExpander(j, k, H1, len(K2[1][1]), 1, dim3=True) * dH2[i][j][k]


