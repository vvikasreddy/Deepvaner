import torch

import torch.nn as nn
from models_ import DeepVANet
from dataset import DEAP
from torch.utils.data import DataLoader

model = DeepVANet(bio_input_size=40)
path = "D:/Vikas/Deepvanet/Deepvaner/results_dominance/DEAP/facebio/s1/DEAP_facebio_dominance_s1_k3_16_64/DEAP_facebio_dominance_s1_k3_16_64.pth"


# Print model's state_dict
# print("Model's state_dict:")
# for param_tensor in model.state_dict():
#     print(param_tensor, "\t", model.state_dict()[param_tensor].size())
# model.load(path)


x = model.face_feature_extractor
y = model.bio_feature_extractor


dataset = DEAP(modal = "facebio", subject= 1, k = 1, kind="all", label = "arousal")

loader = DataLoader(dataset, batch_size=1, shuffle=True)

import numpy as np
vals_x = np.empty((0, 16))
vals_y = np.empty((0,64))
labels = np.empty((0, 1))
count = 0
for ii, (data, label) in enumerate(loader):
    # print(len(data[1]))
    z = y(data[1])
    z_ = x(data[0])
    # print(len(z[0]))

    z =  z.detach().cpu().numpy()
    z_ = z_.detach().cpu().numpy()

    labels = np.append(labels, label.detach().cpu().numpy())

    vals_x = np.append(vals_x, z_, axis = 0)
    vals_y = np.append(vals_y, z, axis = 0)
    # if count == 2:break

    # count +=1
    # if count ==2:
    #     break

# print(vals_x)
# print(vals_y)
#
# print(vals_x.shape)
# print(vals_y.shape)

labels = labels.astype(int)
print(labels)


import numpy as np
import matplotlib.pyplot as plt
from sklearn.manifold import TSNE

# Assuming you have your features stored in 'nx' and 'ny' and labels in 'labels'
# nx and ny should be numpy arrays of shape (number of samples, number of features)
# labels should be a numpy array of shape (number of samples,) containing class labels

# Assuming you have your features stored in 'nx' and 'ny' and labels in 'labels'
# nx and ny should be numpy arrays of shape (number of samples, number of features)
# labels should be a numpy array of shape (number of samples,) containing class labels
from sklearn.preprocessing import StandardScaler
def plot_tSNE(nx, ny, labels):
    # Concatenate nx and ny to get the features matrix
    features = np.concatenate((nx, ny), axis=1)

    # Initialize t-SNE
    tsne = TSNE(n_components=2, random_state=42, perplexity=50)

    scaler = StandardScaler()

    # Fit the scaler to your data and transform it
    features = scaler.fit_transform(features)

    # Perform t-SNE dimensionality reduction
    embedded = tsne.fit_transform(features)

    # Plot t-SNE visualization
    plt.figure(figsize=(8, 6))
    scatter = plt.scatter(embedded[:, 0], embedded[:, 1], c=labels, cmap='viridis')
    plt.title('t-SNE Visualization')
    plt.xlabel('t-SNE Dimension 1')
    plt.ylabel('t-SNE Dimension 2')
    plt.colorbar(scatter, label='Class Label')
    plt.show()

# Call the function to plot t-SNE visualization
plot_tSNE(vals_x, vals_y, labels)
