import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.datasets import make_blobs
from sklearn.cluster import KMeans

# Create random data
data = make_blobs(n_samples=200, n_features=2, centers=4, cluster_std=1.8, random_state=101)

# Plot the data
plt.scatter(data[0][:,0], data[0][:,1], c=data[1], cmap='rainbow')
plt.show()
