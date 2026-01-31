import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from sklearn.cluster import KMeans
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler

data_2d = pd.read_csv("data_2d.csv")
mnist = pd.read_csv("mnist.csv")

data_2d.head()
mnist.head()

print(data_2d.shape)
print(mnist.shape)

X_2d = data_2d.values

inertia_2d = []

K = range(1, 11)
for k in K:
    kmeans = KMeans(n_clusters=k, random_state=42)
    kmeans.fit(X_2d)
    inertia_2d.append(kmeans.inertia_)

plt.figure(figsize=(6,4))
plt.plot(K, inertia_2d, marker='o')
plt.xlabel("Кількість кластерів")
plt.ylabel("Inertia")
plt.title("Ліктевий метод для 2D датасету")
plt.show()

kmeans_2d = KMeans(n_clusters=3, random_state=42)
labels_2d = kmeans_2d.fit_predict(X_2d)

plt.figure(figsize=(6,6))
plt.scatter(X_2d[:, 0], X_2d[:, 1], c=labels_2d, cmap='viridis')
plt.scatter(
    kmeans_2d.cluster_centers_[:, 0],
    kmeans_2d.cluster_centers_[:, 1],
    c='red',
    s=200,
    marker='X'
)
plt.title("Кластеризація 2D датасету (K-means)")
plt.show()

X_mnist = mnist.drop(columns=['label'])
y_mnist = mnist['label']
scaler = StandardScaler()
X_mnist_scaled = scaler.fit_transform(X_mnist)
X_sample = X_mnist_scaled[:2000]
inertia_mnist = []

for k in range(1, 11):
    kmeans = KMeans(n_clusters=k, random_state=42)
    kmeans.fit(X_sample)
    inertia_mnist.append(kmeans.inertia_)
plt.figure(figsize=(6,4))
plt.plot(range(1,11), inertia_mnist, marker='o')
plt.xlabel("Кількість кластерів")
plt.ylabel("Inertia")
plt.title("Ліктевий метод для MNIST")
plt.show()

kmeans_mnist = KMeans(n_clusters=10, random_state=42)
mnist_labels = kmeans_mnist.fit_predict(X_sample)

pca = PCA(n_components=2)
X_mnist_pca = pca.fit_transform(X_sample)
plt.figure(figsize=(7,6))
plt.scatter(
    X_mnist_pca[:, 0],
    X_mnist_pca[:, 1],
    c=mnist_labels,
    cmap='tab10',
    s=10
)
plt.title("MNIST: K-means + PCA (2D)")
plt.xlabel("PC1")
plt.ylabel("PC2")
plt.show()
