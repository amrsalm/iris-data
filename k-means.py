import numpy as np
import matplotlib.pyplot as plt
import math as m
import pandas as pd

file_path = 'irisdata.csv'
df = pd.read_csv(file_path, header=0)
X = df[[df.columns[2], df.columns[3]]].values
original_clus = df.iloc[:, -1].values
class color:
    BOLD = '\033[1m'
    END = '\033[0m'
def distortion(X, centroids, clusters):
    """
    Calculate the distortion for the given data points, centroids, and cluster assignments.

    Parameters:
    - X: Data points
    - centroids: Cluster mean
    - clusters: Cluster assignments for each data point

    Returns:
    - Distortion value
    """
    distortion_value = 0
    for i in range(len(X)):
        centroid_index = clusters[i]
        distortion_value += np.linalg.norm(X[i] - centroids[centroid_index])**2

    return distortion_value
def plot_data(X):
    plt.figure(figsize=(7.5, 6))
    if X.shape[1] == 2:
        plt.scatter(X[:, 0], X[:, 1], color='k')
    elif X.shape[1] == 3:
        fig = plt.figure(figsize=(7.5, 6))
        ax = fig.add_subplot(111, projection='3d')
        ax.scatter(X[:, 0], X[:, 1], X[:, 2], color='k')
    else:
        fig = plt.figure(figsize=(7.5, 6))
        ax = fig.add_subplot(111, projection='3d')
        ax.scatter(X[:, 0], X[:, 1], X[:, 2], c=X[:, 3], cmap='viridis')


def random_centroid(X, k):
    random_idx = np.random.choice(len(X), size=k, replace=False)
    centroids = X[random_idx]
    return centroids

def assign_cluster(X, ini_centroids, k):
    cluster = np.argmin(np.linalg.norm(X[:, np.newaxis, :] - ini_centroids, axis=2), axis=1)
    return cluster

def compute_centroid(X, clusters, k):
    centroids = np.array([np.sum(X[clusters == i], axis=0) / np.sum(clusters == i) for i in range(k)])
    return centroids


def difference(prev, nxt):
    diff = np.linalg.norm(prev - nxt)
    return diff

# Used to plot in each iteration
def show_clusters(X,clusters,centroids,ini_centroids,mark_centroid=True,show_ini_centroid=True,show_plots=True):
    #assigning specific color to each cluster. Assuming 3 for now
    cols={0:'r',1:'b',2:'g',3:'coral',4:'c',5:'lime'}
    fig,ax=plt.subplots(figsize=(7.5,6))
    #plots every cluster points
    for i in range(len(clusters)):
        ax.scatter(X[i][0],X[i][1],color=cols[clusters[i]])
    #plots all the centroids
    for j in range(len(centroids)):
        ax.scatter(centroids[j][0],centroids[j][1],marker='*',color=cols[j])
        if show_ini_centroid==True:
            ax.scatter(ini_centroids[j][0],ini_centroids[j][1],marker="+",s=150,color=cols[j])
    #used to mark the centroid by drawing a circle around it
    if mark_centroid is True:
        for i in range(len(centroids)):
            ax.add_artist(plt.Circle((centroids[i][0], centroids[i][1]), 0.4, linewidth=2, fill=False))
            if show_ini_centroid is True:
                ax.add_artist(plt.Circle((ini_centroids[i][0], ini_centroids[i][1]), 0.4, linewidth=2, color='y', fill=False))

    ax.set_xlabel(header[2])
    ax.set_ylabel(header[3])
    ax.set_title("K-means Clustering")
    
    

"""
Initial Mean is marked with plus marker and yellow circle
Final Mean is marked with * marker and black circle

"""
#Used to perform k means clustering
#if show type input is not given then it will show plot for each loop
def k_means(X, k, header, show_type='all', show_plots=True):
    c_prev = random_centroid(X, k)
    cluster = assign_cluster(X, c_prev, k)
    diff = 100
    ini_centroid = c_prev
    intermediate_centroids = [c_prev.copy()]

    if show_plots:
        print(color.BOLD + "\n\nInitial Plot:\n" + color.END)
        show_clusters(X, cluster, c_prev, ini_centroid, header, show_plots=show_plots)

    # Initialize distortion values
    distortion_values = [distortion(X, c_prev, cluster)]

    while diff > 0.0001:
        cluster = assign_cluster(X, c_prev, k)
        if show_type == 'all' and show_plots:
            show_clusters(X, cluster, c_prev, ini_centroid, header, False, False, show_plots=show_plots)
        c_new = compute_centroid(X, cluster, k)
        diff = difference(c_prev, c_new)
        c_prev = c_new
        intermediate_centroids.append(c_prev.copy())

        # Calculate distortion for the current centroids
        distortion_values.append(distortion(X, c_new, cluster))

    if show_plots:
        print(color.BOLD + "\nInitial Cluster Centers:\n" + color.END)
        print(ini_centroid)
        print(color.BOLD + "\nFinal Cluster Centers:\n" + color.END)
        print(c_prev)
        print(color.BOLD + "\n\nFinal Plot:\n" + color.END)
        show_clusters(X, cluster, c_prev, ini_centroid, header, show_ini_centroid=True)

    return cluster, c_prev, distortion_values, intermediate_centroids



def plot_data(X):
    plt.figure(figsize=(7.5,6))
    for i in range(len(X)):
        plt.scatter(X[i][0],X[i][1],color='k')   

def plot_distortion(distortion_values):
    """
    Plot the distortion values over iterations.
    """
    plt.figure(figsize=(8, 6))
    plt.plot(range(len(distortion_values)), distortion_values, marker='o', linestyle='-')
    plt.title('Distortion Over Iterations')
    plt.xlabel('Iteration')
    plt.ylabel('Distortion')
    plt.grid(True)
    
# Function to plot decision boundaries using Voronoi diagram
def plot_decision_boundaries(X, centroids):
    h = 0.02  # Step size of the meshgrid
    x_min, x_max = X[:, 0].min() - 1, X[:, 0].max() + 1
    y_min, y_max = X[:, 1].min() - 1, X[:, 1].max() + 1

    xx, yy = np.meshgrid(np.arange(x_min, x_max, h), np.arange(y_min, y_max, h))

    # Predict the cluster assignments for each point in the meshgrid
    meshgrid_points = np.c_[xx.ravel(), yy.ravel()]
    meshgrid_clusters = assign_cluster(meshgrid_points, centroids, k).reshape(xx.shape)

    # Plot the decision boundaries
    plt.contourf(xx, yy, meshgrid_clusters, cmap='viridis', alpha=0.3)

    # Plot original data points
    plt.scatter(X[:, 0], X[:, 1], c=cluster, cmap='viridis', edgecolors='k', linewidth=0.7)

    # Plot centroids
    plt.scatter(centroids[:, 0], centroids[:, 1], marker='*', c='red', s=200, label='Centroids')

    plt.title('K-means Decision Boundaries')
    plt.xlabel(header[0])
    plt.ylabel(header[1])
    plt.legend()
k = 3
dimensions = X.shape[1]
plot_data(X)
header = df.columns[:-1]

cluster, centroid, distortion_values, intermediate_centroids = k_means(X, k, header, show_type='ini_fin')

# Print the distortion values during iterations
print(color.BOLD + "\nDistortion Values During Iterations:\n" + color.END)
print(distortion_values)
plot_distortion(distortion_values)
# Plot the learning process (initial, intermediate, and final centroids)
plt.figure(figsize=(10, 8))
for i, centroids in enumerate(intermediate_centroids):
    if i == 0:
        marker = 's'  # Initial centroids
    elif i == len(intermediate_centroids) - 1:
        marker = 'D'  # Final centroids
    else:
        marker = 'o'  # Intermediate centroids
    plt.scatter(centroids[:, 0], centroids[:, 1], label=f'Iteration {i}', marker=marker)

plt.scatter(X[:, 0], X[:, 1], color='k', alpha=0.3, label='Data')
plt.legend()
plt.title('K-means Learning Process')
plt.xlabel(header[0])
plt.ylabel(header[1])
plt.grid(True)

plot_decision_boundaries(X, centroid)
plt.show()