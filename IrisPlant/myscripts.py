import numpy as np
import matplotlib.pyplot as plt
import random as rn
import imageio
import os

def distance(p,ps):
    return np.sum((ps-p)**2,axis=1)

def graf_pt(data):
    plt.scatter(data.T[0],data.T[1],alpha=0.3)

def compute_accuracy(true_indexes,true_centroids,indexes,centroids):
    reorder = MDC(centroids,true_centroids).indexes
    new_indexes = reorder[indexes]
    return np.sum(true_indexes == new_indexes)/(len(indexes))*100




def scatter_clusters_kmeans(clusters, centroids): 
    fig, ax= plt.subplots(figsize=(8,5), dpi=100)
    for i in range(len(clusters)):
        ax.scatter(clusters[i].T[0],clusters[i].T[1])
    ax.scatter(centroids.T[0], centroids.T[1], color="k")

def cluster_scatter(data,indexes,title=""):
    fig, ax= plt.subplots(figsize=(8,5), dpi=100)
    for i in range(len(np.unique(indexes))):
        trues = [indexes == np.unique(indexes)[i]]
        graf_pt(data[trues])
    ax.set_title(title)

def kmeans_gif(data,k,gif=False,gif_file_name="kmeans"):
    centroids = data[rn.sample(range(len(data)),k)]
    
    filenames = []
    for j in range(20):
        # compute distances to centroids
        dists = np.array([distance(centroids[i],data) for i in range(k)])
        
        # list of clusters
        clusters = [np.zeros(len(data[0])) for i in range(k)]
        for i in range(len(data)):
            clusters[int(np.argmin(dists.T[i]))] = np.vstack([clusters[int(np.argmin(dists.T[i]))],data[i]])
        clusters = [clusters[i][1:] for i in range(k)]
        
        if(gif == True):
            # plot the clusters and save the figures
            scatter_clusters_kmeans(clusters,centroids)
            plt.text(-1.45,1.35,f"Iteration: {j+1}")
            filename = f'{j}.png'
            filenames.append(filename)
            plt.savefig(filename)
            plt.close()
        
        # compute new centroids
        centroids = np.array([clusters[i].mean(axis=0) for i in range(k)])
    
    if(gif == True):
        # build gif
        with imageio.get_writer(f'{gif_file_name}.gif', mode='I') as writer:
            for filename in filenames:
                image = imageio.imread(filename)
                for i in range(4):
                    writer.append_data(image)
        # Remove files
        for filename in set(filenames):
            os.remove(filename)

    return centroids




class MDC():
    def __init__(self,data,centroids):
        self.data = data
        self.centroids = centroids
        dists = np.array([distance(centroids[i],data) for i in range(len(centroids))])        
        self.indexes = np.array([int(np.argmin(dists.T[i])) for i in range(len(data))])

class kmeans():
    def __init__(self,data,k,gif=False,gif_file_name="kmeans"):
        self.data = data
        self.k = k
        self.centroids = kmeans_gif(self.data,self.k,gif,gif_file_name)
        self.indexes = MDC(self.data,self.centroids).indexes


def cov_mat(data):
    cov = np.zeros((len(data),len(data)))
    means = np.mean(data,axis=1)
    for i in range(len(data)):
        for j in range(len(data)):
            cov[i][j] = np.sum((data[i]-means[i])*(data[j]-means[j]))/(len(data.T)-1)
    return cov

class PCA():
    def __init__(self,data,n_comps=1):
        self.data = data.T
        self.cov = cov_mat(self.data)
        self.evals, self.evecs = np.linalg.eig(self.cov)

        self.n_comps = n_comps

        self.projection = np.matmul(self.data.T,self.evecs[-self.n_comps:].T)
    
    def project(self,data):
        return np.matmul(data,self.evecs[-self.n_comps:].T)




