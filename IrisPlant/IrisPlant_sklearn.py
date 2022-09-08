import numpy as np
import matplotlib.pyplot as plt

# use the scritps created for PCA, k-means and MDC
import myscripts

# for iris plant data 
from sklearn import datasets

# minimal distance classificator
from sklearn.neighbors import NearestCentroid

# metrics for classifications
from sklearn.metrics import classification_report

# test split data
from sklearn.model_selection import train_test_split

# download iris plant data
iris = datasets.load_iris()

# X is the raw dataset and y the labels
X = iris.data[:,:]
y = iris.target

# compute the true centroids
true_centroids = myscripts.compute_centroids(X,y)

# reduce the data dimensionality from 4 to 3 and 2 with PCA
iris3D = myscripts.PCA(X,n_comps=3).projection
iris2D = myscripts.PCA(X,n_comps=2).projection

# Splitting training and test data
X_train_c, X_test_c, y_train_c, y_test_c = train_test_split(X, y, test_size = 0.2, shuffle = True, random_state = 0)
X_train_3, X_test_3, y_train_3, y_test_3 = train_test_split(iris3D, y, test_size = 0.2, shuffle = True, random_state = 0)
X_train_2, X_test_2, y_train_2, y_test_2 = train_test_split(iris2D, y, test_size = 0.2, shuffle = True, random_state = 0)

# Creating the Nearest Centroid Classifier
model_c = NearestCentroid()
model_3 = NearestCentroid()
model_2 = NearestCentroid()
 
# Training the classifier
model_c.fit(X_train_c, y_train_c)
model_3.fit(X_train_3, y_train_3)
model_2.fit(X_train_2, y_train_2)

# Plots
myscripts.cluster_scatter(X,y,title="True clusters")
myscripts.cluster_scatter(X,model_c.predict(X),title="MDC on 4D dataset", centroids=False)
myscripts.cluster_scatter(X,model_3.predict(iris3D),title="MDC on 3D projection",centroids = False)
myscripts.cluster_scatter(X,model_2.predict(iris2D),title="MDC on 2D projection",centroids = False)

# Accuracy reports
print()
print("Complete dataset: ")
print()
print(f"Training Set Score : {model_c.score(X_train_c, y_train_c) * 100} %")
print(f"Test Set Score : {model_c.score(X_test_c, y_test_c) * 100} %")
print(f"\nModel Classification Report : \n{classification_report(y_test_c, model_c.predict(X_test_c))}")

print()
print("3D dataset: ")
print()
print(f"Training Set Score : {model_3.score(X_train_3, y_train_3) * 100} %")
print(f"Test Set Score : {model_3.score(X_test_3, y_test_3) * 100} %")
print(f"\nModel Classification Report : \n{classification_report(y_test_3, model_3.predict(X_test_3))}")

print()
print("2D dataset: ")
print()
print(f"Training Set Score : {model_2.score(X_train_2, y_train_2) * 100} %")
print(f"Test Set Score : {model_2.score(X_test_2, y_test_2) * 100} %")
print(f"\nModel Classification Report : \n{classification_report(y_test_2, model_2.predict(X_test_2))}")
