import numpy as np
import numpy.linalg as LA
from sklearn import preprocessing
import matplotlib.pyplot as plt

# Load the dataset of orb features from file.
orb_features = np.loadtxt("orb_features.txt")

orb_size = len(orb_features[:, 0])
# Compute covariance (exchange the value for None).
cov = np.cov(orb_features.T)

# Compute eigenvectors and eigenvalues
eig_val_cov, eig_vec_cov = LA.eig(cov)

# Sort eigenvectors and corresponding eigenvalues in descending order.
eig_pairs = [(np.abs(eig_val_cov[i]), eig_vec_cov[:, i]) for i in range(len(eig_val_cov))]
eig_pairs.sort(key=lambda x: x[0], reverse=True)
    
# TODO: Compute 5 dimensional feature vector based on largest eigenvalues and normalize the output (exchange the value for None).
n_pca = 5
pca_features= np.zeros((orb_features.shape[0],n_pca))
for i in range(n_pca-1):
    eigen_vec_i = eig_pairs[i][1]
    pca_features[:,i] = np.dot(orb_features,eigen_vec_i)

# Normalize pca features.
pca_features = preprocessing.scale(pca_features)

# 2D plot of first 2 principal components.
plt.scatter(pca_features[:, 0], pca_features[:, 1], marker = 'o')
plt.xlabel('principal component 1')
plt.ylabel('principal component 2')
plt.title('PCA result')
plt.show()

# Save results.
np.savetxt('results_pca.txt', pca_features)
