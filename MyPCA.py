# Implementation of PCA
# import libraries
import numpy as np


class PCA():
    def __init__(self, percent=0.9, num_dim=None):
        self.num_dim = num_dim  # the number of dimensions to keep, if None, we will refer to the percenage of variance to determine
        self.percent = percent  # placeholders to store the percentage of variance
        self.mean = None  # store the means of training data for normalizing purpose
        self.W = None  # placeholder of projection matrix

    def fit(self,X):
        # normalize the data to make it centered at zero
        self.mean = X.mean(0).reshape(1,-1)  # get the mean
        X = centerData(X, self.mean)

        # finding the projection matrix that maximize the variance (Hint: for eigen computation, use numpy.eigh instead of numpy.eig)
        eig_val, eig_vec = computeE(X)

        # if we do not specify the num_dim, we will compute the num_dim based on the pertange of cov
        if self.num_dim is None:
            # select the reduced dimension that keep >90% of the variance
            self.num_dim = computeDim(eig_val, self.percent)

        # determine the projection matrix and store it as class attribute
        self.W = eig_vec[:,:self.num_dim]

        # project the high-dimensional data to low-dimensional one
        X_pca = project(X, self.W)

        return X_pca, self.num_dim

    def predict(self, X):
        # normalize the test data based on training statistics
        X = centerData(X, self.mean)
        # project the test data
        X_pca = project(X, self.W)
        return X_pca

    def params(self):
        return self.W, self.mean, self.num_dim


# ---------------------------------------------------------
# You are going to implement the following helper functions
# ---------------------------------------------------------


# center the data based on the computed mean
# Input:
#   X: data, (n,d)
#   mean: precomuted mean (1, d)
# Output:
#   centered_X: (n,d)
def centerData(X, mean):
   
    centered_X = np.zeros_like(X)  # placeholder, can be ignored
    centered_X = X - mean # subtract the mean vector from the data matrix
    return centered_X


# compute eigen vectors and eigen values
# Input:
#   centered_X: data, (n,d)
# Output:
#   eig_val: eigenvectors, (d,p)
#   eig_vec: eigenvalues, (p,) p is the number of eigenvectors
def computeE(centered_X):
   
    # (1) get the covariance matrix
    # (2) eigendecomposation of cov, can use np.linalg.eigh(), 
    # plase note the output vals and corresponding vectors are in ascending order, and the shape of vectors are (d, p)
    # (3) reverse the order of eig_val and eig_vec to make it in descending order

    Cov_Data = np.cov(centered_X.T) # find covariance matrix of data
    [eig_val_rev, eig_vec_rev] = np.linalg.eigh(Cov_Data) # find eigenvalues and eigenvectors
    eig_val = np.flip(eig_val_rev, axis = None) # reverse eigenvalue order to match code format
    eig_vec = np.flip(eig_vec_rev, axis = 1) # reverse eigenvector order to match code format
    return eig_val, eig_vec


# compute number of dimensions
# Input:
#   eig_val: eigenvalues (p, )
#   percent: scaler
# Output:
#   num_dim: number of dimensions to keep, scaler
def computeDim(eig_val, percent):
   
    # iterate to add eigen values, until reach the percentage to keep
    # e.g. if eig_val = [0.1, 0.1, 0.3, 0.1, 0.1, 0.3], to keep percent>0.5, num_dim = 4, 
    # beause (0.1+0.1+0.3)/sum(eig_val) = 0.5 is no bigger than 0.5, while (0.1+0.1+0.3+0.1)/sum(eig_val)>0.5

    eig_val_keep = np.array([]) # initialize a new array of eigenvalues to keep

    # iterate through each eigenvalue in the eigenvalue list
    for i in range(len(eig_val)): 
        eig_val_keep = np.append(eig_val_keep, eig_val[i]) # append the newly made array of eigenvalues to keep
        judge = sum(eig_val_keep) / sum(eig_val) # determine the current percentage of eigenvalues
        # judge if they new percentage is greater than the chosen percentage
        if judge > percent:
            num_dim = len(eig_val_keep) # the number dimension = the amount of eigenvalues in the new eigenvalue array
            break
    
    return num_dim


#  project the data to lower dimensions
# Input:
#   X: centered data (n,d)
#   W: projection matrix (d, p)
# Output:
#   X_pca: projected data (n, p)
def project(X, w):
    X_pca = X @ w # multiply data by the projection matrix to lower the dimension of the data
    return X_pca
