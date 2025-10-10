# Implementation of KMEANS
# import libraries
import numpy as np
import sys

class Kmeans:
    def __init__(self,k=8): 
        self.num_cluster = k  # Placeholders for the number of clusters
        self.center = None    # Placeholders for the postion of all k centers
        self.cluster_label = np.zeros([k])  # Placeholders for the labels of all the cluster, you don't need to create it, just access it, the shape is (k,)
        self.error_history = []  # the reconstruction error history during the training, you don't need to handle this

    # the fit function does 2 things: (1) find the optimized k-means cluster centers (2) assign label for each cluster
    # Input: 
    #   X: training data, (n, d)
    #   y: trianing_labels, (n, )
    # Output:
    #   num_iter: number of iterations, scaler
    #   error_history: a list of reconstruction errors, list
    def fit(self, X, y):
        # initialization with pre-defined centers
        dataIndex = np.array([1, 200, 500, 1000, 1001, 1500, 2000, 2005])
        self.center = initCenters(X, dataIndex[:self.num_cluster])

        # reset num_iter
        num_iter = 0

        # initialize the cluster assignment
        # cluster assignment are like follows: if there are 10 data, and 3 centers indexed by 0,1,2
        # the cluster assignment [0,0,0,0,0,1,2,1,2,2] means you assign the first 5 points to cluster 0, 
        # point 6,8 to cluster 1, rest to cluster 2
        prev_cluster_assignment = np.zeros([len(X),]).astype('int')
        cluster_assignment = np.zeros([len(X),]).astype('int')
        is_converged = False  # Flag to check whether the center stops updating

        # main-loop for kmeans update
        while not is_converged:
            # using additional space to reduce time complexity
            # For instance, assuming you have X = [1,2,3,4,5,5,5,7,8], and expect to have 2 centers, new_center['center'] = [[0], [0]] at first
            # during iteration, for each data assessed, you can add the point postion to corresponding center, if you assign x[0],x[1] to cluster 0, new_center['center'] = [[1+2], [0]]
            # and new_center['num_sample'] = [2, 0]. Until finished, you can directly compute the updated center by dividing new_center['center'] with its corresponding count new_center['num_sample']
            new_center = dict()
            new_center['center'] = np.zeros(self.center.shape)  # to save time, this variable can store the summation of point positions that are assigned to the same cluster (k,d)
            new_center['num_sample'] = np.zeros(self.num_cluster)  # this variable stores the number of points that have being assinged for each cluster (k, )

            # iterate through the samples and compute their cluster assignment (E step)
            for i in range(len(X)):
                # compute the euclidean distance between sample and centers
                distances = computeDis(X[i], self.center)
                cur_cluster = assignCen(distances)
                cluster_assignment[i] = cur_cluster
                new_center['center'][cur_cluster] += X[i]
                new_center['num_sample'][cur_cluster] += 1

            # update the centers based on cluster assignment (M step)
            self.center = updateCen(new_center)

            # compute the reconstruction error for the current iteration
            cur_error = computeError(X, cluster_assignment, self.center)
            self.error_history.append(cur_error)

            # reach convergence if the assignment does not change anymore
            is_converged = True if (cluster_assignment==prev_cluster_assignment).sum() == len(X) else False
            prev_cluster_assignment = np.copy(cluster_assignment)
            num_iter += 1


        # compute the class label of each cluster based on majority voting
        contingency_matrix = np.zeros([self.num_cluster,3])
        label2idx = {0:0,8:1,9:2}
        idx2label = {0:0,1:8,2:9}
        for i in range(len(cluster_assignment)):
            contingency_matrix[cluster_assignment[i],label2idx[y[i]]] += 1
        cluster_label = np.argmax(contingency_matrix,-1)
        for i in range(self.num_cluster):
            self.cluster_label[i] = idx2label[cluster_label[i]]

        return num_iter, self.error_history

    def predict(self,X):
        # predicting the labels of test samples based on their clustering results
        prediction = np.ones([len(X),])  # placeholder

        # iterate through the test samples
        for i in range(len(X)):
            # (1) find the cluster of each sample
            # (2) get the label of the cluster as predicted label for that data
            
            Distance = computeDis(X[i], self.center) # Find the distance of a test sample to the decided centers
            assignment = assignCen(Distance) # determine which distance is the smallest and therefore which center the test sample should be categorized to
            prediction[i] = self.cluster_label[assignment] # set i of prediction to be the truth label value of the assignment index
        return prediction

    def params(self):
        return self.center


# ---------------------------------------------------------
# You are going to implement the following helper functions
# ---------------------------------------------------------

# init K data centers specified by the dataIndex
# For example, if X = [1,2,3,4,8,10,8,8], dataIndex = [0,4,5] => centers = [1, 8, 10]
# Input: 
#   X: training data, (n, d)
#   dataIndex: list of center index, (k, )
# Output:
#   centers: (k, d)
def initCenters(X, dataIndex):
    # assign the position of specified points as the centers
    centers = X[dataIndex] # given the initial centers in data index, capture those rows of data as the centers
    return centers


# compute the euclidean distance between x to all the centers
# input:
#   x: single data, (d, )
#   centers:   k center position, (k, d)
# Output:
#   dis: k distances, k
def computeDis(x, centers):    
    # compute distance
    diff = centers - x # create a matrix of the difference between the centers matrix and a single row of x for every iteration of computeDis()
    dis = np.zeros_like(centers[:, 0]) # create an empty dis of length k
    for i in range(len(dis)): # loop for the length of k
        normal = np.linalg.norm(diff[i, :]) # normalize the full row of distances between center feature values and x feature values
        dis[i] = normal
    return dis


# compute the index of closest cluster for assignment
# input:
#   distances:  k distances denote the distance between x and k centers
# Output
#   centers:   k center position, (k, d)
def assignCen(distances):
    assignment = np.argmin(distances)  # Find the index of the smallest distance in each distance array. This assigns each row of data to a center.
    return assignment


# compute center by dividing the sum of points with the corresponding count
# input:
#   new_center:  dict, structre specified in previous commonts line 38-41

## LINES #38-41
# using additional space to reduce time complexity
            # For instance, assuming you have X = [1,2,3,4,5,5,5,7,8], and expect to have 2 centers, new_center['center'] = [[0], [0]] at first
            # during iteration, for each data assessed, you can add the point postion to corresponding center, if you assign x[0],x[1] to cluster 0, new_center['center'] = [[1+2], [0]]
            # and new_center['num_sample'] = [2, 0]. Until finished, you can directly compute the updated center by dividing new_center['center'] with its corresponding count new_center['num_sample']
# Output
#   centers:   k center position, (k, d)
def updateCen(new_center):
    centers = np.zeros_like(new_center['center'])
    # get the postion of each center by dividsion between the 
    # sum of points at one center and their corresponding count
    # new_center captures K new center rows and their data in 'center' and the amount of times they are the lowest in 'num_sample' 
    
    center = new_center['center'] # sum of all data that has the minimum distance to its correlated center
    num_sample = new_center['num_sample'] # number of data points that were closest to a given center

    for i in range(len(center)): # index through K centers
        centers[i] = center[i] / num_sample[i] # divide each feature value sum by the number of times that collective data row was closest to center number i
    return centers


# compute reconstruction error, assume data X = [1,2,5,7] has assignment [0,0,1,1], and the center positions are [1.5, 6].
# the reconstruction error is (1-1.5)^2+(2-1.5)^2+(5-6)^2+(7-6)^2
def computeError(X, assign, centers):
    error = 0   # placeholders of errors, can be ignored or removed

    for i in range(len(assign)): # Loop through all rows of data
        error += sum((X[i] - centers[assign[i]]) ** 2) # complete the J equation, but add all data error together into one value
  
    return error
