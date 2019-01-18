import numpy as np
import matplotlib.pyplot as plt
from sklearn.datasets import make_blobs
from sklearn.datasets import make_circles

pts = make_blobs(n_samples=250)[0]
#pts = make_circles(n_samples=250)[0]
#pts = np.random.randint(-10, 10, (100, 2))
eps = 1 # maximum distance to be considered a neighbour
minPoints = 4 # minimum neighbours to be considered a cluster

# create a distance matrix given a list of points
def create_distance_matrix(points):
    distance_matrix = np.zeros((len(points), len(points))) # initialize matrix
    for i in range(len(points)):
        for j in range(len(points)):
            dist = np.sqrt(np.square(points[i]-points[j]).sum()) # calculate distance between two points
            distance_matrix[i, j] = dist # set value of i,j'th point on matrix to distance between point i and point j
    return distance_matrix

# calculate number of neighbours of each point given a distance matrix
def calculate_neighbours(distance_matrix):
    num_neighbours = np.zeros(len(distance_matrix)) # initialize array
    for i in range(len(num_neighbours)):
        num_neighbours[i] = (distance_matrix[i] <= eps).sum() - 1 # calculate number of points within epsilon radius, -1 because don't want to include itself
    return num_neighbours

# calculate whether a point is a core point given number of neighbours
def calculate_core(num_neighbours):
    core = np.zeros(len(num_neighbours)) # initialize array
    for i in range(len(num_neighbours)):
        core[i] = 1 if num_neighbours[i] >= minPoints else 0 # assign 1 if a point has more than minPoints neighbours
    return core

def get_neighbours(pt_idx, distance_matrix):
    neighbours = np.where(distance_matrix[pt_idx] <= eps)[0]
    return neighbours

def calculate_border_noise(distance_matrix, core):
    for i in range(len(core)):
        if core[i] == 1: # if index contains a core point
            #core_neighbours = get_neighbours(i, distance_matrix)
            core_neighbours = np.where(distance_matrix[i] <= eps)[0] # find neighbours of a core point
            core[core_neighbours] = np.array([2 if x==0 else x for x in core[core_neighbours]]) # if the neighbour is not already assigned (is another core) then set it to a border point
    core[core==0] = 3 # points that are neither core or border are noise
    return core

"""
for an unassociated core
get its neighbours
for its neighbours which are core
repeat the above
once you have all those linked cores
assign them all to one cluster
increment cluster
repeat the above until all cores are associated
"""

dtmat = create_distance_matrix(pts)
neighbours = calculate_neighbours(dtmat)
cores = calculate_core(neighbours)
types = calculate_border_noise(dtmat, cores)


from sklearn.cluster import DBSCAN
cluster = DBSCAN(eps=eps, min_samples=minPoints).fit(pts)

f, (ax0, ax1, ax2) = plt.subplots(1, 3, sharey=True)
ax0.scatter(pts[:,0], pts[:,1])
ax1.scatter(pts[:,0], pts[:,1], c=types)
ax2.scatter(pts[:,0], pts[:,1], c=cluster.labels_)
plt.show()
