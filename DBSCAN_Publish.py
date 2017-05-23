import csv
import numpy as np

import matplotlib.pyplot as plt

from sklearn.cluster import DBSCAN
from sklearn.preprocessing import StandardScaler
from sklearn.neighbors import kneighbors_graph

def select_column(matrix, column_name):

    column_index = matrix[0].index(column_name)
    selected_column = [row[column_index] for row in matrix][1:]

    return selected_column

def load_csv(filename, filter_parallax_error = False, rel_parallax_error_limit = 1):

    csvf = open(filename,'rU')
    rows = csv.reader(csvf)
    loaded_file = [row for row in rows]
    csvf.close()

    if "parallax" in loaded_file[0]:

        distances = 1000/np.array(select_column(loaded_file, "parallax")).astype(np.float)
        distances = distances.tolist()
        distances.insert(0, "distance")

        parallaxes = np.array(select_column(loaded_file, "parallax")).astype(np.float)

        parallax_errors = np.array(select_column(loaded_file, "parallax_error")).astype(np.float)

        corrected_parallax_errors = parallax_errors + 0.3

        rel_parallax_errors = (parallax_errors/parallaxes).tolist()
        rel_parallax_errors.insert(0, "rel_parallax_error")

        distance_errors = (1000 * corrected_parallax_errors/np.power(parallaxes, 2)).tolist()
        distance_errors.insert(0, "distance_error")

        parsed_file = np.c_[np.array(loaded_file), distances, rel_parallax_errors, distance_errors].tolist()

        print "\n" + filename + " Loaded!\n"
        print "Sample Size: " + str(len(parsed_file) - 1) + "\n"

        if filter_parallax_error:

            print "Filtering by rel_parallax_error\n"
            print "Max rel_parallax_error: " + str(rel_parallax_error_limit) + "\n"

            filtered_file = []
            filtered_file.append(parsed_file[0])

            for index, rel_parallax_error in enumerate(select_column(parsed_file, "rel_parallax_error")):
                float_rel_parallax_error = float(rel_parallax_error)
                if (float_rel_parallax_error < rel_parallax_error_limit) & (float_rel_parallax_error >= 0):
                    filtered_file.append(parsed_file[index + 1])

            print "Filtered Sample Size: " + str(len(filtered_file) - 1) + "\n"
            return filtered_file

        else:

            print "Not filtering by rel_parallax_error\n"
            return parsed_file

    else:
        print "No \"parallax\" column in file\n"

def k_neighbours_graph(data, columns, min_pts):

    selected_data = []

    for index, column in enumerate(columns):
        selected_data.append(select_column(data, column))

    clustering_data = np.array(map(list, zip(*selected_data))).astype(np.float)
    scaled_clustering_data = StandardScaler().fit_transform(clustering_data)

    k_neighbours_array = kneighbors_graph(scaled_clustering_data, min_pts, mode = 'distance').toarray()
    sorted_k_neighbours_array = np.flipud(np.sort(np.amax(k_neighbours_array, axis=1)))

    plt.grid(which='major')

    plt.xlabel("$\mathrm{point}$")
    plt.ylabel("$\mathrm{" + str(min_pts) + "-distance}$")

    plt.xlim([0,len(sorted_k_neighbours_array) - 1])
    plt.ylim([0,2]) # Set this to a sensible value!

    plt.scatter(np.arange(len(scaled_clustering_data)), sorted_k_neighbours_array)

    choice = raw_input("Show k-distance plot? [y/n] ")

    if choice == "y":
        plt.show()

    print ""

def DBSCAN_cluster(data, columns, eps, min_pts, cluster_errors = False):

    print "DBSCAN Started!\n"
    print "eps: " + str(eps)
    print "minPts: " + str(min_pts) + "\n"

    selected_data = []

    for column in columns:
        selected_data.append(select_column(data, column))

    clustering_data = np.array(map(list, zip(*selected_data))).astype(np.float)
    scaled_clustering_data = StandardScaler().fit_transform(clustering_data)

    dbscan_object = DBSCAN(eps = eps, min_samples = min_pts) # minPts is called min_samples in scikit-learn
    dbscan_object.fit(scaled_clustering_data)
    dbscan_labels = dbscan_object.labels_
    dbscan_labels_count = len(set(dbscan_labels)) - (1 if -1 in dbscan_labels else 0)

    labels_list = np.unique(dbscan_labels)

    if cluster_errors:

        distance_errors = select_column(data, "distance_error")
        cluster_distance_errors = []

        for label in labels_list:
            if label != -1:
                temp_errors = []
                for index, entry in enumerate(dbscan_labels):
                    if entry == label:
                        temp_errors.append(distance_errors[index])
                cluster_distance_errors.append(temp_errors)

        mean_cluster_distance_errors = []

        for cluster in cluster_distance_errors:
            mean_cluster_distance_errors.append(np.mean(np.array(cluster).astype(np.float)))

    if "distance" in data[0]:
        calculated_distances = np.array(select_column(data, "distance")).astype(np.float)
    elif "parallax" in data[0]:
        calculated_distances = 1000/np.array(select_column(data, "parallax")).astype(np.float)

    cluster_distances = []

    for label in labels_list:
        if label != -1:
            temp_distances = []
            for index, entry in enumerate(dbscan_labels):
                if entry == label:
                    temp_distances.append(calculated_distances[index])
            cluster_distances.append(temp_distances)

    cluster_distances.insert(0, calculated_distances.tolist())

    print "DBSCAN Completed!\n"
    print "Number of Clusters: ", dbscan_labels_count, "\n"

    if -1 in dbscan_labels:
        print "Orphans: " + str(dbscan_labels.tolist().count(-1)) + "\n"

    for index in range(0, dbscan_labels_count):
        print "Cluster #" + str(index) + " Size: " + str(dbscan_labels.tolist().count(index))
        if cluster_errors:
            print "Cluster " + "#" + str(index) + " Mean Distance (Corrected) Error: " + str(mean_cluster_distance_errors[index])
        print "Mean distance of cluster: " + str(np.mean(cluster_distances[index + 1])) + "\n"

    return dbscan_labels

def main():

    clustering_columns = ["pmra", "pmdec", "distance"]
    parameters = {'eps': 0.103, 'min_pts': 15}

    my_file = load_csv(filename = "../Data/Radius Tests/10deg.csv", filter_parallax_error = True, rel_parallax_error_limit = 0.1)

    k_neighbours_graph(data = my_file, columns = clustering_columns, min_pts = parameters['min_pts'])

    eps = raw_input("What's the eps? ")
    parameters['eps'] = eps

    cluster_labels = DBSCAN_cluster(data = my_file, columns = clustering_columns, eps = parameters['eps'], min_pts = parameters['min_pts'], cluster_errors = True)

if __name__ == "__main__":
    main()
