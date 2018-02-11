import pandas as pd
import numpy as np

from sklearn.cluster import DBSCAN
from sklearn.preprocessing import StandardScaler

from .settings import FEATURES


def _add_dbscan_labels(df, parameters):

    feature_df = df[FEATURES].copy()
    feature_df[FEATURES] = StandardScaler().fit_transform(feature_df)
    feature_matrix = feature_df.as_matrix()

    dbscan_object = DBSCAN(eps=parameters['eps'], min_samples=parameters['min_pts'])
    dbscan_object.fit(feature_matrix)

    labelled_df = df.copy()
    labelled_df['cluster'] = dbscan_object.labels_

    return labelled_df


def _get_cluster_info(labelled_df):

    cluster_info = []

    for cluster_label in np.unique(labelled_df['cluster']):
        cluster_df = labelled_df[labelled_df['cluster'] == cluster_label]

        label_info = {'label': cluster_label,
                      'count': len(cluster_df),
                      'median_distance': None,
                      'mean_distance_error': None}

        if cluster_label != -1:
            label_info['median_distance'] = round(cluster_df['distance'].median(), 2)
            label_info['mean_distance_error'] = round(cluster_df['distance_error'].mean(), 2)

        cluster_info.append(label_info)

    return cluster_info


def _present_cluster_info(cluster_info):

    print("Number of Clusters: {}".format(len(cluster_info) - 1))

    for cluster in cluster_info:
        if cluster['label'] != -1:
            print('Cluster #{} Count: {}'.format(cluster['label'], cluster['count']))
            print('Cluster #{} Median Distance (pc): {}'.format(cluster['label'], cluster['median_distance']))
            print('Cluster #{} Mean Distance Error (pc): {}'.format(cluster['label'], cluster['mean_distance_error']))
        else:
            print('Number of Orphans: {}'.format(cluster['count']))


def cluster(df, parameters):

    labelled_df = _add_dbscan_labels(df, parameters)
    cluster_info = _get_cluster_info(labelled_df)

    _present_cluster_info(cluster_info)

    return labelled_df
