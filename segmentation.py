#!/usr/bin/env python3
from __future__ import print_function
import pandas as pd
import numpy as np
import math
import matplotlib.pyplot as plt
from sklearn.cluster import KMeans
from sklearn.cluster import AgglomerativeClustering
from univariate_analysis import FORCE_TYPES as forced_types
from univariate_analysis import OUTPUT_CSV as FROM_ANALYSIS
#from mpl_toolkits.mplot3d import Axes3D
from mpl_toolkits import mplot3d


# Cluster 0=red, 1=lime, 2=black, 3=blue, 4=brown, 5=green
colormap = np.array(['red', 'lime', 'black', 'blue', 'brown', 'green', 'pink', 'teal', 'purple', 'cyan', 'olive', 'orange', 'yellow', 'tan'])

def import_data():
    forced_types['Gender'] = 'category'
    forced_types['JobCategory'] = 'category'
    forced_types['HouseholdIncome'] = 'category'
    forced_types['TownSize'] = 'category'
    forced_types['HomeOwner'] = 'category'
    forced_types['Education'] = 'category'
    forced_types['InHeavyDebt'] = 'category'
    forced_types['HasPets'] = 'category'
    forced_types['EquipmentLastMonth'] = 'category'
    forced_types['EquipmentOverTenure'] = 'category'
    df = pd.read_csv(FROM_ANALYSIS, dtype=forced_types)
    cluster_df = pd.DataFrame()
    cluster_df['LogEmployment'] = df['LogEmployment']
    cluster_df['LogAvgTotal'] = df['LogAvgTotal']
    cluster_df['PhoneCoTenure'] = df['PhoneCoTenure']
    return cluster_df

def kmeans_cluster_graphs(df):
    headers = list(df)
    kmeans_results = {}
    for num_clusters in range(2, 7):
        title = str(num_clusters) + "KmeansClusters"
        kmodel = KMeans(n_clusters=num_clusters)
        kfit = kmodel.fit(df)
        kcluster = kmodel.predict(df)
        fig = plt.figure(num_clusters - 1)
        ax = plt.axes(projection='3d')
        ax.scatter3D(
            df[df.columns[0]],
            df[df.columns[1]],
            df[df.columns[2]],
        c=colormap[kcluster])
        kmeans_results[title] = kfit
        ax.set_xlabel(headers[0])
        ax.set_ylabel(headers[1])
        ax.set_zlabel(headers[2])
        plt.title(title)
        fig.show()
    return kmeans_results

def ward_cluster_graphs(df):
    headers = list(df)
    ward_results = {}
    for num_clusters in range(7, 12):
        title = str(num_clusters) + "WardHierarchical"
        wmodel = AgglomerativeClustering(
            n_clusters=num_clusters,
            linkage='ward')
        wfit = wmodel.fit(df)
        wcluster = wmodel.fit_predict(df)
        wfig = plt.figure(num_clusters - 1)
        ax = plt.axes(projection='3d')
        ax.scatter3D(df[df.columns[0]],
                     df[df.columns[1]],
                     df[df.columns[2]],
                    c=colormap[wcluster])
        ward_results[title] = wfit
        ax.set_xlabel(headers[0])
        ax.set_ylabel(headers[1])
        ax.set_zlabel(headers[2])
        plt.title(title)
        wfig.show()
    return ward_results

def k_means_analysis(df):
    results = kmeans_cluster_graphs(df)
    # Do the math to convert them back to not-logarithm
    # Perform ANOVA on clusters vs data

    i = 2
    log_results = {}
    for result in sorted(results):
        #print(result + " centers\n")
        #print(results[result].cluster_centers_)
        log_results[result] = results[result].cluster_centers_
        i = i+1

    real_results = denormalize_centers(log_results)
    #print("Real_Results\n")
    for center in sorted(real_results):
        print(center + '\n')
        print(real_results[center])
        print('\n')

    input()

def ward_analysis(df):
    results = ward_cluster_graphs(df)
    input()

def denormalize_centers(results):
    # first cal is log of employment length +1, change back
    dresults = results
    for result in dresults:
        for row in dresults[result]:
            row[0] = math.exp(row[0]) - 1
            row[1] = math.exp(row[1]) - 1
    return dresults


def main():
    cluster_df = import_data()
    #k_means_analysis(cluster_df)
    ward_analysis(cluster_df)


if __name__ == "__main__":
    main()
