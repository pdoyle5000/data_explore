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
    cluster_df['Age'] = df['Age'].astype(np.int64)
    cluster_df['PhoneCoTenure'] = df['PhoneCoTenure']
    cluster_df['LogAvgTotal'] = df['LogAvgTotal']
    return cluster_df

def kmeans_cluster_graphs_2d(df):
    headers = list(df)
    kmeans_results = {}
    for num_clusters in range(2,7):
        title = str(num_clusters) + "KmeansClusters2D"
        kmodel = KMeans(n_clusters=num_clusters)
        kfit = kmodel.fit(df)
        kcluster = kmodel.predict(df)
        fig = plt.figure(num_clusters-1)
        ax = plt.scatter(
            df[df.columns[0]],
            df[df.columns[1]],
            c=colormap[kcluster])
        kmeans_results[title] = kfit
        plt.xlabel(headers[0])
        plt.ylabel(headers[1])
        plt.title(title)
        fig.show()
    return kmeans_results

def kmeans_cluster_graphs_3d(df):
    headers = list(df)
    kmeans_results = {}
    for num_clusters in range(2, 7):
        title = str(num_clusters) + "KmeansClusters3D"
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
        #fig.show()
    return kmeans_results

def ward_cluster_graphs(df):
    headers = list(df)
    ward_results = {}
    for num_clusters in range(2, 7):
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
        #wfig.show()
    return ward_results

def k_means_analysis(df):
    emp_df = pd.DataFrame()
    emp_df['LogEmployment'] = df['LogEmployment']
    emp_df['PhoneCoTenure'] = df['PhoneCoTenure']
    result_emp_len = kmeans_cluster_graphs_2d(emp_df)
    #input()

    bill_df = pd.DataFrame()
    bill_df['LogAvgTotal'] = df['LogAvgTotal']
    bill_df['PhoneCoTenure'] = df['PhoneCoTenure']
    result_avg_bill = kmeans_cluster_graphs_2d(bill_df)
    #input()

    region_df = pd.DataFrame()
    #region_df['Age'] = df['Age']
    region_df['CardTenure'] = df['CardTenure']
    region_df['PhoneCoTenure'] = df['PhoneCoTenure']
    result_region = kmeans_cluster_graphs_2d(region_df)
    #input()

    results3d = kmeans_cluster_graphs_3d(df)
    input()
    log_results = {}
    for result in sorted(results3d):
        log_results[result] = results3d[result].cluster_centers_
    for result in sorted(result_emp_len):
        log_results[result] = result_emp_len[result].cluster_centers_
    for result in sorted(result_avg_bill):
        log_results[result] = result_avg_bill[result].cluster_centers_

    real_results = denormalize_centers(log_results)
    for center in sorted(real_results):
        print(center + '\n')
        print(real_results[center])
        print('\n')

    

def ward_analysis(df):
    results = ward_cluster_graphs(df)
    # Perform Anova Test using df and generated grouping.
    #input()

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
    k_means_analysis(cluster_df)
    #ward_analysis(cluster_df)


if __name__ == "__main__":
    main()
