#!/usr/bin/env python3
from __future__ import print_function
import pandas as pd
import numpy as np
import math
import scipy.stats as stats
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
    cluster_df['Age'] = df['Age'].astype(np.int64)
    cluster_df['LogEmployment'] = df['LogEmployment']
    cluster_df['PhoneCoTenure'] = df['PhoneCoTenure']
    cluster_df['LogAvgTotal'] = df['LogAvgTotal']
    return cluster_df

def kmeans_cluster_graphs_2d(df):
    headers = list(df)
    kmeans_results = {}
    for num_clusters in range(2,7):
        title = str(num_clusters) + "KmeansClusters2D-" + headers[0] 
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
        #fig.show()
    return kmeans_results

def kmeans_cluster_graphs_3d(df):
    headers = list(df)
    kmeans_results = {}
    for num_clusters in range(2, 7):
        title = str(num_clusters) + "KmeansClusters3D-" + headers[0] + "-v-" + headers[1]
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

def ward_cluster_graphs_2d(df):
    headers=list(df)
    ward_results = {}
    for num_clusters in range(2, 7):
        title = str(num_clusters) + "WardHeirarchical2D"
        wmodel = AgglomerativeClustering(
                n_clusters=num_clusters,
                linkage='ward')
        wfit = wmodel.fit(df)
        wcluster = wmodel.fit_predict(df)
        wfig = plt.figure(num_clusters -1)
        ax = plt.scatter(
            df[df.columns[0]],
            df[df.columns[1]],
            c=colormap[wcluster])
        ward_results[title] = wfit
        plt.xlabel(headers[0])
        plt.ylabel(headers[1])
        plt.title(title)
        #wfig.show()
    return ward_results

def ward_cluster_graphs_3d(df):
    headers = list(df)
    ward_results = {}
    for num_clusters in range(2, 7):
        title = str(num_clusters) + "WardHierarchical3D"
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

def stats_analysis(df, results, is_kmeans):
    n_lim = 2
    for result in sorted(results):
        if is_kmeans:
            print(result + " Centers\n")
            print(results[result].cluster_centers_)
            print("Noramlized Centers:")
            print(denormalize_centers(results[result].cluster_centers_))
        else:
            print(result + " Ward Alg\n")

        tmp_df = df
        tmp_df['Labels'] = pd.Series(results[result].labels_, dtype='category')
        c_count = 0
        while (c_count < n_lim):
            single_cluster_df = tmp_df[tmp_df['Labels'] == c_count]
            print("\n")
            print(str(n_lim) + " Clusters.  Cluster Group " + str(c_count + 1))
            print(single_cluster_df.describe())
            c_count = c_count + 1
        n_lim = n_lim + 1
        print("\n")

        print("ANOVA Test\n")
        for k in tmp_df:
            if k != 'Labels':
                pval = stats.f_oneway(tmp_df[k], tmp_df['Labels'])
                print(k + " vs Labels p-value: " + str(pval))
        print("\n")


def k_means_analysis(df):
    emp_df = pd.DataFrame()
    emp_df['LogEmployment'] = df['LogEmployment']
    emp_df['PhoneCoTenure'] = df['PhoneCoTenure']
    result_emp_len = kmeans_cluster_graphs_2d(emp_df)
    stats_analysis(emp_df, result_emp_len, True)
    #input()

    bill_df = pd.DataFrame()
    bill_df['LogAvgTotal'] = df['LogAvgTotal']
    bill_df['PhoneCoTenure'] = df['PhoneCoTenure']
    result_avg_bill = kmeans_cluster_graphs_2d(bill_df)
    stats_analysis(bill_df, result_avg_bill, True)
    #input()

    age_df = pd.DataFrame()
    age_df['Age'] = df['Age']
    age_df['PhoneCoTenure'] = df['PhoneCoTenure']
    result_age = kmeans_cluster_graphs_2d(age_df)
    stats_analysis(age_df, result_age, True)
    #input()


    elp_df = pd.DataFrame()
    elp_df['LogEmployment'] = df['LogEmployment']
    elp_df['LogAvgTotal'] = df['LogAvgTotal']
    elp_df['PhoneCoTenure'] = df['PhoneCoTenure']
    results3d = kmeans_cluster_graphs_3d(elp_df)
    stats_analysis(elp_df, results3d, True)
    #input()

    alp_df = pd.DataFrame()
    alp_df['Age'] = df['Age']
    alp_df['LogEmployment'] = df['LogEmployment']
    alp_df['PhoneCoTenure'] = df['PhoneCoTenure']
    results3d1 = kmeans_cluster_graphs_3d(alp_df)
    stats_analysis(alp_df, results3d1, True)
    #input()
    
def ward_analysis(df):
    emp_df = pd.DataFrame()
    emp_df['LogEmployment'] = df['LogEmployment']
    emp_df['PhoneCoTenure'] = df['PhoneCoTenure']
    result_emp_len = ward_cluster_graphs_2d(emp_df)
    stats_analysis(emp_df, result_emp_len, False)
    #input()

    bill_df = pd.DataFrame()
    bill_df['LogAvgTotal'] = df['LogAvgTotal']
    bill_df['PhoneCoTenure'] = df['PhoneCoTenure']
    result_avg_bill = ward_cluster_graphs_2d(bill_df)
    stats_analysis(bill_df, result_avg_bill, False)
    #input()

    age_df = pd.DataFrame()
    age_df['Age'] = df['Age']
    age_df['PhoneCoTenure'] = df['PhoneCoTenure']
    result_age = ward_cluster_graphs_2d(age_df)
    stats_analysis(age_df, result_age, False)
    #input()

    elp_df = pd.DataFrame()
    elp_df['LogEmployment'] = df['LogEmployment']
    elp_df['LogAvgTotal'] = df['LogAvgTotal']
    elp_df['PhoneCoTenure'] = df['PhoneCoTenure']
    results = ward_cluster_graphs_3d(elp_df)
    stats_analysis(elp_df, results, False)
    #input()

    alp_df = pd.DataFrame()
    alp_df['Age'] = df['Age']
    alp_df['LogEmployment'] = df['LogEmployment']
    alp_df['PhoneCoTenure'] = df['PhoneCoTenure']
    results3d1 = ward_cluster_graphs_3d(alp_df)
    stats_analysis(alp_df, results3d1, False)
    #input()

def denormalize_centers(results):
    # Will not make sense for Age
    dresults = results
    for row in dresults:
        row[0] = math.exp(row[0]) - 1
        if len(row) > 2:
             row[1] = math.exp(row[1]) - 1
    return dresults


def main():
    cluster_df = import_data()
    print("K Means Testing!\n")
    k_means_analysis(cluster_df)
    #print("\n\nWard Testing!\n")
    #ward_analysis(cluster_df)


if __name__ == "__main__":
    main()
