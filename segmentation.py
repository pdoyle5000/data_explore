#!/usr/bin/env python3
from __future__ import print_function
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.cluster import KMeans
from univariate_analysis import FORCE_TYPES as forced_types
from univariate_analysis import OUTPUT_CSV as FROM_ANALYSIS
from mpl_toolkits.mplot3d import Axes3D

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
    
    return pd.read_csv(FROM_ANALYSIS, dtype=forced_types)

def plot_k_means_cluster(df):

	# View the results
	# Set the size of the plot
	plt.figure()
	 
	# Create a colormap
	colormap = np.array(['red', 'lime', 'black'])
	 
	# Plot the Models Classifications
	#plt.subplot(1, 2, 2)
	plt.scatter(df['LogEmployment'], df['LogAvgTotal'], c=colormap[df['kmeans_model']], s=40)
	plt.title('K Mean Classification')
	plt.show()

def main():
    df = import_data()
    cluster_df = pd.DataFrame()
    cluster_df['LogEmployment'] = df['LogEmployment']
    cluster_df['LogAvgTotal'] = df['LogAvgTotal']
    cluster_df['PhoneCoTenure'] = df['PhoneCoTenure']
    # TODO: Visualize data first maybe.
    print(cluster_df['LogEmployment'].describe())
    print(cluster_df['LogAvgTotal'].describe())
    print(cluster_df['PhoneCoTenure'].describe())

    # Perform K Nearest Neighbor Clustering
    kmeans_model = KMeans(n_clusters=3).fit(cluster_df)

    cluster_df['kmeans_model'] = kmeans_model.labels_

    # 3d array code?
    plot_k_means_cluster(cluster_df)
    """
    fig = plt.figure()
    ax = fig.add_subplot(projection='3d')
    ax.scatter(
    	cluster_df['LogEmployment'], 
    	cluster_df['LogAvgTotal'], 
    	cluster_df['PhoneCoTenure'],
    	'gray')
    	"""
if __name__ == "__main__":
    main()
