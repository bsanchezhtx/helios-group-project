import pandas as pd
import numpy as np
from helios import Helios

if __name__ == '__main__':
    # reading the 04-05 data from the csv
    df_04 = pd.read_csv('dataset/Solar_flare_RHESSI_2004_05.csv')

    # changing the month values for the year 2005 (1 = 13, 2 = 14, etc.) to make subdivision simpler later on
    # in rows where the year value is 2005, add 12 to the month value and replace the old month value
    df_04['month'] = np.where((df_04['year'] == 2005), df_04['month'] + 12, df_04['month'])

    # list that will hold sets of 4 with a two month overlap
    dfs_04 = []

    # grouping the data frame by month nummber
    df_m = df_04.groupby(['month'])

    # looping starting at i = 4, in steps of 2, while i < 25
    # creates an overlap of two months between subsets
    for i in range(4, 25, 2):
        # get groups for i - 3th, i - 2th, i - 1th, and ith months, 
        # concatenate them into a single data frame, then append this to the dfs list
        dfs_04.append(pd.concat([df_m.get_group(i - 3), df_m.get_group(i - 2), df_m.get_group(i - 1), df_m.get_group(i)]))

    # same thing but with the 15-16 data
    df_15 = pd.read_csv('dataset/Solar_flare_RHESSI_2004_05.csv')
    df_15['month'] = np.where((df_15['year'] == 2005), df_15['month'] + 12, df_15['month'])
    dfs_15 = []
    df_m = df_15.groupby(['month'])

    for i in range(4, 25, 2):
        dfs_15.append(pd.concat([df_m.get_group(i - 3), df_m.get_group(i - 2), df_m.get_group(i - 1), df_m.get_group(i)]))

    # some examples

    # frequency based intensity plot for the entire 04-05 dataset (no scatter plot overlaid)
    Helios.intensity_estimation_frequency(df_04, plot=True, scatter=False, thresh=False, levels=100)
    # energy based intensity plot for just the last 4 months of the 15-16 dataset
    Helios.intensity_estimation_frequency(dfs_15[-1], plot=True, levels=100)

    