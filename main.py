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

    for i in range(4, 25, 2):
        # get groups for i - 3th, i - 2th, i - 1th, and ith months, 
        # concatenate them into a single data frame, then append this to the dfs list
        dfs_04.append(pd.concat([df_m.get_group(i - 3), df_m.get_group(i - 2), df_m.get_group(i - 1), df_m.get_group(i)]))



    # same thing but with the 15-16 data
    df_15 = pd.read_csv('dataset/Solar_flare_RHESSI_2015_16.csv')
    df_15['month'] = np.where((df_15['year'] == 2016), df_15['month'] + 12, df_15['month'])
    dfs_15 = []
    df_n = df_15.groupby(['month'])

    for i in range(4, 14, 2):
        dfs_15.append(pd.concat([df_n.get_group(i - 3), df_n.get_group(i - 2), df_n.get_group(i - 1), df_n.get_group(i)]))

    for i in range(20, 25, 2):
        dfs_15.append(pd.concat([df_n.get_group(i - 3), df_n.get_group(i - 2), df_n.get_group(i - 1), df_n.get_group(i)]))



    helios = Helios()

    s1 = dfs_04[0]
    s2 = dfs_15[0]

    helios.intensity_estimation_frequency(s1, True, levels=100)
    helios.intensity_estimation_energy(s1, plot=True, levels=100)
    helios.intensity_estimation_frequency(s2, plot=True, levels=100)
    helios.intensity_estimation_energy(s2, plot=True, levels=100)
