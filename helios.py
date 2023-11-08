"""Helper module to pass various functions between jupyter notebooks"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from scipy import stats

# from task1.ipynb
def intensity_estimation(data, plot=False, levels=10):
    # getting the x, y, and total.counts values in numpy arrays
    x = data['x.pos.asec'].values.flatten()
    y = data['y.pos.asec'].values.flatten()
    counts = data['total.counts'].values.flatten()
    
    if plot:
        # new matplotlib figure
        plt.figure()

        # seaborn kde plot, uses scipy gaussian_kde underneath
        ax = sns.kdeplot(data=data, x='x.pos.asec', y='y.pos.asec', 
                           weights='total.counts', fill=True, levels=levels, 
                           thresh=0, cmap='gist_heat', cbar=True)
        
        # retrieving the tick values for the colorbar, can be used as threshold values 
        ticks = ax.collections[-1].colorbar.get_ticks()

        # adjusting domain and range
        plt.xlim(x.min(), x.max())
        plt.ylim(y.min(), y.max())

        # title
        plt.title(f"Solar Flare Intensities for Months {data['month'].min()}-{data['month'].max()}")

        # plotting the scatterplot on top of the kde plot
        plt.scatter(x, y, s=0.5, facecolor='white')
        
        # saving the figure to the output folder
        date_range = f"{data['year'].iloc[0]}_{data['month'].min()}-{data['month'].max()}"
        plt.savefig(f"./output/task2/intensity_frequency{date_range}.png")
        
        # prevent the plot from being shown
        plt.close()

        # return the values for the contours, which can serve as threshold values for hotspot analysis
        return ticks
    else:
        # data from the subset to be passed to gaussian kernel
        training_locations = np.vstack([x, y])

        # gaussian kernel, bandwidth using "scott's rule", using the total.counts attribute as weights
        kde = stats.gaussian_kde(training_locations, bw_method='scott', weights=counts.T)

        # returning the intensity values for each location
        return kde.evaluate(training_locations)