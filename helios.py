"""Helper module to pass various functions between jupyter notebooks"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from scipy import stats

class Helios:

    def __init__(self):
        pass
    
    def map_and_normalize(data):
        # maps energy band to average energy
        energy_band_mapping = {    
            '6-12': 9,
            '12-25': 18.5,
            '25-50': 37.5,
            '50-100': 75,
            '100-300': 200,
            '300-800': 550,
            '800-7000': 3900,
            '7000-20000': 13500
        }

        # new column from mapping
        data['energy.median.kev'] = data['energy.kev'].map(energy_band_mapping)

        # retrieving max duration and max energy value
        max_duration = data['duration.s'].max()
        max_energy = data['energy.median.kev'].max()

        # normalizing columns
        data['normalized.duration'] = data['duration.s'] / max_duration
        data['normalized.energy'] = data['energy.median.kev'] / max_energy

        # calculating intensity based on duration * average energy of band
        data['intensity.method.2'] = data['normalized.duration'] * data['normalized.energy']

        # returning the modified data frame
        return data

    # from task1.ipynb
    # used to either get an array of intensity estimations, 
    # a plot if plot=True and thresh=False, 
    # or just list of probability density values if plot=True and thresh=True
    def intensity_estimation_frequency(data, plot=False, scatter=True, thresh=False, levels=10):
        # getting the x, y, and total.counts values in numpy arrays
        x = data['x.pos.asec'].values.flatten()
        y = data['y.pos.asec'].values.flatten()
        counts = data['total.counts'].values.flatten()
        
        if plot:
            # new matplotlib figure
            fig, ax = plt.subplots()

            # seaborn kde plot, uses scipy gaussian_kde underneath
            sns.kdeplot(data=data, x='x.pos.asec', y='y.pos.asec', 
                            weights='total.counts', fill=True, levels=levels, 
                            thresh=0, cmap='magma', cbar=True, ax=ax)
            
            # getting colorbar
            cb = ax.collections[-1].colorbar
            
            # retrieving the tick values for the colorbar, can be used as threshold values 
            ticks = cb.get_ticks()

            if thresh:
                # just return the values for the contours, which can serve as threshold values for hotspot analysis
                plt.close()
                return ticks
            else:
                # Setting the background color of the plot 
                # using set_facecolor() method
                ax.set_facecolor("black")

                # color bar label
                cb.set_label('Intensity')


                # adjusting domain and range
                ax.set_xlim(-1100, 1100)
                ax.set_ylim(-1100, 1100)

                if scatter:
                    # plotting the scatterplot on top of the kde plot
                    plt.scatter(x, y, s=0.5, facecolor='white')

                # title
                ax.set_title(f"Solar Flare Intensity Method 1 for Months {data['month'].min()}-{data['month'].max()}")
                
                # saving the figure to the output folder
                date_range = f"{data['year'].iloc[0]}_{data['month'].min()}-{data['month'].max()}"
                plt.savefig(f"./output/intensity_frequency_{date_range}.png")
                plt.show()

                return fig, ax
        else:
            # data from the subset to be passed to gaussian kernel
            training_locations = np.vstack([x, y])

            # gaussian kernel, bandwidth using "scott's rule", using the total.counts attribute as weights
            kde = stats.gaussian_kde(training_locations, bw_method='scott', weights=counts.T)

            # returning the intensity values for each location
            return kde.evaluate(training_locations)

    # from task1.ipynb    
    # gets the energy based intensity values if plot=False,
    # or a plot if plot=True    
    def intensity_estimation_energy(self, plot=False, scatter=True, levels=10):
        # adding the new intensity values for the data
        #data = self.__map_and_normalize(data)

        # energy-based intensity data
        intensity = self['intensity.method.2'].values.flatten()
        # x and y values
        x = self['x.pos.asec'].values.flatten()
        y = self['y.pos.asec'].values.flatten()

        if plot:
            # new matplotlib figure
            fig, ax = plt.subplots()
            
            # seaborn kde plot, uses scipy gaussian_kde underneath
            ax = sns.kdeplot(data=self, x='x.pos.asec', y='y.pos.asec',
                            weights='intensity.method.2', fill=True, levels=levels, 
                            thresh=0, cmap='inferno', cbar=True, bw_method='scott')

            # Setting the background color of the plot 
            # using set_facecolor() method
            ax.set_facecolor("black")

            # adjusting domain and range
            plt.xlim(-1100, 1100)
            plt.ylim(-1100, 1100)

            # color bar and label
            cb = ax.collections[-1].colorbar
            cb.set_label('Mean Intensity')

            if scatter:
                # plotting the scatterplot on top of the kde plot
                plt.scatter(x, y, s=0.2, facecolor='white')

            ax.set_title(f"Solar Flare Intensity Method 2 for Months {self['month'].min()}-{self['month'].max()}")
            date_range = f"{self['year'].iloc[0]}_{self['month'].min()}-{self['month'].max()}"
            plt.savefig(f"./output/intensity_energy_{date_range}.png")
            plt.show()

            return fig, ax
        else:
            return intensity
        
    # from task2.ipynb
    # averages all the potential threshold values across a set of pandas dataframes
    # and returns d2 and d1
    # levels should have two values for this to work right
    def thresholds(self, dataframes, levels=[0.5, 0.99]):
        ticks = []

        # loop through each subset
        for set in dataframes:
            # append the potential threshold values to the ticks list
            # a level value of [0, 0.25, 0.5, 0.75, 1] will return 5 intensity values where each
            # corresponds with a probability mass of 0%, 25%, and so on
            ticks.append(self.intensity_estimation_frequency(set, plot=True, thresh=True, levels=levels))
        
        # numpy mean will get the element-wise mean for all the potential threshold values
        # from here, we can select the d1 and d2 threshold values from this list, like t[2] for the med
        # hotspots and t[3] for the intense hotspots
        t = np.mean(ticks, axis=0)

        # returning d2, d1
        return (t[0], t[1])
