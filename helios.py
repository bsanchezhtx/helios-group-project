import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from scipy import stats

class Helios:

    def __init__(self):
        pass
    
    def __map_and_normalize(self, data):
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
    # a plot if plot=True, 
    # or just list of probability density values if thresh=True
    def intensity_estimation_frequency(self, data, plot=False, show=True, scatter=False, thresh=False, levels=10):
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
            
            # setting the background color of the plot 
            ax.set_facecolor("black")

            # getting colorbar
            cb = ax.collections[-1].colorbar

            # color bar label
            cb.set_label('Intensity (Frequency-Based)')

            # adjusting domain and range
            ax.set_xlim(-1100, 1100)
            ax.set_ylim(-1100, 1100)

            if scatter:
                # plotting the scatterplot on top of the kde plot
                plt.scatter(x, y, s=0.5, facecolor='white')

            # title
            plt.suptitle('Method 1 (Frequency-Based)')
            ax.set_title(f"Year {data['year'].iloc[0]}, Months {data['month'].min()}-{data['month'].max()}")
            
            # saving the figure to the output folder
            date_range = f"{data['year'].iloc[0]}_{data['month'].min()}-{data['month'].max()}"
            plt.savefig(f"./output/intensity_frequency_{date_range}.png")
            
            if show:
                plt.show()
            else:
                plt.close()

            return fig, ax
        elif thresh:
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

            # don't show the plot
            plt.close()

            return ticks
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
    def intensity_estimation_energy(self, data, plot=False, show=True, scatter=True, levels=10):
        # adding the new intensity values for the data
        data = self.__map_and_normalize(data)

        # energy-based intensity data
        intensity = data['intensity.method.2'].values.flatten()

        # x and y values
        x = data['x.pos.asec'].values.flatten()
        y = data['y.pos.asec'].values.flatten()

        if plot:
            # new matplotlib figure
            fig, ax = plt.subplots()
            
            # seaborn kde plot, uses scipy gaussian_kde underneath
            ax = sns.kdeplot(data=data, x='x.pos.asec', y='y.pos.asec',
                            weights='intensity.method.2', fill=True, levels=levels, 
                            thresh=0, cmap='inferno', cbar=True, bw_method='scott')

            # setting the background color of the plot 
            ax.set_facecolor("black")

            # adjusting domain and range
            plt.xlim(-1100, 1100)
            plt.ylim(-1100, 1100)

            # color bar and label
            cb = ax.collections[-1].colorbar
            cb.set_label('Intensity (Energy-Based)')

            if scatter:
                # plotting the scatterplot on top of the kde plot
                plt.scatter(x, y, s=0.2, facecolor='white')
            plt.suptitle('Method 2 (Energy-Based)')
            ax.set_title(f"Year {data['year'].iloc[0]}, Months {data['month'].min()}-{data['month'].max()}")

            date_range = f"{data['year'].iloc[0]}_{data['month'].min()}-{data['month'].max()}"
            plt.savefig(f"./output/intensity_energy_{date_range}.png")

            if show:
                plt.show()
            else:
                plt.close()

            return fig, ax
        else:
            return intensity
        
    # from task2.ipynb
    # averages all the potential threshold values across a set of pandas dataframes and returns them
    # from there you can select threhold values with t[2], t[3] etc.
    def thresholds(self, data, levels=[0, 0.25, 0.5, 0.95, 1]):
        
        ticks = []

        # loop through each subset
        for set in data:
            # append the potential threshold values to the ticks list
            # a level value of [0, 0.25, 0.5, 0.75, 1] will return 5 intensity values where each
            # corresponds with a probability mass of 25%, 50%, and so on
            ticks.append(self.intensity_estimation_frequency(data=set, plot=True, thresh=True, levels=levels))
        
        # numpy mean will get the element-wise mean for all the potential threshold values
        t = np.mean(ticks, axis=0)

        # returning the average threshold values
        return t
