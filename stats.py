import array

import numpy as np
import pandas as pd
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error, r2_score
import statsmodels.api as sm
import matplotlib.pyplot as plt
import seaborn as sns
from scipy import stats
from helios import Helios

df_04 = pd.read_csv('dataset/Solar_flare_RHESSI_2004_05.csv')
#df_04 = pd.DataFrame(df_04['peak.c/s'], df_04['duration.s'], df_04['energy.kev'])
df_04x = df_04[['peak.c/s','duration.s']]
print(df_04x)

df_04y = df_04[['energy.kev']]
print(df_04y)

df_15 = pd.read_csv('dataset/Solar_flare_RHESSI_2015_16.csv')
df_15x = df_15[['peak.c/s', 'duration.s']]
df_15y = df_15[['energy.kev']]

model = LinearRegression()
model.fit(df_04x, df_04y)
predict = model.predict(df_04x)

print("Intercept: \n", model.intercept_)
print("Coefficients: ")
list(zip(df_04x, model.coef_))
print("Mean squared error: %.2f" % mean_squared_error(df_04y, predict))
print("Coefficient of determination: %.2f" % r2_score(df_04y, predict))

# peak.c/s
# duration.s
# energy.kev

