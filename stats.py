import array

import numpy as np
import pandas as pd
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error, r2_score
from sklearn import preprocessing
import statsmodels.api as sm
import matplotlib.pyplot as plt
import seaborn as sns
from scipy import stats
from helios import Helios

df_04 = pd.read_csv('dataset/Solar_flare_RHESSI_2004_05.csv')
df_04x = df_04[['peak.c/s','duration.s']]
print(df_04x)

#df_04y = df_04[['energy.kev']]
df_04y= df_04['energy.kev'].astype('category')
le= preprocessing.LabelEncoder()
le.fit(df_04y)
df_04y = le.transform(df_04y)
print(df_04y)

df_15 = pd.read_csv('dataset/Solar_flare_RHESSI_2015_16.csv')
df_15x = df_15[['peak.c/s', 'duration.s']]
df_15y = df_15[['energy.kev']]
le.fit(df_15y)
df_15y = le.transform(df_15y)
# 0,1,2,3,4,5

model1 = LinearRegression()
model1.fit(df_04x, df_04y)
predict04 = model1.predict(df_04x)

model2 = LinearRegression()
model2.fit(df_15x, df_15y)
predict15 = model2.predict(df_15x)

print("Intercept: \n", model1.intercept_)
print("Coefficients: ")
list(zip(df_04x, model1.coef_))
print("Mean squared error: %.2f" % mean_squared_error(df_04y, predict04))
print("Coefficient of determination: %.2f" % r2_score(df_04y, predict04))

print("Intercept: \n", model2.intercept_)
print("Coefficients: ")
list(zip(df_15x, model2.coef_))
print("Mean squared error: %.2f" % mean_squared_error(df_15y, predict15))
print("Coefficient of determination: %.2f" % r2_score(df_15y, predict15))

colors = {'0':'tab:blue', '1':'tab:orange', '2':'tab:green', '3':'tab:red', '4':'tab:purple', '5':'tab:brown'}
fig, ax = plt.subplots(figsize=(6, 6))
ax.scatter(df_04x['peak.c/s'], df_04x['duration.s'],s = 5,  c=df_04y)

plt.show()
# peak.c/s
# duration.s
# energy.kev

