import numpy as np
import pandas as pd
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error, r2_score
from sklearn import preprocessing
import matplotlib.pyplot as plt


df_04 = pd.read_csv('dataset/Solar_flare_RHESSI_2004_05.csv')
df_04x = df_04[['peak.c/s','duration.s']]
df_04y= df_04['energy.kev'].astype('category')
le= preprocessing.LabelEncoder()
le.fit(df_04y)
df_04y = le.transform(df_04y)


df_15 = pd.read_csv('dataset/Solar_flare_RHESSI_2015_16.csv')
df_15x = df_15[['peak.c/s', 'duration.s']]
df_15y = df_15['energy.kev'].astype('category')
le.fit(df_15y)
df_15y = le.transform(df_15y)
# 0,1,2,3,4,5

model1 = LinearRegression()
model1.fit(df_04x, df_04y)
predict04 = model1.predict(df_04x)

model2 = LinearRegression()
model2.fit(df_15x, df_15y)
predict15 = model2.predict(df_15x)

# 04-05 linear regression attempt, unused
print("Intercept: \n", model1.intercept_)
print("Coefficients: ")
list(zip(df_04x, model1.coef_))
print("Mean squared error: %.2f" % mean_squared_error(df_04y, predict04))
print("Coefficient of determination: %.2f" % r2_score(df_04y, predict04))

# 15-16 linear regression attempt, unused
print("Intercept: \n", model2.intercept_)
print("Coefficients: ")
list(zip(df_15x, model2.coef_))
print("Mean squared error: %.2f" % mean_squared_error(df_15y, predict15))
print("Coefficient of determination: %.2f" % r2_score(df_15y, predict15))


# 04-05 scatter plot
fig1, ax1 = plt.subplots(figsize=(6, 6))
ax1.scatter(df_04x['duration.s'], df_04x['peak.c/s'], s = 5,  c=df_04y)
plt.title('2004-2005 Scatter Plot Comparing Peak.c/s and Duration.s')
plt.ylabel('Peak.c/s')
plt.xlabel('Duration.s')
plt.savefig(f"./output/04_scatter.png", dpi = 300)

# 04-05 histogram, unused
fig3, axs3 = plt.subplots()
axs3.hist(df_04y, bins=[0,1,2,3,4,5])
plt.title('Histogram of 2004-05 Energy.kev in Numerical Categories')
plt.savefig(f"./output/04_hist.png", dpi = 300)

# 15-16 scatter plot
fig2, ax2 = plt.subplots(figsize=(6,6))
ax2.scatter(df_15x['duration.s'], df_15x['peak.c/s'], s = 5,  c=df_15y)
plt.title('2015-2016 Scatter Plot Comparing Peak.c/s and Duration.s')
plt.ylabel('Peak.c/s')
plt.xlabel('Duration.s')
plt.savefig(f"./output/15_scatter.png", dpi = 300)

# 15-16 histogram, unused
fig4, axs4 = plt.subplots()
axs4.hist(df_15y, bins=[0,1,2,3,4,5])
plt.title('Histogram of 2015-16 Energy.kev in Numerical Categories')
plt.savefig(f"./output/15_hist.png", dpi = 300)

# print all plots
plt.show()
# peak.c/s
# duration.s
# energy.kev

