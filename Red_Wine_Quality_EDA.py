
# Import packages
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import matplotlib
matplotlib.use('TkAgg')

#Read csv files
df = pd.read_csv('winequality-red.csv')

#Gets information on dataframe
#There are 1599 records with 12 attributes.
#Quality is an integer, and all others are floats
df.info()

#Check for missing values
#There are none
df.isnull() .sum()

#Generate summary statistics using describe function
df_stat = df.describe().drop(["count"]).round(3)

#Generate .csv file to add to report
df_stat.to_csv('Red_Summary_Stats.csv')

#Generate histograms to observe distributions
fig, axes = plt.subplots(nrows = 4, ncols = 3)    # axes is 2d array (3x3)
axes = axes.flatten()         # Convert axes to 1d array of length 9
fig.set_size_inches(15, 15)

for ax, col in zip(axes, df.columns):
  sns.histplot(df[col], ax = ax)

#Save figure to add to report
plt.savefig('Variable_Distributions.jpeg')

#Create pair plots and save figures to add to report
p1 = sns.pairplot(df, hue='quality', palette='colorblind',
             x_vars=('fixed acidity', 'volatile acidity', 'citric acid'),
             y_vars='quality')
plt.savefig('Pair_Plot1')

p2 = sns.pairplot(df, hue='quality', palette='colorblind',
             x_vars=('residual sugar', 'chlorides', 'free sulfur dioxide'),
             y_vars='quality')
plt.savefig('Pair_Plot2')

p3 = sns.pairplot(df, hue='quality', palette='colorblind',
             x_vars=('total sulfur dioxide', 'density', 'pH'),
             y_vars='quality')
plt.savefig('Pair_Plot3')

p4 = sns.pairplot(df, hue='quality', palette='colorblind',
             x_vars=('sulphates', 'alcohol', 'quality'),
             y_vars='quality')
plt.savefig('Pair_Plot4')

#Create correlation table and save as .csv
corr = df.corrwith(df['quality']).round(2)
print(corr)
corr.to_csv('Quality_Correlation.csv')
