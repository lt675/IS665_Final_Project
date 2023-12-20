
# Importing the required libraries
import matplotlib
import matplotlib.pyplot as plt
import numpy as np
import pandas
import pandas as pd
import seaborn as sns
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report
from scipy import stats
from sklearn.model_selection import train_test_split

matplotlib.use('TkAgg')

# Read csv files
df = pd.read_csv('../winequality-red.csv')

# Find z-scores to determine outliers. IQR method is too time-consuming.
z = np.abs(stats.zscore(df))

# Remove data points with a z-score greater than 3
# Based on the 68-95-99.7 rule, this should leave us with points that lie within
# 99.7% of the data since those are within 3 standard deviations of the mean.
df_o = df[(z < 3).all(axis=1)]

# Define independent variables by removing quality
X = df_o.drop(columns='quality')

# Define dependent variable or label
y = df_o['quality']

# Split data into training and test sets
X_train, X_test, y_train, y_test = train_test_split(X, y, train_size=0.7, random_state=42)

# Define parameters for random forest using the ones from hyperparameter tuning
rf4 = RandomForestClassifier(max_depth=40, n_estimators=30, n_jobs=-1, oob_score=True,
                             random_state=42)

# Fit model
rf4.fit(X_train, y_train)

# Make predictions for the test set
y_pred_test = rf4.predict(X_test)

# View accuracy scores
Acc = (accuracy_score(y_test, y_pred_test))
obscore = rf4.oob_score_
f = open('RF_Outlier_Accuracy_Scores.txt', 'w+')
print('The Accuracy score for the multiclass model with outlier removal is', format(Acc, '.2%'), file=f)
print('The out of bag score for the multiclass model with outlier removal is', format(obscore, '.2%'), file=f)
f.close()

# Get and reshape confusion matrix data
matrix = confusion_matrix(y_test, y_pred_test)
matrix = matrix.astype('float') / matrix.sum(axis=1)[:, np.newaxis]

# Create .csv version of confusion matrix
class_names = ['4', '5', '6', '7', '8']
mcsv = pandas.DataFrame(matrix, index=class_names, columns=class_names).transpose()
mcsv.to_csv("RF_Outlier_Confusion_Matrix.csv")

# Build a plot for a confusion matrix heatmap
plt.figure(figsize=(7, 7))
sns.set(font_scale=1)
sns.color_palette("colorblind", as_cmap=True)
sns.heatmap(mcsv, center=0, annot=True, annot_kws={'size': 10},
            linewidths=0.2)

# Add labels to the plot
plt.xticks(rotation=0)
plt.yticks(rotation=0)
plt.xlabel('Predicted label')
plt.ylabel('True label')
plt.title('Confusion Matrix for Multiclass\n Outlier Removal Random Forest Model\n')

# Save the Heatmap
plt.savefig('RF_Outlier_CM_Heat.jpeg')

# Create the classification report for test data and predictions
report = (classification_report(y_test, y_pred_test, output_dict=True))
cr = pandas.DataFrame(report).transpose()
cr.to_csv('RF_Outlier_Class_Report.csv')
