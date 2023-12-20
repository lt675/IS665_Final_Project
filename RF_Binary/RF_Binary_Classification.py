
# Importing the required libraries
import matplotlib
import matplotlib.pyplot as plt
import numpy as np
import pandas
import pandas as pd
import seaborn as sns
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report
from sklearn.model_selection import train_test_split
matplotlib.use('TkAgg')

# Read csv files
df = pd.read_csv('../winequality-red.csv')

# Define binary data split and create new dataframe
ddf = pd.DataFrame()
ddf['class'] = df['quality']
ddf.loc[(ddf['class'] < 6), 'new_class'] = 0
ddf.loc[(ddf['class'] >= 6), 'new_class'] = 1

# Concatenate original and data split dataframes
df_bin = pd.concat((ddf, df), axis=1)

# Remove unnecessary columns
df_bin = df_bin.drop(['class'], axis=1)
df_bin = df_bin.drop(['quality'], axis=1)

# Define independent variables by removing new class column
X = df_bin.drop('new_class', axis=1)

# Define dependent variable or label
y = df_bin['new_class']

# Split data into training and test sets
X_train, X_test, y_train, y_test = train_test_split(X, y, train_size=0.7, random_state=42)

# Define parameters for random forest using the ones from hyperparameter tuning
rf3 = RandomForestClassifier(max_depth=40, n_estimators=30, n_jobs=-1, oob_score=True,
                             random_state=42)

# Fit model
rf3.fit(X_train, y_train)

# Make predictions for the test set
y_pred_test = rf3.predict(X_test)

# View accuracy scores
Acc = (accuracy_score(y_test, y_pred_test))
obscore = rf3.oob_score_
f = open('RF_Binary_Accuracy_Scores.txt', 'w+')
print('The Accuracy score for the binary classification model is', format(Acc, '.2%'), file=f)
print('The out of bag score for the binary classification model is', format(obscore, '.2%'), file=f)
f.close()

# Get and reshape confusion matrix data
matrix = confusion_matrix(y_test, y_pred_test)
matrix = matrix.astype('float') / matrix.sum(axis=1)[:, np.newaxis]

# Build the plot
plt.figure(figsize=(5, 5))
sns.set(font_scale=1)
sns.color_palette("colorblind", as_cmap=True)
sns.heatmap(matrix, center=0, annot=True, annot_kws={'size': 10},
            linewidths=0.2)

# Add labels to the plot
plt.xticks(rotation=0)
plt.yticks(rotation=0)
plt.xlabel('Predicted label')
plt.ylabel('True label')
plt.title('Confusion Matrix for Binary, \n Classification Random Forest Model\n')

# Save the Heatmap
plt.savefig('RF_Binary_CM_Heat.jpeg')

# Create the classification report for test data and predictions
report = (classification_report(y_test, y_pred_test, output_dict=True))
cr = pandas.DataFrame(report).transpose()
cr.to_csv('RF_Binary_Class_Report.csv')
