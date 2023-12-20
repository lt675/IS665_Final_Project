
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

# Define independent variables by removing quality
X = df.drop('quality', axis=1)

# Define dependent variable or label
y = df['quality']

# Split data into training and test sets
X_train, X_test, y_train, y_test = train_test_split(X, y, train_size=0.7, random_state=42)

# Define parameters for random forest using the ones from hyperparameter tuning
rf2 = RandomForestClassifier(max_depth=40, n_estimators=30, n_jobs=-1, oob_score=True,
                             random_state=42)

# Fit model
rf2.fit(X_train, y_train)

# Make predictions for the test set
y_pred_test = rf2.predict(X_test)

Acc = (accuracy_score(y_test, y_pred_test))
obscore = rf2.oob_score_
f = open('RF_Multi_Accuracy_Scores.txt', 'w+')
print('The Accuracy score for the multiclass model is', format(Acc, '.2%'), file=f)
print('The out of bag score for the multiclass model is', format(obscore, '.2%'), file=f)
f.close()

# Get and reshape confusion matrix data
matrix = confusion_matrix(y_test, y_pred_test)
matrix = matrix.astype('float') / matrix.sum(axis=1)[:, np.newaxis]

# Create .csv version of confusion matrix
class_names = ['3', '4', '5', '6', '7', '8']
mcsv = pandas.DataFrame(matrix, index=class_names, columns=class_names).transpose()
mcsv.to_csv("RF_Multi_Confusion_Matrix.csv")


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
plt.title('Confusion Matrix for\n Multiclass Random Forest Model\n')

#Save the Heatmap
plt.savefig('RF_Multi_CM_Heat.jpeg')

# Create the classification report for test data and predictions
report = (classification_report(y_test, y_pred_test, output_dict=True))
cr = pandas.DataFrame(report).transpose()
cr.to_csv('RF_Multi_Class_Report.csv')
