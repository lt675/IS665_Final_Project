
# Importing the required libraries
import matplotlib
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import GridSearchCV
matplotlib.use('TkAgg')

# Read csv files
df = pd.read_csv('winequality-red.csv')

# Define independent variables by removing quality
X = df.drop('quality', axis=1)

# Define dependent variable or label
y = df['quality']

# Split data into training and test sets
# random_state value generally used is Hitch-hikers reference so keep it there
# Keep random_state so we can get the same results from the train/test splits

X_train, X_test, y_train, y_test = train_test_split(X, y, train_size=0.7, random_state=42)

# Fit the model
# Natural criterion is gini index. Leave it as such
# Not much gained by switching to entropy, which is slower than gini
# oob_score stays true to keep out of the bag cross-validation method
# Not a large dataset so keeping oob_score should be fine
# Keep n_jobs at -1. Let it use all the powers

classifier_rf = RandomForestClassifier(random_state=42, n_jobs=-1, oob_score=True)
classifier_rf.fit(X_train, y_train)

# Checking the oob score
print(classifier_rf.oob_score_)

# Show current parameters
print(classifier_rf.get_params())

'''
print(classifier_rf.oob_score_)
0.6863270777479893

print(classifier_rf.get_params())
{'bootstrap': True, 'ccp_alpha': 0.0, 'class_weight': None, 'criterion': 'gini', 'max_depth': None,
 'max_features': 'sqrt', 'max_leaf_nodes': None, 'max_samples': None, 'min_impurity_decrease': 0.0,
  'min_samples_leaf': 1, 'min_samples_split': 2, 'min_weight_fraction_leaf': 0.0, 'n_estimators': 100,
   'n_jobs': -1, 'oob_score': True, 'random_state': 42, 'verbose': 0, 'warm_start': False}
'''

# Hyperparameter tuning
rf = RandomForestClassifier(random_state=42, n_jobs=-1, oob_score=True)

params = {
    'max_depth': [2, 3, 5, 10, 20, 40, 50],
    'min_samples_leaf': [1, 2, 5, 10, 20, 50],
    'n_estimators': [10, 25, 30, 50, 100, 200, 350, 500, 750, 1000],
    'max_features': [5, 7, 9, 'sqrt', 'log2'],
    'min_samples_split': [2, 4, 5, 10]
}

# Instantiate the grid search model
grid_search = GridSearchCV(estimator=rf,
                           param_grid=params,
                           cv=4,
                           n_jobs=-1, verbose=3, scoring="accuracy")

'''Fitting 4 folds for each of 8400 candidates, totalling 33600 fits
This took a long time to compile. In the future, prune the parameters,
 lower the cv, or get a better computer'''

grid_search.fit(X_train, y_train)

print(grid_search.best_score_)

rf_best = grid_search.best_estimator_
print(rf_best)

# Show current parameters
print(rf_best.get_params())

'''#Below are the results from the Hyperparameter Tuning

print(grid_search.best_score_)
0.6720718125960061
 
print(rf_best)
RandomForestClassifier(max_depth=40, n_estimators=30, n_jobs=-1, oob_score=True,
                        random_state=42)

print(rf_best.get_params())
{'bootstrap': True, 'ccp_alpha': 0.0, 'class_weight': None, 'criterion': 'gini', 'max_depth': 40,
'max_features': 'sqrt', 'max_leaf_nodes': None, 'max_samples': None, 'min_impurity_decrease': 0.0,
'min_samples_leaf': 1, 'min_samples_split': 2, 'min_weight_fraction_leaf': 0.0, 'n_estimators': 30,
'n_jobs': -1, 'oob_score': True, 'random_state': 42, 'verbose': 0, 'warm_start': False}'''
