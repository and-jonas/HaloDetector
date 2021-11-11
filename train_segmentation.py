
# ======================================================================================================================
# Author: Jonas Anderegg, jonas.anderegg@usys.ethz.ch
# Extract training data from labelled patches and train segmentation algorithm
# Last modified: 2021-11-10
# ======================================================================================================================

from pathlib import Path
import pickle
import numpy as np
import pandas as pd
import glob
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import RandomizedSearchCV  # Number of trees in random forest
from sklearn.model_selection import GridSearchCV  # Create the parameter grid based on the results of random search
import SegmentationFunctions

# ======================================================================================================================
# (1) extract features and save to .csv
# ======================================================================================================================

workdir = 'Z:/Public/Jonas/001_LesionZoo/MRE'

# set directories to previously selected training patches
dir_positives = f'{workdir}/train_data_segmentation/Positives'
dir_negatives = f'{workdir}/train_data_segmentation/Negatives'

# extract feature data for all pixels in all patches
# output is stores in .csv files in the same directories
SegmentationFunctions.iterate_patches(dir_positives, dir_negatives)

# ======================================================================================================================
# (2) combine training data from all patches into single file
# ======================================================================================================================

# import all training data
# get list of files
files_pos = glob.glob(f'{dir_positives}/*.csv')
files_neg = glob.glob(f'{dir_negatives}/*.csv')
all_files = files_pos + files_neg

# load data
train_data = []
for i, file in enumerate(all_files):
    print(i)
    data = pd.read_csv(file)
    data = data.iloc[::10, :]  # only keep every 10th pixel of the patch
    train_data.append(data)
# to single df
train_data = pd.concat(train_data)
# export, this may take a while
train_data.to_csv(f'{workdir}/train_data_segmentation/training_data.csv', index=False)

# ======================================================================================================================
# (3) train random forest classifier
# ======================================================================================================================

train_data = pd.read_csv(f'{workdir}/train_data_segmentation/training_data.csv')

# OPTIONAL: sample an equal number of rows per class for training
n_pos = train_data.groupby('response').count().iloc[0, 0]
n_neg = train_data.groupby('response').count().iloc[1, 0]
n_min = min(n_pos, n_neg)
train_data = train_data.groupby(['response']).apply(lambda grp: grp.sample(n=n_min))

n_estimators = [int(x) for x in np.linspace(start=20, stop=200, num=10)]
# Number of features to consider at every split
max_features = ['auto', 'sqrt']
# Maximum number of levels in tree
max_depth = [int(x) for x in np.linspace(10, 110, num=11)]
max_depth.append(None)
# Minimum number of samples required to split a node
min_samples_split = [2, 5, 10]
# Minimum number of samples required at each leaf node
min_samples_leaf = [1, 2, 4, 8]
# Method of selecting samples for training each tree
bootstrap = [True, False]  # Create the random grid
random_grid = {'n_estimators': n_estimators,
               'max_features': max_features,
               'max_depth': max_depth,
               'min_samples_split': min_samples_split,
               'min_samples_leaf': min_samples_leaf,
               'bootstrap': bootstrap}


# Use the random grid to search for best hyperparameters
# First create the base model to tune
rf = RandomForestClassifier()
# Random search of parameters, using 3 fold cross validation,
# search across 100 different combinations, and use all available cores
rf_random = RandomizedSearchCV(estimator=rf,
                               param_distributions= random_grid,
                               n_iter=100, cv=10,
                               verbose=3, random_state=42,
                               n_jobs=-1)  # Fit the random search model

# predictor matrix
X = np.asarray(train_data)[:, 0:21]
# response vector
y = np.asarray(train_data)[:, 21]

model = rf_random.fit(X, y)
rf_random.best_params_
best_random = rf_random.best_estimator_

param_grid = {
    'bootstrap': [True],
    'max_depth': [60, 70, 80],
    'max_features': [2, 4, 6, 8, 10],
    'min_samples_leaf': [2, 4, 6, 8],
    'min_samples_split': [8, 10, 12],
    'n_estimators': [200, 300, 400]
}
# Create a based model
rf = RandomForestClassifier()  # Instantiate the grid search model
grid_search = GridSearchCV(estimator=rf, param_grid=param_grid,
                           cv=10, n_jobs=-1, verbose=3)

# Fit the grid search to the data
grid_search.fit(X, y)
grid_search.best_params_

# specify model hyper-parameters
clf = RandomForestClassifier(
    max_depth=80,  # maximum depth of 95 decision nodes for each decision tree
    max_features=2,  # maximum of 6 features (channels) are considered when forming a decision node
    min_samples_leaf=8,  # minimum of 6 samples needed to form a final leaf
    min_samples_split=12,  # minimum 4 samples needed to create a decision node
    n_estimators=200,  # maximum of 55 decision trees
    bootstrap=False,  # don't reuse samples
    random_state=1,
    n_jobs=-1
)

# fit random forest
model = clf.fit(X, y)
score = model.score(X, y)

# save model
path = f'{workdir}/Output/Models'
if not Path(path).exists():
    Path(path).mkdir(parents=True, exist_ok=True)
pkl_filename = f'{path}/rf_segmentation.pkl'
with open(pkl_filename, 'wb') as file:
    pickle.dump(model, file)

# ======================================================================================================================
