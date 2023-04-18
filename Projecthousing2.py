# -*- coding: utf-8 -*-
"""
Created on Tue Apr 18 05:54:34 2023

@author: KMarg
"""

import pandas as pd
import numpy as np
from sklearn.impute import SimpleImputer
from sklearn.model_selection import train_test_split
from sklearn.decomposition import PCA


filepath = "C:/Users/KMarg/Downloads/assignment_rev2.csv"

data = pd.read_csv(filepath)

# Drop columns which contain only NaN values
data.dropna(how='all', axis=1, inplace=True)

# Drop the "agent_id" and "id" columns in order to drop duplicates
data.drop(["agent_id", "id"], axis=1, inplace=True)

# Drop duplicates
data.drop_duplicates(inplace=True)

# Separate the target variable from the features
target = data["price"]
data.drop("price", axis=1, inplace=True)

# Create a SimpleImputer object with the "most_frequent" strategy
imputer = SimpleImputer(strategy='most_frequent')

# Fit and transform the imputer on all columns with missing values
data = pd.DataFrame(imputer.fit_transform(data), columns=data.columns)

data.columns

#We seperate the numerical with the categorical values. The decisions of which variables
#to keep was made based on criteria like clolumns with too many missing values were not chosen, columns
#with different types were dropped in order to avoid too many extra clomuns out of one hot encoding.
#Columns that seem that may become heavily imputed again where not chosen.
#We also had a view of the data from power bi to see which columns seemed problematic
numcols = ['ranking_score',  'sq_meters', 'year_of_construction',
         'rooms']
       

# Get the string columns of the dataframe
#Again some columns were left out
cols_to_encode = ['geography_name','floor',  
       'subtype','ad_type',         
       'lux_home', 'new_development', 'garage', 'furnished',
       'preserved', 'investment', 'unfinished', 'heating_under_floor',
       'renovated', 'requires_renovation', 'airy', 'bright', 'painted',
       'garden', 'ac', 'storage', 'balcony', 'internal_stairs', 'view',
       'secure_door', 'penthouse', 'alarm', 'corner',
       'accessible_for_disabled', 'holiday_home', 'student_home',
       'night_power', 'neoclassic', 'double_glass', 'for_commercial_use',
       'fireplace', 'elevator', 'satellite', 'solar_heater', 'attic', 'pool',
       'playroom']

# Now we will one hot encode the categorical columns
data_encoded = pd.get_dummies(data[cols_to_encode], prefix=cols_to_encode)

# And we will combine again all together
data_processed = pd.concat([data[numcols], data_encoded], axis=1)

# we will perform split of the data
X_train, X_test, y_train, y_test = train_test_split(data_processed, target, test_size=0.2, random_state=42)

# Initialize PCA with 10 components
pca = PCA(n_components=8)

# Fit PCA on standardized training data
pca.fit(X_train)

# Get the explained variance ratio for each component
explained_variance = pca.explained_variance_ratio_

# Print the explained variance for each component
for i in range(len(explained_variance)):
    print(f"Explained variance for component {i+1}: {explained_variance[i]:.4f}")

# Get the index of the components sorted in descending order of explained variance
sorted_components = np.argsort(explained_variance)[::-1]

# Print the most important features for each component
for i in sorted_components:
    print(f"Component {i+1}:")
    component_loadings = pca.components_[i]
    sorted_loadings = np.argsort(abs(component_loadings))[::-1]
    for j in sorted_loadings[:10]:
        print(f"{data_processed.columns[j]}: {component_loadings[j]:.3f}")
    print()

# Transform the training and testing data using the trained PCA
X_train_pca = pca.transform(X_train)
X_test_pca = pca.transform(X_test)

#It is important here to stress out that the random forest model was not the only model that we ran
#Actually we ran a gradientboost regressor, a decision tree regressor, a neural net regressor, a linear regression
#a lasso regression, a polynomial regression and a ridge regression and a SVM.
#The Random Forest regressor outperformed all the rest.
#The interesting thing is that the Random forest without pca had explained variance of 0.73
#With the application of pca the model loses explanatory power. We performed the pca
#because it was asked and we wanted to gain an insight of the explained variance and the important features

from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_squared_error, r2_score, explained_variance_score, max_error, mean_absolute_error, median_absolute_error


# Initialize the Random Forest model with 100 trees and a random state of 42
rf_model = RandomForestRegressor(n_estimators=100, random_state=42)

# Train the Random Forest model on the training data
rf_model.fit(X_train_pca, y_train)

# Predict the target variable on the test data using the trained Random Forest model
y_pred_rf = rf_model.predict(X_test_pca)

# Calculate and print the evaluation metrics for the Random Forest model
mse_rf = mean_squared_error(y_test, y_pred_rf)
r2_rf = r2_score(y_test, y_pred_rf)
evs_rf = explained_variance_score(y_test, y_pred_rf)
me_rf = max_error(y_test, y_pred_rf)
mae_rf = mean_absolute_error(y_test, y_pred_rf)
medae_rf = median_absolute_error(y_test, y_pred_rf)
print("Random Forest Model Evaluation Metrics:")
print(f"Mean Squared Error: {mse_rf:.2f}")
print(f"R^2 score: {r2_rf:.2f}")
print(f"Explained Variance Score: {evs_rf:.2f}")
print(f"Max Error: {me_rf:.2f}")
print(f"Mean Absolute Error: {mae_rf:.2f}")
print(f"Median Absolute Error: {medae_rf:.2f}\n")


#Although most of the variance is concertrated in the 2 first components, the 8 components
#provide the best variance explanation in the random forest, it is possible that patterns that were not 
#captured in the first two components were captured in some of the rest and provided these results.
#More investigation is needed. The explained variance is less than in the non pca version but still it outperformed he rest of the models
#The fact that the target variable is in 1000s of euros plays a role in big mse, but still
#mean abosulte error is high and median error is 69535. This may indicate that there is still an uncaptured factor
#or that some other method like rfe is need. More investigation is needed

#Here we provide the random forest without PCA. The results are better.
#Seems like random forest is a good choice for a model but more research and more optimization is needed.
#This is a project under development and we will continue to work in this direction till the model 
#performs better.

# Initialize the Random Forest model with 100 trees and a random state of 42
rf_model = RandomForestRegressor(n_estimators=100, random_state=42)

# Train the Random Forest model on the training data
rf_model.fit(X_train, y_train)

# Predict the target variable on the test data using the trained Random Forest model
y_pred_rf = rf_model.predict(X_test)

# Calculate and print the evaluation metrics for the Random Forest model
mse_rf = mean_squared_error(y_test, y_pred_rf)
r2_rf = r2_score(y_test, y_pred_rf)
evs_rf = explained_variance_score(y_test, y_pred_rf)
me_rf = max_error(y_test, y_pred_rf)
mae_rf = mean_absolute_error(y_test, y_pred_rf)
medae_rf = median_absolute_error(y_test, y_pred_rf)
print("Random Forest Model Evaluation Metrics:")
print(f"Mean Squared Error: {mse_rf:.2f}")
print(f"R^2 score: {r2_rf:.2f}")
print(f"Explained Variance Score: {evs_rf:.2f}")
print(f"Max Error: {me_rf:.2f}")
print(f"Mean Absolute Error: {mae_rf:.2f}")
print(f"Median Absolute Error: {medae_rf:.2f}\n")

