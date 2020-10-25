from sklearn.datasets import fetch_california_housing
from pandas.plotting import scatter_matrix
import matplotlib.pyplot as plt
import pandas as pd
from sklearn.linear_model import LinearRegression

california_data = fetch_california_housing()

california = pd.DataFrame(data=california_data.data, columns=california_data.feature_names)
california['target']=california_data.target

train=california.sample(frac=0.8, random_state=200)
test=california.drop(train.index)

mlr = LinearRegression()
mlr.fit(train[['MedInc','HouseAge','AveRooms','AveBedrms','Population','AveOccup','Latitude','Longitude']],
        train['target'])

sum_difference = 0
for i, row in test.iterrows():
    estimate = row['MedInc']*mlr.coef_[0] + row['HouseAge']*mlr.coef_[1] + row['AveRooms']*mlr.coef_[2] + \
               row['AveBedrms']*mlr.coef_[3] + row['Population']*mlr.coef_[4] + row['AveOccup']*mlr.coef_[5] + \
               row['Latitude']*mlr.coef_[6] + row['Longitude']*mlr.coef_[7] +\ 
               mlr.intercept_

    difference = abs(row['target'] - estimate)
    sum_difference += difference
print(difference)
