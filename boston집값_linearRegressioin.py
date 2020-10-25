from sklearn.datasets import load_boston
import pandas as pd
from sklearn.linear_model import LinearRegression

boston_data = load_boston()

boston = pd.DataFrame(data=boston_data.data, columns=boston_data.feature_names)
boston['target'] = boston_data.target

train = boston.sample(frac=0.8, random_state=200)
test = boston.drop(train.index)

mlr = LinearRegression()
mlr.fit(train[['PTRATIO','INDUS','NOX','B','CHAS','RAD','TAX','ZN','DIS','CRIM','RM','LSTAT','AGE']],
        train['target'])
#print(mlr.intercept_)
#print(mlr.coef_)

sum_difference = 0
for i, row in test.iterrows():
    estimate = row['PTRATIO']*mlr.coef_[0] + row['INDUS']*mlr.coef_[1] + row['NOX']*mlr.coef_[2] + \
               row['B']*mlr.coef_[3] + row['CHAS']*mlr.coef_[4] + row['RAD']*mlr.coef_[5] + \
               row['TAX']*mlr.coef_[6] + row['ZN']*mlr.coef_[7] + row['DIS']*mlr.coef_[8] + \
               row['CRIM']*mlr.coef_[9] + row['RM']*mlr.coef_[10] + row['LSTAT']*mlr.coef_[11] + \
               row['AGE']*mlr.coef_[12] + mlr.intercept_

    difference = abs(row['target'] - estimate)
    sum_difference += difference
print(difference)
