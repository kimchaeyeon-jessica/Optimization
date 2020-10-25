from sklearn.datasets import load_boston
import pandas as pd
import statsmodels.formula.api as sm

boston_data = load_boston()

boston = pd.DataFrame(data=boston_data.data, columns=boston_data.feature_names)
boston['target'] = boston_data.target

train = boston.sample(frac=0.8, random_state=200)
test = boston.drop(train.index)

result = sm.gls(formula= 'target ~ CRIM + ZN +CHAS + NOX + RM + DIS + RAD + TAX + PTRATIO + B + LSTAT',
                data=train).fit()

print(result.summary())

#오차의 합들이 가장 작아지는 것이 무엇인지 찾아 최적의 변숫값을 찾는것
for i, row in test.iterrows():
    params = result.params
    r_estimate = row['PTRATIO']*params['PTRATIO'] + row['NOX']*params['NOX'] + row['B']*params['B'] + \
                 row['CHAS']*params['CHAS'] + row['RAD']*params['RAD'] + row['TAX']*params['TAX'] + row['ZN']*params['ZN'] + \
                 row['DIS']*params['DIS'] + row['CRIM']*params['CRIM'] + row['RM']*params['RM'] + \
                 row['LSTAT']*params['LSTAT'] + params['Intercept']
    difference = abs(row['target'] - estimate)
    sum_difference += difference
print(difference)
