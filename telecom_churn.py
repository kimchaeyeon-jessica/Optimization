#One Hot Encoding 사용
#2개->logistic 회귀
#2개 이상 ->knn
#개수,횟수, 가격 등 ->linear regression
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.preprocessing import LabelEncoder  # Label Encoding 사용 (근데 label encoding보다는 one hot encoding이 훨씬 나음)
from sklearn.linear_model import LogisticRegression
import seaborn as sns

df = pd.read_csv('telecom_churn_data.csv')
df.dropna(inplace=True) #없는 값이 있는 행 제거하기 == 값이 없는 행 제거하기

le = LabelEncoder()
df['gender'] = le.fit_transform(df['gender'])  # 하나씩 "치환"해주기!!
df['SeniorCitizen'] = le.fit_transform(df['SeniorCitizen'])
df['Partner'] = le.fit_transform(df['Partner'])
df['Dependents'] = le.fit_transform(df['Dependents'])
df['tenure'] = le.fit_transform(df['tenure'])
df['PhoneService'] = le.fit_transform(df['PhoneService'])
df['MultipleLines'] = le.fit_transform(df['MultipleLines'])
df['InternetService'] = le.fit_transform(df['InternetService'])
df['OnlineSecurity'] = le.fit_transform(df['OnlineSecurity'])
df['OnlineBackup'] = le.fit_transform(df['OnlineBackup'])
df['DeviceProtection'] = le.fit_transform(df['DeviceProtection'])
df['TechSupport'] = le.fit_transform(df['TechSupport'])
df['StreamingTV'] = le.fit_transform(df['StreamingTV'])
df['StreamingMovies'] = le.fit_transform(df['StreamingMovies'])
df['Contract'] = le.fit_transform(df['Contract'])
df['PaperlessBilling'] = le.fit_transform(df['PaperlessBilling'])
df['PaymentMethod'] = le.fit_transform(df['PaymentMethod'])
df['MonthlyCharges'] = le.fit_transform(df['MonthlyCharges'])
df['TotalCharges'] = le.fit_transform(df['TotalCharges'])

sns.pairplot(data=df[['gender', 'SeniorCitizen', 'Partner', 'Dependents', 'tenure', 'PhoneService', 'MultipleLines', 'InternetService',
     'OnlineSecurity', 'OnlineBackup', 'DeviceProtection', 'TechSupport', 'StreamingTV', 'StreamingMovies', 'Contract',
     'PaperlessBilling', 'PaymentMethod', 'MonthlyCharges', 'TotalCharges','Churn']], hue='Churn') #ctrl+alt+l로 예쁘게 정렬 가능!
plt.show()

'''train = df.sample(frac=0.8, random_state=200)
test = df.drop(train.index)''' #_test데이터로 테스트 할거기 때문에 없애준다.


#logistic 회귀분석 진행
logistic=LogisticRegression(solver="newton-cg") #학습시키기
logistic.fit(df[['gender', 'SeniorCitizen', 'Partner', 'Dependents', 'tenure', 'PhoneService', 'MultipleLines', 'InternetService',
     'OnlineSecurity', 'OnlineBackup', 'DeviceProtection', 'TechSupport', 'StreamingTV', 'StreamingMovies', 'Contract',
     'PaperlessBilling', 'PaymentMethod', 'MonthlyCharges', 'TotalCharges']], df['Churn']) #train지웠으니까 그 자리에 df써주기(test할 때) **test를 하는 경우에는 아예 test와 df를 분리해주는 경우가 많음
test = pd.read_csv('telecom_churn_data.csv')
test.dropna(inplace=True) #없는 값이 있는 행 제거하기 == 값이 없는 행 제거하기

le = LabelEncoder()
test['gender'] = le.fit_transform(df['gender'])  # 하나씩 "치환"해주기!!
test['SeniorCitizen'] = le.fit_transform(df['SeniorCitizen'])
test['Partner'] = le.fit_transform(df['Partner'])
test['Dependents'] = le.fit_transform(df['Dependents'])
test['tenure'] = le.fit_transform(df['tenure'])
test['PhoneService'] = le.fit_transform(df['PhoneService'])
test['MultipleLines'] = le.fit_transform(df['MultipleLines'])
test['InternetService'] = le.fit_transform(df['InternetService'])
test['OnlineSecurity'] = le.fit_transform(df['OnlineSecurity'])
test['OnlineBackup'] = le.fit_transform(df['OnlineBackup'])
test['DeviceProtection'] = le.fit_transform(df['DeviceProtection'])
test['TechSupport'] = le.fit_transform(df['TechSupport'])
test['StreamingTV'] = le.fit_transform(df['StreamingTV'])
test['StreamingMovies'] = le.fit_transform(df['StreamingMovies'])
test['Contract'] = le.fit_transform(df['Contract'])
test['PaperlessBilling'] = le.fit_transform(df['PaperlessBilling'])
test['PaymentMethod'] = le.fit_transform(df['PaymentMethod'])
test['MonthlyCharges'] = le.fit_transform(df['MonthlyCharges'])
test['TotalCharges'] = le.fit_transform(df['TotalCharges'])

score = logistic.score(test[['gender', 'SeniorCitizen', 'Partner', 'Dependents', 'tenure', 'PhoneService', 'MultipleLines', 'InternetService',
     'OnlineSecurity', 'OnlineBackup', 'DeviceProtection', 'TechSupport', 'StreamingTV', 'StreamingMovies', 'Contract',
     'PaperlessBilling', 'PaymentMethod', 'MonthlyCharges', 'TotalCharges','Churn']], test['Churn'])
print(score)

'''sns.pairplot(data=[['gender', 'SeniorCitizen', 'Partner', 'Dependents', 'tenure', 'PhoneService', 'MultipleLines', 'InternetService',
     'OnlineSecurity', 'OnlineBackup', 'DeviceProtection', 'TechSupport', 'StreamingTV', 'StreamingMovies', 'Contract',
     'PaperlessBilling', 'PaymentMethod', 'MonthlyCharges', 'TotalCharges','Churn']], hue='Churn')
plt.show()'''
