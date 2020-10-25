import pandas as pd
import matplotlib.pyplot as plt
from sklearn.neighbors import KNeighborsClassifier
import seaborn as sns

df = pd.read_csv('sharing_bike_train_data.csv')

#df에서 ""와 같은 값들을 쏙쏙 뺴오겠다.라는 뜻 (치환)
#컴퓨터가 읽을 수 있는 날짜로 되어 있어서?? 가져온 것
df["year"] = pd.to_datetime(df["datetime"]).dt.year
df["month"] = pd.to_datetime(df["datetime"]).dt.year
df["day"] = pd.to_datetime(df["datetime"]).dt.year
df["hour"] = pd.to_datetime(df["datetime"]).dt.year



#이제 데이터를 보고 어떤 속성을 사용할지, 어떤 분석 방법을 사용할지, 이를 결정하고 분석 실행해야함
#season,holiday,workingday,weather과 count의 관계를 가장 쓸모있는 데이터들이라고 판단함
#knn, kmeans, linear regression, logistic regression 중에 골라서 해야함
'''k_means 어떤거 의미하는지 모르는데 뭉쳐있음
knn 찾고싶은 카테고리가 정해져있는 경우
linear regression 어떠한 값이 어떻게 바뀌냐에 따라서
                      일정하게 바뀔 거다!
logistic regression 선택지가 두개일 때'''





