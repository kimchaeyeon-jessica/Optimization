import pandas as pd
import matplotlib.pyplot as plt
from sklearn.neighbors import KNeighborsClassifier
import seaborn as sns

#데이터 불러오기
df=pd.read_csv('seoul_data.csv')
label_count=len(df['name'].unique())

#데이터 시각적으로 확인
sns.lmplot('lat','lon',data=df, hue='name', fit_reg=False)
#plt.show()

#테스트 데이터 분리
train=df.sample(frac=0.8, random_state=200)
test=df.drop(train.index)

#knn구하기
knn=KNeighborsClassifier(n_neighbors=label_count)
knn.fit(train[['lat','lon']],train['name'])
score=knn.score(test[['lat','lon']],test['name'])
print(score)

#레이블 예측하기
guess=pd.DataFrame(columns=['lat','lon'])
guess.loc[0]=[37.520040,127.110136]
print(knn.predict(guess))