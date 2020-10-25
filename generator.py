#원자력 발전소
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.neighbors import KNeighborsClassifier
import seaborn as sns

#데이터 시각적으로 확인하기
df = pd.read_csv('generator_data.csv')
sns.lmplot('RPM','VIBRATION',data=df, hue='STATUS', fit_reg=False)
#plt.show

#테스트 데이터 분리
train=df.sample(frac=0.8, random_state=200)
test=df.drop(train.index)

#knn구하기
knn=KNeighborsClassifier(n_neighbors=2) #내 옆에 있는 2개를 보고 정상상태인지 비정상상태인지 확인
knn.fit(train[['RPM','VIBRATION']], train['STATUS']) #fit 학습시켜라
score=knn.score(test[['RPM','VIBRATION']], test['STATUS']) #score test데이터랑 얼마나 맞는지 채점해봐라 ->결과 1.0==다 맞음 (train과 test값이)

#레이블 예측하기
guess=pd.DataFrame(columns=['RPM','VIBRATION'])
guess.loc[0]=[790,550] #800,200등등 시 faulty뜸 #예측을 위해 임의의 값을 사용함
print(knn.predict(guess))

print(score)
