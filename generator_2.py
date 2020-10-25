import pandas as pd
import matplotlib.pyplot as plt
from sklearn.linear_model import LogisticRegression
import seaborn as sns

df=pd.read_csv('generator.csv')
sns.lmplot('RPM', 'VIBRATION', data=df, hue='STATUS',fit_reg=False)
plt.show()

train=df.sample(frac=0.8,random_state=200)
test=df.drop(train.index)

#트레이닝 코드로 예측해보라 한 후 실제 데이터와 비교를 해본 후 맞으면 1 틀리면 0해서 평균을 내서 구하는 것
logistic=LogisticRegression()
logistic.fit(train[['RPM','VIBRATION']],train['STATUS'])
score=logistic.score(test[['RPM','VIBRATION']],test['STATUS'])

print(score)

#테스트 데이터 사용
guess = pd.DataFrame(columns=['RPM','VIBRATION'])
guess.loc[0] = [600, 600]
print(logistic.predict(guess))