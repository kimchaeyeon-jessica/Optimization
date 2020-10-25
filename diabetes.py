import pandas as pd
import matplotlib.pyplot as plt
from sklearn.linear_model import LogisticRegression
import seaborn as sns

df = pd.read_csv('diabetes_data.csv', names=['a', 'b', 'c', 'd', 'e', 'f', 'g', 'h', 'result']) #names는 이름이 없어서 붙여주려고 만드는거!!

#데이터를 구하는게 어렵기 때문에 주어진 sample data를 최대한 활용
#학습/테스트 데이터 분리
train = df.sample(frac=0.8, random_state=200)
test = df.drop(train.index)

#logistic 회귀분석 진행
logistic=LogisticRegression(solver="newton-cg") #학습시키기
logistic.fit(train[['a', 'b', 'c', 'd', 'e', 'f', 'g', 'h']], train['result'])
score = logistic.score(test[['a', 'b', 'c', 'd', 'e', 'f', 'g', 'h']],test['result'])
print(score)

sns.pairplot(data=df[['a', 'b', 'c', 'd', 'e', 'f', 'g', 'h','result']], hue='result')
plt.show()