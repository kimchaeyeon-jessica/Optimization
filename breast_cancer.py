#암에 걸린 사람들이 악성인지 아닌지 판단하는 것임
#암에 걸린 사람의 수치들을 보고 악성인 것 같으면 1, 아닌 것 같으면 0으로 매겨서 평균을 내보고 그것의 정확도가 약 96퍼센트라는 결과가 나왔다.
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.linear_model import LogisticRegression
import seaborn as sns

df = pd.read_csv('breast_cancer_data.csv')
'''sns.lmplot('radius_mean', 'smoothness_mean', data=df, hue='diagnosis', fit_reg=False) #데이터는 df를 쓸거고 앞에 2개를 x축y축으로 쓸 것임, hue는 색깔로 진단결과 0,1을 보여주는 것
plt.show()'''

train = df.sample(frac=0.8, random_state=200)  #데이터의 80퍼센트를 가지고 할 것이다 /random_state은 난수생성을 위한 시드값 설정
test = df.drop(train.index)

#Logistic Regression
logistic = LogisticRegression(solver='newton-cg')
logistic.fit(train[["radius_mean", "texture_mean", "perimeter_mean", "area_mean", "smoothness_mean",
                    "compactness_mean", "concavity_mean", "concave points_mean", "symmetry_mean",
                    "fractal_dimension_mean", "radius_se", "texture_se", "perimeter_se", "area_se", "smoothness_se",
                    "compactness_se", "concavity_se", "concave points_se", "symmetry_se", "fractal_dimension_se",
                    "radius_worst", "texture_worst", "perimeter_worst", "area_worst", "smoothness_worst",
                    "compactness_worst", "concavity_worst", "concave points_worst", "symmetry_worst",
                    "fractal_dimension_worst", ]], train['diagnosis'])
score = logistic.score(test[["radius_mean", "texture_mean", "perimeter_mean", "area_mean",
                             "smoothness_mean", "compactness_mean", "concavity_mean", "concave points_mean",
                             "symmetry_mean", "fractal_dimension_mean", "radius_se", "texture_se", "perimeter_se",
                             "area_se", "smoothness_se", "compactness_se", "concavity_se", "concave points_se",
                             "symmetry_se", "fractal_dimension_se", "radius_worst", "texture_worst", "perimeter_worst",
                             "area_worst", "smoothness_worst", "compactness_worst", "concavity_worst",
                             "concave points_worst", "symmetry_worst", "fractal_dimension_worst", ]], test['diagnosis'])
print(score)

sns.lmplot('radius_mean', 'smoothness_mean', data=df, hue='diagnosis', fit_reg=False) #데이터는 df를 쓸거고 앞에 2개를 x축y축으로 쓸 것임, hue는 색깔로 진단결과 0,1을 보여주는 것
plt.show()
