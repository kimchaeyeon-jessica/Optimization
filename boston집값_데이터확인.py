from sklearn.datasets import load_boston #보스턴 집값에 대한 데이터(정리 짱)
from pandas.plotting import scatter_matrix
import matplotlib.pyplot as plt
import pandas as pd
#집값 예측 회귀분석
boston_data = load_boston() #데이터 불러오는 기능

boston = pd.DataFrame(data=boston_data.data, columns=boston_data.feature_names)
#DataFrame은 pandas에서 데이터를 처리하기 위한 기준
#data는 실제 데이터 columns는 각 칼럼의 이름 리스트
#target은 데이터의 결과 (y값)을 의미
#데이터를 묶어서 보기 위해 boston에 target데이터 추
boston['target'] = boston_data.target
train = boston.sample(frac=0.8, random_state=200)
test=boston.drop(train.index)
#frac =몇퍼센트하겠냐?
#데이터의 80%를 샘플링
#샘플링 된 데이터들을 제외한 데이터를 검증 데이터로 이용
scatter_matrix(boston.drop(columns=["B","LSTAT","ZN","INDUS","NOX","RM","AGE","RAD","TAX"]))
#여기 리스트에 들어있는 것은 빼고 보여줌
plt.show()
