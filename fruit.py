import pandas as pd
import matplotlib.pyplot as plt
from sklearn.neighbors import KNeighborsClassifier
import seaborn as sns

#데이터 불러오기
df=pd.read_csv('fruit_data_with_colors.csv')
sns.lmplot('mass','width','height',data=df, hue='color_score', fit_reg=False)

label_count = len(df["fruit_label"].unique())

scores = []
for k in range(1, label_count + 1):
    knn = KNeighborsClassifier(n_neighbors=k)
    knn.fit(train[['mass','color_score','width','height']],train['fruit_label'])
    score = knn.score(test[['mass','color_score','width','height']],test['fruit_label'])
    scores.append(score)
plt.plot(range(1, label_count + 1),scores)
plt.show
