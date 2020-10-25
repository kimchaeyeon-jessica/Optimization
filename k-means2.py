import pandas as pd
import numpy as np
from sklearn.cluster import KMeans
import matplotlib.pyplot as plt
from scipy.spatial.distance import cdist

data = [[7,1],[2,1],[4,2],[9,4],[10,5],[10,6],[11,5],[11,6],[15,3],[15,2],[16,4],[16,1]]
df = pd.DataFrame(columns=['x','y'], data=data)

distortions = []
#1~10개를 가지고 테스팅 (최적인 것을 찾기 위해서 범위를 정해주는 것)
for cluster in range(1,10):
    km = KMeans(n_clusters=cluster).fit(df[['x','y']])

    #중심점과 모든 좌표들간의 거리 (N:M)
    distance = cdist(df[['x','y']],km.cluster_centers_, 'euclidean')
    #유클리디언 디스턴스로 계산 ,cluster_centers는 값의 중심점(가장 평균),
    #모든 점과의 거리에 대해서 나옴

    #중심점-좌표간 거리 중 최저인 값
    min_distance = np.min(distance, axis=1)
    #열두개의 점과 중심점과의 거리가 나옴, 각 점별로 중심점과의 거리가 최저인 것이 나옴
    #그리고 거리가 제일 짧은 점들의 합을 구한 것이 아래 코드
    sum_distance = sum(min_distance)

    #평균 최소거리의 합
    distortions.append(sum_distance/df[['x','y']].shape[0])

#차트로 출력하여 확인
plt.plot(range(1,10),distortions)
plt.show()
#그래프에서 꺾이는 부분이 최적
