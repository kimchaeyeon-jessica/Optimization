import pandas as pd
import numpy as np
from sklearn.cluster import KMeans
import matplotlib.pyplot as plt
from scipy.spatial.distance import cdist

df = pd.read_csv('cluster_sample_1.csv')
km = KMeans(n_clusters=3).fit(df[['x','y']])


distortions = []
for cluster in range(1,20):
    km = KMeans(n_clusters=cluster).fit(df[['x','y']])

    #중심점과 모든 좌표들간의 거리 (N:M)
    distance = cdist(df[['x','y']],km.cluster_centers_, 'euclidean')

    #중심점-좌표간 거리 중 최저인 값
    min_distance = np.min(distance, axis=1)
    sum_distance = sum(min_distance)

    #평균 최소거리의 합
    distortions.append(sum_distance/df[['x','y']].shape[0])

#차트로 출력하여 확인
plt.plot(range(1,20),distortions)
plt.show()