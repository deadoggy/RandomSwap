import numpy as np 
import sklearn.datasets as db 
from random_swap import RandomSwapClustering
from sklearn.metrics import normalized_mutual_info_score as nmi 
from sklearn.metrics import adjusted_mutual_info_score as ari 
from sklearn.cluster import KMeans

iris_db = db.load_iris()

X = iris_db['data']
label = iris_db['target']

rs = RandomSwapClustering(X, 3, 100, km_t=2)
C, P = rs.fit()

nmi_idx = nmi(label, P)
ari_idx = ari(label, P)

print('rs nmi: %f'%nmi_idx)
print('rs ari: %f'%ari_idx)

km = KMeans(n_clusters=3, max_iter=200, algorithm='full')
km_P = km.fit_predict(X)

km_nmi_idx = nmi(label, km_P)
km_ari_idx = ari(label, km_P)

print('km nmi: %f'%km_nmi_idx)
print('km ari: %f'%km_ari_idx)


