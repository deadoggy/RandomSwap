import numpy as np 
from sklearn.cluster import KMeans 
from random_swap import RandomSwapClustering
from nltk.cluster import KMeansClusterer
from sklearn.metrics import normalized_mutual_info_score as nmi 
from sklearn.metrics import adjusted_mutual_info_score as ari
import sys 


k = int(sys.argv[1])

l1_dist = lambda x,y: np.sum(np.abs(x-y))

def load_db(fn):
    '''
        load database
    '''

    X = []
    with open(fn) as fin:
        lines = fin.readlines()

    X = [list(map(lambda x:eval(x), l.strip().split(' '))) for l in lines]

    return np.array(X)

fn = '/home/yinjia/encryption_clustering/data/yelp3000'

X = load_db(fn)
label = [ 1 ] * 1000 + [ 2 ] * 600 + [ 3 ] * 600 + [ 4 ] * 800

rs = RandomSwapClustering(X, k, 100, metric=l1_dist, km_t=3)
C,P = rs.fit()

rs_nmi = nmi(label, P)
rs_ari = ari(label, P)
rs_mse = rs.loss

print('rs_nmi: %f'%rs_nmi)
print('rs_ari: %f'%rs_ari)
print('rs_mse: %f'%rs_mse)


km = KMeansClusterer(k, l1_dist, avoid_empty_clusters=True)
km_P = km.cluster(X, True)
km_C = km._means

km_nmi = nmi(label, km_P)
km_ari = ari(label, km_P)
km_mse = rs._mean_squared_error(km_C, km_P, X)

print('km_nmi: %f'%km_nmi)
print('km_ari: %f'%km_ari)
print('km_mse: %f'%km_mse)

