import numpy as np
from random_swap import RandomSwapClustering
import time 
import json

class KMeans(RandomSwapClustering):

    def fit(self):
        
        self._select_random_rep()
        self._optimal_partition()

        for t in range(self.T):
            self._kmeans(1)
        


        

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


rs_time = []
for k in range(2, 15):
    t = 0.
    for t in range(10):
        s = time.time()
        rs = RandomSwapClustering(X, k, 100)
        rs.fit()
        e = time.time()
        t += e-s
    rs_time.append(t/10.) 

km_time = []
for k in range(2, 15):
    t = 0.
    for t in range(10):
        s = time.time()
        km = KMeans(X, k, 300) # cause 100 * 3 in RandomSwap
        km.fit()
        e = time.time()
        t += e-s
    km_time.append(t/10.)




with open('out/time.json', 'w') as out:
    json.dump({'rs': rs_time, 'km': km_time}, out)




