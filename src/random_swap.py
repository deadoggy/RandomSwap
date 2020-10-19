import numpy as np
from sklearn.cluster import KMeans
from random import randrange


ZERO_THRESHOLD = 1E-8


class RandomSwapClustering:

    def __init__(self, X, k, T, metric=lambda x,y: np.sqrt((x-y).dot(x-y)), km_t=3):
        '''
            Initialization func of RandomSwap

            argv:
                @X:
                    np.ndarray, shape=(N, D), data
                @k:
                    int, number of clusters
                @T:
                    int, number of iterations
                @metric:
                    callable, the distance calcuating function
                @km_t:
                    int, times of iterations in KMeans
        '''

        self.X = np.array(X)
        self.k = k 
        self.T = T 
        self.metric = metric
        self.N = self.X.shape[0]
        self.D = self.X.shape[1]
        self.C = np.zeros((k, self.D))
        self.P = np.array([-1] * self.N)
        self.km_t = km_t

    def _find_nearest_rep(self, x):
        '''
            find the nearest centroid using self.C

            argv:
                @x:
                    np.ndarray, shape=(self.D,)

            return:
                the index of the nearest centroid            
        '''
        j = 0

        for k in range(1, self.k):
            if self.metric(x, self.C[k]) < self.metric(x, self.C[j]):
                j = k
        
        return j

    def _optimal_partition(self):
        '''
            relocate the data objects to new centroids
        '''
        for i in range(0, self.N):
            self.P[i] = self._find_nearest_rep(self.X[i])
        
    def _optimal_rep(self):
        '''
            recalcuate the representation of a cluster
        '''

        s = np.zeros((self.k, self.D))
        cnt = np.array([0.]*self.k)

        for i in range(self.N):
            j = self.P[i]
            s[j] = s[j] + self.X[i]
            cnt[j] = cnt[j] + 1
        
        for i in range(self.k):
            if cnt[i] != 0:
                self.C[i] = s[i] / cnt[i]

    def _kmeans(self,t):
        '''
            run @t iterations of kmeans

            argv:
                @t:
                    int, times of iteration
        '''

        for itr_t in range(t):
            self._optimal_rep()
            self._optimal_partition()
    
    def _select_random_obj(self, k_lmt):
        '''
            select a data obj randomly and make sure that the selected one
            is not one of the range(C[0]:C[k_lmt])

            argv:
                @k_lmt:
                    int, the range of centroids to be checked
        '''

        while True:
            i = randrange(0, self.N)
            flag = True 

            for j in range(k_lmt):
                if self.metric(self.X[i], self.C[j]) < ZERO_THRESHOLD:
                    flag = False 
            
            if flag:
                break
        
        return i
    
    def _select_random_rep(self):
        '''
            pick centroids randomly
        '''

        for i in range(self.k):
            x_i = self._select_random_obj(i)
            self.C[i] = self.X[x_i]

    def _random_swap(self):
        '''
            change an random centroid 
        '''

        j = randrange(0, self.k)
        x_i = self._select_random_obj(self.k)
        self.C[j] = self.X[x_i]
        return j

    def _local_repartition(self, j):
        '''
            update the partition after random swap

            argv:
                @j: the index of centroid which is replaced
        '''

        for i in range(self.N):
            if self.P[i] == j:
                self.P[i] = self._find_nearest_rep(self.X[i])
        
        for i in range(self.N):
            if self.metric(self.X[i],self.C[j]) < self.metric(self.X[i], self.C[self.P[i]]):
                self.P[i] = j
    
    def _mean_squared_error(self, C, P):
        sum = 0. 
        for i in range(self.N):
            sum = sum + self.metric(self.X[i], C[P[i]]) ** 2
        
        return sum
        

    def fit(self):

        # Initialization 
        self._select_random_rep()
        self._optimal_partition()

        for t in range(self.T):

            old_C = np.copy(self.C)
            old_P = np.copy(self.P)
            j = self._random_swap()
            self._local_repartition(j)
            self._kmeans(self.km_t)

            if self._mean_squared_error(self.C, self.P) > self._mean_squared_error(old_C, old_P):
                self.C = old_C
                self.P = old_P

        return self.C, self.P





