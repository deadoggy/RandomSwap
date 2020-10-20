# Random Swap v.s. KMeans


## Experiment on quality

- Dataset: Yelp3k, size = [1000, 600, 600, 800] 
- Algorithms: RandomSwap and KMeans
- K: 2 - 14
- Index: NMI, ARI, MSE
- Setup: For each k, run RandomSwap and KMeans for 40 times separately


### Box

![](box_nmi.png)

![](box_ari.png)

![](box_mse.png)

### Mean Index

![](mean_nmi.png)

![](mean_ari.png)

![](mean_mse.png)


## Experiment on efficiency

- Dataset: Yelp3k, size = [1000, 600, 600, 800]
- K: 2 - 14
- Index: second
- Setup: For each k, run RandomSwap and KMeans for 20 times separately and calcuate the mean time of each run

![](mean_time.png)
