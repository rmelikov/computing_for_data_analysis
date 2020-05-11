# Clustering via $k$-means

We previously studied the classification problem using the logistic regression algorithm. Since we had labels for each data point, we may regard the problem as one of _supervised learning_. However, in many applications, the data have no labels but we wish to discover possible labels (or other hidden patterns or structures). This problem is one of _unsupervised learning_. How can we approach such problems?

**Clustering** is one class of unsupervised learning methods. In this lab, we'll consider the following form of the clustering task. Suppose you are given

- a set of observations, $X \equiv \{\hat{x}_i \,|\, 0 \leq i < n\}$, and
- a target number of _clusters_, $k$.

Your goal is to partition the points into $k$ subsets, $C_0,\dots, C_{k-1} \subseteq X$, which are

- disjoint, i.e., $i \neq j \implies C_i \cap C_j = \emptyset$;
- but also complete, i.e., $C_0 \cup C_1 \cup \cdots \cup C_{k-1} = X$.

Intuitively, each cluster should reflect some "sensible" grouping. Thus, we need to specify what constitutes such a grouping.

## Setup: Dataset

The following cell will download the data you'll need for this lab. Run it now.


```python
import requests
import os
import hashlib
import io

def on_vocareum():
    return os.path.exists('.voc')

def download(file, local_dir="", url_base=None, checksum=None):
    local_file = "{}{}".format(local_dir, file)
    print (local_file)
    if not os.path.exists(local_file):
        if url_base is None:
            url_base = "https://cse6040.gatech.edu/datasets/"
        url = "{}{}".format(url_base, file)
        print("Downloading: {} ...".format(url))
        r = requests.get(url)
        with open(local_file, 'wb') as f:
            f.write(r.content)
            
    if checksum is not None:
        with io.open(local_file, 'rb') as f:
            body = f.read()
            body_checksum = hashlib.md5(body).hexdigest()
            assert body_checksum == checksum, \
                "Downloaded file '{}' has incorrect checksum: '{}' instead of '{}'".format(local_file,
                                                                                           body_checksum,
                                                                                           checksum)
    print("'{}' is ready!".format(file))
    
if on_vocareum():
    URL_BASE = "https://cse6040.gatech.edu/datasets/kmeans/"
    DATA_PATH = "./resource/asnlib/publicdata/"
else:
    URL_BASE = "https://github.com/cse6040/labs-fa17/raw/master/datasets/kmeans/"
    DATA_PATH = ""

datasets = {'logreg_points_train.csv': '9d1e42f49a719da43113678732491c6d',
            'centers_initial_testing.npy': '8884b4af540c1d5119e6e8980da43f04',
            'compute_d2_soln.npy': '980fe348b6cba23cb81ddf703494fb4c',
            'y_test3.npy': 'df322037ea9c523564a5018ea0a70fbf',
            'centers_test3_soln.npy': '0c594b28e512a532a2ef4201535868b5',
            'assign_cluster_labels_S.npy': '37e464f2b79dc1d59f5ec31eaefe4161',
            'assign_cluster_labels_soln.npy': 'fc0e084ac000f30948946d097ed85ebc'}

for filename, checksum in datasets.items():
    download(filename, local_dir=DATA_PATH, url_base=URL_BASE, checksum=checksum)
    
print("\n(All data appears to be ready.)")


```

    logreg_points_train.csv
    'logreg_points_train.csv' is ready!
    centers_initial_testing.npy
    'centers_initial_testing.npy' is ready!
    compute_d2_soln.npy
    'compute_d2_soln.npy' is ready!
    y_test3.npy
    'y_test3.npy' is ready!
    centers_test3_soln.npy
    'centers_test3_soln.npy' is ready!
    assign_cluster_labels_S.npy
    'assign_cluster_labels_S.npy' is ready!
    assign_cluster_labels_soln.npy
    'assign_cluster_labels_soln.npy' is ready!
    
    (All data appears to be ready.)
    


```python
! dir {"."}
```

     Volume in drive D is Data
     Volume Serial Number is 34D5-C3C3
    
     Directory of D:\OneDrive - Erisal\School\CSE 6040\Module 2\Notebook 14
    
    04/13/2020  10:52 PM    <DIR>          .
    04/13/2020  10:52 PM    <DIR>          ..
    04/13/2020  08:52 AM    <DIR>          .ipynb_checkpoints
    04/12/2020  01:24 PM               304 assign_cluster_labels_S.npy
    04/12/2020  01:25 PM               136 assign_cluster_labels_soln.npy
    04/12/2020  01:25 PM               112 centers_initial_testing.npy
    04/12/2020  01:25 PM               144 centers_test3_soln.npy
    04/12/2020  01:25 PM             6,080 compute_d2_soln.npy
    04/12/2020  01:25 PM           766,374 football.bmp
    04/12/2020  01:25 PM             7,695 logreg_points_train.csv
    04/13/2020  10:52 PM           394,283 main.ipynb
    04/12/2020  01:25 PM             3,080 y_test3.npy
                   9 File(s)      1,178,208 bytes
                   3 Dir(s)  949,705,867,264 bytes free
    

## The $k$-means clustering criterion

Here is one way to measure the quality of a set of clusters. For each cluster $C$, consider its center $\mu$ and measure the distance $\|x-\mu\|$ of each observation $x \in C$ to the center. Add these up for all points in the cluster; call this sum is the _within-cluster sum-of-squares (WCSS)_. Then, set as our goal to choose clusters that minimize the total WCSS over _all_ clusters.

More formally, given a clustering $C = \{C_0, C_1, \ldots, C_{k-1}\}$, let

$$
  \mathrm{WCSS}(C) \equiv \sum_{i=0}^{k-1} \sum_{x\in C_i} \|x - \mu_i\|^2,
$$

where $\mu_i$ is the center of $C_i$. This center may be computed simply as the mean of all points in $C_i$, i.e.,

$$
  \mu_i \equiv \dfrac{1}{|C_i|} \sum_{x \in C_i} x.
$$

Then, our objective is to find the "best" clustering, $C_*$, which is the one that has a minimum WCSS.

$$
  C_* = \arg\min_C \mathrm{WCSS}(C).
$$

## The standard $k$-means algorithm (Lloyd's algorithm)

Finding the global optimum is [NP-hard](https://en.wikipedia.org/wiki/NP-hardness), which is computer science mumbo jumbo for "we don't know whether there is an algorithm to calculate the exact answer in fewer steps than exponential in the size of the input." Nevertheless, there is an iterative method, Lloyd’s algorithm, that can quickly converge to a _local_ (as opposed to _global_) minimum. The procedure alternates between two operations: _assignment_ and _update_.

**Step 1: Assignment.** Given a fixed set of $k$ centers, assign each point to the nearest center:

$$
  C_i = \{\hat{x}: \| \hat{x} - \mu_i \| \le \| \hat{x} - \mu_j \|, 1 \le j \le k \}.
$$

**Step 2: Update.** Recompute the $k$ centers ("centroids") by averaging all the data points belonging to each cluster, i.e., taking their mean:

$$
  \mu_i = \dfrac{1}{|C_i|} \sum_{\hat{x} \in C_i} \hat{x}
$$

![Illustration of $k$-means](https://github.com/cse6040/labs-fa17/raw/master/lab14-kmeans/base21-small-transparent.png)

> Figure adapted from: http://stanford.edu/~cpiech/cs221/img/kmeansViz.png

In the code that follows, it will be convenient to use our usual "data matrix" convention, that is, each row of a data matrix $X$ is one of $m$ observations and each column (coordinate) is one of $d$ predictors. However, we will _not_ need a dummy column of ones since we are not fitting a function.

$$
  X
  \equiv \left(\begin{array}{c} \hat{x}_0^T \\ \vdots \\ \hat{x}_{m}^T \end{array}\right)
  = \left(\begin{array}{ccc} x_0 & \cdots & x_{d-1} \end{array}\right).
$$


```python
import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt

%matplotlib inline

import matplotlib as mpl
mpl.rc("savefig", dpi=100) # Adjust for higher-resolution figures
```

We will use the following data set which some of you may have seen previously.


```python
df = pd.read_csv(f'{DATA_PATH}logreg_points_train.csv')
df.head()
```




<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>x_1</th>
      <th>x_2</th>
      <th>label</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>-0.234443</td>
      <td>-1.075960</td>
      <td>1</td>
    </tr>
    <tr>
      <th>1</th>
      <td>0.730359</td>
      <td>-0.918093</td>
      <td>0</td>
    </tr>
    <tr>
      <th>2</th>
      <td>1.432270</td>
      <td>-0.439449</td>
      <td>0</td>
    </tr>
    <tr>
      <th>3</th>
      <td>0.026733</td>
      <td>1.050300</td>
      <td>0</td>
    </tr>
    <tr>
      <th>4</th>
      <td>1.879650</td>
      <td>0.207743</td>
      <td>0</td>
    </tr>
  </tbody>
</table>
</div>




```python
# Helper functions from Logistic Regression Lesson
def make_scatter_plot(df, x="x_1", y="x_2", hue="label",
                      palette={0: "orange", 1: "blue"},
                      size=5,
                      centers=None):
    sns.lmplot(x=x, y=y, hue=hue, data=df, palette=palette,
               fit_reg=False)
    if centers is not None:
        plt.scatter(centers[:,0], centers[:,1],
                    marker=u'*', s=500,
                    c=[palette[0], palette[1]])
    
def mark_matches(a, b, exact=False):
    """
    Given two Numpy arrays of {0, 1} labels, returns a new boolean
    array indicating at which locations the input arrays have the
    same label (i.e., the corresponding entry is True).
    
    This function can consider "inexact" matches. That is, if `exact`
    is False, then the function will assume the {0, 1} labels may be
    regarded as the same up to a swapping of the labels. This feature
    allows
    
      a == [0, 0, 1, 1, 0, 1, 1]
      b == [1, 1, 0, 0, 1, 0, 0]
      
    to be regarded as equal. (That is, use `exact=False` when you
    only care about "relative" labeling.)
    """
    assert a.shape == b.shape
    a_int = a.astype(dtype=int)
    b_int = b.astype(dtype=int)
    all_axes = tuple(range(len(a.shape)))
    assert ((a_int == 0) | (a_int == 1)).all()
    assert ((b_int == 0) | (b_int == 1)).all()
    
    exact_matches = (a_int == b_int)
    if exact:
        return exact_matches

    assert exact == False
    num_exact_matches = np.sum(exact_matches)
    if (2*num_exact_matches) >= np.prod (a.shape):
        return exact_matches
    return exact_matches == False # Invert
    
def count_matches(a, b, exact=False):
    """
    Given two sets of {0, 1} labels, returns the number of mismatches.
    
    This function can consider "inexact" matches. That is, if `exact`
    is False, then the function will assume the {0, 1} labels may be
    regarded as similar up to a swapping of the labels. This feature
    allows
    
      a == [0, 0, 1, 1, 0, 1, 1]
      b == [1, 1, 0, 0, 1, 0, 0]
      
    to be regarded as equal. (That is, use `exact=False` when you
    only care about "relative" labeling.)
    """
    matches = mark_matches(a, b, exact=exact)
    return np.sum(matches)
```


```python
make_scatter_plot(df)
```


![png](output_11_0.png)


Let's extract the data points as a data matrix, `points`, and the labels as a vector, `labels`. Note that the k-means algorithm you will implement should **not** reference `labels` -- that's the solution we will try to predict given only the point coordinates (`points`) and target number of clusters (`k`).


```python
points = df[['x_1', 'x_2']].values
labels = df['label'].values
n, d = points.shape
k = 2
```

Note that the labels should _not_ be used in the $k$-means algorithm. We use them here only as ground truth for later verification.

### How to start? Initializing the $k$ centers

To start the algorithm, you need an initial guess. Let's randomly choose $k$ observations from the data.

**Exercise 1** (2 points). Complete the following function, `init_centers(X, k)`, so that it randomly selects $k$ of the given observations to serve as centers. It should return a Numpy array of size `k`-by-`d`, where `d` is the number of columns of `X`.


```python
def init_centers(X, k):
    """
    Randomly samples k observations from X as centers.
    Returns these centers as a (k x d) numpy array.
    """
    ind = np.arange(X.shape[0])[:k]
    return X[ind, :]
    
    #from numpy.random import choice
    #samples = choice(len(X), size = k, replace = False)
    #return X[samples, :]
```


```python
# Test cell: `init_centers_test`

centers_initial = init_centers(points, k)
print("Initial centers:\n", centers_initial)

assert type(centers_initial) is np.ndarray, "Your function should return a Numpy array instead of a {}".format(type(centers_initial))
assert centers_initial.shape == (k, d), "Returned centers do not have the right shape ({} x {})".format(k, d)
assert (sum(centers_initial[0, :] == points) == [1, 1]).all(), "The centers must come from the input."
assert (sum(centers_initial[1, :] == points) == [1, 1]).all(), "The centers must come from the input."

print("\n(Passed!)")
```

    Initial centers:
     [[-0.234443 -1.07596 ]
     [ 0.730359 -0.918093]]
    
    (Passed!)
    

### Computing the distances

**Exercise 2** (3 points). Implement a function that computes a distance matrix, $S = (s_{ij})$ such that $s_{ij} = d_{ij}^2$ is the _squared_ distance from point $\hat{x}_i$ to center $\mu_j$. It should return a Numpy matrix `S[:m, :k]`.


```python
def compute_d2(X, centers):
    return np.sum((X[:, np.newaxis, :] - centers[np.newaxis, :, :]) ** 2, axis = -1)
    
    #m = len(X)
    #k = len(centers)
    #S = np.empty((m, k))
    #
    #for i in range(m):
    #    S[i, :] = np.linalg.norm(X[i, :] - centers, ord = 2, axis = 1) ** 2
    #return S
```


```python
np.finfo
```




    numpy.finfo




```python
# Test cell: `compute_d2_test`

centers_initial_testing = np.load("{}centers_initial_testing.npy".format(DATA_PATH))
compute_d2_soln = np.load("{}compute_d2_soln.npy".format(DATA_PATH))

S = compute_d2 (points, centers_initial_testing)
assert (np.linalg.norm (S - compute_d2_soln, axis=1) <= (20.0 * np.finfo(float).eps)).all ()

print("\n(Passed!)")
```

    
    (Passed!)
    

**Exercise 3** (2 points). Write a function that uses the (squared) distance matrix to assign a "cluster label" to each point.

That is, consider the $m \times k$ squared distance matrix $S$. For each point $i$, if $s_{i,j}$ is the minimum squared distance for point $i$, then the index $j$ is $i$'s cluster label. In other words, your function should return a (column) vector $y$ of length $m$ such that

$$
  y_i = \underset{j \in \{0, \ldots, k-1\}}{\operatorname{argmin}} s_{ij}.
$$

> Hint: Judicious use of Numpy's [`argmin()`](https://docs.scipy.org/doc/numpy/reference/generated/numpy.argmin.html) makes for a nice one-line solution.


```python
def assign_cluster_labels(S):
    return np.argmin(S, axis = 1)

# Cluster labels:     0    1
S_test1 = np.array([[0.3, 0.2],  # --> cluster 1
                    [0.1, 0.5],  # --> cluster 0
                    [0.4, 0.2]]) # --> cluster 1
y_test1 = assign_cluster_labels(S_test1)
print("You found:", y_test1)

assert (y_test1 == np.array([1, 0, 1])).all()
```

    You found: [1 0 1]
    


```python
# Test cell: `assign_cluster_labels_test`

S_test2 = np.load("{}assign_cluster_labels_S.npy".format(DATA_PATH))
y_test2_soln = np.load("{}assign_cluster_labels_soln.npy".format(DATA_PATH))
y_test2 = assign_cluster_labels(S_test2)
assert (y_test2 == y_test2_soln).all()

print("\n(Passed!)")
```

    
    (Passed!)
    

**Exercise 4** (2 points). Given a clustering (i.e., a set of points and assignment of labels), compute the center of each cluster.


```python
def update_centers(X, y):
    # X[:m, :d] == m points, each of dimension d
    # y[:m] == cluster labels
    m, d = X.shape
    k = max(y) + 1
    assert m == len(y)
    assert (min(y) >= 0)
    
    centers = np.empty((k, d))
    for j in range(k):
        # Compute the new center of cluster j,
        # i.e., centers[j, :d].
        centers[j, :d] = np.mean(X[y == j, :], axis = 0)
    return centers
```


```python
# Test cell: `update_centers_test`

y_test3 = np.load("{}y_test3.npy".format(DATA_PATH))
centers_test3_soln = np.load("{}centers_test3_soln.npy".format(DATA_PATH))
centers_test3 = update_centers(points, y_test3)

delta_test3 = np.abs(centers_test3 - centers_test3_soln)
assert (delta_test3 <= 2.0*len(centers_test3_soln)*np.finfo(float).eps).all()

print("\n(Passed!)")
```

    
    (Passed!)
    

**Exercise 5** (2 points). Given the squared distances, return the within-cluster sum of squares.

In particular, your function should have the signature,

```python
    def WCSS(S):
        ...
```

where `S` is an array of distances as might be computed from Exercise 2.

For example, suppose `S` is defined as follows:

```python
    S = np.array([[0.3, 0.2],
                  [0.1, 0.5],
                  [0.4, 0.2]])
```

Then `WCSS(S) == 0.2 + 0.1 + 0.2 == 0.5.`

> _Hint_: See [numpy.amin](https://docs.scipy.org/doc/numpy/reference/generated/numpy.amin.html#numpy.amin).


```python
def WCSS(S):
    return np.sum(np.amin(S, axis = 1))
    
# Quick test:
print("S ==\n", S_test1)
WCSS_test1 = WCSS(S_test1)
print("\nWCSS(S) ==", WCSS(S_test1))
```

    S ==
     [[0.3 0.2]
     [0.1 0.5]
     [0.4 0.2]]
    
    WCSS(S) == 0.5
    


```python
# Test cell: `WCSS_test`

assert np.abs(WCSS_test1 - 0.5) <= 3.0*np.finfo(float).eps, "WCSS(S_test1) should be close to 0.5, not {}".format(WCSS_test1)
print("\n(Passed!)")
```

    
    (Passed!)
    

Lastly, here is a function to check whether the centers have "moved," given two instances of the center values. It accounts for the fact that the order of centers may have changed.


```python
def has_converged(old_centers, centers):
    return set([tuple(x) for x in old_centers]) == set([tuple(x) for x in centers])
```

**Exercise 6** (3 points). Put all of the preceding building blocks together to implement Lloyd's $k$-means algorithm.


```python
def kmeans(X, k,
           starting_centers=None,
           max_steps=np.inf):
    if starting_centers is None:
        centers = init_centers(X, k)
    else:
        centers = starting_centers
        
    converged = False
    labels = np.zeros(len(X))
    i = 1
    while (not converged) and (i <= max_steps):
        old_centers = centers
        S = compute_d2(X, centers)
        labels = assign_cluster_labels(S)
        centers = update_centers(X, labels)
        converged = has_converged(old_centers, centers)
        print ("iteration", i, "WCSS = ", WCSS (S))
        i += 1
    return labels

clustering = kmeans(points, k, starting_centers=points[[0, 187], :])
```

    iteration 1 WCSS =  549.9175535488309
    iteration 2 WCSS =  339.800663302551
    iteration 3 WCSS =  300.33011292232806
    iteration 4 WCSS =  289.80700777322045
    iteration 5 WCSS =  286.0745591062787
    iteration 6 WCSS =  284.1907705579879
    iteration 7 WCSS =  283.22732249939105
    iteration 8 WCSS =  282.456491302569
    iteration 9 WCSS =  281.84838225337074
    iteration 10 WCSS =  281.57242082723724
    iteration 11 WCSS =  281.5315627987326
    

Let's visualize the results.


```python
# Test cell: `kmeans_test`
print(clustering)
df['clustering'] = clustering
centers = update_centers(points, clustering)
make_scatter_plot(df, hue='clustering', centers=centers)

n_matches = count_matches(df['label'], df['clustering'])
print(n_matches,
      "matches out of",
      len(df), "possible",
      "(~ {:.1f}%)".format(100.0 * n_matches / len(df)))

assert n_matches >= 320
```

    [0 0 1 1 1 1 0 0 0 0 0 1 1 0 1 1 1 1 0 1 1 1 0 1 1 0 1 0 1 0 0 1 1 0 1 1 1
     1 0 0 1 0 0 1 0 0 1 1 1 0 0 1 1 0 1 1 1 0 1 0 0 1 1 1 0 1 0 0 0 0 0 0 0 1
     0 1 0 0 1 0 1 1 1 0 1 0 0 1 1 0 0 1 1 0 1 1 1 0 1 0 1 0 1 0 1 0 0 0 1 0 0
     0 1 0 1 0 0 1 1 1 1 0 0 0 1 0 0 0 0 1 1 1 1 0 0 1 0 1 0 0 0 1 1 0 1 1 0 1
     1 0 0 0 0 1 0 1 0 0 0 0 1 1 0 1 0 0 1 0 0 0 0 0 0 1 1 1 0 0 1 0 0 1 1 0 0
     0 0 1 1 0 0 0 0 0 1 0 0 0 0 0 1 0 1 0 0 0 1 1 1 0 0 1 1 0 1 1 0 0 1 0 0 1
     1 1 1 1 1 0 0 0 0 0 0 1 0 1 1 1 1 1 0 0 1 0 0 1 0 1 1 0 1 0 0 1 1 1 0 0 0
     1 0 0 0 1 1 1 1 0 0 1 0 0 1 1 1 1 1 0 1 0 0 1 1 0 1 0 1 1 0 0 0 0 1 1 1 1
     0 0 0 1 1 1 0 0 1 1 1 0 1 1 1 1 0 1 0 0 1 0 1 1 0 1 0 1 0 1 1 1 1 1 1 1 1
     0 0 0 0 0 0 0 1 1 1 0 0 1 1 1 1 0 1 0 1 0 0 1 1 0 0 0 0 0 0 1 0 0 0 1 1 1
     1 0 1 1 1]
    329 matches out of 375 possible (~ 87.7%)
    


![png](output_36_1.png)


**Applying k-means to an image.** In this section of the notebook, you will apply k-means to an image, for the purpose of doing a "stylized recoloring" of it. (You can view this example as a primitive form of [artistic style transfer](http://genekogan.com/works/style-transfer/), which state-of-the-art methods today [accomplish using neural networks](https://medium.com/artists-and-machine-intelligence/neural-artistic-style-transfer-a-comprehensive-look-f54d8649c199).)

In particlar, let's take an input image and cluster pixels based on the similarity of their colors. Maybe it can become the basis of your own [Instagram filter](https://blog.hubspot.com/marketing/instagram-filters)!


```python
from PIL import Image
from matplotlib.pyplot import imshow
%matplotlib inline

def read_img(path):
    """
    Read image and store it as an array, given the image path. 
    Returns the 3 dimensional image array.
    """
    img = Image.open(path)
    img_arr = np.array(img, dtype='int32')
    img.close()
    return img_arr

def display_image(arr):
    """
    display the image
    input : 3 dimensional array
    """
    arr = arr.astype(dtype='uint8')
    img = Image.fromarray(arr, 'RGB')
    imshow(np.asarray(img))
    

img_arr = read_img("football.bmp")
display_image(img_arr)
print("Shape of the matrix obtained by reading the image")
print(img_arr.shape)
```

    Shape of the matrix obtained by reading the image
    (412, 620, 3)
    


![png](output_38_1.png)


Note that the image is stored as a "3-D" matrix. It is important to understand how matrices help to store a image. Each pixel corresponds to a intensity value for Red, Green and Blue. If you note the properties of the image, its resolution is 620 x 412. The image width is 620 pixels and height is 412 pixels, and each pixel has three values - **R**, **G**, **B**. This makes it a 412 x 620 x 3 matrix.

**Exercise 7** (1 point). Write some code to *reshape* the matrix into "img_reshaped" by transforming "img_arr" from a "3-D" matrix to a flattened "2-D" matrix which has 3 columns corresponding to the RGB values for each pixel. In this form, the flattened matrix must contain all pixels and their corresponding RGB intensity values. Remember in the previous modules we had discussed a C type indexing style and a Fortran type indexing style. In this problem, refer to the C type indexing style. The numpy reshape function may be of help here.


```python
img_reshaped = img_arr.reshape(-1, 3)
```


```python
# Test cell - 'reshape_test'
r, c, l = img_arr.shape
# The reshaped image is a flattened '2-dimensional' matrix
assert len(img_reshaped.shape) == 2
r_reshaped, c_reshaped = img_reshaped.shape
assert r * c * l == r_reshaped * c_reshaped
assert c_reshaped == 3
print("Passed")
```

    Passed
    

**Exercise 8** (1 point). Now use the k-means function that you wrote above to divide the image in **3** clusters. The result would be a vector named labels, which assigns the label to each pixel.


```python
labels = kmeans(img_reshaped, 3)
```

    iteration 1 WCSS =  341319534
    iteration 2 WCSS =  1121891275.269317
    iteration 3 WCSS =  844608813.518326
    iteration 4 WCSS =  706494229.4621991
    iteration 5 WCSS =  660134124.8736633
    iteration 6 WCSS =  644306603.2367755
    iteration 7 WCSS =  638271661.5531305
    iteration 8 WCSS =  635851541.8145165
    iteration 9 WCSS =  634867373.5522717
    iteration 10 WCSS =  634463387.7574959
    iteration 11 WCSS =  634300534.5525875
    iteration 12 WCSS =  634227311.3237957
    iteration 13 WCSS =  634196210.4521514
    iteration 14 WCSS =  634182513.5892371
    iteration 15 WCSS =  634178251.2860513
    iteration 16 WCSS =  634175708.1669554
    iteration 17 WCSS =  634174624.9946772
    iteration 18 WCSS =  634174059.4714676
    iteration 19 WCSS =  634173882.9721487
    iteration 20 WCSS =  634173783.4825951
    iteration 21 WCSS =  634173756.5604436
    iteration 22 WCSS =  634173755.1569637
    


```python
# Test cell - 'labels'
assert len(labels) == r_reshaped
assert set(labels) == {0, 1, 2}
print("\nPassed!")
```

    
    Passed!
    

**Exercise 9** (2 points). Write code to calculate the mean of each cluster and store it in a dictionary, named centers, as label:array(cluster_center). For 3 clusters, the dictionary should have three keys as the labels and their corresponding cluster centers as values, i.e. {0:array(center0), 1: array(center1), 2:array(center2)}.


```python
combined = np.column_stack((img_reshaped, labels))

cluster_ids = combined[:,3]

unique_cluster_ids = np.unique(cluster_ids)

centers = {i : combined[cluster_ids == i].mean(axis = 0)[0:3] for i in unique_cluster_ids}
```


```python
print("Free points here! But you need to implement the above section correctly for you to see what we want you to see later.")
print("\nPassed!")
```

    Free points here! But you need to implement the above section correctly for you to see what we want you to see later.
    
    Passed!
    

Below, we have written code to generate a matrix "img_clustered" of the same dimensions as img_reshaped, where each pixel is replaced by the cluster center to which it belongs.


```python
img_clustered = np.array([centers[i] for i in labels])
```

Let us display the clustered image and see how kmeans works on the image.


```python
r, c, l = img_arr.shape
img_disp = np.reshape(img_clustered, (r, c, l), order="C")
display_image(img_disp)
```


![png](output_52_0.png)


You can visually inspect the original image and the clustered image to get a sense of what kmeans is doing here. You can also try to vary the number of clusters to see how the output image changes

## Built-in $k$-means

The preceding exercises walked you through how to implement $k$-means, but as you might have imagined, there are existing implementations as well! The following shows you how to use Scipy's implementation, which should yield similar results. If you are asked to use $k$-means in a future lab (or exam!), you can use this one.


```python
from scipy.cluster import vq
```


```python
# `distortion` below is the similar to WCSS.
# It is called distortion in the Scipy documentation
# since clustering can be used in compression.
k = 2
centers_vq, distortion_vq = vq.kmeans(points, k)

# vq return the clustering (assignment of group for each point)
# based on the centers obtained by the kmeans function.
# _ here means ignore the second return value
clustering_vq, _ = vq.vq(points, centers_vq)

print("Centers:\n", centers_vq)
print("\nCompare with your method:\n", centers, "\n")
print("Distortion (WCSS):", distortion_vq)

df['clustering_vq'] = clustering_vq
make_scatter_plot(df, hue='clustering_vq', centers=centers_vq)

n_matches_vq = count_matches(df['label'], df['clustering_vq'])
print(n_matches_vq,
      "matches out of",
      len(df), "possible",
      "(~ {:.1f}%)".format(100.0 * n_matches_vq / len(df)))
```

    Centers:
     [[ 0.64980076  0.4667703 ]
     [-0.37382602 -1.18565619]]
    
    Compare with your method:
     {0: array([202.52108949, 198.84707504, 192.62337998]), 1: array([134.23584777, 125.34568766, 101.93988768]), 2: array([38.7890798 , 45.35163548, 48.17869416])} 
    
    Distortion (WCSS): 0.7502431668235036
    329 matches out of 375 possible (~ 87.7%)
    


![png](output_56_1.png)


**Fin!** That marks the end of this notebook. Don't forget to submit it!
