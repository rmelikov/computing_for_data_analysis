# Problem 0

Winter is coming (or rather, is here). So let's do some warm-up exercises to get you in the right mindset for the rest of the exam.

This problem should be **easy**. Its purpose is to assess whether you can read and understand the Python documentation of some (likely unfamiliar) functionality and apply it.

This problem has three (3) exercises (numbered 0-2) and is worth a total of ten (10) points.

## Setup

This problem depends on the following modules. Make sure these are available on your system before beginning.


```python
import pandas as pd

from IPython.display import display

import seaborn as sns
%matplotlib inline

from scipy.cluster.vq import kmeans2

# Utility function for later on
def make_scatter_plot (df, x="x_1", y="x_2", hue="label",
                       palette={0: "red", 1: "olive", 2: "blue", 3: "green"},
                       size=5,
                       centers=None):
    from seaborn import lmplot
    from matplotlib.pyplot import scatter
    if (hue is not None) and (hue in df.columns):
        lmplot (x=x, y=y, hue=hue, data=df, palette=palette,
                fit_reg=False)
    else:
        lmplot (x=x, y=y, data=df, fit_reg=False)

    if centers is not None:
        scatter (centers[:,0], centers[:,1],
                 marker=u'*', s=500,
                 c=[palette[0], palette[1]])
```

## Download and unpack a compressed file

**Exercise 0** (4 points). Write a function that

- downloads a compressed gzip (`xxx.gz`) file from the interwebs, given its URL; and
- returns a file-like handle to it.

In particular, find and read the documentation for [`urllib.request.urlopen()`](https://docs.python.org/3/library/urllib.request.html) and [`gzip.open()`](https://docs.python.org/3/library/gzip.html), and use them to implement the desired function.

> Note: `gzip.open()` accepts an optional argument named `mode`, which specifies how to interpret the contents of the data when decompressing it. The `url_open_gz()` function you implement should simply pass its `mode` argument onto `gzip.open()`.

The test code will check your implementation and uses it to download a dataset, which you will need for the rest of this problem.


```python
def open_url_gz (url_gz, mode='rt'):
    """
    Given a URL to a compressed gzip (.gz) file, downloads it to a
    temporary location, unpacks it, and returns a file-like handle
    for reading its uncompressed contents.
    """
    print ("Downloading", url_gz, "...")
    
    from urllib.request import urlopen

    from gzip import open

    downloaded_file = urlopen(url_gz)

    decompressed_file = open(downloaded_file, mode = mode)

    return decompressed_file
```


```python
# Test code for Part 1-A

# First, check a small message.
url_msg_gz = 'https://cse6040.gatech.edu/datasets/message_in_a_bottle.txt.gz'
with open_url_gz (url_msg_gz) as fp_msg:
    line = fp_msg.readline ()
    msg = line.strip ()
    print ("\nDownloaded the message: '%s'" % msg)
    
assert msg == 'Good luck, kiddos!'
print ("\n(Passed check 1 of 2!)\n")

# Next, use this function to download a dataset.
url_data_gz = 'http://cse6040.gatech.edu/datasets/faithful.dat.gz'
with open_url_gz (url_data_gz) as fp:
    fp_local = open ('faithful.dat', 'wt')
    fp_local.write (fp.read ())
    fp_local.close ()
with open ('faithful.dat', 'rt') as fp_faithful:
    assert fp_faithful.readline () == 'Old Faithful Geyser Data\n'

print ("\n(Passed check 2 of 2!)")
```

    Downloading https://cse6040.gatech.edu/datasets/message_in_a_bottle.txt.gz ...
    
    Downloaded the message: 'Good luck, kiddos!'
    
    (Passed check 1 of 2!)
    
    Downloading http://cse6040.gatech.edu/datasets/faithful.dat.gz ...
    
    (Passed check 2 of 2!)
    

## Load the "Old Faithful" dataset

The test code above, assuming you implemented `open_url_gz()` correctly, should have downloaded and unpacked a file in the local directory called `faithful.dat`.

If you did not manage to get a working implementation that does that, then manually download a copy of this file now from here: http://www.stat.cmu.edu/~larry/all-of-statistics/=data/faithful.dat

(In either case, you should probably click on the above URL to see what is in the file, as you'll be working with this data.)

This particular dataset comes from a study of the [Old Faithful geyser](https://en.wikipedia.org/wiki/Old_Faithful) in Yellowstone National Park. Amazingly, this geyser erupts very regularly, hence the name! The dataset contains a bunch of observations, where each observation consists of a) the duration of an eruption (in minutes), and b) the time until the next eruption (again in minutes).

**Exercise 1** (2 points). Use the Pandas function, [`pd.read_table()`](http://pandas.pydata.org/pandas-docs/version/0.18.1/generated/pandas.read_table.html), to read this dataset from `faithful.dat` and store it in a data frame with two columns, one named `eruptions` and one named `waiting`.

> Hint: There is a one-line solution, which requires only that you set the right arguments to `pd.read_table()`.



```python
faithful = pd.read_table(
    'faithful.dat',
    names = ['eruptions', 'waiting'],
    sep = '\s+',
    skiprows = 26
)

# Quickly inspect this data
display (faithful.head ())
print ("...")
display (faithful.tail ())
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
      <th>eruptions</th>
      <th>waiting</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>1</th>
      <td>3.600</td>
      <td>79</td>
    </tr>
    <tr>
      <th>2</th>
      <td>1.800</td>
      <td>54</td>
    </tr>
    <tr>
      <th>3</th>
      <td>3.333</td>
      <td>74</td>
    </tr>
    <tr>
      <th>4</th>
      <td>2.283</td>
      <td>62</td>
    </tr>
    <tr>
      <th>5</th>
      <td>4.533</td>
      <td>85</td>
    </tr>
  </tbody>
</table>
</div>


    ...
    


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
      <th>eruptions</th>
      <th>waiting</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>268</th>
      <td>4.117</td>
      <td>81</td>
    </tr>
    <tr>
      <th>269</th>
      <td>2.150</td>
      <td>46</td>
    </tr>
    <tr>
      <th>270</th>
      <td>4.417</td>
      <td>90</td>
    </tr>
    <tr>
      <th>271</th>
      <td>1.817</td>
      <td>46</td>
    </tr>
    <tr>
      <th>272</th>
      <td>4.467</td>
      <td>74</td>
    </tr>
  </tbody>
</table>
</div>



```python
# Test code

make_scatter_plot (faithful, x='eruptions', y='waiting', hue=None)

# Sanity check a few values
assert set (faithful.columns) == set (['eruptions', 'waiting'])
assert sum (faithful['waiting'] == 54) == 9
assert sum (faithful['waiting'] == 90) == 6
print ("\n(Passed!)")
```

    
    (Passed!)
    


![png](output_9_1.png)


## Cluster this data

**Exercise 2** (2 points). The plot of the data suggests that there is a relationship between how long an eruption lasts and the time between eruptions. Use Scipy's [`kmeans2()`](https://docs.scipy.org/doc/scipy-0.14.0/reference/generated/scipy.cluster.vq.kmeans2.html) to estimate this cluster structure, specifically by doing the following.

- Assign each point of `faithful` an integer cluster label that is either 0 or 1. Store these labels in a new column of the data frame named `label`.
- Compute the centers of the two clusters, storing them in a $2 \times 2$ Numpy array named `centers`, where each row `centers[i, :]` is a center whose columns contain its coordinates.


```python
centers, labels = kmeans2(faithful, k = 2)
faithful['label'] = labels
num_clusters = 2
class_sizes = [sum (labels==c) for c in range (num_clusters)]
print ("\n=== Clusters ===")
for c in range (num_clusters):
    print ("- Cluster {}: {} points centered at {}".format (c,
                                                            class_sizes[c],
                                                            centers[c, :2]))

make_scatter_plot (faithful, x='eruptions', y='waiting', centers=centers)
```

    
    === Clusters ===
    - Cluster 0: 172 points centered at [ 4.29793023 80.28488372]
    - Cluster 1: 100 points centered at [ 2.09433 54.75   ]
    


![png](output_11_1.png)



```python
assert 90 <= min (class_sizes) <= max (class_sizes) <= 180
print ("\n(Passed!)")
```

    
    (Passed!)
    

**Fin!** If you've reached this point and all tests above pass, you are ready to submit your solution to this problem. Don't forget to save you work prior to submitting.
