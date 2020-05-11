# Part 3: Sparse matrix storage

This part is about sparse matrix storage in Numpy/Scipy.

> **Note:** When you submit this notebook to the autograder, there will be a time-limit of about 180 seconds (3 minutes). If your notebook takes more time than that to run, it's likely you are not writing efficient enough code.

Start by running the following code cell to get some of the key modules you'll need.


```python
import sys
print(sys.version)

from random import sample # Used to generate a random sample

import numpy as np
print(np.__version__)

import pandas as pd
print(pd.__version__)

from IPython.display import display
```

    3.7.4 (tags/v3.7.4:e09359112e, Jul  8 2019, 20:34:20) [MSC v.1916 64 bit (AMD64)]
    1.18.1
    0.25.3
    

## Sample data

For this part, you'll need to download the dataset below. It's a list of pairs of strings. The strings, it turns out, correspond to anonymized Yelp! user IDs; a pair $(a, b)$ exists if user $a$ is friends on Yelp! with user $b$.

**Exercise 0** (ungraded). Verify that you can obtain the dataset and take a peek by running the two code cells that follow.


```python
import requests
import os
import hashlib
import io

def is_vocareum():
    return os.path.exists('.voc')

if is_vocareum():
    local_filename = './resource/asnlib/publicdata/UserEdges-1M.csv'
else:
    local_filename = 'UserEdges-1M.csv'
    url = 'https://cse6040.gatech.edu/datasets/{}'.format(local_filename)
    if os.path.exists(local_filename):
        print("[{}]\n==> '{}' is already available.".format(url, local_filename))
    else:
        print("[{}] Downloading...".format(url))
        r = requests.get(url)
        with open(file, 'w', encoding=r.encoding) as f:
            f.write(r.text)
            
checksum = '4668034bbcd2fa120915ea2d15eafa8d'
with io.open(local_filename, 'r', encoding='utf-8', errors='replace') as f:
    body = f.read()
    body_checksum = hashlib.md5(body.encode('utf-8')).hexdigest()
    assert body_checksum == checksum, \
        "Downloaded file '{}' has incorrect checksum: '{}' instead of '{}'".format(local_filename,
                                                                                   body_checksum,
                                                                                   checksum)
    print("==> Checksum test passes: {}".format(checksum))
    
print("==> '{}' is ready!\n".format(local_filename))
print("(Auxiliary files appear to be ready.)")
```

    [https://cse6040.gatech.edu/datasets/UserEdges-1M.csv]
    ==> 'UserEdges-1M.csv' is already available.
    ==> Checksum test passes: 4668034bbcd2fa120915ea2d15eafa8d
    ==> 'UserEdges-1M.csv' is ready!
    
    (Auxiliary files appear to be ready.)
    


```python
# Peek at the data:
edges_raw = pd.read_csv(local_filename)
display(edges_raw.head ())
print("...\n`edges_raw` has {} entries.".format(len(edges_raw)))
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
      <th>Source</th>
      <th>Target</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>18kPq7GPye-YQ3LyKyAZPw</td>
      <td>rpOyqD_893cqmDAtJLbdog</td>
    </tr>
    <tr>
      <th>1</th>
      <td>18kPq7GPye-YQ3LyKyAZPw</td>
      <td>4U9kSBLuBDU391x6bxU-YA</td>
    </tr>
    <tr>
      <th>2</th>
      <td>18kPq7GPye-YQ3LyKyAZPw</td>
      <td>fHtTaujcyKvXglE33Z5yIw</td>
    </tr>
    <tr>
      <th>3</th>
      <td>18kPq7GPye-YQ3LyKyAZPw</td>
      <td>8J4IIYcqBlFch8T90N923A</td>
    </tr>
    <tr>
      <th>4</th>
      <td>18kPq7GPye-YQ3LyKyAZPw</td>
      <td>wy6l_zUo7SN0qrvNRWgySw</td>
    </tr>
  </tbody>
</table>
</div>


    ...
    `edges_raw` has 1000000 entries.
    

Evidently, this dataframe has one million entries.

**Exercise 1** (ungraded). Explain what the following code cell does.


```python
edges_raw_trans = pd.DataFrame({'Source': edges_raw['Target'],
                                'Target': edges_raw['Source']})
edges_raw_symm = pd.concat([edges_raw, edges_raw_trans])
edges = edges_raw_symm.drop_duplicates()

V_names = set(edges['Source'])
V_names.update(set(edges['Target']))

num_edges = len(edges)
num_verts = len(V_names)
print("==> |V| == {}, |E| == {}".format(num_verts, num_edges))
```

    ==> |V| == 107456, |E| == 882640
    

**Answer.** Give this question some thought before peeking at our suggested answer, which follows.

Recall that the input dataframe, `edges_raw`, has a row $(a, b)$ if $a$ and $b$ are friends. But here is what is unclear at the outset: if $(a, b)$ is an entry in this table, is $(b, a)$ also an entry? The code in the above cell effectively figures that out, by computing a dataframe, `edges`, that contains both $(a, b)$ and $(b, a)$, with no additional duplicates, i.e., no copies of $(a, b)$.

It also uses sets to construct a set, `V_names`, that consists of all the names. Evidently, the dataset consists of 107,456 unique names and 441,320 unique pairs, or 882,640 pairs when you "symmetrize" to ensure that both $(a, b)$ and $(b, a)$ appear.

## Graphs

One way a computer scientist thinks of this collection of pairs is as a _graph_: 
https://en.wikipedia.org/wiki/Graph_(discrete_mathematics%29)

The names or user IDs are _nodes_ or _vertices_ of this graph; the pairs are _edges_, or arrows that connect vertices. That's why the final output objects are named `V_names` (for vertex names) and `edges` (for the vertex-to-vertex relationships). The process or calculation to ensure that both $(a, b)$ and $(b, a)$ are contained in `edges` is sometimes referred to as _symmetrizing_ the graph: it ensures that if an edge $a \rightarrow b$ exists, then so does $b \rightarrow a$. If that's true for all edges, then the graph is _undirected_. The Wikipedia page linked to above explains these terms with some examples and helpful pictures, so take a moment to review that material before moving on.

We'll also refer to this collection of vertices and edges as the _connectivity graph_.

## Sparse matrix storage: Baseline methods

Let's start by reminding ourselves how our previous method for storing sparse matrices, based on nested default dictionaries, works and performs.


```python
def sparse_matrix(base_type=float):
    """Returns a sparse matrix using nested default dictionaries."""
    from collections import defaultdict
    return defaultdict(lambda: defaultdict (base_type))

def dense_vector(init, base_type=float):
    """
    Returns a dense vector, either of a given length
    and initialized to 0 values or using a given list
    of initial values.
    """
    # Case 1: `init` is a list of initial values for the vector entries
    if type(init) is list:
        initial_values = init
        return [base_type(x) for x in initial_values]
    
    # Else, case 2: `init` is a vector length.
    assert type(init) is int
    return [base_type(0)] * init
```

**Exercise 2** (3 points). Implement a function to compute $y \leftarrow A x$. Assume that the keys of the sparse matrix data structure are integers in the interval $[0, s)$ where $s$ is the number of rows or columns as appropriate.

> **Hint**: Recall that you implemented a _dense_ matrix-vector multiply in Part 2. Think about how to adapt that same piece of code when the data structure for storing `A` has changed to the sparse representation given in this exercise.


```python
def spmv(A, x, num_rows=None): 
    if num_rows is None:
        num_rows = max(A.keys()) + 1
    y = dense_vector(num_rows) 
    
    # Recall: y = A*x is, conceptually,
    # for all i, y[i] == sum over all j of (A[i, j] * x[j])
    for i, row_i in A.items():
        s = 0.
        for j, a_ij in row_i.items():
            s += a_ij * x[j]
            y[i] = s
    return y
```


```python
# Test cell: `spmv_baseline_test`

#   / 0.   -2.5   1.2 \   / 1. \   / -1.4 \
#   | 0.1   1.    0.  | * | 2. | = |  2.1 |
#   \ 6.   -1.    0.  /   \ 3. /   \  4.0 /

A = sparse_matrix ()
A[0][1] = -2.5
A[0][2] = 1.2
A[1][0] = 0.1
A[1][1] = 1.
A[2][0] = 6.
A[2][1] = -1.

x = dense_vector ([1, 2, 3])
y0 = dense_vector ([-1.4, 2.1, 4.0])


# Try your code:
y = spmv(A, x)

max_abs_residual = max([abs(a-b) for a, b in zip(y, y0)])

print ("==> A:", A)
print ("==> x:", x)
print ("==> True solution, y0:", y0)
print ("==> Your solution, y:", y)
print ("==> Residual (infinity norm):", max_abs_residual)
assert max_abs_residual <= 1e-14

print ("\n(Passed.)")
```

    ==> A: defaultdict(<function sparse_matrix.<locals>.<lambda> at 0x000001EC7002D948>, {0: defaultdict(<class 'float'>, {1: -2.5, 2: 1.2}), 1: defaultdict(<class 'float'>, {0: 0.1, 1: 1.0}), 2: defaultdict(<class 'float'>, {0: 6.0, 1: -1.0})})
    ==> x: [1.0, 2.0, 3.0]
    ==> True solution, y0: [-1.4, 2.1, 4.0]
    ==> Your solution, y: [-1.4000000000000004, 2.1, 4.0]
    ==> Residual (infinity norm): 4.440892098500626e-16
    
    (Passed.)
    

Next, let's convert the `edges` input into a sparse matrix representing its connectivity graph. To do so, we'll first want to map names to integers.


```python
id2name = {} # id2name[id] == name
name2id = {} # name2id[name] == id

for k, v in enumerate (V_names):
    # for debugging
    if k <= 5: print ("Name %s -> Vertex id %d" % (v, k))
    if k == 6: print ("...")
        
    id2name[k] = v
    name2id[v] = k
```

    Name Ngek8nnEgtekhEruT2LYdg -> Vertex id 0
    Name 26m_MINkAvKTlK1RAdyMnA -> Vertex id 1
    Name 0a8cQ9Iu-x8oOhL2c5nHPQ -> Vertex id 2
    Name mXWHEDnybDLtTjn60PCasw -> Vertex id 3
    Name XtLYmzV9dzVUgekysK4kbA -> Vertex id 4
    Name 9s3VoRIf1POcN5_4BeQ2iw -> Vertex id 5
    ...
    

**Exercise 3** (3 points). Given `id2name` and `name2id` as computed above, convert `edges` into a sparse matrix, `G`, where there is an entry `G[s][t] == 1.0` wherever an edge `(s, t)` exists.

**Note** - This step might take time for the kernel to process as there are 1 million rows


```python
G = sparse_matrix()

for i in range(len(edges)):
    s = edges['Source'].iloc[i]
    t = edges['Target'].iloc[i]
    s_id = name2id[s]
    t_id = name2id[t]
    G[s_id][t_id] = 1.0
```


```python
# Test cell: `edges2spmat1_test`

G_rows_nnz = [len(row_i) for row_i in G.values()]
print ("G has {} vertices and {} edges.".format(len(G.keys()), sum(G_rows_nnz)))

assert len(G.keys()) == num_verts
assert sum(G_rows_nnz) == num_edges

# Check a random sample
for k in sample(range(num_edges), 1000):
    i = name2id[edges['Source'].iloc[k]]
    j = name2id[edges['Target'].iloc[k]]
    assert i in G
    assert j in G[i]
    assert G[i][j] == 1.0

print ("\n(Passed.)")
```

    G has 107456 vertices and 882640 edges.
    
    (Passed.)
    

**Exercise 4** (3 points). In the above, we asked you to construct `G` using integer keys. However, since we are, after all, using default dictionaries, we could also use the vertex _names_ as keys. Construct a new sparse matrix, `H`, which uses the vertex names as keys instead of integers.


```python
H = sparse_matrix()
for i in range(len(edges)):
    s = edges['Source'].iloc[i]
    t = edges['Target'].iloc[i]
    H[s][t] = 1.0
```


```python
# Test cell: `create_H_test`

H_rows_nnz = [len(h) for h in H.values()]
print("`H` has {} vertices and {} edges.".format(len(H.keys()), sum(H_rows_nnz)))

assert len(H.keys()) == num_verts
assert sum(H_rows_nnz) == num_edges

# Check a random sample
for i in sample(G.keys(), 100):
    i_name = id2name[i]
    assert i_name in H
    assert len(G[i]) == len(H[i_name])
    
print ("\n(Passed.)")
```

    `H` has 107456 vertices and 882640 edges.
    
    (Passed.)
    

**Exercise 5** (3 points). Implement a sparse matrix-vector multiply for matrices with named keys. In this case, it will be convenient to have vectors that also have named keys; assume we use dictionaries to hold these vectors as suggested in the code skeleton, below.

> **Hint:** Go back to **Exercise 2** and see what you did there. If it was implemented well, a modest change to the solution for Exercise 2 is likely to be all you need here.


```python
def vector_keyed(keys=None, values=0, base_type=float):
    if keys is not None:
        if type(values) is not list:
            values = [base_type(values)] * len(keys)
        else:
            values = [base_type(v) for v in values]
        x = dict(zip(keys, values))
    else:
        x = {}
    return x

def spmv_keyed(A, x):
    """Performs a sparse matrix-vector multiply for keyed matrices and vectors."""
    assert type(x) is dict
    
    y = vector_keyed(keys=A.keys(), values=0.0)
    for i, A_i in A.items():
        for j, a_ij in A_i.items():
            y[i] += a_ij * x[j]
    return y
```


```python
# Test cell: `spmv_keyed_test`

#   'row':  / 0.   -2.5   1.2 \   / 1. \   / -1.4 \
#  'your':  | 0.1   1.    0.  | * | 2. | = |  2.1 |
#  'boat':  \ 6.   -1.    0.  /   \ 3. /   \  4.0 /

KEYS = ['row', 'your', 'boat']

A_keyed = sparse_matrix ()
A_keyed['row']['your'] = -2.5
A_keyed['row']['boat'] = 1.2
A_keyed['your']['row'] = 0.1
A_keyed['your']['your'] = 1.
A_keyed['boat']['row'] = 6.
A_keyed['boat']['your'] = -1.

x_keyed = vector_keyed (KEYS, [1, 2, 3])
y0_keyed = vector_keyed (KEYS, [-1.4, 2.1, 4.0])


# Try your code:
y_keyed = spmv_keyed (A_keyed, x_keyed)

# Measure the residual:
residuals = [(y_keyed[k] - y0_keyed[k]) for k in KEYS]
max_abs_residual = max ([abs (r) for r in residuals])

print ("==> A_keyed:", A_keyed)
print ("==> x_keyed:", x_keyed)
print ("==> True solution, y0_keyed:", y0_keyed)
print ("==> Your solution:", y_keyed)
print ("==> Residual (infinity norm):", max_abs_residual)
assert max_abs_residual <= 1e-14

print ("\n(Passed.)")
```

    ==> A_keyed: defaultdict(<function sparse_matrix.<locals>.<lambda> at 0x000001EC7002DCA8>, {'row': defaultdict(<class 'float'>, {'your': -2.5, 'boat': 1.2}), 'your': defaultdict(<class 'float'>, {'row': 0.1, 'your': 1.0}), 'boat': defaultdict(<class 'float'>, {'row': 6.0, 'your': -1.0})})
    ==> x_keyed: {'row': 1.0, 'your': 2.0, 'boat': 3.0}
    ==> True solution, y0_keyed: {'row': -1.4, 'your': 2.1, 'boat': 4.0}
    ==> Your solution: {'row': -1.4000000000000004, 'your': 2.1, 'boat': 4.0}
    ==> Residual (infinity norm): 4.440892098500626e-16
    
    (Passed.)
    

Let's benchmark `spmv()` against `spmv_keyed()` on the full data set. Do they perform differently?


```python
x = dense_vector ([1.] * num_verts)
%timeit spmv (G, x)

x_keyed = vector_keyed (keys=[v for v in V_names], values=1.)
%timeit spmv_keyed (H, x_keyed)
```


    ---------------------------------------------------------------------------

    AttributeError                            Traceback (most recent call last)

    <ipython-input-25-d71472cb5b5b> in <module>
          1 x = dense_vector ([1.] * num_verts)
    ----> 2 get_ipython().run_line_magic('timeit', 'spmv (G, x)')
          3 
          4 x_keyed = vector_keyed (keys=[v for v in V_names], values=1.)
          5 get_ipython().run_line_magic('timeit', 'spmv_keyed (H, x_keyed)')
    

    c:\program files\python37\lib\site-packages\IPython\core\interactiveshell.py in run_line_magic(self, magic_name, line, _stack_depth)
       2305                 kwargs['local_ns'] = sys._getframe(stack_depth).f_locals
       2306             with self.builtin_trap:
    -> 2307                 result = fn(*args, **kwargs)
       2308             return result
       2309 
    

    <c:\program files\python37\lib\site-packages\decorator.py:decorator-gen-61> in timeit(self, line, cell, local_ns)
    

    c:\program files\python37\lib\site-packages\IPython\core\magic.py in <lambda>(f, *a, **k)
        185     # but it's overkill for just that one bit of state.
        186     def magic_deco(arg):
    --> 187         call = lambda f, *a, **k: f(*a, **k)
        188 
        189         if callable(arg):
    

    c:\program files\python37\lib\site-packages\IPython\core\magics\execution.py in timeit(self, line, cell, local_ns)
       1134         tc_min = 0.1
       1135 
    -> 1136         t0 = clock()
       1137         code = self.shell.compile(timeit_ast, "<magic-timeit>", "exec")
       1138         tc = clock()-t0
    

    c:\program files\python37\lib\site-packages\IPython\utils\timing.py in clock()
         49         avoids the wraparound problems in time.clock()."""
         50 
    ---> 51         u,s = resource.getrusage(resource.RUSAGE_SELF)[:2]
         52         return u+s
         53 
    

    AttributeError: module 'resource' has no attribute 'getrusage'


## Alternative formats: 

Take a look at the following slides: [link](https://www.dropbox.com/s/4fwq21dy60g4w4u/cse6040-matrix-storage-notes.pdf?dl=0). These slides cover the basics of two list-based sparse matrix formats known as _coordinate format_ (COO) and _compressed sparse row_ (CSR). We will also discuss them briefly below.

### Coordinate Format (COO)
In this format we store three lists, one each for rows, columns and the elements of the matrix. Look at the below picture to understand how these lists are formed.

![Coordinate (COO) storage](https://github.com/cse6040/labs-fa17/raw/master/lab10-numpy/coo.png)

**Exercise 6** (3 points). Convert the `edges[:]` data into a coordinate (COO) data structure in native Python using three lists, `coo_rows[:]`, `coo_cols[:]`, and `coo_vals[:]`, to store the row indices, column indices, and matrix values, respectively. Use integer indices and set all values to 1.

**Hint** - Think of what rows, columns and values mean conceptually when you relate it with our dataset of edges


```python
coo_rows = [name2id[s] for s in edges['Source']]
coo_cols = [name2id[t] for t in edges['Target']]
coo_vals = [1.0]*len(edges)

```


```python
# Test cell: `create_coo_test`

assert len (coo_rows) == num_edges
assert len (coo_cols) == num_edges
assert len (coo_vals) == num_edges
assert all ([v == 1. for v in coo_vals])

# Randomly check a bunch of values
coo_zip = zip (coo_rows, coo_cols, coo_vals)
for i, j, a_ij in sample (list (coo_zip), 1000):
    assert (i in G) and j in G[i]
    
print ("\n(Passed.)")
```

    
    (Passed.)
    

**Exercise 7** (3 points). Implement a sparse matrix-vector multiply routine for COO implementation.


```python
def spmv_coo(R, C, V, x, num_rows=None):
    """
    Returns y = A*x, where A has 'm' rows and is stored in
    COO format by the array triples, (R, C, V).
    """
    assert type(x) is list
    assert type(R) is list
    assert type(C) is list
    assert type(V) is list
    assert len(R) == len(C) == len(V)
    if num_rows is None:
        num_rows = max(R) + 1
    
    y = dense_vector(num_rows)
    
    for i, j, a_ij in zip(R, C, V):
        y[i] += a_ij * x[j]

    
    return y
```


```python
# Test cell: `spmv_coo_test`

#   / 0.   -2.5   1.2 \   / 1. \   / -1.4 \
#   | 0.1   1.    0.  | * | 2. | = |  2.1 |
#   \ 6.   -1.    0.  /   \ 3. /   \  4.0 /

A_coo_rows = [0, 0, 1, 1, 2, 2]
A_coo_cols = [1, 2, 0, 1, 0, 1]
A_coo_vals = [-2.5, 1.2, 0.1, 1., 6., -1.]

x = dense_vector([1, 2, 3])
y0 = dense_vector([-1.4, 2.1, 4.0])

# Try your code:
y_coo = spmv_coo(A_coo_rows, A_coo_cols, A_coo_vals, x)

max_abs_residual = max ([abs(a-b) for a, b in zip(y_coo, y0)])

print("==> A_coo:", list(zip(A_coo_rows, A_coo_cols, A_coo_vals)))
print("==> x:", x)
print("==> True solution, y0:", y0)
print("==> Your solution:", y_coo)
print("==> Residual (infinity norm):", max_abs_residual)
assert max_abs_residual <= 1e-15

print("\n(Passed.)")
```

    ==> A_coo: [(0, 1, -2.5), (0, 2, 1.2), (1, 0, 0.1), (1, 1, 1.0), (2, 0, 6.0), (2, 1, -1.0)]
    ==> x: [1.0, 2.0, 3.0]
    ==> True solution, y0: [-1.4, 2.1, 4.0]
    ==> Your solution: [-1.4000000000000004, 2.1, 4.0]
    ==> Residual (infinity norm): 4.440892098500626e-16
    
    (Passed.)
    

Let's see how fast this is...


```python
x = dense_vector([1.] * num_verts)
%timeit spmv_coo(coo_rows, coo_cols, coo_vals, x)
```

### Compressed Sparse Row Format
This is similar to the COO format excpet that it is much more compact and takes up less storage. Look at the picture below to understand more about this representation

![Compressed sparse row (CSR) format](https://github.com/cse6040/labs-fa17/raw/master/lab10-numpy/csr.png)

**Exercise 8** (3 points). Now create a CSR data structure, again using native Python lists. Name your output CSR lists `csr_ptrs`, `csr_inds`, and `csr_vals`.

It's easiest to start with the COO representation. We've given you some starter code. Unlike most of the exercises, instead of creating a function, you have to compute csr_ptrs here


```python
from operator import itemgetter
C = sorted(zip(coo_rows, coo_cols, coo_vals), key=itemgetter(0))
nnz = len(C)
assert nnz >= 1

assert (C[-1][0] + 1) == num_verts  # Why?

csr_inds = [j for _, j, _ in C]
csr_vals = [a_ij for _, _, a_ij in C]

# Your task: Compute `csr_ptrs`
C_rows = [i for i, _, _ in C] # sorted rows
csr_ptrs = [0] * (num_verts + 1)
i_cur = -1 # a known, invalid row index
for k in range(nnz):
    if C_rows[k] != i_cur:
        i_cur = C_rows[k]
        csr_ptrs[i_cur] = k
from itertools import accumulate
csr_ptrs = list(accumulate(csr_ptrs, max))
csr_ptrs[-1] = nnz


```


```python
# Test cell: `create_csr_test`

assert type(csr_ptrs) is list, "`csr_ptrs` is not a list."
assert type(csr_inds) is list, "`csr_inds` is not a list."
assert type(csr_vals) is list, "`csr_vals` is not a list."

assert len(csr_ptrs) == (num_verts + 1), "`csr_ptrs` has {} values instead of {}".format(len(csr_ptrs), num_verts+1)
assert len(csr_inds) == num_edges, "`csr_inds` has {} values instead of {}".format(len(csr_inds), num_edges)
assert len(csr_vals) == num_edges, "`csr_vals` has {} values instead of {}".format(len(csr_vals), num_edges)
assert csr_ptrs[num_verts] == num_edges, "`csr_ptrs[{}]` == {} instead of {}".format(num_verts, csr_ptrs[num_verts], num_edges)

# Check some random entries
for i in sample(range(num_verts), 10000):
    assert i in G
    a, b = csr_ptrs[i], csr_ptrs[i+1]
    msg_prefix = "Row {} should have these nonzeros: {}".format(i, G[i])
    assert (b-a) == len(G[i]), "{}, which is {} nonzeros; instead, it has just {}.".format(msg_prefix, len(G[i]), b-a)
    assert all([(j in G[i]) for j in csr_inds[a:b]]), "{}. However, it may have missing or incorrect column indices: csr_inds[{}:{}] == {}".format(msg_prefix, a, b, csr_inds[a:b])
    assert all([(j in csr_inds[a:b] for j in G[i].keys())]), "{}. However, it may have missing or incorrect column indices: csr_inds[{}:{}] == {}".format(msg_prefix, a, b, csr_inds[a:b])

print ("\n(Passed.)")
```

    
    (Passed.)
    

**Exercise 9** (3 points). Now implement a CSR-based sparse matrix-vector multiply.


```python
def spmv_csr(ptr, ind, val, x, num_rows=None):
    assert type(ptr) == list
    assert type(ind) == list
    assert type(val) == list
    assert type(x) == list
    if num_rows is None: num_rows = len(ptr) - 1
    assert len(ptr) >= (num_rows+1)  # Why?
    assert len(ind) >= ptr[num_rows]  # Why?
    assert len(val) >= ptr[num_rows]  # Why?
    
    y = dense_vector(num_rows)
    for i in range(num_rows):
        for k in range(ptr[i], ptr[i+1]):
            y[i] += val[k] * x[ind[k]]
    return y
```


```python
# Test cell: `spmv_csr_test`

#   / 0.   -2.5   1.2 \   / 1. \   / -1.4 \
#   | 0.1   1.    0.  | * | 2. | = |  2.1 |
#   \ 6.   -1.    0.  /   \ 3. /   \  4.0 /

A_csr_ptrs = [ 0,        2,       4,       6]
A_csr_cols = [ 1,   2,   0,   1,  0,   1]
A_csr_vals = [-2.5, 1.2, 0.1, 1., 6., -1.]

x = dense_vector([1, 2, 3])
y0 = dense_vector([-1.4, 2.1, 4.0])

# Try your code:
y_csr = spmv_csr(A_csr_ptrs, A_csr_cols, A_csr_vals, x)

max_abs_residual = max([abs(a-b) for a, b in zip(y_csr, y0)])

print ("==> A_csr_ptrs:", A_csr_ptrs)
print ("==> A_csr_{cols, vals}:", list(zip(A_csr_cols, A_csr_vals)))
print ("==> x:", x)
print ("==> True solution, y0:", y0)
print ("==> Your solution:", y_csr)
print ("==> Residual (infinity norm):", max_abs_residual)
assert max_abs_residual <= 1e-14

print ("\n(Passed.)")
```

    ==> A_csr_ptrs: [0, 2, 4, 6]
    ==> A_csr_{cols, vals}: [(1, -2.5), (2, 1.2), (0, 0.1), (1, 1.0), (0, 6.0), (1, -1.0)]
    ==> x: [1.0, 2.0, 3.0]
    ==> True solution, y0: [-1.4, 2.1, 4.0]
    ==> Your solution: [-1.4000000000000004, 2.1, 4.0]
    ==> Residual (infinity norm): 4.440892098500626e-16
    
    (Passed.)
    


```python
x = dense_vector([1.] * num_verts)
%timeit spmv_csr(csr_ptrs, csr_inds, csr_vals, x)
```


    ---------------------------------------------------------------------------

    AttributeError                            Traceback (most recent call last)

    <ipython-input-34-37da58f15562> in <module>
          1 x = dense_vector([1.] * num_verts)
    ----> 2 get_ipython().run_line_magic('timeit', 'spmv_csr(csr_ptrs, csr_inds, csr_vals, x)')
    

    c:\program files\python37\lib\site-packages\IPython\core\interactiveshell.py in run_line_magic(self, magic_name, line, _stack_depth)
       2305                 kwargs['local_ns'] = sys._getframe(stack_depth).f_locals
       2306             with self.builtin_trap:
    -> 2307                 result = fn(*args, **kwargs)
       2308             return result
       2309 
    

    <c:\program files\python37\lib\site-packages\decorator.py:decorator-gen-61> in timeit(self, line, cell, local_ns)
    

    c:\program files\python37\lib\site-packages\IPython\core\magic.py in <lambda>(f, *a, **k)
        185     # but it's overkill for just that one bit of state.
        186     def magic_deco(arg):
    --> 187         call = lambda f, *a, **k: f(*a, **k)
        188 
        189         if callable(arg):
    

    c:\program files\python37\lib\site-packages\IPython\core\magics\execution.py in timeit(self, line, cell, local_ns)
       1134         tc_min = 0.1
       1135 
    -> 1136         t0 = clock()
       1137         code = self.shell.compile(timeit_ast, "<magic-timeit>", "exec")
       1138         tc = clock()-t0
    

    c:\program files\python37\lib\site-packages\IPython\utils\timing.py in clock()
         49         avoids the wraparound problems in time.clock()."""
         50 
    ---> 51         u,s = resource.getrusage(resource.RUSAGE_SELF)[:2]
         52         return u+s
         53 
    

    AttributeError: module 'resource' has no attribute 'getrusage'


## Using Scipy's implementations

What you should have noticed is that the list-based COO and CSR formats do not really lead to sparse matrix-vector multiply implementations that are much faster than the dictionary-based methods. Let's instead try Scipy's native COO and CSR implementations.


```python
import numpy as np
import scipy.sparse as sp

A_coo_sp = sp.coo_matrix((coo_vals, (coo_rows, coo_cols)))
A_csr_sp = A_coo_sp.tocsr() # Alternatively: sp.csr_matrix((val, ind, ptr))
x_sp = np.ones(num_verts)

print ("\n==> COO in Scipy:")
%timeit A_coo_sp.dot (x_sp)

print ("\n==> CSR in Scipy:")
%timeit A_csr_sp.dot (x_sp)
```

    
    ==> COO in Scipy:
    


    ---------------------------------------------------------------------------

    AttributeError                            Traceback (most recent call last)

    <ipython-input-35-ba49db574884> in <module>
          7 
          8 print ("\n==> COO in Scipy:")
    ----> 9 get_ipython().run_line_magic('timeit', 'A_coo_sp.dot (x_sp)')
         10 
         11 print ("\n==> CSR in Scipy:")
    

    c:\program files\python37\lib\site-packages\IPython\core\interactiveshell.py in run_line_magic(self, magic_name, line, _stack_depth)
       2305                 kwargs['local_ns'] = sys._getframe(stack_depth).f_locals
       2306             with self.builtin_trap:
    -> 2307                 result = fn(*args, **kwargs)
       2308             return result
       2309 
    

    <c:\program files\python37\lib\site-packages\decorator.py:decorator-gen-61> in timeit(self, line, cell, local_ns)
    

    c:\program files\python37\lib\site-packages\IPython\core\magic.py in <lambda>(f, *a, **k)
        185     # but it's overkill for just that one bit of state.
        186     def magic_deco(arg):
    --> 187         call = lambda f, *a, **k: f(*a, **k)
        188 
        189         if callable(arg):
    

    c:\program files\python37\lib\site-packages\IPython\core\magics\execution.py in timeit(self, line, cell, local_ns)
       1134         tc_min = 0.1
       1135 
    -> 1136         t0 = clock()
       1137         code = self.shell.compile(timeit_ast, "<magic-timeit>", "exec")
       1138         tc = clock()-t0
    

    c:\program files\python37\lib\site-packages\IPython\utils\timing.py in clock()
         49         avoids the wraparound problems in time.clock()."""
         50 
    ---> 51         u,s = resource.getrusage(resource.RUSAGE_SELF)[:2]
         52         return u+s
         53 
    

    AttributeError: module 'resource' has no attribute 'getrusage'


**Fin!** If your notebook runs to this point without error, try submitting it.
