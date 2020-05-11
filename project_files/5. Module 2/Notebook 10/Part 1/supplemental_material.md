# Part 1: Supplemental Background on Numpy

This notebook is a quick overview of additional functionality in Numpy. It is intended to supplement the videos and the other parts of this assignment. It does **not** contain any exercises that you need to submit.


```python
import sys
print(sys.version)

import numpy as np
print(np.__version__)
```

    3.7.4 (tags/v3.7.4:e09359112e, Jul  8 2019, 20:34:20) [MSC v.1916 64 bit (AMD64)]
    1.18.1
    

# Random numbers

Numpy has a rich collection of (pseudo) random number generators. Here is an example; 
see the documentation for [numpy.random()](https://docs.scipy.org/doc/numpy/reference/routines.random.html) for more details.


```python
A = np.random.randint(-10, 10, size=(4, 3)) # return random integers from -10 (inclusive) to 10 (exclusive)
print(A)

```

    [[ -4   4   4]
     [ -3   9   3]
     [ -4   3  -6]
     [-10   4  -9]]
    


```python
np.mean(A.T, axis=1)
```




    array([-5.25,  5.  , -2.  ])




```python
A.T
```




    array([[ -4,  -3,  -4, -10],
           [  4,   9,   3,   4],
           [  4,   3,  -6,  -9]])




```python
np.mean([  3,  -8,   6,   7])
```




    2.0



# Aggregations or reductions

Suppose you want to reduce the values of a Numpy array to a smaller number of values. Numpy provides a number of such functions that _aggregate_ values. Examples of aggregations include sums, min/max calculations, and averaging, among others.


```python
print("np.max =", np.max(A),"; np.amax =", np.amax(A)) # np.max() and np.amax() are synonyms
print("np.min =",np.min(A),"; np.amin =", np.amin(A)) # same
print("np.sum =",np.sum(A))
print("np.mean =",np.mean(A))
print("np.std =",np.std(A))
```

    np.max = 9 ; np.amax = 9
    np.min = -10 ; np.amin = -10
    np.sum = -9
    np.mean = -0.75
    np.std = 5.7608593109014565
    

The above examples aggregate over all values. But you can also aggregate along a dimension using the optional `axis` parameter.


```python
print("Max in each column:", np.amax(A, axis=0)) # i.e., aggregate along axis 0, the rows, producing column maxes
print("Max in each row:", np.amax(A, axis=1)) # i.e., aggregate along axis 1, the columns, producing row maxes
```

    Max in each column: [-3  9  4]
    Max in each row: [4 9 3 4]
    

# Universal functions

Universal functions apply a given function _elementwise_ to one or more Numpy objects.

For instance, `np.abs(A)` takes the absolute value of each element.


```python
print(A, "\n==>\n", np.abs(A))
```

    [[ -4   4   4]
     [ -3   9   3]
     [ -4   3  -6]
     [-10   4  -9]] 
    ==>
     [[ 4  4  4]
     [ 3  9  3]
     [ 4  3  6]
     [10  4  9]]
    

Some universal functions accept multiple, compatible arguments. For instance, here, we compute the _elementwise maximum_ between two matrices, $A$ and $B$, producing a new matrix $C$ such that $c_{ij} = \max(a_{ij}, b_{ij})$.

> The matrices must have compatible shapes, which we will elaborate on below when we discuss Numpy's _broadcasting rule_.


```python
print(A) # recall A
```

    [[ -4   4   4]
     [ -3   9   3]
     [ -4   3  -6]
     [-10   4  -9]]
    


```python
B = np.random.randint(-10, 10, size=A.shape)
print(B)
```

    [[ 2  3 -5]
     [-9 -4  3]
     [ 5 -2 -4]
     [ 9 -7 -9]]
    


```python
C = np.maximum(A, B) # elementwise comparison
print(C)
```

    [[ 2  4  4]
     [-3  9  3]
     [ 5  3 -4]
     [ 9  4 -9]]
    

You can also build your own universal functions! For instance, suppose we want a way to compute, elementwise, $f(x) = e^{-x^2}$ and we have a scalar function that can do so:


```python
def f(x):
    from math import exp
    return exp(-(x**2))
```

This function accepts one input (`x`) and returns a single output. The following will create a new Numpy universal function `f_np`.
See the documentation for [np.frompyfunc()](https://docs.scipy.org/doc/numpy/reference/generated/numpy.frompyfunc.html) for more details.


```python
f_np = np.frompyfunc(f, 1, 1)  

print(A, "\n=>\n", f_np(A))
```

    [[ -4   4   4]
     [ -3   9   3]
     [ -4   3  -6]
     [-10   4  -9]] 
    =>
     [[1.1253517471925912e-07 1.1253517471925912e-07 1.1253517471925912e-07]
     [0.00012340980408667956 6.639677199580735e-36 0.00012340980408667956]
     [1.1253517471925912e-07 0.00012340980408667956 2.3195228302435696e-16]
     [3.720075976020836e-44 1.1253517471925912e-07 6.639677199580735e-36]]
    

# Broadcasting

Sometimes we want to combine operations on Numpy arrays that have different shapes but are _compatible_.

In the following example, we want to add 3 elementwise to every value in `A`.


```python
print(A)
print()
print(A + 3)
```

    [[ -4   4   4]
     [ -3   9   3]
     [ -4   3  -6]
     [-10   4  -9]]
    
    [[-1  7  7]
     [ 0 12  6]
     [-1  6 -3]
     [-7  7 -6]]
    

Technically, `A` and `3` have different shapes: the former is a $4 \times 3$ matrix, while the latter is a scalar ($1 \times 1$). However, they are compatible because Numpy knows how to _extend_---or **broadcast**---the value 3 into an equivalent matrix object of the same shape in order to combine them.

To see a more sophisticated example, suppose each row `A[i, :]` are the coordinates of a data point, and we want to compute the centroid of all the data points (or center-of-mass, if we imagine each point is a unit mass). That's the same as computing the mean coordinate for each column:


```python
A_row_means = np.mean(A, axis=0)

print(A, "\n=>\n", A_row_means)
```

    [[ -4   4   4]
     [ -3   9   3]
     [ -4   3  -6]
     [-10   4  -9]] 
    =>
     [-5.25  5.   -2.  ]
    

Now, suppose you want to shift the points so that their mean is zero. Even though they don't have the same shape, Numpy will interpret `A - A_row_means` as precisely this operation, effectively extending or "replicating" `A_row_means` into rows of a matrix of the same shape as `A`, in order to then perform elementwise subtraction.


```python
A_row_centered = A - A_row_means
A_row_centered
```




    array([[ 1.25, -1.  ,  6.  ],
           [ 2.25,  4.  ,  5.  ],
           [ 1.25, -2.  , -4.  ],
           [-4.75, -1.  , -7.  ]])



Suppose you instead want to mean-center the _columns_ instead of the rows. You could start by computing column means:


```python
A_col_means = np.mean(A, axis=1)

print(A, "\n=>\n", A_col_means)
```

    [[ -4   4   4]
     [ -3   9   3]
     [ -4   3  -6]
     [-10   4  -9]] 
    =>
     [ 1.33333333  3.         -2.33333333 -5.        ]
    

But the same operation will fail!


```python
A - A_col_means # Fails!
```


    ---------------------------------------------------------------------------

    ValueError                                Traceback (most recent call last)

    <ipython-input-18-d3357eda1460> in <module>
    ----> 1 A - A_col_means # Fails!
    

    ValueError: operands could not be broadcast together with shapes (4,3) (4,) 


The error reports that these shapes are not compatible. So how can you fix it?

**Broadcasting rule.** One way is to learn Numpy's convention for **[broadcasting](https://docs.scipy.org/doc/numpy/reference/ufuncs.html#broadcasting)**. Numpy starts by looking at the shapes of the objects:


```python
print(A.shape, A_row_means.shape)
```

    (4, 3) (3,)
    

These are compatible if, starting from _right_ to _left_, the dimensions match **or** one of the dimensions is 1. This convention of moving from right to left is referred to as matching the _trailing dimensions_. In this example, the rightmost dimensions of each object are both 3, so they match. Since `A_row_means` has no more dimensions, it can be replicated to match the remaining dimensions of `A`.

By contrast, consider the shapes of `A` and `A_col_means`:


```python
print(A.shape, A_col_means.shape)
```

    (4, 3) (4,)
    

In this case, per the broadcasting rule, the trailing dimensions of 3 and 4 do not match. Therefore, the broadcast rule fails. To make it work, we need to modify `A_col_means` to have a unit trailing dimension. Use Numpy's [`reshape()`](https://docs.scipy.org/doc/numpy/reference/generated/numpy.reshape.html) to convert `A_col_means` into a shape that has an explicit trailing dimension of size 1.


```python
A_col_means2 = np.reshape(A_col_means, (len(A_col_means), 1))
print(A_col_means2, "=>", A_col_means2.shape)
```

    [[ 1.33333333]
     [ 3.        ]
     [-2.33333333]
     [-5.        ]] => (4, 1)
    

Now the trailing dimension equals 1, so it can be matched against the trailing dimension of `A`. The next dimension is the same between the two objects, so Numpy knows it can replicate accordingly.


```python
print("A - A_col_means2\n\n", A, "\n-", A_col_means2) 
print("\n=>\n", A - A_col_means2)
```

    A - A_col_means2
    
     [[ -4   4   4]
     [ -3   9   3]
     [ -4   3  -6]
     [-10   4  -9]] 
    - [[ 1.33333333]
     [ 3.        ]
     [-2.33333333]
     [-5.        ]]
    
    =>
     [[-5.33333333  2.66666667  2.66666667]
     [-6.          6.          0.        ]
     [-1.66666667  5.33333333 -3.66666667]
     [-5.          9.         -4.        ]]
    

**Thought exercise.** As a thought exercise, you might see if you can generalize and apply the broadcasting rule to a 3-way array.

**Fin!** That marks the end of this notebook. If you want to learn more, checkout the second edition of [Python for Data Analysis](http://shop.oreilly.com/product/0636920050896.do) (released in October 2017).


```python
pass # Dummy cell
```
