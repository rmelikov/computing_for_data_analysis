# Problem 0: Two algorithms to calculate sample variance

This problem is related to floating-point arithmetic and the _sample variance_, a commonly used measure in statistics. However, the problem should go quickly -- so, if you find yourself spending a lot of time on it, you may be overthinking it or consider returning to it later.

There are two exercises, numbered 0 and 1, which are worth a total of ten (10) points.

## Setup

Python has a built-in function, [`statistics.variance`](https://docs.python.org/3.5/library/statistics.html#statistics.variance), that computes the sample variance. However, for this problem we want you to implement it from scratch in two different ways and compare their accuracy. (The test codes will use Python's function as a baseline for comparison against your implementations.)


```python
# Run this cell.
from statistics import variance

SAVE_VARIANCE = variance # Ignore me
```

## A baseline algorithm to compute the sample variance

Suppose we observe $n$ samples from a much larger population. Denote these observations by $x_0, x_1, \ldots, x_{n-1}$. Then the _sample mean_ (sample average), $\bar{x}$, is defined to be

$$
  \bar{x} \equiv \frac{1}{n} \sum_{i=0}^{n-1} x_i.
$$

Given both the samples and the sample mean, a standard formula for the (unbiased) _sample variance_, $\bar{s}^2$, is

$$
  \bar{s}^2 \equiv \frac{1}{n-1} \sum_{i=0}^{n-1} (x_i - \bar{x})^2.
$$

**Exercise 0** (5 points). Write a function, `var_method_0(x)`, that implements this formula for the sample variance given a list `x[:]` of observed sample values.

> Remember **not** to use Python's built-in `variance()`.


```python
def var_method_0(x):
    n = len(x) # Number of samples
    x_bar = sum(x) / n
    squared_errors = [(x_i - x_bar)**2 for x_i in x]
    return sum(squared_errors) / (n - 1)
```


```python
# Test cell: `exercise_0_test`

from random import gauss

n = 100000
mu = 1e7  # True mean
sigma = 100.0  # True variance

for _ in range(5): # 5 trials
    X = [gauss(mu, sigma) for _ in range(n)]
    var_py = variance(X)
    try:
        del variance
        var_you_0 = var_method_0(X)
    except NameError as n:
        if n.args[0] == "name 'variance' is not defined":
            assert False, "Did you try to use `variance()` instead of implementing it from scratch?"
        else:
            raise n
    finally:
        variance = SAVE_VARIANCE
        
    rel_diff = abs(var_you_0 - var_py) / var_py
    print("\nData: n={} samples from a Gaussian with mean {} and standard deviation {}".format(n, mu, sigma))
    print("\tPython's variance function computed {}.".format(var_py))
    print("\tYour var_method_0(X) computed {}.".format(var_you_0))
    print("\tThe relative difference is |you - python| / |python| ~= {}.".format(rel_diff))
    assert rel_diff <= n*(2.0**(-52)), "Relative difference is larger than expected..."
    
print("\n(Passed!)")
```

    
    Data: n=100000 samples from a Gaussian with mean 10000000.0 and standard deviation 100.0
    	Python's variance function computed 9969.287244336085.
    	Your var_method_0(X) computed 9969.287244336087.
    	The relative difference is |you - python| / |python| ~= 1.824593232158388e-16.
    
    Data: n=100000 samples from a Gaussian with mean 10000000.0 and standard deviation 100.0
    	Python's variance function computed 9962.668891152425.
    	Your var_method_0(X) computed 9962.668891152241.
    	The relative difference is |you - python| / |python| ~= 1.84406339069731e-14.
    
    Data: n=100000 samples from a Gaussian with mean 10000000.0 and standard deviation 100.0
    	Python's variance function computed 9970.453229873041.
    	Your var_method_0(X) computed 9970.45322987317.
    	The relative difference is |you - python| / |python| ~= 1.2953096983074693e-14.
    
    Data: n=100000 samples from a Gaussian with mean 10000000.0 and standard deviation 100.0
    	Python's variance function computed 10017.454818169896.
    	Your var_method_0(X) computed 10017.454818169901.
    	The relative difference is |you - python| / |python| ~= 5.447459768662587e-16.
    
    Data: n=100000 samples from a Gaussian with mean 10000000.0 and standard deviation 100.0
    	Python's variance function computed 9914.466729491589.
    	Your var_method_0(X) computed 9914.466729491667.
    	The relative difference is |you - python| / |python| ~= 7.889132767959043e-15.
    
    (Passed!)
    

## A one-pass algorithm

If there are a huge number of samples, the preceding formula can be slow. The reason is that it makes *two* passes (or loops) over the data: once to sum the samples and another to sum the squares of the samples.

So if there are a huge number of samples and these were stored on disk, for instance, you would have to read each sample from disk twice. (For reference, the cost of accessing data on disk can be orders of magnitude slower than reading it from memory.)

However, there is an alternative that would touch each observation only once. It is based on this formula:

$$\begin{eqnarray*}
  \bar{s}^2
  & = & \dfrac{\left( \sum_{i=0}^{n-1} x_i^2 \right) - \dfrac{1}{n}\left( \sum_{i=0}^{n-1} x_i \right)^2}{n-1}.
\end{eqnarray*}$$

In exact arithmetic, it is the same as the previous formula. And it can be implemented using only **one pass** of the data, using an algorithm of the following form:

```
  temp_sum = 0
  temp_sum_squares = 0
  for each observation x_i: # Read x_i once, but use twice!
     temp_sum += x_i
     temp_sum_squares += (x_i * x_i)
  (calculate final variance)
```

But there is a catch, related to the numerical stability of this scheme.

**Exercise 1** (5 points). Implement a function, `var_method_1(x)`, for the one-pass scheme shown above.

The test cell below will run several experiments comparing its accuracy to the accuracy of the first method. You should observe that the one-pass method can be highly inaccurate!


```python
def var_method_1(x):
    temp_sum = 0
    temp_sum_squares = 0
    for x_i in x:
        temp_sum += x_i
        temp_sum_squares += (x_i * x_i)
    return (temp_sum_squares - temp_sum**2 / n) / (n - 1)
```


```python
# Test cell: `exercise_1_test`

from random import gauss
from statistics import variance

n = 100000
mu = 1e7
sigma = 1.0

for _ in range(5): # 5 trials
    X = [gauss(mu, sigma) for _ in range(n)]
    var_py = variance(X)
    try:
        del variance
        var_you_0 = var_method_0(X)
        var_you_1 = var_method_1(X)
    except NameError as n:
        if n.args[0] == "name 'variance' is not defined":
            assert False, "Did you try to use `variance()` instead of implementing it from scratch?"
        else:
            raise n
    finally:
        variance = SAVE_VARIANCE
        
    rel_diff_0 = abs(var_you_0 - var_py) / var_py
    rel_diff_1 = abs(var_you_1 - var_py) / var_py
    print("\nData: n={} samples from a Gaussian with mean {} and standard deviation {}".format(n, mu, sigma))
    print("\tPython's variance function computed {}.".format(var_py))
    print("\tvar_method_0(X) computed {}, with a relative difference of {}.".format(var_you_0, rel_diff_0))
    assert rel_diff_0 <= n*(2.0**(-52)), "The relative difference is larger than expected."
    print("\tvar_method_1(X) computed {}, with a relative difference of {}.".format(var_you_1, rel_diff_1))
    assert rel_diff_1 > n*(2.0**(-52)), "The relative difference is smaller than expected!"
    
print("\n(Passed!)")
```

    
    Data: n=100000 samples from a Gaussian with mean 10000000.0 and standard deviation 1.0
    	Python's variance function computed 0.9966184525543424.
    	var_method_0(X) computed 0.9966184525543377, with a relative difference of 4.790157149561571e-15.
    	var_method_1(X) computed 1.2902529025290252, with a relative difference of 0.29463075786134096.
    
    Data: n=100000 samples from a Gaussian with mean 10000000.0 and standard deviation 1.0
    	Python's variance function computed 1.002209593461886.
    	var_method_0(X) computed 1.002209593461882, with a relative difference of 3.987991049701084e-15.
    	var_method_1(X) computed 0.10240102401024011, with a relative difference of 0.8978247417723063.
    
    Data: n=100000 samples from a Gaussian with mean 10000000.0 and standard deviation 1.0
    	Python's variance function computed 0.9977123385392327.
    	var_method_0(X) computed 0.9977123385392533, with a relative difference of 2.069749712453405e-14.
    	var_method_1(X) computed 1.1673716737167372, with a relative difference of 0.17004834823021794.
    
    Data: n=100000 samples from a Gaussian with mean 10000000.0 and standard deviation 1.0
    	Python's variance function computed 1.00551535063407.
    	var_method_0(X) computed 1.0055153506340713, with a relative difference of 1.3249600105160708e-15.
    	var_method_1(X) computed 1.2288122881228811, with a relative difference of 0.2220721318157917.
    
    Data: n=100000 samples from a Gaussian with mean 10000000.0 and standard deviation 1.0
    	Python's variance function computed 1.0044276996581831.
    	var_method_0(X) computed 1.0044276996582098, with a relative difference of 2.6527895039206345e-14.
    	var_method_1(X) computed 3.153951539515395, with a relative difference of 2.1400483485159922.
    
    (Passed!)
    

**Fin!** If you've reached this point and all tests above pass, you are ready to submit your solution to this problem. Don't forget to save you work prior to submitting.


```python
print("\n(This notebook ran to completion.)")
```

    
    (This notebook ran to completion.)
    
