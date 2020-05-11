# Problem 10: Polynomials

This exercise is similar to Notebook 1, which deals with compressed vectors. Here, we represent polynomials as compressed vectors and perform simple operations like addition and multiplication in this compressed form. (While the concept is similar, the operations you will need to implement are different from what you did in Notebook 1.)

This problem has four (4) exercises, numbered 0 through 3, and is worth a total of ten (10) points. Depending on your approach, Exercise 1 depends on 0 and 3 depends on 2, but 0 and 2 are independent. Therefore, partial credit is possible.

## Definition: $n$-th order polynomials

Let $P(x, n)$ denote a _polynomial of order $n$_, which has the mathematical form,

$P(x, n) = a_n x^n + a_{n-1} x^{n-1} + \cdots + a_2 x^2 + a_1 x + a_0.$

The values $a_n, a_{n-1}, \ldots, a_{0}$ are the polynomial's _coefficients_. (Observe that we have ordered the coefficients in **descending order** of power, $x^k$.) Knowing the coefficients is enough to specify the polynomial.

Suppose we expect that a typical polynomial is likely to have several zero coefficients. Then, we can represent the polynomial's coefficients in our computer code using a _compressed vector_ format. For example, consider this 5th-order polynomial, $P_A(x, 5)$:

$ P_A(x, 5) = x^5 + 2x^4 + 8x^2 + 9x^1 + 3 $

Observe that $a_3=0$ and so omitted from the sum. In code, we can represent this polynomial using two lists that record the power or "index" (`idx`) and the coefficient value (`coef`) of each term:


```python
idx = [5, 4, 2, 1, 0]
coef = [1, 2, 8, 9, 3]
```

**Important note.** For the rest of this notebook, your solutions should enforce the following conventions for storing `idx` and `coef`.

1. The powers in `idx` are always sorted in descending order.
2. The terms stored are the ones that have a non-zero coefficient. 
3. You may assume all coefficients (in `coef`) are non-negative.
4. All elements in the `idx` vector are unique.

Here is another example of a polynomial and its list representation:

$ P_B(x, 5) = x^5 + 5x^3 + 2x^2.$


```python
idx = [5, 3, 2]
coef = [1, 5, 2]
```

## Adding polynomials

Given two polynomials, $P_1(x, n_1)$ and $P_2(x, n_2)$, let's consider their sum, $P_3(x, n_3) \equiv P_1(x, n_1) + P_2(x, n_2)$. (_Aside:_ How will $n_3$ relate to $n_1$ and $n_2$?)

For instance, let

$$\begin{eqnarray}
  P_1(x, 4) & = & 3x^4 + 2x^2 + 2x + 5 \\
  P_2(x, 3) & = & 4x^3 + 3x^2 + x + 2.
\end{eqnarray}$$

Then their sum is

$$\begin{eqnarray}
  P_3(x, 4) \equiv P_1(x, 4) + P_2(x, 3) & = & 3x^4 + 4x^3 + 5x^2 + 3x + 7.
\end{eqnarray}$$

Now suppose we store $P_1$ in compressed form using the lists `idx1` and `coef`, $P_2$ using `idx2` and `coef2`, and $P_3$ by `idx3` and `coef3`. Then,


```python
idx1 = [4, 2, 1, 0]
coef1 = [3, 2, 2, 5]

idx2 = [3, 2, 1, 0]
coef2 = [4, 3, 1, 2]

idx3 = [4, 3, 2, 1, 0]
coef3 = [3, 4, 5, 3, 7]
```

**Exercise 0** (2 points). Write a function `add_idx(idx1, idx2)`, that takes as inputs the index vectors of two polynomials and returns the index vector of their sum.

For instance, if `idx1` and `idx2` are defined as in the preceding example, then

```python
    add_idx(idx1, idx2) == idx3
```


```python
def add_idx(idx1, idx2):
    return sorted(list(set(idx1 + idx2)), reverse = True)
```


```python
# Test cell : `test_add_idx`

import random
import numpy as np
def generate_polynomial(max_n=9):
    len_poly = random.randint(2,5)
    idx = random.sample(range(max_n+1), len_poly)
    idx.sort(reverse=True)
    coef = [random.randint(2, 8) for _ in range(len_poly)]
    assert len(idx) == len(coef), 'Unequal lengths for generated idx and coef'
    return idx, coef

for trial in range(5):
    print("\n=== trial {} ===".format(trial))
    idx1, coef1 = generate_polynomial()
    idx2, coef2 = generate_polynomial()
    print("idx1 is {}".format(idx1))
    print("idx2 is {}".format(idx2))
    idx = add_idx(idx1, idx2)
    idx_inv = idx[::-1]
    idx.sort()
    assert idx == idx_inv, "The idx list is not sorted correctly"
    idx.sort(reverse=True)
    print("\nyour solution:")
    print("idx = {}".format(idx))
    for i in idx1:
        assert i in idx, "{} in idx1 is absent from idx".format(i)
    for j in idx2:
        assert j in idx, "{} in idx2 is absent from idx".format(j)
        
print("\nPassed!")
```

    
    === trial 0 ===
    idx1 is [8, 4, 2, 1, 0]
    idx2 is [9, 7, 3, 2]
    
    your solution:
    idx = [9, 8, 7, 4, 3, 2, 1, 0]
    
    === trial 1 ===
    idx1 is [6, 0]
    idx2 is [9, 7, 2, 1]
    
    your solution:
    idx = [9, 7, 6, 2, 1, 0]
    
    === trial 2 ===
    idx1 is [4, 3, 2, 1, 0]
    idx2 is [8, 6, 5, 4]
    
    your solution:
    idx = [8, 6, 5, 4, 3, 2, 1, 0]
    
    === trial 3 ===
    idx1 is [7, 5, 3, 0]
    idx2 is [7, 4, 1]
    
    your solution:
    idx = [7, 5, 4, 3, 1, 0]
    
    === trial 4 ===
    idx1 is [9, 5]
    idx2 is [8, 5, 4]
    
    your solution:
    idx = [9, 8, 5, 4]
    
    Passed!
    

**Exercise 1** (3 points). Write a function, `add_coef(idx1, coef1, idx2, coef2)`, which takes in the compressed vector form of two polynomials, `(idx1, coef)` and `(idx2, coef2)`, and returns the **coefficient vector** corresponding to their sum. (_Note:_ You only need to return the coefficients, not the indices.)

For the earlier example,

```python
    idx1 = [4, 2, 1, 0]
    coef1 = [3, 2, 2, 5]

    idx2 = [3, 2, 1, 0]
    coef2 = [4, 3, 1, 2]

    add_coef(idx1, coef1, idx2, coef2) == [3, 4, 5, 3, 7]
```


```python
def add_coef(idx1, coef1, idx2, coef2):
    from collections import Counter
    return [
        v 
        for k, v in sorted(
            dict(
                Counter(dict(zip(idx1, coef1))) + Counter(dict(zip(idx2, coef2)))
            ).items(),
            key = lambda v : v,
            reverse = True
        )
    ]
```


```python
# Test cell: `test_add_coef`

for trial in range(5):
    print("\n=== trial {} ===".format(trial))
    idx1, coef1 = generate_polynomial()
    idx2, coef2 = generate_polynomial()
    # idx1 and coef1
    print("idx1 is {}".format(idx1))
    print("coef1 is {} \n".format(coef1))
    # idx2 and coef2
    print("idx2 is {}".format(idx2))
    print("coef2 is {} \n".format(coef2))
    # Student's solution
    print("your solutions -")
    idx = add_idx(idx1, idx2)
    coef = add_coef(idx1, coef1, idx2, coef2)
    print("idx : {}".format(idx))
    print("coef : {}".format(coef))
    for i in idx:
        pos1 = idx1.index(i) if i in idx1 else -1
        pos2 = idx2.index(i) if i in idx2 else -1
        c1 = coef1[pos1] if pos1 != -1 else 0
        c2 = coef2[pos2] if pos2 != -1 else 0
        c = c1 + c2
        pos = idx.index(i)
        assert c == coef[pos], "The correct coefficient for power {} should be {}, you get {}".format(i, c, coef[pos])

print("\nPassed!")
```

    
    === trial 0 ===
    idx1 is [4, 3, 2]
    coef1 is [8, 3, 4] 
    
    idx2 is [9, 5]
    coef2 is [7, 4] 
    
    your solutions -
    idx : [9, 5, 4, 3, 2]
    coef : [7, 4, 8, 3, 4]
    
    === trial 1 ===
    idx1 is [8, 4, 0]
    coef1 is [2, 8, 5] 
    
    idx2 is [6, 1]
    coef2 is [7, 6] 
    
    your solutions -
    idx : [8, 6, 4, 1, 0]
    coef : [2, 7, 8, 6, 5]
    
    === trial 2 ===
    idx1 is [5, 4]
    coef1 is [8, 5] 
    
    idx2 is [7, 3]
    coef2 is [8, 3] 
    
    your solutions -
    idx : [7, 5, 4, 3]
    coef : [8, 8, 5, 3]
    
    === trial 3 ===
    idx1 is [9, 5]
    coef1 is [8, 7] 
    
    idx2 is [8, 6, 3, 2, 1]
    coef2 is [3, 4, 4, 2, 6] 
    
    your solutions -
    idx : [9, 8, 6, 5, 3, 2, 1]
    coef : [8, 3, 4, 7, 4, 2, 6]
    
    === trial 4 ===
    idx1 is [9, 8, 4, 2, 1]
    coef1 is [8, 3, 8, 4, 7] 
    
    idx2 is [9, 7, 5, 3, 2]
    coef2 is [3, 5, 4, 2, 8] 
    
    your solutions -
    idx : [9, 8, 7, 5, 4, 3, 2, 1]
    coef : [11, 3, 5, 4, 8, 2, 12, 7]
    
    Passed!
    

## Polynomial multiplication

Now let's consider multiplying two polynomials.

For example, let

$$\begin{eqnarray}
P_1(x, 5) & = & x^5 + 2x^4 + 3 \\
P_2(x, 5) & = & x^5 + 5x^3.
\end{eqnarray}$$

Then their product, $P_1(x, 5) \times P_2(x, 5)$ is

$$P_3(x, 10) \equiv P_1(x, 5) \times P2(x, 5) = x^{10} + 2 x^9 + 5 x^8 + 10 x^7 + 3x^5 + 15 x^3.$$

> _Aside._ For the product $P_3(x, n_3) \equiv P_1(x, n_1) \times P_2(x, n_2)$, what is the relationship between $n_3$ and $n_1$ and $n_2$?

You may wish to verify the following fact for the special case of multiplying polynomials with two terms each:

$(c_a x^a + c_b x^b)$ $\times$ $(c_d x^d + c_e x^e) = c_a c_d x^{a+d} + c_a c_e x^{a+e}+ c_b c_d x^{b+d} + c_b c_e x^{b + e}.$ 

(When storing the result of this product in compressed vector form, recall that the index vector must be sorted in descending order.)


```python
idx1 = [5, 4, 0]
coef1 = [1, 2, 3]

idx2 = [5, 3]
coef2 = [1, 5]

# idx_mult and coef_mult for the final polynomial obtained by multiplying P1 and P2
idx_mult = [10, 9, 8, 7, 5, 3]
coef_mult = [1, 2, 5, 10, 3, 15]
```

**Exercise 2** (2 points). Write a function, `mult_idx(idx1, idx2)`, that takes as input the index lists of two polynomials and returns the index list for their product.

For instance, given `idx1` and `idx2` from the previous example, your function would return `idx_mult`.


```python
def mult_idx(idx1, idx2):
    return sorted(
        list(
            set(
                [x + y for x in idx1 for y in idx2]
            )
        ),
        reverse = True
    )
```


```python
# Test cell: `test_mult_idx`

for trial in range(5):
    print("\n=== trial {} ===".format(trial))
    idx1, coef1 = generate_polynomial()
    idx2, coef2 = generate_polynomial()
    # idx1 and coef1
    print("idx1 is {}".format(idx1))
    # idx2 and coef2
    print("idx2 is {}".format(idx2))

    idx = mult_idx(idx1, idx2)
    l1 = len(idx1)
    l2 = len(idx2)

    p1_mat = np.repeat([idx1], [l2], axis=0)
    p2_mat = np.repeat([idx2], [l1], axis=0)
    p2_mat = p2_mat.T
    prod_powers = p1_mat + p2_mat
    prod_powers = list(np.unique(prod_powers))
    prod_powers.sort(reverse=True)
    print("your solution - {}".format(idx))
    assert idx == prod_powers, "Your output is {}, correct output is {}".format(idx, prod_powers)
    
print("Passed!")
```

    
    === trial 0 ===
    idx1 is [8, 7]
    idx2 is [9, 8, 7, 4, 2]
    your solution - [17, 16, 15, 14, 12, 11, 10, 9]
    
    === trial 1 ===
    idx1 is [6, 4, 1]
    idx2 is [8, 2]
    your solution - [14, 12, 9, 8, 6, 3]
    
    === trial 2 ===
    idx1 is [9, 6, 0]
    idx2 is [8, 7, 6, 2]
    your solution - [17, 16, 15, 14, 13, 12, 11, 8, 7, 6, 2]
    
    === trial 3 ===
    idx1 is [9, 8, 5, 3]
    idx2 is [9, 6, 1]
    your solution - [18, 17, 15, 14, 12, 11, 10, 9, 6, 4]
    
    === trial 4 ===
    idx1 is [9, 3, 2, 0]
    idx2 is [9, 4, 2]
    your solution - [18, 13, 12, 11, 9, 7, 6, 5, 4, 2]
    Passed!
    

**Exercise 3** (3 points). Write a function, `mult_coef(idx1, coef1, idx2, coef2)`, that takes as input two polynomials in compressed vector form, `(idx1, coef1)` and `(idx2, coef2)`, and returns the **coefficients** of their product.

In the previous example, your function should return `coef_mult`.


```python
def mult_coef(idx1, coef1, idx2, coef2):

    p1 = list(zip(idx1, coef1))
    
    p2 = list(zip(idx2, coef2))
    
    combinations = [((x[0] + y[0]), (x[1] * y[1])) for x in p1 for y in p2]

    from collections import defaultdict
    
    d = defaultdict(list)
    
    for k, v in combinations:
        d[k].append(v)
    
    return [
        sum(v) 
        for k, v in sorted(
            d.items(),
            key = lambda v : v[0],
            reverse = True
        )
    ]
```


```python
# Test cell: `test_mult_coef`

for trial in range(5):
    print("\n=== trial {} ===".format(trial))
    idx1, coef1 = generate_polynomial()
    idx2, coef2 = generate_polynomial()
    # idx1 and coef1
    print("idx1 is {}".format(idx1))
    print("coef1 is {} \n".format(coef1))
    # idx2 and coef2
    print("idx2 is {}".format(idx2))
    print("coef2 is {} \n".format(coef2))
    print("Your solution :")
    idx = mult_idx(idx1, idx2)
    coef = mult_coef(idx1, coef1, idx2, coef2)
    print("idx = {}".format(idx))
    print("coef = {}".format(coef))
    for e, i in enumerate(idx):
        cp = []
        for i1 in idx1:
            for i2 in idx2:
                if i == i1 + i2:
                    c1 = coef1[idx1.index(i1)]
                    c2 = coef2[idx2.index(i2)]
                    c = c1*c2
                    cp.append(c)
        assert coef[e] == sum(cp), "incorrect coefficient for power {}".format(i)
        
print("\nPassed!")
```

    
    === trial 0 ===
    idx1 is [3, 1]
    coef1 is [2, 4] 
    
    idx2 is [8, 6, 5, 3, 2]
    coef2 is [8, 4, 7, 8, 3] 
    
    Your solution :
    idx = [11, 9, 8, 7, 6, 5, 4, 3]
    coef = [16, 40, 14, 16, 44, 6, 32, 12]
    
    === trial 1 ===
    idx1 is [9, 4]
    coef1 is [5, 5] 
    
    idx2 is [9, 7, 6, 3]
    coef2 is [4, 6, 4, 6] 
    
    Your solution :
    idx = [18, 16, 15, 13, 12, 11, 10, 7]
    coef = [20, 30, 20, 20, 30, 30, 20, 30]
    
    === trial 2 ===
    idx1 is [6, 0]
    coef1 is [4, 2] 
    
    idx2 is [8, 7, 6, 5, 0]
    coef2 is [6, 2, 4, 2, 4] 
    
    Your solution :
    idx = [14, 13, 12, 11, 8, 7, 6, 5, 0]
    coef = [24, 8, 16, 8, 12, 4, 24, 4, 8]
    
    === trial 3 ===
    idx1 is [9, 7, 6, 5, 2]
    coef1 is [4, 3, 8, 2, 4] 
    
    idx2 is [7, 4]
    coef2 is [3, 5] 
    
    Your solution :
    idx = [16, 14, 13, 12, 11, 10, 9, 6]
    coef = [12, 9, 44, 6, 15, 40, 22, 20]
    
    === trial 4 ===
    idx1 is [8, 7, 6, 3, 2]
    coef1 is [6, 6, 3, 6, 4] 
    
    idx2 is [6, 4, 1]
    coef2 is [4, 5, 6] 
    
    Your solution :
    idx = [14, 13, 12, 11, 10, 9, 8, 7, 6, 4, 3]
    coef = [24, 24, 42, 30, 15, 60, 52, 48, 20, 36, 24]
    
    Passed!
    

**Fin!** Remember to test your solutions by running them as the autograder will: restart the kernel and run all cells from "top-to-bottom." Also remember to submit to the autograder; otherwise, you will **not** get credit for your hard work!
