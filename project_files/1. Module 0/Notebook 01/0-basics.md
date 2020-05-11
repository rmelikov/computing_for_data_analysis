# Python review: Values, variables, types, lists, and strings

These first few notebooks are a set of exercises designed to reinforce various aspects of Python programming.

**Study hint: Read the test code!** You'll notice that most of the exercises below have a place for you to code up your answer followed by a "test cell." That's a code cell that checks the output of your code to see whether it appears to produce correct results. You can often learn a lot by reading the test code. In fact, sometimes it gives you a hint about how to approach the problem. As such, we encourage you to try to read the test cells even if they seem cryptic, which is deliberate!

**Debugging tip: Read assertions.** The test cells often run an `assert` statement to see whether some condition that it thinks should be true is true. If an assertion fails, look at the condition being checked and use that as a guide to help you debug. For example, if an assertion reads, `assert a + b == 3`, and that fails, inspect the values and types of `a` and `b` to help determine why their sum does not equal 3.

**Exercise 0** (1 point). Run the code cell below. It should display the output string, `Hello, world!`.


```python
print("Hello, world!")
```

    Hello, world!
    

**Exercise 1** (`x_float_test`: 1 point). Create a variable named `x_float` whose numerical value is one (1) and whose type is *floating-point* (i.e., `float`).


```python
x_float = 1.0
```


```python
# `x_float_test`: Test cell
assert x_float == 1, f"`x_float` has the wrong value ({x_float} rather than 1.0)"
assert type(x_float) is float, f"`type(x_float)` == {type(x_float)} rather than `float`"
print("\n(Passed!)")
```

    
    (Passed!)
    

**Exercise 2** (`strcat_ba_test`: 1 point). Complete the following function, `strcat_ba(a, b)`, so that given two strings, `a` and `b`, it returns the concatenation of `b` followed by `a` (pay attention to the order in these instructions!).


```python
def strcat_ba(a, b):
    assert type(a) is str, f"Input argument `a` has `type(a)` is {type(a)} rather than `str`"
    assert type(b) is str, f"Input argument `b` has `type(b)` is {type(b)} rather than `str`"
    return b + a

```


```python
# `strcat_ba_test`: Test cell

# Workaround:  # Python 3.5.2 does not have `random.choices()` (available in 3.6+)
def random_letter():
    from random import choice
    return choice('abcdefghijklmnopqrstuvwxyz')

def random_string(n, fun=random_letter):
    return ''.join([str(fun()) for _ in range(n)])

a = random_string(5)
b = random_string(3)
c = strcat_ba(a, b)
print('strcat_ba("{}", "{}") == "{}"'.format(a, b, c))
assert len(c) == len(a) + len(b), "`c` has the wrong length: {len(c)} rather than {len(a)+len(b)}"
assert c[:len(b)] == b
assert c[-len(a):] == a
print("\n(Passed!)")
```

    strcat_ba("ojxke", "zlf") == "zlfojxke"
    
    (Passed!)
    

**Exercise 3** (`strcat_list_test`: 2 points). Complete the following function, `strcat_list(L)`, which generalizes the previous function: given a *list* of strings, `L[:]`, returns the concatenation of the strings in reverse order. For example:

```python
    strcat_list(['abc', 'def', 'ghi']) == 'ghidefabc'
```


```python
def strcat_list(L):
    assert type(L) is list
    string = ''
    for i in range(len(L)):
        string += L[len(L) - 1 - i]
    return string
```


```python
# `strcat_list_test`: Test cell
n = 3
nL = 6
L = [random_string(n) for _ in range(nL)]
Lc = strcat_list(L)

print('L == {}'.format(L))
print('strcat_list(L) == \'{}\''.format(Lc))
assert all([Lc[i*n:(i+1)*n] == L[nL-i-1] for i, x in zip(range(nL), L)])
print("\n(Passed!)")
```

    L == ['jyr', 'rlt', 'uon', 'bie', 'xnv', 'yvn']
    strcat_list(L) == 'yvnxnvbieuonrltjyr'
    
    (Passed!)
    

**Exercise 4** (`floor_fraction_test`: 1 point). Suppose you are given two variables, `a` and `b`, whose values are the real numbers, $a \geq 0$ (non-negative) and $b > 0$ (positive). Complete the function, `floor_fraction(a, b)` so that it returns $\left\lfloor\frac{a}{b}\right\rfloor$, that is, the *floor* of $\frac{a}{b}$. The *type* of the returned value must be `int` (an integer).


```python
def is_number(x):
    """Returns `True` if `x` is a number-like type, e.g., `int`, `float`, `Decimal()`, ..."""
    from numbers import Number
    return isinstance(x, Number)
    
def floor_fraction(a, b):
    assert is_number(a) and a >= 0
    assert is_number(b) and b > 0
    from math import floor
    return floor(a/b)

```


```python
# `floor_fraction_test`: Test cell
from random import random
a = random()
b = random()
c = floor_fraction(a, b)

print('floor_fraction({}, {}) == floor({}) == {}'.format(a, b, a/b, c))
assert b*c <= a <= b*(c+1)
assert type(c) is int, f"type(c) == {type(c)} rather than `int`"
print('\n(Passed!)')
```

    floor_fraction(0.7819786240630128, 0.3438168541622424) == floor(2.2744045691663737) == 2
    
    (Passed!)
    

**Exercise 5** (`ceiling_fraction_test`: 1 point). Complete the function, `ceiling_fraction(a, b)`, which for any numeric inputs, `a` and `b`, corresponding to real numbers, $a \geq 0$ and $b > 0$, returns $\left\lceil\frac{a}{b}\right\rceil$, that is, the *ceiling* of $\frac{a}{b}$. The type of the returned value must be `int`.


```python
def ceiling_fraction(a, b):
    assert is_number(a) and a >= 0
    assert is_number(b) and b > 0
    from math import ceil
    return ceil(a/b)

```


```python
# `ceiling_fraction_test`: Test cell
from random import random
a = random()
b = random()
c = ceiling_fraction(a, b)
print('ceiling_fraction({}, {}) == ceiling({}) == {}'.format(a, b, a/b, c))
assert b*(c-1) <= a <= b*c
assert type(c) is int
print("\n(Passed!)")
```

    ceiling_fraction(0.06657157648057277, 0.28789287234149885) == ceiling(0.2312373208101015) == 1
    
    (Passed!)
    


```python
a = 0.3
b = 0.1
c = ceiling_fraction(a, b)
print(f"{a/b}")
print('ceiling_fraction({}, {}) == ceiling({}) == {}'.format(a, b, a/b, c))
assert b*(c-1) <= a <= b*c
assert type(c) is int
```

    2.9999999999999996
    ceiling_fraction(0.3, 0.1) == ceiling(2.9999999999999996) == 3
    

**Exercise 6** (`report_exam_avg_test`: 1 point). Let `a`, `b`, and `c` represent three exam scores as numerical values. Complete the function, `report_exam_avg(a, b, c)` so that it computes the average score (equally weighted) and returns the string, `'Your average score: XX'`, where `XX` is the average rounded to one decimal place. For example:

```python
    report_exam_avg(100, 95, 80) == 'Your average score: 91.7'
```


```python
def report_exam_avg(a, b, c):
    assert is_number(a) and is_number(b) and is_number(c)
    from inspect import signature
    sig = signature(report_exam_avg)
    params = sig.parameters
    numerator = sum(list(locals().values())[3 : (3 + len(params))])
    denominator = len(params)
    rounded_avg = round(numerator / denominator, 1)
    return f'Your average score: {rounded_avg}'

```


```python
# `report_exam_avg_test`: Test cell
msg = report_exam_avg(100, 95, 80)
print(msg)
assert msg == 'Your average score: 91.7'

print("Checking some additional randomly generated cases:")
for _ in range(10):
    ex1 = random() * 100
    ex2 = random() * 100
    ex3 = random() * 100
    msg = report_exam_avg(ex1, ex2, ex3)
    ex_rounded_avg = float(msg.split()[-1])
    abs_err = abs(ex_rounded_avg*3 - (ex1 + ex2 + ex3)) / 3
    print("{}, {}, {} -> '{}' [{}]".format(ex1, ex2, ex3, msg, abs_err))
    assert abs_err <= 0.05

print("\n(Passed!)")
```

    Your average score: 91.7
    Checking some additional randomly generated cases:
    31.753771194987657, 42.11144616838948, 27.84964058100745 -> 'Your average score: 33.9' [0.004952648128195847]
    85.93155231065201, 17.150522694756944, 13.514732016127972 -> 'Your average score: 38.9' [0.03439765948768544]
    97.3468834405844, 27.904894718791862, 85.25767740497494 -> 'Your average score: 70.2' [0.030181478549602996]
    28.779598660418294, 30.499781942095694, 59.81141266929072 -> 'Your average score: 39.7' [0.0030689093984364035]
    55.54683807651768, 71.40005920532234, 23.81916348309453 -> 'Your average score: 50.3' [0.044646411688470757]
    27.506559262518206, 72.94700788009428, 54.705374426232304 -> 'Your average score: 51.7' [0.019647189614924326]
    3.362295895177947, 82.22526948077234, 1.3707212646226874 -> 'Your average score: 29.0' [0.013904453142340381]
    83.93448245849984, 2.5526581748962496, 64.47855092889267 -> 'Your average score: 50.3' [0.021897187429601672]
    61.986719108727804, 19.81026593869376, 53.708383419598704 -> 'Your average score: 45.2' [0.03154384432658238]
    81.78396942207017, 94.27233513626477, 56.92444644194974 -> 'Your average score: 77.7' [0.039749666571784324]
    
    (Passed!)
    

**Exercise 7** (`count_word_lengths_test`: 2 points). Write a function `count_word_lengths(s)` that, given a string consisting of words separated by spaces, returns a list containing the length of each word. Words will consist of lowercase alphabetic characters, and they may be separated by multiple consecutive spaces. If a string is empty or has no spaces, the function should return an empty list.

For instance, in this code sample,

```python
   count_word_lengths('the quick  brown   fox jumped over     the lazy  dog') == [3, 5, 5, 3, 6, 4, 3, 4, 3]`
```

the input string consists of nine (9) words whose respective lengths are shown in the list.


```python
def count_word_lengths(s):
    assert all([x.isalpha() or x == ' ' for x in s])
    assert type(s) is str
    #return [len(w) for w in s.split()]    # Also works
    return list(map(len, s.split()))

```


```python
# `count_word_lengths_test`: Test cell

# Test 1: Example
qbf_str = 'the quick brown fox jumped over the lazy dog'
qbf_lens = count_word_lengths(qbf_str)
print("Test 1: count_word_lengths('{}') == {}".format(qbf_str, qbf_lens))
assert qbf_lens == [3, 5, 5, 3, 6, 4, 3, 4, 3]

# Test 2: Random strings
from random import choice # 3.5.2 does not have `choices()` (available in 3.6+)
#return ''.join([choice('abcdefghijklmnopqrstuvwxyz') for _ in range(n)])

def random_letter_or_space(pr_space=0.15):
    from random import choice, random
    is_space = (random() <= pr_space)
    if is_space:
        return ' '
    return random_letter()

S_LEN = 40
W_SPACE = 1 / 6
rand_str = random_string(S_LEN, fun=random_letter_or_space)
rand_lens = count_word_lengths(rand_str)
print("Test 2: count_word_lengths('{}') == '{}'".format(rand_str, rand_lens))
c = 0
while c < len(rand_str) and rand_str[c] == ' ':
    c += 1
for k in rand_lens:
    print("  => '{}'".format (rand_str[c:c+k]))
    assert (c+k) == len(rand_str) or rand_str[c+k] == ' '
    c += k
    while c < len(rand_str) and rand_str[c] == ' ':
        c += 1
    
# Test 3: Empty string
print("Test 3: Empty strings...")
assert count_word_lengths('') == []
assert count_word_lengths('   ') == []
print(count_word_lengths('the'))

print("\n(Passed!)")
```

    Test 1: count_word_lengths('the quick brown fox jumped over the lazy dog') == [3, 5, 5, 3, 6, 4, 3, 4, 3]
    Test 2: count_word_lengths('ilfg ehz gmgibo agml mkmwb gapcj kri kaf') == '[4, 3, 6, 4, 5, 5, 3, 3]'
      => 'ilfg'
      => 'ehz'
      => 'gmgibo'
      => 'agml'
      => 'mkmwb'
      => 'gapcj'
      => 'kri'
      => 'kaf'
    Test 3: Empty strings...
    [3]
    
    (Passed!)
    

**Fin!** You've reached the end of this part. Don't forget to restart and run all cells again to make sure it's all working when run in sequence; and make sure your work passes the submission process. Good luck!
