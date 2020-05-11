# Part 2 of 2 (OPTIONAL): An extreme case of regular expression processing

> This part is **OPTIONAL**. That is, while there are exercises, they are worth 0 points each. Rather, this notebook is designed for those of you who may have a deeper interest in computational aspects of the course material and would like to explore that.

There is a beautiful theory underlying regular expressions, and efficient regular expression processing is regarded as one of the classic problems of computer science. In the last part of this lab, you will explore a bit of that theory, albeit by experiment.

In particular, the code cells below will walk you through a simple example of the potentially **hidden cost** of regular expression parsing. And if you really want to geek out, look at the article on which this example is taken: https://swtch.com/~rsc/regexp/regexp1.html

## Quick review

**Exercise 0** (ungraded) Let $a^n$ be a shorthand notation for a string in which $a$ is repeated $n$ times. For example, $a^3$ is the same as $aaa$ and $a^6$ is the same as $aaaaaa$. Write a function to generate the string for $a^n$, given a string $a$ and an integer $n \geq 1$.


```python
def rep_str (a, n):
    """Returns a string consisting of an input string repeated a given number of times."""
    assert type(a) is str and n >= 1
    return a * n

```


```python
# Test cell: `rep_str_test`

def check_fixed(a, n, ans):
    msg = "Testing: '{}'^{} -> '{}'".format(a, n, ans)
    print(msg)
    assert rep_str(a, n) == ans, "Case failed!"
    
check_fixed('a', 3, 'aaa')
check_fixed('cat', 4, 'catcatcatcat')
check_fixed('', 100, '')

def check_rand():
    from random import choice, randint
    a = ''.join([choice([chr(k) for k in range(ord('a'), ord('z')+1)]) for _ in range(randint(1, 5))])
    n = randint(1, 10)
    msg = "Testing: '{}'^{}".format(a, n)
    print(msg)
    s_you = rep_str(a, n)
    for k in range(0, n*len(a), len(a)):
        assert s_you[k:(k+len(a))] == a, "Your result, '{}', is not correct at position {} [{}].".format(s_you, k)
    
for _ in range(10):
    check_rand()

print("\n(Passed!)")
```

    Testing: 'a'^3 -> 'aaa'
    Testing: 'cat'^4 -> 'catcatcatcat'
    Testing: ''^100 -> ''
    Testing: 'txq'^5
    Testing: 'ayimi'^5
    Testing: 'spe'^6
    Testing: 'jnmy'^3
    Testing: 'n'^8
    Testing: 'yylv'^9
    Testing: 'rbpaw'^8
    Testing: 'cce'^1
    Testing: 'igbp'^10
    Testing: 'iwczk'^3
    
    (Passed!)
    

## An initial experiment

Intuitively, you should expect (or hope) that the time to determine whether a string of length $n$ matches a given pattern will be proportional to $n$. Let's see if this holds when matching simple input strings of repeated letters against a pattern designed to match such strings.


```python
import re
```


```python
# Set up an input problem
n = 3
s_n = rep_str ('a', n) # Input string
pattern = '^a{%d}$' % n # Pattern to match it exactly

# Test it
print ("Matching input '{}' against pattern '{}'...".format (s_n, pattern))
assert re.match (pattern, s_n) is not None

# Benchmark it & report time, normalized to 'n'
timing = %timeit -q -o re.match (pattern, s_n)
t_avg = sum (timing.all_runs) / len (timing.all_runs) / timing.loops / n * 1e9
print ("Average time per match per `n`: {:.1f} ns".format (t_avg))
```

    Matching input 'aaa' against pattern '^a{3}$'...
    Average time per match per `n`: 628.9 ns
    

Before moving on, be sure you understand what the above benchmark is doing. For more on the Jupyter "magic" command, `%timeit`, see: http://ipython.readthedocs.io/en/stable/interactive/magics.html?highlight=magic#magic-magic

**Exercise 1** (ungraded) Repeat the above experiment for various values of `n`. To help keep track of the results, feel free to create new code cells that repeat the benchmark for different values of `n`. Explain what you observe. Can you conclude that matching simple regular expression patterns of the form `^a{n}$` against input strings of the form $a^n$ does, indeed, scale linearly?


```python
# Use this code cell (and others, if you wish) to set up an experiment
# to test whether matching simple patterns behaves at worst linearly
# in the length of the input.

def string_pattern(l, n):
    s = rep_str(l, n)
    p = '^' + l + '{%d}$' % n   
    assert re.match(p, s) is not None
    return (s, p)

print(string_pattern('a', 10))

N = [1000, 10000, 100000, 1000000, 10000000, 100000000]
T = []
for n in N:
    s, p = string_pattern('a', n)
    #print ("Matching input '{}' against pattern '{}'...".format (s, p))
    timing = %timeit -q -o re.match(p, s)
    T.append(sum(timing.all_runs) / len(timing.all_runs) / timing.loops / n * 1e9)
    print ("==> Average time per match per `n`: {:.1f} ns".format(T[-1]))
```

    ('aaaaaaaaaa', '^a{10}$')
    ==> Average time per match per `n`: 2.7 ns
    ==> Average time per match per `n`: 0.9 ns
    ==> Average time per match per `n`: 0.6 ns
    ==> Average time per match per `n`: 0.6 ns
    ==> Average time per match per `n`: 0.7 ns
    ==> Average time per match per `n`: 0.6 ns
    

**Answer.** To see asymptotically linear behavior, you'll need to try some fairly large values of $n$, e.g., a thousand, ten thousand, a hundred thousand, and a million. Even then, it may **appear** as though the time continues to decrease, but that does not mean you have not reached an asymptote; why not?

> Regarding the latter question, suppose matching time as a function of input size is $t(n) = \alpha + \beta n$, so that the time per match per $n$ is $t(n)/n$.

## A more complex pattern

Consider a regular expression of the form:

$$(a?)^n(a^n) \quad$$

For instance, $n=3$, the regular expression pattern is `(a?){3}a{3} == a?a?a?aaa`. Start by convincing yourself that an input string of the form,

$$a^n = \underbrace{aa\cdots a}_{n \mbox{ occurrences}}$$

should match this pattern. Here is some code to set up an experiment to benchmark this case.


```python
def setup_inputs(n):
    """Sets up the 'complex pattern example' above."""
    s_n = rep_str('a', n)
    p_n = "^(a?){%d}(a{%d})$" % (n, n)
    print ("[n={}] Matching pattern '{}' against input '{}'...".format(n, p_n, s_n))
    assert re.match(p_n, s_n) is not None
    return (p_n, s_n)

n = 3
p_n, s_n = setup_inputs(n)
timing = %timeit -q -o re.match(p_n, s_n)
t_n = sum(timing.all_runs) / len(timing.all_runs) / timing.loops / n * 1e9
print ("==> Time per run per `n`: {} ns".format(t_n))
```

    [n=3] Matching pattern '^(a?){3}(a{3})$' against input 'aaa'...
    ==> Time per run per `n`: 990.8471428553595 ns
    

**Exercise 3** (ungraded) Repeat the above experiment but for different values of $n$, such as $n \in \{3, 6, 9, 12, 15, 18\}$. As before, feel free to use the code cell below or make new code cells to contain the code for your experiments. Summarize what you observe. How does the execution time vary with $n$? Can you explain this behavior?


```python
# Use this code cell (and others, if you wish) to set up an experiment
# to test whether matching simple patterns behaves at worst linearly
# in the length of the input.

N = [3, 6, 9, 12, 15, 18]
T = []
for n in N:
    p_n, s_n = setup_inputs (n)
    timing = %timeit -q -o re.match (p_n, s_n)
    t_n = sum (timing.all_runs) / len (timing.all_runs) / timing.loops / n * 1e9
    print ("Time per run per `n`: {} ns".format (t_n))
    T.append (t_n)

```

    [n=3] Matching pattern '^(a?){3}(a{3})$' against input 'aaa'...
    Time per run per `n`: 958.7144761947761 ns
    [n=6] Matching pattern '^(a?){6}(a{6})$' against input 'aaaaaa'...
    Time per run per `n`: 1673.5274285747435 ns
    [n=9] Matching pattern '^(a?){9}(a{9})$' against input 'aaaaaaaaa'...
    Time per run per `n`: 7443.157777800926 ns
    [n=12] Matching pattern '^(a?){12}(a{12})$' against input 'aaaaaaaaaaaa'...
    Time per run per `n`: 43285.49523805643 ns
    [n=15] Matching pattern '^(a?){15}(a{15})$' against input 'aaaaaaaaaaaaaaa'...
    Time per run per `n`: 197484.27619023938 ns
    [n=18] Matching pattern '^(a?){18}(a{18})$' against input 'aaaaaaaaaaaaaaaaaa'...
    Time per run per `n`: 1757576.507935439 ns
    

**Answer.** Here, you should observe something more like polynomial growth. Here are some results we collected, for instance.

|    n    |  t (ns)   |
|---------|-----------|
|       3 |     412.3 |
|       6 |     728.4 |
|       9 |   3,259.1 |
|      12 |  20,201.9 |
|      15 | 131,392.2 |
|      18 | 861,721.7 |

**Fin!** This cell marks the end of Part 2, which is the final part of this assignment. Don't forget to save, restart and rerun all cells, and submit it.
