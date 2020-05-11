# Problem 12: Major League Baseball

This part of the exam will test your skills maneuvering in Pandas, as well as your ability to think of ways to improve computing and memory efficiency.

The dataset used in these exercises, ["Offensive Baseball Stats Through 2016"](https://www.kaggle.com/baseballstatsonline/offensive-baseball-stats-through-2016), Version 1, is from Kaggle user baseballstatsonline.com. The dataset is a fairly rich set of players and statistics, but for the purposes of these exercises you do not need to know anything in particular about the columns' meaning unless it is otherwise explained below. The dataset has been modified slightly for the purpose of some exercises. 

Run the following code cell to download the data and create some necessary functions. Good luck!

**Note:** If you are running this notebook locally please make sure you run the same version of pandas as in Vocareum enviroment.


```python
import pandas as pd
import numpy as np
from cse6040utils import canonicalize_tibble, tibbles_are_equivalent    

def get_data_path(filebase):
    return f"./mlb/{filebase}"

baseball = pd.read_csv(get_data_path('mlb_off_stats_modified.csv'),header=0)
baseball_test = baseball.iloc[np.arange(1, baseball.shape[0], 200)]
baseball.head(5)
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
      <th>Player Name</th>
      <th>playerID</th>
      <th>yearID</th>
      <th>stint</th>
      <th>teamID</th>
      <th>name</th>
      <th>park</th>
      <th>lgID</th>
      <th>G</th>
      <th>AB</th>
      <th>...</th>
      <th>birthCountry</th>
      <th>birthState</th>
      <th>birthDay</th>
      <th>birthMonth</th>
      <th>birthYear</th>
      <th>Weight</th>
      <th>Height</th>
      <th>bats</th>
      <th>200HitSeason</th>
      <th>Decade</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>Carroll</td>
      <td>carroch01</td>
      <td>1884-01-01</td>
      <td>1</td>
      <td>WSU</td>
      <td>Washington Nationals</td>
      <td>NaN</td>
      <td>UA</td>
      <td>4</td>
      <td>16</td>
      <td>...</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0.000000</td>
      <td>NaN</td>
      <td>0</td>
      <td>1800</td>
    </tr>
    <tr>
      <th>1</th>
      <td>Gavern</td>
      <td>gaver01</td>
      <td>1874-01-01</td>
      <td>1</td>
      <td>BR2</td>
      <td>Brooklyn Atlantics</td>
      <td>Union Grounds</td>
      <td>NaN</td>
      <td>1</td>
      <td>0</td>
      <td>...</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0.000000</td>
      <td>NaN</td>
      <td>0</td>
      <td>1800</td>
    </tr>
    <tr>
      <th>2</th>
      <td>McRemer</td>
      <td>mcrem01</td>
      <td>1884-01-01</td>
      <td>1</td>
      <td>WSU</td>
      <td>Washington Nationals</td>
      <td>NaN</td>
      <td>UA</td>
      <td>1</td>
      <td>0</td>
      <td>...</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0.000000</td>
      <td>NaN</td>
      <td>0</td>
      <td>1800</td>
    </tr>
    <tr>
      <th>3</th>
      <td>Sterling</td>
      <td>sterljo01</td>
      <td>1890-01-01</td>
      <td>1</td>
      <td>PH4</td>
      <td>Philadelphia Athletics</td>
      <td>Jefferson Street Grounds</td>
      <td>AA</td>
      <td>1</td>
      <td>0</td>
      <td>...</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0.000000</td>
      <td>NaN</td>
      <td>0</td>
      <td>1800</td>
    </tr>
    <tr>
      <th>4</th>
      <td>A. J. Ellis</td>
      <td>ellisaj01</td>
      <td>1/1/14</td>
      <td>1</td>
      <td>LAN</td>
      <td>Los Angeles Dodgers</td>
      <td>Dodger Stadium</td>
      <td>NL</td>
      <td>0</td>
      <td>0</td>
      <td>...</td>
      <td>USA</td>
      <td>MO</td>
      <td>9</td>
      <td>4</td>
      <td>1981</td>
      <td>0</td>
      <td>6.166666</td>
      <td>R</td>
      <td>0</td>
      <td>2020</td>
    </tr>
  </tbody>
</table>
<p>5 rows × 77 columns</p>
</div>



First things first, let's delete unnecessary columns.

**Exercise 0** (1 point): Complete the `remove_redundant_columns` function that takes dataframe `df` and returns a _**new**_ dataframe  _excluding_ the following columns:
- Any column with the word 'Career' in it's name
- Any column with the string 'birth' in it's name
- 200HitSeason, Decade, Player Name, name, park, lgID, bats



```python
def remove_redundant_columns(df):
    filter_cols = (
        list(df.filter(regex = 'Career|birth')) + 
        ['200HitSeason', 'Decade', 'Player Name', 'name', 'park', 'lgID', 'bats']
    )
    #return df[[x for x in df.columns if x not in filter_cols]]
    return df[df.columns.difference(filter_cols)]
```

To test your solution we have created a data frame `baseball_test` which is a much smaller sample of the orginial dataframe `baseball`. We also provided how the output for `remove_redundant_columns(baseball_test)` should look like in data-frame `df_ex0_soln_instructor`. 

**Note**: The below test case is designed just for the purpose of debugging and will not be used for grading. 


```python
# (0 Points) `remove_redundant_columns_dummy`: Test cell 1
from pandas.util.testing import assert_frame_equal

df_ex0_soln_instructor = canonicalize_tibble(pd.read_csv(get_data_path('ex0_soln.csv')))
df_ex0_soln_yours = canonicalize_tibble(remove_redundant_columns(baseball_test))

assert type(df_ex0_soln_yours) == type(df_ex0_soln_instructor), 'Your output does not return a pandas dataframe'
assert_frame_equal(df_ex0_soln_instructor, df_ex0_soln_yours)

print('Passed!')

del df_ex0_soln_instructor
del df_ex0_soln_yours
```

    Passed!
    

Testing your solution on the original dataframe



```python
# (1 Point) `remove_redundant_columns`: Test cell 2
assert tibbles_are_equivalent(remove_redundant_columns(baseball), -8278288771535832348), "Tibbles don't match!" 
print("Passed!")
```

    Passed!
    

Great! If the above test case passed then let's remove columns from the original dataset `df` using `remove_redundant_columns(baseball)`


```python
baseball_test = remove_redundant_columns(baseball_test)
baseball = remove_redundant_columns(baseball)
baseball.shape
```




    (101766, 37)



#### Shrinking the dataset.

**Exercise 1** (1 point). Write a function `shrink_data()` which takes the dataframe `df` and returns a **new** dataframe where:

* the column `yearID` should be converted to `pandas datetime` format; **and**
* only rows such that `yearID` is from 2000 (inclusive) to 2016 (inclusive) are returned.

> Hint: Regarding the first condition, see [`pandas.to_datetime()`](https://pandas.pydata.org/pandas-docs/stable/generated/pandas.to_datetime.html).


```python
def shrink_data(df):
    #df['yearID'] = pd.to_datetime(df['yearID'])
    df.yearID = pd.to_datetime(df.yearID)
    year = df.yearID.dt.year
    #return df[(year >= 2000) & (year <= 2016)]
    return df[year.isin(range(2000, 2017))]
```

The test below will test your solution on a the smaller sample `baseball_test`. 

**Note**: The below test case is designed just for the purpose of debugging and will not be used for grading. 


```python
# (0 Points) `shrink_data_dummy`:  Test cell 1

from pandas.util.testing import assert_frame_equal

df_ex1_soln_instructor = canonicalize_tibble(pd.read_csv(get_data_path('ex1_soln.csv'), parse_dates=['yearID']))
df_ex1_soln_yours = canonicalize_tibble(shrink_data(baseball_test))

assert type(df_ex1_soln_yours) == type(df_ex1_soln_instructor), 'Your output does not return a pandas dataframe'
assert_frame_equal(df_ex1_soln_instructor, df_ex1_soln_yours)

print('Passed!')
del df_ex1_soln_instructor
del df_ex1_soln_yours
```

    Passed!
    

Testing solution on original dataframe


```python
# (1 Points) `shrink_data`: Test cell
assert tibbles_are_equivalent(shrink_data(baseball), 1828659959833542576), "Tibbles don't match!" 
print("Passed!")
```

    Passed!
    

Let's shrink the orginial dataframe now.


```python
baseball_test = shrink_data(baseball_test)
baseball = shrink_data(baseball)
baseball.shape
```




    (31805, 37)



Several players appear in the dataset more than once. This is because they played for one team, then were traded or moved to another. Currently, the combination of `playerID`, `stint`, and `teamID` is unique within each row. We want to transform this into a dataset that contains the all _minimum_ characteristics of each player over all the teams and stints the player had.

**Exercise 2** (3 points). Complete the function `transform_baseball_data(df)` and return a **new** dataframe such that, for each unique player (`playerID`):
- only the earliest `yearID` is retained;
- only the _lowest_ value of every numerical column is retained;
- the columns `stint` and `teamID` are not retained;
- after tranformation the numerical values are rounded to nearest 10 and converted to integer. For example, the value 15.33 rounded to nearest 10 is 20, and the value 14.999 is rounded to 10. For cases like 5, 15, 25, etc., such a value `v` would be the same as that produced by `round(v, -1)` in standard Python.

The final data frame will have one row per unique player with the columns retained and transformed as outlined above.

A natural way to start is to group the data frame `df` by `playerID`. However, for your final result, be sure that `playerID` is **not** the index. (That is, be sure your final result is a tibble.)

> Hint: A relatively clean solution may be had by exploiting features of the [`.agg()` method](https://pandas.pydata.org/pandas-docs/stable/generated/pandas.core.groupby.DataFrameGroupBy.agg.html) available for `.groupby()` objects produced when called on `DataFrame` objects.


```python
def transform_baseball_data(df):
    function_mix = {x : 'min' for x in baseball.columns if x not in ['stint','teamID','playerID']}
    df_sub = df.groupby(['playerID'], as_index = False).agg(function_mix)
    select_col = [x for x in df_sub.columns if x not in ['playerID','yearID']]
    df_sub[select_col] = df_sub[select_col].round(-1).astype('int64')
    return df_sub
```

To test your solution we have provided solution for the dataframe `baseball_test` 

**Note**: The below test case is designed just to help you debug. 


```python
# (0 Points) `transform_baseball_data_dummy`: Test cell
from pandas.util.testing import assert_frame_equal

df_ex2_soln_instructor = canonicalize_tibble(pd.read_csv(get_data_path('ex2_soln.csv'), parse_dates=['yearID']))
df_ex2_soln_yours = canonicalize_tibble(transform_baseball_data(baseball_test))

assert type(df_ex2_soln_yours) == type(df_ex2_soln_instructor), 'Your output does not return a pandas dataframe'
assert_frame_equal(df_ex2_soln_instructor, df_ex2_soln_yours)

print('Passed!')
del df_ex2_soln_instructor
del df_ex2_soln_yours
```

    Passed!
    

Testing on original dataframe


```python
# (3 Points) `transform_baseball_data`: Test cell
assert tibbles_are_equivalent(baseball, 1828659959833542576), "Tibbles don't match!" 
print("Passed!")
```

    Passed!
    

Let's transform the original dataframe `baseball`


```python
baseball_test = transform_baseball_data(baseball_test)
baseball = transform_baseball_data(baseball)
baseball.shape
```




    (7403, 35)



## Faster matrix products

Now, we're going to change direction just a little. Don't worry, you don't need to know anything conceptual about what we're about to do; all you need to do is think of a faster way to apply the formula to generate its output based on the course material that's already been covered.

One way of calculating the variance matrix $\Sigma$ of a dataset is given by the formula

$$\Sigma=E(XX^T)-\mu\mu^T$$

where $X$ is an $n \times m$ matrix containing each data point and $\mu$ is an $n \times m$ matrix containing the column means of those data points.

For this exercise, we will simply be calculating the parameter given to the expectation $E(\cdot)$ function, $XX^T$. You can see in the cell below that this has already been done for you, but your task will be to figure out how to run `X.dot(X.T)` faster than we did.

First, run the code cell below to establish an estimate time for the output.


```python
import numpy as np
import timeit

X = baseball[[x for x in baseball.columns if x not in ['playerID','yearID']]].values

print("Matrix X is of size {}".format(X.shape))
```

    Matrix X is of size (7403, 33)
    


```python
def slow_calc(X):
    return X.dot(X.T)

slow_time = timeit.timeit("slow_calc(X)",setup="from __main__ import slow_calc, X", number = 1)
print("Your estimated time for X.dot(X.T) is "+' {0:.4f}'.format(slow_time) + " seconds.")
```

    Your estimated time for X.dot(X.T) is  1.6628 seconds.
    

**Exercise 3** (5 points): Come up with a way to make the matrix-times-transpose function faster than `slow_calc(X)`. Implement your method as the function `fast_calc()`. To get the full 5 points, your method must be **2.5 times faster**. You can get partial credit: 3 points if your function is at least 2 times faster and 1 point if your function is at least 1.5 times faster.

The input to `fast_calc` is of type `numpy.ndarray` and expected output is also of type `numpy.ndarray`

**Note**: The variable named `number_of_runs` determines how many times `fast_calc` will be run against the timer. You may lower `number_of_runs` for debugging purposes, but must increase it to at least 5 to pass the test cell. You may also import libraries you would like to use.

**The benchmarks for this question are set according to Vocareum environment.** You might get different results if you test on your system. So please test your results here.

> This exercise requires some creativity in thinking about how to exploit structure present in the problem.


```python
number_of_runs = 5

def fast_calc(X):
    from scipy import linalg as la
    return la.blas.sgemm(1.0, X, X.T)
```

Run this test to get on 1 point for a solution that is at least 1.5 times faster


```python
# (1 point) `speed_test_1`: Test cell
for i in range(5):
    nrows = np.random.randint(5) + 1
    ncols = np.random.randint(5) + 1
    A = np.random.rand(nrows, ncols)
    your_out = fast_calc(A)
    instructor_out = slow_calc(A)
    assert type(your_out) == type(A), "Please return object of type {}".format(type(A))
    np.testing.assert_array_almost_equal(instructor_out, your_out, decimal = 5)

slow_time = timeit.timeit("slow_calc(X)",setup="from __main__ import slow_calc, X", number = number_of_runs)/number_of_runs
student_time = timeit.timeit("fast_calc(X)", setup="from __main__ import fast_calc, X",number = number_of_runs)/number_of_runs
print("Your baseline time for X.dot(X.T) is "+'{0:.4f}'.format(student_time)+" seconds, which is "+'{0:.2f}'.format(slow_time/student_time)+ " times faster than our method.")
assert student_time/slow_time <= 0.75, "Your solution isn't at least 1.5 times faster than our solution."
assert number_of_runs >= 5, "number_of_runs needs to be >=5 to pass this cell."

print("Passed!")
```

    Your baseline time for X.dot(X.T) is 0.1526 seconds, which is 10.59 times faster than our method.
    Passed!
    

Run this test to get 2 points for a solution that is at least 2 times faster


```python
# (2 point) `speed_test_2`: Test cell
for i in range(5):
    nrows = np.random.randint(5) + 1
    ncols = np.random.randint(5) + 1
    A = np.random.rand(nrows, ncols)
    your_out = fast_calc(A)
    instructor_out = slow_calc(A)
    assert type(your_out) == type(A), "Please return object of type {}".format(type(A))
    np.testing.assert_array_almost_equal(instructor_out, your_out, decimal = 5)

slow_time = timeit.timeit("slow_calc(X)",setup="from __main__ import slow_calc, X", number = number_of_runs)/number_of_runs
student_time = timeit.timeit("fast_calc(X)", setup="from __main__ import fast_calc, X",number = number_of_runs)/number_of_runs
print("Your baseline time for X.dot(X.T) is "+'{0:.4f}'.format(student_time)+" seconds, which is "+'{0:.2f}'.format(slow_time/student_time)+ " times faster than our method.")
assert student_time/slow_time <= 0.50, "Your solution isn't at least 2 times faster than our solution."
assert number_of_runs >= 5, "number_of_runs needs to be >=5 to pass this cell."

print("Passed!")
```

    Your baseline time for X.dot(X.T) is 0.1564 seconds, which is 10.20 times faster than our method.
    Passed!
    

Run this test to get 2 points for a solution that is at least 2.5 times faster


```python
# (2 point) `speed_test_3`: Test cell
for i in range(5):
    nrows = np.random.randint(5) + 1
    ncols = np.random.randint(5) + 1
    A = np.random.rand(nrows, ncols)
    your_out = fast_calc(A)
    instructor_out = slow_calc(A)
    assert type(your_out) == type(A), "Please return object of type {}".format(type(A))
    np.testing.assert_array_almost_equal(instructor_out, your_out, decimal = 5)

slow_time = timeit.timeit("slow_calc(X)",setup="from __main__ import slow_calc, X", number = number_of_runs)/number_of_runs
student_time = timeit.timeit("fast_calc(X)", setup="from __main__ import fast_calc, X",number = number_of_runs)/number_of_runs
print("Your baseline time for X.dot(X.T) is "+'{0:.4f}'.format(student_time)+" seconds, which is "+'{0:.2f}'.format(slow_time/student_time)+ " times faster than our method.")
assert student_time/slow_time <= 0.40, "Your solution isn't at least 2.5 times faster than our solution."
assert number_of_runs >= 5, "number_of_runs needs to be >=5 to pass this cell."

print("Passed!")
```

    Your baseline time for X.dot(X.T) is 0.1539 seconds, which is 10.41 times faster than our method.
    Passed!
    

**Fin!** Remember to test your solutions by running them as the autograder will: restart the kernel and run all cells from "top-to-bottom." Also remember to submit to the autograder; otherwise, you will **not** get credit for your hard work!
