# Part 0: Getting the data

Before beginning, you'll need to download several files containing the data for the exercises below.

**Exercise 0** (ungraded). Run the code cell below to download the data. (This code will check if each dataset has already been downloaded and, if so, will avoid re-downloading it.)


```python
dataset = {'iris.csv': 'd1175c032e1042bec7f974c91e4a65ae',
           'table1.csv': '556ffe73363752488d6b41462f5ff3c9',
           'table2.csv': '16e04efbc7122e515f7a81a3361e6b87',
           'table3.csv': '531d13889f191d6c07c27c3c7ea035ff',
           'table4a.csv': '3c0bbecb40c6958df33a1f9aa5629a80',
           'table4b.csv': '8484bcdf07b50a7e0932099daa72a93d',
           'who.csv': '59fed6bbce66349bf00244b550a93544',
           'who2_soln.csv': 'f6d4875feea9d6fca82ae7f87f760f44',
           'who3_soln.csv': 'fba14f1e088d871e4407f5f737cfbc06'}

from cse6040utils import download_dataset
local_data = download_dataset(dataset, url_suffix='tidy/')

print("\n(All data appear to be ready.)")
```

    'iris.csv' is ready!
    'table1.csv' is ready!
    'table2.csv' is ready!
    'table3.csv' is ready!
    'table4a.csv' is ready!
    'table4b.csv' is ready!
    'who.csv' is ready!
    'who2_soln.csv' is ready!
    'who3_soln.csv' is ready!
    
    (All data appear to be ready.)
    

# Part 1: Tidy data 

The overall topic for this lab is what we'll refer to as representing data _relationally_. The topic of this part is a specific type of relational representation sometimes referred to as the _tidy_ (as opposed to _untidy_ or _messy_) form. The concept of tidy data was developed by [Hadley Wickham](http://hadley.nz/), a statistician and R programming maestro. Much of this lab is based on his tutorial materials (see below).

If you know [SQL](https://en.wikipedia.org/wiki/SQL), then you are already familiar with relational data representations. However, we might discuss it a little differently from the way you may have encountered the subject previously. The main reason is our overall goal in the class: to build data _analysis_ pipelines. If our end goal is analysis, then we often want to extract or prepare data in a way that makes analysis easier.

You may find it helpful to also refer to the original materials on which this lab is based:

* Wickham's R tutorial on making data tidy: http://r4ds.had.co.nz/tidy-data.html
* The slides from a talk by Wickham on the concept: http://vita.had.co.nz/papers/tidy-data-pres.pdf
* Wickham's more theoretical paper of "tidy" vs. "untidy" data: http://www.jstatsoft.org/v59/i10/paper

------------------------------------------------------------

## What is tidy data?

To build your intuition, consider the following data set collected from a survey or study.

**Representation 1.** [Two-way contigency table](https://en.wikipedia.org/wiki/Contingency_table).

|            | Pregnant | Not pregnant |
|-----------:|:--------:|:------------:|
| **Male**   |     0    |      5       |
| **Female** |     1    |      4       |

**Representation 2.** Observation list or "data frame."

| Gender  | Pregnant | Count |
|:-------:|:--------:|:-----:|
| Male    | Yes      | 0     |
| Male    | No       | 5     |
| Female  | Yes      | 1     |
| Female  | No       | 4     |

These are two entirely equivalent ways of representing the same data. However, each may be suited to a particular task.

For instance, Representation 1 is a typical input format for statistical routines that implement Pearson's $\chi^2$-test, which can check for independence between factors. (Are gender and pregnancy status independent?) By contrast, Representation 2 might be better suited to regression. (Can you predict relative counts from gender and pregnancy status?)

While [Representation 1 has its uses](http://simplystatistics.org/2016/02/17/non-tidy-data/), Wickham argues that Representation 2 is often the cleaner and more general way to supply data to a wide variety of statistical analysis and visualization tasks. He refers to Representation 2 as _tidy_ and Representation 1 as _untidy_ or _messy_.

> The term "messy" is, as Wickham states, not intended to be perjorative since "messy" representations may be exactly the right ones for particular analysis tasks, as noted above.

**Definition: Tidy datasets.** More specifically, Wickham defines a tidy data set as one that can be organized into a 2-D table such that

1. each column represents a _variable_;
2. each row represents an _observation_;
3. each entry of the table represents a single _value_, which may come from either categorical (discrete) or continuous spaces.

Here is a visual schematic of this definition, taken from [another source](http://r4ds.had.co.nz/images/tidy-1.png):

![Wickham's illustration of the definition of tidy](http://r4ds.had.co.nz/images/tidy-1.png)

This definition appeals to a statistician's intuitive idea of data he or she wishes to analyze. It is also consistent with tasks that seek to establish a functional relationship between some response (output) variable from one or more independent variables.

> A computer scientist with a machine learning outlook might refer to columns as _features_ and rows as _data points_, especially when all values are numerical (ordinal or continuous).

**Definition: Tibbles.** Here's one more bit of terminology: if a table is tidy, we will call it a _tidy table_, or _tibble_, for short.

## Part 2: Tidy Basics and Pandas

In Python, the [Pandas](http://pandas.pydata.org/) module is a convenient way to store tibbles. If you know [R](http://r-project.org), you will see that the design and API of Pandas's data frames derives from [R's data frames](https://stat.ethz.ch/R-manual/R-devel/library/base/html/data.frame.html).

In this part of this notebook, let's look at how Pandas works and can help us store Tidy data.

You may find this introduction to the Pandas module's data structures useful for reference:

* https://pandas.pydata.org/pandas-docs/stable/dsintro.html

Consider the famous [Iris data set](https://en.wikipedia.org/wiki/Iris_flower_data_set). It consists of 50 samples from each of three species of Iris (_Iris setosa_, _Iris virginica_, and _Iris versicolor_). Four features were measured from each sample: the lengths and the widths of the [sepals](https://en.wikipedia.org/wiki/Sepal) and [petals](https://en.wikipedia.org/wiki/Petal).

The following code uses Pandas to read and represent this data in a Pandas data frame object, stored in a variable named `irises`.


```python
# Some modules you'll need in this part
import pandas as pd
from io import StringIO
from IPython.display import display

# Ignore this line. It will be used later.
SAVE_APPLY = getattr(pd.DataFrame, 'apply')

irises = pd.read_csv(local_data['iris.csv'])
print("=== Iris data set: {} rows x {} columns. ===".format(irises.shape[0], irises.shape[1]))
display (irises.head())
```

    === Iris data set: 150 rows x 5 columns. ===
    


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
      <th>sepal length</th>
      <th>sepal width</th>
      <th>petal length</th>
      <th>petal width</th>
      <th>species</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>5.1</td>
      <td>3.5</td>
      <td>1.4</td>
      <td>0.2</td>
      <td>Iris-setosa</td>
    </tr>
    <tr>
      <th>1</th>
      <td>4.9</td>
      <td>3.0</td>
      <td>1.4</td>
      <td>0.2</td>
      <td>Iris-setosa</td>
    </tr>
    <tr>
      <th>2</th>
      <td>4.7</td>
      <td>3.2</td>
      <td>1.3</td>
      <td>0.2</td>
      <td>Iris-setosa</td>
    </tr>
    <tr>
      <th>3</th>
      <td>4.6</td>
      <td>3.1</td>
      <td>1.5</td>
      <td>0.2</td>
      <td>Iris-setosa</td>
    </tr>
    <tr>
      <th>4</th>
      <td>5.0</td>
      <td>3.6</td>
      <td>1.4</td>
      <td>0.2</td>
      <td>Iris-setosa</td>
    </tr>
  </tbody>
</table>
</div>


In a Pandas data frame, every column has a name (stored as a string) and all values within the column must have the same primitive type. This fact makes columns different from, for instance, lists.

In addition, every row has a special column, called the data frame's _index_. (Try printing `irises.index`.) Any particular index value serves as a name for its row; these index values are usually integers but can be more complex types, like tuples.


```python
print(irises.index)
```

    RangeIndex(start=0, stop=150, step=1)
    

Separate from the index values (row names), you can also refer to rows by their integer offset from the top, where the first row has an offset of 0 and the last row has an offset of `n-1` if the data frame has `n` rows. You'll see that in action in Exercise 1, below.

**Exercise 1** (ungraded). Run the following commands to understand what each one does. If it's not obvious, try reading the [Pandas documentation](http://pandas.pydata.org/) or going online to get more information.

```python
irises.describe()
irises['sepal length'].head()
irises[["sepal length", "petal width"]].head()
irises.iloc[5:10]
irises[irises["sepal length"] > 5.0]
irises["sepal length"].max()
irises['species'].unique()
irises.sort_values(by="sepal length", ascending=False).head(1)
irises.sort_values(by="sepal length", ascending=False).iloc[5:10]
irises.sort_values(by="sepal length", ascending=False).loc[5:10]
irises['x'] = 3.14
irises.rename(columns={'species': 'type'})
del irises['x']
```


```python
irises.describe()
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
      <th>sepal length</th>
      <th>sepal width</th>
      <th>petal length</th>
      <th>petal width</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>count</th>
      <td>150.000000</td>
      <td>150.000000</td>
      <td>150.000000</td>
      <td>150.000000</td>
    </tr>
    <tr>
      <th>mean</th>
      <td>5.843333</td>
      <td>3.057333</td>
      <td>3.758000</td>
      <td>1.199333</td>
    </tr>
    <tr>
      <th>std</th>
      <td>0.828066</td>
      <td>0.435866</td>
      <td>1.765298</td>
      <td>0.762238</td>
    </tr>
    <tr>
      <th>min</th>
      <td>4.300000</td>
      <td>2.000000</td>
      <td>1.000000</td>
      <td>0.100000</td>
    </tr>
    <tr>
      <th>25%</th>
      <td>5.100000</td>
      <td>2.800000</td>
      <td>1.600000</td>
      <td>0.300000</td>
    </tr>
    <tr>
      <th>50%</th>
      <td>5.800000</td>
      <td>3.000000</td>
      <td>4.350000</td>
      <td>1.300000</td>
    </tr>
    <tr>
      <th>75%</th>
      <td>6.400000</td>
      <td>3.300000</td>
      <td>5.100000</td>
      <td>1.800000</td>
    </tr>
    <tr>
      <th>max</th>
      <td>7.900000</td>
      <td>4.400000</td>
      <td>6.900000</td>
      <td>2.500000</td>
    </tr>
  </tbody>
</table>
</div>




```python
irises['sepal length'].head()
```




    0    5.1
    1    4.9
    2    4.7
    3    4.6
    4    5.0
    Name: sepal length, dtype: float64




```python
irises[["sepal length", "petal width"]].head()
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
      <th>sepal length</th>
      <th>petal width</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>5.1</td>
      <td>0.2</td>
    </tr>
    <tr>
      <th>1</th>
      <td>4.9</td>
      <td>0.2</td>
    </tr>
    <tr>
      <th>2</th>
      <td>4.7</td>
      <td>0.2</td>
    </tr>
    <tr>
      <th>3</th>
      <td>4.6</td>
      <td>0.2</td>
    </tr>
    <tr>
      <th>4</th>
      <td>5.0</td>
      <td>0.2</td>
    </tr>
  </tbody>
</table>
</div>




```python
irises.iloc[5:10]
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
      <th>sepal length</th>
      <th>sepal width</th>
      <th>petal length</th>
      <th>petal width</th>
      <th>species</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>5</th>
      <td>5.4</td>
      <td>3.9</td>
      <td>1.7</td>
      <td>0.4</td>
      <td>Iris-setosa</td>
    </tr>
    <tr>
      <th>6</th>
      <td>4.6</td>
      <td>3.4</td>
      <td>1.4</td>
      <td>0.3</td>
      <td>Iris-setosa</td>
    </tr>
    <tr>
      <th>7</th>
      <td>5.0</td>
      <td>3.4</td>
      <td>1.5</td>
      <td>0.2</td>
      <td>Iris-setosa</td>
    </tr>
    <tr>
      <th>8</th>
      <td>4.4</td>
      <td>2.9</td>
      <td>1.4</td>
      <td>0.2</td>
      <td>Iris-setosa</td>
    </tr>
    <tr>
      <th>9</th>
      <td>4.9</td>
      <td>3.1</td>
      <td>1.5</td>
      <td>0.1</td>
      <td>Iris-setosa</td>
    </tr>
  </tbody>
</table>
</div>




```python
irises["sepal length"] > 5.0
```




    0       True
    1      False
    2      False
    3      False
    4      False
           ...  
    145     True
    146     True
    147     True
    148     True
    149     True
    Name: sepal length, Length: 150, dtype: bool




```python
irises[irises["sepal length"] > 5.0]
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
      <th>sepal length</th>
      <th>sepal width</th>
      <th>petal length</th>
      <th>petal width</th>
      <th>species</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>5.1</td>
      <td>3.5</td>
      <td>1.4</td>
      <td>0.2</td>
      <td>Iris-setosa</td>
    </tr>
    <tr>
      <th>5</th>
      <td>5.4</td>
      <td>3.9</td>
      <td>1.7</td>
      <td>0.4</td>
      <td>Iris-setosa</td>
    </tr>
    <tr>
      <th>10</th>
      <td>5.4</td>
      <td>3.7</td>
      <td>1.5</td>
      <td>0.2</td>
      <td>Iris-setosa</td>
    </tr>
    <tr>
      <th>14</th>
      <td>5.8</td>
      <td>4.0</td>
      <td>1.2</td>
      <td>0.2</td>
      <td>Iris-setosa</td>
    </tr>
    <tr>
      <th>15</th>
      <td>5.7</td>
      <td>4.4</td>
      <td>1.5</td>
      <td>0.4</td>
      <td>Iris-setosa</td>
    </tr>
    <tr>
      <th>...</th>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
    </tr>
    <tr>
      <th>145</th>
      <td>6.7</td>
      <td>3.0</td>
      <td>5.2</td>
      <td>2.3</td>
      <td>Iris-virginica</td>
    </tr>
    <tr>
      <th>146</th>
      <td>6.3</td>
      <td>2.5</td>
      <td>5.0</td>
      <td>1.9</td>
      <td>Iris-virginica</td>
    </tr>
    <tr>
      <th>147</th>
      <td>6.5</td>
      <td>3.0</td>
      <td>5.2</td>
      <td>2.0</td>
      <td>Iris-virginica</td>
    </tr>
    <tr>
      <th>148</th>
      <td>6.2</td>
      <td>3.4</td>
      <td>5.4</td>
      <td>2.3</td>
      <td>Iris-virginica</td>
    </tr>
    <tr>
      <th>149</th>
      <td>5.9</td>
      <td>3.0</td>
      <td>5.1</td>
      <td>1.8</td>
      <td>Iris-virginica</td>
    </tr>
  </tbody>
</table>
<p>118 rows × 5 columns</p>
</div>




```python
irises["sepal length"].max()
```




    7.9




```python
list(irises['species'].unique())
```




    ['Iris-setosa', 'Iris-versicolor', 'Iris-virginica']




```python
irises.sort_values(by="sepal length", ascending=False).head(1)
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
      <th>sepal length</th>
      <th>sepal width</th>
      <th>petal length</th>
      <th>petal width</th>
      <th>species</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>131</th>
      <td>7.9</td>
      <td>3.8</td>
      <td>6.4</td>
      <td>2.0</td>
      <td>Iris-virginica</td>
    </tr>
  </tbody>
</table>
</div>




```python
irises.sort_values(by="sepal length", ascending=False).iloc[5:10]
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
      <th>sepal length</th>
      <th>sepal width</th>
      <th>petal length</th>
      <th>petal width</th>
      <th>species</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>105</th>
      <td>7.6</td>
      <td>3.0</td>
      <td>6.6</td>
      <td>2.1</td>
      <td>Iris-virginica</td>
    </tr>
    <tr>
      <th>130</th>
      <td>7.4</td>
      <td>2.8</td>
      <td>6.1</td>
      <td>1.9</td>
      <td>Iris-virginica</td>
    </tr>
    <tr>
      <th>107</th>
      <td>7.3</td>
      <td>2.9</td>
      <td>6.3</td>
      <td>1.8</td>
      <td>Iris-virginica</td>
    </tr>
    <tr>
      <th>125</th>
      <td>7.2</td>
      <td>3.2</td>
      <td>6.0</td>
      <td>1.8</td>
      <td>Iris-virginica</td>
    </tr>
    <tr>
      <th>109</th>
      <td>7.2</td>
      <td>3.6</td>
      <td>6.1</td>
      <td>2.5</td>
      <td>Iris-virginica</td>
    </tr>
  </tbody>
</table>
</div>




```python
irises.sort_values(by="sepal length", ascending=False).loc[5:10]
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
      <th>sepal length</th>
      <th>sepal width</th>
      <th>petal length</th>
      <th>petal width</th>
      <th>species</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>5</th>
      <td>5.4</td>
      <td>3.9</td>
      <td>1.7</td>
      <td>0.4</td>
      <td>Iris-setosa</td>
    </tr>
    <tr>
      <th>10</th>
      <td>5.4</td>
      <td>3.7</td>
      <td>1.5</td>
      <td>0.2</td>
      <td>Iris-setosa</td>
    </tr>
  </tbody>
</table>
</div>




```python
irises['x'] = 3.14
```


```python
irises.rename(columns={'species': 'type'})
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
      <th>sepal length</th>
      <th>sepal width</th>
      <th>petal length</th>
      <th>petal width</th>
      <th>type</th>
      <th>x</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>5.1</td>
      <td>3.5</td>
      <td>1.4</td>
      <td>0.2</td>
      <td>Iris-setosa</td>
      <td>3.14</td>
    </tr>
    <tr>
      <th>1</th>
      <td>4.9</td>
      <td>3.0</td>
      <td>1.4</td>
      <td>0.2</td>
      <td>Iris-setosa</td>
      <td>3.14</td>
    </tr>
    <tr>
      <th>2</th>
      <td>4.7</td>
      <td>3.2</td>
      <td>1.3</td>
      <td>0.2</td>
      <td>Iris-setosa</td>
      <td>3.14</td>
    </tr>
    <tr>
      <th>3</th>
      <td>4.6</td>
      <td>3.1</td>
      <td>1.5</td>
      <td>0.2</td>
      <td>Iris-setosa</td>
      <td>3.14</td>
    </tr>
    <tr>
      <th>4</th>
      <td>5.0</td>
      <td>3.6</td>
      <td>1.4</td>
      <td>0.2</td>
      <td>Iris-setosa</td>
      <td>3.14</td>
    </tr>
    <tr>
      <th>...</th>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
    </tr>
    <tr>
      <th>145</th>
      <td>6.7</td>
      <td>3.0</td>
      <td>5.2</td>
      <td>2.3</td>
      <td>Iris-virginica</td>
      <td>3.14</td>
    </tr>
    <tr>
      <th>146</th>
      <td>6.3</td>
      <td>2.5</td>
      <td>5.0</td>
      <td>1.9</td>
      <td>Iris-virginica</td>
      <td>3.14</td>
    </tr>
    <tr>
      <th>147</th>
      <td>6.5</td>
      <td>3.0</td>
      <td>5.2</td>
      <td>2.0</td>
      <td>Iris-virginica</td>
      <td>3.14</td>
    </tr>
    <tr>
      <th>148</th>
      <td>6.2</td>
      <td>3.4</td>
      <td>5.4</td>
      <td>2.3</td>
      <td>Iris-virginica</td>
      <td>3.14</td>
    </tr>
    <tr>
      <th>149</th>
      <td>5.9</td>
      <td>3.0</td>
      <td>5.1</td>
      <td>1.8</td>
      <td>Iris-virginica</td>
      <td>3.14</td>
    </tr>
  </tbody>
</table>
<p>150 rows × 6 columns</p>
</div>




```python
del irises['x']
```


```python
irises.iloc[0:5]
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
      <th>sepal length</th>
      <th>sepal width</th>
      <th>petal length</th>
      <th>petal width</th>
      <th>species</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>5.1</td>
      <td>3.5</td>
      <td>1.4</td>
      <td>0.2</td>
      <td>Iris-setosa</td>
    </tr>
    <tr>
      <th>1</th>
      <td>4.9</td>
      <td>3.0</td>
      <td>1.4</td>
      <td>0.2</td>
      <td>Iris-setosa</td>
    </tr>
    <tr>
      <th>2</th>
      <td>4.7</td>
      <td>3.2</td>
      <td>1.3</td>
      <td>0.2</td>
      <td>Iris-setosa</td>
    </tr>
    <tr>
      <th>3</th>
      <td>4.6</td>
      <td>3.1</td>
      <td>1.5</td>
      <td>0.2</td>
      <td>Iris-setosa</td>
    </tr>
    <tr>
      <th>4</th>
      <td>5.0</td>
      <td>3.6</td>
      <td>1.4</td>
      <td>0.2</td>
      <td>Iris-setosa</td>
    </tr>
  </tbody>
</table>
</div>



## Merging data frames: join operations

Another useful operation on data frames is [merging](http://pandas.pydata.org/pandas-docs/stable/generated/pandas.DataFrame.merge.html).

For instance, consider the following two tables, `A` and `B`:

| country     | year | cases  |
|:------------|-----:|-------:|
| Afghanistan | 1999 |    745 |
| Brazil      | 1999 |  37737 |
| China       | 1999 | 212258 |
| Afghanistan | 2000 |   2666 |
| Brazil      | 2000 |  80488 |
| China       | 2000 | 213766 |

| country     | year | population |
|:------------|-----:|-----------:|
| Afghanistan | 1999 |   19987071 |
| Brazil      | 1999 |  172006362 |
| China       | 1999 | 1272915272 |
| Afghanistan | 2000 |   20595360 |
| Brazil      | 2000 |  174504898 |
| China       | 2000 | 1280428583 |

Suppose we wish to combine these into a single table, `C`:

| country     | year | cases  | population |
|:------------|-----:|-------:|-----------:|
| Afghanistan | 1999 |    745 |   19987071 |
| Brazil      | 1999 |  37737 |  172006362 |
| China       | 1999 | 212258 | 1272915272 |
| Afghanistan | 2000 |   2666 |   20595360 |
| Brazil      | 2000 |  80488 |  174504898 |
| China       | 2000 | 213766 | 1280428583 |

In Pandas, you can perform this merge using the [`.merge()` function](http://pandas.pydata.org/pandas-docs/stable/generated/pandas.DataFrame.merge.html):

```python
C = A.merge (B, on=['country', 'year'])
```

In this call, the `on=` parameter specifies the list of column names to use to align or "match" the two tables, `A` and `B`. By default, `merge()` will only include rows from `A` and `B` where all keys match between the two tables.

The following code cell demonstrates this functionality.


```python
A_csv = """country,year,cases
Afghanistan,1999,745
Brazil,1999,37737
China,1999,212258
Afghanistan,2000,2666
Brazil,2000,80488
China,2000,213766"""

with StringIO(A_csv) as fp:
    A = pd.read_csv(fp)
print("=== A ===")
display(A)
```

    === A ===
    


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
      <th>country</th>
      <th>year</th>
      <th>cases</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>Afghanistan</td>
      <td>1999</td>
      <td>745</td>
    </tr>
    <tr>
      <th>1</th>
      <td>Brazil</td>
      <td>1999</td>
      <td>37737</td>
    </tr>
    <tr>
      <th>2</th>
      <td>China</td>
      <td>1999</td>
      <td>212258</td>
    </tr>
    <tr>
      <th>3</th>
      <td>Afghanistan</td>
      <td>2000</td>
      <td>2666</td>
    </tr>
    <tr>
      <th>4</th>
      <td>Brazil</td>
      <td>2000</td>
      <td>80488</td>
    </tr>
    <tr>
      <th>5</th>
      <td>China</td>
      <td>2000</td>
      <td>213766</td>
    </tr>
  </tbody>
</table>
</div>



```python
B_csv = """country,year,population
Afghanistan,1999,19987071
Brazil,1999,172006362
China,1999,1272915272
Afghanistan,2000,20595360
Brazil,2000,174504898
China,2000,1280428583"""

with StringIO(B_csv) as fp:
    B = pd.read_csv(fp)
print("\n=== B ===")
display(B)
```

    
    === B ===
    


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
      <th>country</th>
      <th>year</th>
      <th>population</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>Afghanistan</td>
      <td>1999</td>
      <td>19987071</td>
    </tr>
    <tr>
      <th>1</th>
      <td>Brazil</td>
      <td>1999</td>
      <td>172006362</td>
    </tr>
    <tr>
      <th>2</th>
      <td>China</td>
      <td>1999</td>
      <td>1272915272</td>
    </tr>
    <tr>
      <th>3</th>
      <td>Afghanistan</td>
      <td>2000</td>
      <td>20595360</td>
    </tr>
    <tr>
      <th>4</th>
      <td>Brazil</td>
      <td>2000</td>
      <td>174504898</td>
    </tr>
    <tr>
      <th>5</th>
      <td>China</td>
      <td>2000</td>
      <td>1280428583</td>
    </tr>
  </tbody>
</table>
</div>



```python
C = A.merge(B, on=['country', 'year'])
print("\n=== C = merge(A, B) ===")
display(C)
```

    
    === C = merge(A, B) ===
    


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
      <th>country</th>
      <th>year</th>
      <th>cases</th>
      <th>population</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>Afghanistan</td>
      <td>1999</td>
      <td>745</td>
      <td>19987071</td>
    </tr>
    <tr>
      <th>1</th>
      <td>Brazil</td>
      <td>1999</td>
      <td>37737</td>
      <td>172006362</td>
    </tr>
    <tr>
      <th>2</th>
      <td>China</td>
      <td>1999</td>
      <td>212258</td>
      <td>1272915272</td>
    </tr>
    <tr>
      <th>3</th>
      <td>Afghanistan</td>
      <td>2000</td>
      <td>2666</td>
      <td>20595360</td>
    </tr>
    <tr>
      <th>4</th>
      <td>Brazil</td>
      <td>2000</td>
      <td>80488</td>
      <td>174504898</td>
    </tr>
    <tr>
      <th>5</th>
      <td>China</td>
      <td>2000</td>
      <td>213766</td>
      <td>1280428583</td>
    </tr>
  </tbody>
</table>
</div>


**Joins.** This default behavior of keeping only rows that match both input frames is an example of what relational database systems call an _inner-join_ operation. But there are several other types of joins.

- _Inner-join (`A`, `B`)_ (default): Keep only rows of `A` and `B` where the on-keys match in both.
- _Outer-join (`A`, `B`)_: Keep all rows of both frames, but merge rows when the on-keys match. For non-matches, fill in missing values with not-a-number (`NaN`) values.
- _Left-join (`A`, `B`)_: Keep all rows of `A`. Only merge rows of `B` whose on-keys match `A`.
- _Right-join (`A`, `B`)_: Keep all rows of `B`. Only merge rows of `A` whose on-keys match `B`.

You can use `merge`'s `how=...` parameter, which takes the (string) values, `'inner`', `'outer'`, `'left'`, and `'right'`. Here are some examples of these types of joins.


```python
with StringIO("""x,y,z
bug,1,d
rug,2,d
lug,3,d
mug,4,d""") as fp:
    D = pd.read_csv(fp)
print("=== D ===")
display(D)

with StringIO("""x,y,w
hug,-1,e
smug,-2,e
rug,-3,e
tug,-4,e
bug,1,e""") as fp:
    E = pd.read_csv(fp)
print("\n=== E ===")
display(E)

print("\n=== Outer-join (D, E) ===")
display(D.merge(E, on=['x', 'y'], how='outer'))

print("\n=== Left-join (D, E) ===")
display(D.merge(E, on=['x', 'y'], how='left'))

print("\n=== Right-join (D, E) ===")
display(D.merge(E, on=['x', 'y'], how='right'))


print("\n=== Inner-join (D, E) ===")
display(D.merge(E, on=['x', 'y']))
```

    === D ===
    


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
      <th>x</th>
      <th>y</th>
      <th>z</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>bug</td>
      <td>1</td>
      <td>d</td>
    </tr>
    <tr>
      <th>1</th>
      <td>rug</td>
      <td>2</td>
      <td>d</td>
    </tr>
    <tr>
      <th>2</th>
      <td>lug</td>
      <td>3</td>
      <td>d</td>
    </tr>
    <tr>
      <th>3</th>
      <td>mug</td>
      <td>4</td>
      <td>d</td>
    </tr>
  </tbody>
</table>
</div>


    
    === E ===
    


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
      <th>x</th>
      <th>y</th>
      <th>w</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>hug</td>
      <td>-1</td>
      <td>e</td>
    </tr>
    <tr>
      <th>1</th>
      <td>smug</td>
      <td>-2</td>
      <td>e</td>
    </tr>
    <tr>
      <th>2</th>
      <td>rug</td>
      <td>-3</td>
      <td>e</td>
    </tr>
    <tr>
      <th>3</th>
      <td>tug</td>
      <td>-4</td>
      <td>e</td>
    </tr>
    <tr>
      <th>4</th>
      <td>bug</td>
      <td>1</td>
      <td>e</td>
    </tr>
  </tbody>
</table>
</div>


    
    === Outer-join (D, E) ===
    


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
      <th>x</th>
      <th>y</th>
      <th>z</th>
      <th>w</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>bug</td>
      <td>1</td>
      <td>d</td>
      <td>e</td>
    </tr>
    <tr>
      <th>1</th>
      <td>rug</td>
      <td>2</td>
      <td>d</td>
      <td>NaN</td>
    </tr>
    <tr>
      <th>2</th>
      <td>lug</td>
      <td>3</td>
      <td>d</td>
      <td>NaN</td>
    </tr>
    <tr>
      <th>3</th>
      <td>mug</td>
      <td>4</td>
      <td>d</td>
      <td>NaN</td>
    </tr>
    <tr>
      <th>4</th>
      <td>hug</td>
      <td>-1</td>
      <td>NaN</td>
      <td>e</td>
    </tr>
    <tr>
      <th>5</th>
      <td>smug</td>
      <td>-2</td>
      <td>NaN</td>
      <td>e</td>
    </tr>
    <tr>
      <th>6</th>
      <td>rug</td>
      <td>-3</td>
      <td>NaN</td>
      <td>e</td>
    </tr>
    <tr>
      <th>7</th>
      <td>tug</td>
      <td>-4</td>
      <td>NaN</td>
      <td>e</td>
    </tr>
  </tbody>
</table>
</div>


    
    === Left-join (D, E) ===
    


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
      <th>x</th>
      <th>y</th>
      <th>z</th>
      <th>w</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>bug</td>
      <td>1</td>
      <td>d</td>
      <td>e</td>
    </tr>
    <tr>
      <th>1</th>
      <td>rug</td>
      <td>2</td>
      <td>d</td>
      <td>NaN</td>
    </tr>
    <tr>
      <th>2</th>
      <td>lug</td>
      <td>3</td>
      <td>d</td>
      <td>NaN</td>
    </tr>
    <tr>
      <th>3</th>
      <td>mug</td>
      <td>4</td>
      <td>d</td>
      <td>NaN</td>
    </tr>
  </tbody>
</table>
</div>


    
    === Right-join (D, E) ===
    


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
      <th>x</th>
      <th>y</th>
      <th>z</th>
      <th>w</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>bug</td>
      <td>1</td>
      <td>d</td>
      <td>e</td>
    </tr>
    <tr>
      <th>1</th>
      <td>hug</td>
      <td>-1</td>
      <td>NaN</td>
      <td>e</td>
    </tr>
    <tr>
      <th>2</th>
      <td>smug</td>
      <td>-2</td>
      <td>NaN</td>
      <td>e</td>
    </tr>
    <tr>
      <th>3</th>
      <td>rug</td>
      <td>-3</td>
      <td>NaN</td>
      <td>e</td>
    </tr>
    <tr>
      <th>4</th>
      <td>tug</td>
      <td>-4</td>
      <td>NaN</td>
      <td>e</td>
    </tr>
  </tbody>
</table>
</div>


    
    === Inner-join (D, E) ===
    


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
      <th>x</th>
      <th>y</th>
      <th>z</th>
      <th>w</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>bug</td>
      <td>1</td>
      <td>d</td>
      <td>e</td>
    </tr>
  </tbody>
</table>
</div>


## Apply functions to data frames

Another useful primitive is `apply()`, which can apply a function to a data frame or to a series (column of the data frame).

For instance, suppose we wish to convert the year column in `C` into an abbrievated two-digit form. The following code will do it:


```python
display(C)
G = C.copy() # If you do not use copy function the original data frame is modified
G['year'] = G['year'].apply(lambda x: "'{:02d}".format(x % 100))
display(G)
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
      <th>country</th>
      <th>year</th>
      <th>cases</th>
      <th>population</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>Afghanistan</td>
      <td>1999</td>
      <td>745</td>
      <td>19987071</td>
    </tr>
    <tr>
      <th>1</th>
      <td>Brazil</td>
      <td>1999</td>
      <td>37737</td>
      <td>172006362</td>
    </tr>
    <tr>
      <th>2</th>
      <td>China</td>
      <td>1999</td>
      <td>212258</td>
      <td>1272915272</td>
    </tr>
    <tr>
      <th>3</th>
      <td>Afghanistan</td>
      <td>2000</td>
      <td>2666</td>
      <td>20595360</td>
    </tr>
    <tr>
      <th>4</th>
      <td>Brazil</td>
      <td>2000</td>
      <td>80488</td>
      <td>174504898</td>
    </tr>
    <tr>
      <th>5</th>
      <td>China</td>
      <td>2000</td>
      <td>213766</td>
      <td>1280428583</td>
    </tr>
  </tbody>
</table>
</div>



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
      <th>country</th>
      <th>year</th>
      <th>cases</th>
      <th>population</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>Afghanistan</td>
      <td>'99</td>
      <td>745</td>
      <td>19987071</td>
    </tr>
    <tr>
      <th>1</th>
      <td>Brazil</td>
      <td>'99</td>
      <td>37737</td>
      <td>172006362</td>
    </tr>
    <tr>
      <th>2</th>
      <td>China</td>
      <td>'99</td>
      <td>212258</td>
      <td>1272915272</td>
    </tr>
    <tr>
      <th>3</th>
      <td>Afghanistan</td>
      <td>'00</td>
      <td>2666</td>
      <td>20595360</td>
    </tr>
    <tr>
      <th>4</th>
      <td>Brazil</td>
      <td>'00</td>
      <td>80488</td>
      <td>174504898</td>
    </tr>
    <tr>
      <th>5</th>
      <td>China</td>
      <td>'00</td>
      <td>213766</td>
      <td>1280428583</td>
    </tr>
  </tbody>
</table>
</div>


**Exercise 2** (2 points). Suppose you wish to compute the prevalence, which is the ratio of cases to the population.

The simplest way to do it is as follows:

```python
    G['prevalence'] = G['cases'] / G['population']
```

However, for this exercise, try to figure out how to use `apply()` to do it instead. To figure that out, you'll need to consult the documentation for [`apply()`](http://pandas.pydata.org/pandas-docs/stable/generated/pandas.DataFrame.apply.html) or go online to find some hints.

Implement your solution in a function, `calc_prevalence(G)`, which given `G` returns a **new copy** `H` that has a column named `'prevalence'` holding the correctly computed prevalence values.

> **Note 0.** The emphasis on "new copy" is there to remind you that your function should *not* modify the input dataframe, `G`.
>
> **Note 1.** Although there is the easy solution above, the purpose of this exercise is to force you to learn more about how `apply()` works, so that you can "apply" it in more settings in the future.


```python
def calc_prevalence(G):
    assert 'cases' in G.columns and 'population' in G.columns
    H = G.copy()
    H['prevalence'] = G.apply(lambda x : x['cases'] / x['population'], axis = 1)
    return H
```


```python
# Test cell: `prevalence_test`

G_copy = G.copy()
H = calc_prevalence(G)
display(H) # Displayed `H` should have a 'prevalence' column

assert (G == G_copy).all().all(), "Did your function modify G? It shouldn't..."
assert set(H.columns) == (set(G.columns) | {'prevalence'}), "Check `H` again: it should have the same columns as `G` plus a new column, `prevalence`."

Easy_prevalence_method = G['cases'] / G['population']
assert (H['prevalence'] == Easy_prevalence_method).all(), "One or more prevalence values is incorrect."

print("Prevalance values seem correct. But did you use `apply()?` Let's see...")

# Tests that you actually used `apply()` in your function:
def apply_fail():
    raise ValueError("Did you really use apply?")
    
setattr(pd.DataFrame, 'apply', apply_fail)

try:
    calc_prevalence(G)
except (ValueError, TypeError):
    print("You used `apply()`. You may have even used it as intended.")
else:
    assert False, "Are you sure you used `apply()`?"
finally:
    setattr(pd.DataFrame, 'apply', SAVE_APPLY)

print("\n(Passed!)")
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
      <th>country</th>
      <th>year</th>
      <th>cases</th>
      <th>population</th>
      <th>prevalence</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>Afghanistan</td>
      <td>'99</td>
      <td>745</td>
      <td>19987071</td>
      <td>0.000037</td>
    </tr>
    <tr>
      <th>1</th>
      <td>Brazil</td>
      <td>'99</td>
      <td>37737</td>
      <td>172006362</td>
      <td>0.000219</td>
    </tr>
    <tr>
      <th>2</th>
      <td>China</td>
      <td>'99</td>
      <td>212258</td>
      <td>1272915272</td>
      <td>0.000167</td>
    </tr>
    <tr>
      <th>3</th>
      <td>Afghanistan</td>
      <td>'00</td>
      <td>2666</td>
      <td>20595360</td>
      <td>0.000129</td>
    </tr>
    <tr>
      <th>4</th>
      <td>Brazil</td>
      <td>'00</td>
      <td>80488</td>
      <td>174504898</td>
      <td>0.000461</td>
    </tr>
    <tr>
      <th>5</th>
      <td>China</td>
      <td>'00</td>
      <td>213766</td>
      <td>1280428583</td>
      <td>0.000167</td>
    </tr>
  </tbody>
</table>
</div>


    Prevalance values seem correct. But did you use `apply()?` Let's see...
    You used `apply()`. You may have even used it as intended.
    
    (Passed!)
    

## Part 3 : Tibbles and Bits 

Now let's start creating and manipulating tibbles.


```python
import pandas as pd  # The suggested idiom
from io import StringIO

from IPython.display import display # For pretty-printing data frames
```

**Exercise 3** (3 points). Write a function, `canonicalize_tibble(X)`, that, given a tibble `X`, returns a new copy `Y` of `X` in _canonical order_. We say `Y` is in canonical order if it has the following properties.

1. The variables appear in sorted order by name, ascending from left to right.
2. The rows appear in lexicographically sorted order by variable, ascending from top to bottom.
3. The row labels (`Y.index`) go from 0 to `n-1`, where `n` is the number of observations.

For instance, here is a **non-canonical tibble** ...

|   |  c  | a | b |
|:-:|:---:|:-:|:-:|
| 2 | hat | x | 1 |
| 0 | rat | y | 4 |
| 3 | cat | x | 2 |
| 1 | bat | x | 2 |


... and here is its **canonical counterpart.**

|   | a | b |  c  |
|:-:|:-:|:-:|:---:|
| 0 | x | 1 | hat |
| 1 | x | 2 | bat |
| 2 | x | 2 | cat |
| 3 | y | 4 | rat |

A partial solution appears below, which ensures that Property 1 above holds. Complete the solution to ensure Properties 2 and 3 hold. Feel free to consult the [Pandas API](http://pandas.pydata.org/pandas-docs/stable/api.html).

> **Hint**. For Property 3, you may find `reset_index()` handy: https://pandas.pydata.org/pandas-docs/stable/generated/pandas.DataFrame.reset_index.html


```python
def canonicalize_tibble(X):
    # Enforce Property 1:
    var_names = sorted(X.columns)
    Y = X[var_names].copy()

    Y.sort_values(by = var_names, inplace = True)
    
    #Y.set_index([list(range(len(Y)))], inplace = True)
    Y.reset_index(drop = True, inplace = True)
    
    return Y
```


```python
# Test: `canonicalize_tibble_test`

# Test input
canonical_in_csv = """,c,a,b
2,hat,x,1
0,rat,y,4
3,cat,x,2
1,bat,x,2"""

with StringIO(canonical_in_csv) as fp:
    canonical_in = pd.read_csv(fp, index_col=0)
print("=== Input ===")
display(canonical_in)
print("")
    
# Test output solution
canonical_soln_csv = """,a,b,c
0,x,1,hat
1,x,2,bat
2,x,2,cat
3,y,4,rat"""

with StringIO(canonical_soln_csv) as fp:
    canonical_soln = pd.read_csv(fp, index_col=0)
print("=== True solution ===")
display(canonical_soln)
print("")

canonical_out = canonicalize_tibble(canonical_in)
print("=== Your computed solution ===")
display(canonical_out)
print("")

canonical_matches = (canonical_out == canonical_soln)
print("=== Matches? (Should be all True) ===")
display(canonical_matches)
assert canonical_matches.all().all()

print ("\n(Passed.)")
```

    === Input ===
    


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
      <th>c</th>
      <th>a</th>
      <th>b</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>2</th>
      <td>hat</td>
      <td>x</td>
      <td>1</td>
    </tr>
    <tr>
      <th>0</th>
      <td>rat</td>
      <td>y</td>
      <td>4</td>
    </tr>
    <tr>
      <th>3</th>
      <td>cat</td>
      <td>x</td>
      <td>2</td>
    </tr>
    <tr>
      <th>1</th>
      <td>bat</td>
      <td>x</td>
      <td>2</td>
    </tr>
  </tbody>
</table>
</div>


    
    === True solution ===
    


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
      <th>a</th>
      <th>b</th>
      <th>c</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>x</td>
      <td>1</td>
      <td>hat</td>
    </tr>
    <tr>
      <th>1</th>
      <td>x</td>
      <td>2</td>
      <td>bat</td>
    </tr>
    <tr>
      <th>2</th>
      <td>x</td>
      <td>2</td>
      <td>cat</td>
    </tr>
    <tr>
      <th>3</th>
      <td>y</td>
      <td>4</td>
      <td>rat</td>
    </tr>
  </tbody>
</table>
</div>


    
    === Your computed solution ===
    


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
      <th>a</th>
      <th>b</th>
      <th>c</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>x</td>
      <td>1</td>
      <td>hat</td>
    </tr>
    <tr>
      <th>1</th>
      <td>x</td>
      <td>2</td>
      <td>bat</td>
    </tr>
    <tr>
      <th>2</th>
      <td>x</td>
      <td>2</td>
      <td>cat</td>
    </tr>
    <tr>
      <th>3</th>
      <td>y</td>
      <td>4</td>
      <td>rat</td>
    </tr>
  </tbody>
</table>
</div>


    
    === Matches? (Should be all True) ===
    


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
      <th>a</th>
      <th>b</th>
      <th>c</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>True</td>
      <td>True</td>
      <td>True</td>
    </tr>
    <tr>
      <th>1</th>
      <td>True</td>
      <td>True</td>
      <td>True</td>
    </tr>
    <tr>
      <th>2</th>
      <td>True</td>
      <td>True</td>
      <td>True</td>
    </tr>
    <tr>
      <th>3</th>
      <td>True</td>
      <td>True</td>
      <td>True</td>
    </tr>
  </tbody>
</table>
</div>


    
    (Passed.)
    

**Exercise 4** (1 point). Write a function, `tibbles_are_equivalent(A, B)` to determine if two tibbles, `A` and `B`, are equivalent. "Equivalent" means that `A` and `B` have identical variables and observations, up to permutations. If `A` and `B` are equivalent, then the function should return `True`. Otherwise, it should return `False`.

The last condition, "up to permutations," means that the variables and observations might not appear in the table in the same order. For example, the following two tibbles are equivalent:


| a | b |  c  |
|:-:|:-:|:---:|
| x | 1 | hat |
| y | 2 | cat |
| z | 3 | bat |
| w | 4 | rat |

| b |  c  | a |
|:-:|:---:|:-:|
| 2 | cat | y |
| 3 | bat | z |
| 1 | hat | x |
| 4 | rat | w |

By contrast, the following table would not be equivalent to either of the above tibbles.

| a | b |  c  |
|:-:|:-:|:---:|
| 2 | y | cat |
| 3 | z | bat |
| 1 | x | hat |
| 4 | w | rat |

> **Note**: Unlike Pandas data frames, tibbles conceptually do not have row labels. So you should ignore row labels.


```python
def tibbles_are_equivalent(A, B):
    """Given two tidy tables ('tibbles'), returns True iff they are
    equivalent.
    """
    #return (canonicalize_tibble(A) == canonicalize_tibble(B)).all().all()
    return canonicalize_tibble(A).equals(canonicalize_tibble(B))
```


```python
# Test: `tibble_are_equivalent_test`

A = pd.DataFrame(columns=['a', 'b', 'c'],
                 data=list(zip (['x', 'y', 'z', 'w'],
                                [1, 2, 3, 4],
                                ['hat', 'cat', 'bat', 'rat'])))
print("=== Tibble A ===")
display(A)

# Permute rows and columns, preserving equivalence
import random

obs_ind_orig = list(range(A.shape[0]))
var_names = list(A.columns)

obs_ind = obs_ind_orig.copy()
while obs_ind == obs_ind_orig:
    random.shuffle(obs_ind)
    
while var_names == list(A.columns):
    random.shuffle(var_names)

B = A[var_names].copy()
B = B.iloc[obs_ind]

print ("=== Tibble B == A ===")
display(B)

print ("=== Tibble C != A ===")
C = A.copy()
C.columns = var_names
display(C)

assert tibbles_are_equivalent(A, B)
assert not tibbles_are_equivalent(A, C)
assert not tibbles_are_equivalent(B, C)

print ("\n(Passed.)")
```

    === Tibble A ===
    


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
      <th>a</th>
      <th>b</th>
      <th>c</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>x</td>
      <td>1</td>
      <td>hat</td>
    </tr>
    <tr>
      <th>1</th>
      <td>y</td>
      <td>2</td>
      <td>cat</td>
    </tr>
    <tr>
      <th>2</th>
      <td>z</td>
      <td>3</td>
      <td>bat</td>
    </tr>
    <tr>
      <th>3</th>
      <td>w</td>
      <td>4</td>
      <td>rat</td>
    </tr>
  </tbody>
</table>
</div>


    === Tibble B == A ===
    


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
      <th>c</th>
      <th>b</th>
      <th>a</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>hat</td>
      <td>1</td>
      <td>x</td>
    </tr>
    <tr>
      <th>3</th>
      <td>rat</td>
      <td>4</td>
      <td>w</td>
    </tr>
    <tr>
      <th>1</th>
      <td>cat</td>
      <td>2</td>
      <td>y</td>
    </tr>
    <tr>
      <th>2</th>
      <td>bat</td>
      <td>3</td>
      <td>z</td>
    </tr>
  </tbody>
</table>
</div>


    === Tibble C != A ===
    


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
      <th>c</th>
      <th>b</th>
      <th>a</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>x</td>
      <td>1</td>
      <td>hat</td>
    </tr>
    <tr>
      <th>1</th>
      <td>y</td>
      <td>2</td>
      <td>cat</td>
    </tr>
    <tr>
      <th>2</th>
      <td>z</td>
      <td>3</td>
      <td>bat</td>
    </tr>
    <tr>
      <th>3</th>
      <td>w</td>
      <td>4</td>
      <td>rat</td>
    </tr>
  </tbody>
</table>
</div>


    
    (Passed.)
    

# Basic tidying transformations: Melting and casting

Given a data set and a target set of variables, there are at least two common issues that require tidying.
## Melting
First, values often appear as columns. Table 4a is an example. To tidy up, you want to turn columns into rows:

![Gather example](http://r4ds.had.co.nz/images/tidy-9.png)

Because this operation takes columns into rows, making a "fat" table more tall and skinny, it is sometimes called _melting_.




To melt the table, you need to do the following.

1. Extract the _column values_ into a new variable. In this case, columns `"1999"` and `"2000"` of `table4` need to become the values of the variable, `"year"`.
2. Convert the values associated with the column values into a new variable as well. In this case, the values formerly in columns `"1999"` and `"2000"` become the values of the `"cases"` variable.

In the context of a melt, let's also refer to `"year"` as the new _key_ variable and `"cases"` as the new _value_ variable.

**Exercise 5** (4 points). Implement the melt operation as a function,

```python
    def melt(df, col_vals, key, value):
        ...
```

It should take the following arguments:
- `df`: the input data frame, e.g., `table4` in the example above;
- `col_vals`: a list of the column names that will serve as values;  column `1999` & `2000` in example  table
- `key`: name of the new variable, e.g., `year` in the example above;
- `value`: name of the column to hold the values. `cases` in the example above

> You may need to refer to the Pandas documentation to figure out how to create and manipulate tables. The bits related to [indexing](http://pandas.pydata.org/pandas-docs/stable/indexing.html) and [merging](http://pandas.pydata.org/pandas-docs/stable/merging.html) may be especially helpful.


```python
def melt(df, col_vals, key, value):
    assert type(df) is pd.DataFrame
    keep_vars = df.columns.difference(col_vals)
    melted_sections = []
    for c in col_vals:
        melted_c = df[keep_vars].copy()
        melted_c[key] = c
        melted_c[value] = df[c]
        melted_sections.append(melted_c)
    melted = pd.concat(melted_sections)
    return melted
```


```python
# Test: `melt_test`

table4a = pd.read_csv(local_data['table4a.csv'])
print("\n=== table4a ===")
display(table4a)

m_4a = melt(table4a, col_vals=['1999', '2000'], key='year', value='cases')
print("=== melt(table4a) ===")
display(m_4a)

table4b = pd.read_csv(local_data['table4b.csv'])
print("\n=== table4b ===")
display(table4b)

m_4b = melt(table4b, col_vals=['1999', '2000'], key='year', value='population')
print("=== melt(table4b) ===")
display(m_4b)

m_4 = pd.merge(m_4a, m_4b, on=['country', 'year'])
print ("\n=== inner-join(melt(table4a), melt (table4b)) ===")
display(m_4)

m_4['year'] = m_4['year'].apply (int)

table1 = pd.read_csv(local_data['table1.csv'])
print ("=== table1 (target solution) ===")
display(table1)
assert tibbles_are_equivalent(table1, m_4)
print ("\n(Passed.)")
```

    
    === table4a ===
    


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
      <th>country</th>
      <th>1999</th>
      <th>2000</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>Afghanistan</td>
      <td>745</td>
      <td>2666</td>
    </tr>
    <tr>
      <th>1</th>
      <td>Brazil</td>
      <td>37737</td>
      <td>80488</td>
    </tr>
    <tr>
      <th>2</th>
      <td>China</td>
      <td>212258</td>
      <td>213766</td>
    </tr>
  </tbody>
</table>
</div>


    === melt(table4a) ===
    


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
      <th>country</th>
      <th>year</th>
      <th>cases</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>Afghanistan</td>
      <td>1999</td>
      <td>745</td>
    </tr>
    <tr>
      <th>1</th>
      <td>Brazil</td>
      <td>1999</td>
      <td>37737</td>
    </tr>
    <tr>
      <th>2</th>
      <td>China</td>
      <td>1999</td>
      <td>212258</td>
    </tr>
    <tr>
      <th>0</th>
      <td>Afghanistan</td>
      <td>2000</td>
      <td>2666</td>
    </tr>
    <tr>
      <th>1</th>
      <td>Brazil</td>
      <td>2000</td>
      <td>80488</td>
    </tr>
    <tr>
      <th>2</th>
      <td>China</td>
      <td>2000</td>
      <td>213766</td>
    </tr>
  </tbody>
</table>
</div>


    
    === table4b ===
    


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
      <th>country</th>
      <th>1999</th>
      <th>2000</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>Afghanistan</td>
      <td>19987071</td>
      <td>20595360</td>
    </tr>
    <tr>
      <th>1</th>
      <td>Brazil</td>
      <td>172006362</td>
      <td>174504898</td>
    </tr>
    <tr>
      <th>2</th>
      <td>China</td>
      <td>1272915272</td>
      <td>1280428583</td>
    </tr>
  </tbody>
</table>
</div>


    === melt(table4b) ===
    


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
      <th>country</th>
      <th>year</th>
      <th>population</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>Afghanistan</td>
      <td>1999</td>
      <td>19987071</td>
    </tr>
    <tr>
      <th>1</th>
      <td>Brazil</td>
      <td>1999</td>
      <td>172006362</td>
    </tr>
    <tr>
      <th>2</th>
      <td>China</td>
      <td>1999</td>
      <td>1272915272</td>
    </tr>
    <tr>
      <th>0</th>
      <td>Afghanistan</td>
      <td>2000</td>
      <td>20595360</td>
    </tr>
    <tr>
      <th>1</th>
      <td>Brazil</td>
      <td>2000</td>
      <td>174504898</td>
    </tr>
    <tr>
      <th>2</th>
      <td>China</td>
      <td>2000</td>
      <td>1280428583</td>
    </tr>
  </tbody>
</table>
</div>


    
    === inner-join(melt(table4a), melt (table4b)) ===
    


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
      <th>country</th>
      <th>year</th>
      <th>cases</th>
      <th>population</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>Afghanistan</td>
      <td>1999</td>
      <td>745</td>
      <td>19987071</td>
    </tr>
    <tr>
      <th>1</th>
      <td>Brazil</td>
      <td>1999</td>
      <td>37737</td>
      <td>172006362</td>
    </tr>
    <tr>
      <th>2</th>
      <td>China</td>
      <td>1999</td>
      <td>212258</td>
      <td>1272915272</td>
    </tr>
    <tr>
      <th>3</th>
      <td>Afghanistan</td>
      <td>2000</td>
      <td>2666</td>
      <td>20595360</td>
    </tr>
    <tr>
      <th>4</th>
      <td>Brazil</td>
      <td>2000</td>
      <td>80488</td>
      <td>174504898</td>
    </tr>
    <tr>
      <th>5</th>
      <td>China</td>
      <td>2000</td>
      <td>213766</td>
      <td>1280428583</td>
    </tr>
  </tbody>
</table>
</div>


    === table1 (target solution) ===
    


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
      <th>country</th>
      <th>year</th>
      <th>cases</th>
      <th>population</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>Afghanistan</td>
      <td>1999</td>
      <td>745</td>
      <td>19987071</td>
    </tr>
    <tr>
      <th>1</th>
      <td>Afghanistan</td>
      <td>2000</td>
      <td>2666</td>
      <td>20595360</td>
    </tr>
    <tr>
      <th>2</th>
      <td>Brazil</td>
      <td>1999</td>
      <td>37737</td>
      <td>172006362</td>
    </tr>
    <tr>
      <th>3</th>
      <td>Brazil</td>
      <td>2000</td>
      <td>80488</td>
      <td>174504898</td>
    </tr>
    <tr>
      <th>4</th>
      <td>China</td>
      <td>1999</td>
      <td>212258</td>
      <td>1272915272</td>
    </tr>
    <tr>
      <th>5</th>
      <td>China</td>
      <td>2000</td>
      <td>213766</td>
      <td>1280428583</td>
    </tr>
  </tbody>
</table>
</div>


    
    (Passed.)
    

## Casting
The second most common issue is that an observation might be split across multiple rows. Table 2 is an example. To tidy up, you want to merge rows:

![Spread example](http://r4ds.had.co.nz/images/tidy-8.png)

Because this operation is the moral opposite of melting, and "rebuilds" observations from parts, it is sometimes called _casting_.

> Melting and casting are Wickham's terms from [his original paper on tidying data](http://www.jstatsoft.org/v59/i10/paper). In his more recent writing, [on which this tutorial is based](http://r4ds.had.co.nz/tidy-data.html), he refers to the same operation as _gathering_. Again, this term comes from Wickham's original paper, whereas his more recent summaries use the term _spreading_.

The signature of a cast is similar to that of melt. However, you only need to know the `key`, which is column of the input table containing new variable names, and the `value`, which is the column containing corresponding values.

**Exercise 6** (4 points). Implement a function to cast a data frame into a tibble, given a key column containing new variable names and a value column containing the corresponding cells.

We've given you a partial solution that

- verifies that the given `key` and `value` columns are actual columns of the input data frame;
- computes the list of columns, `fixed_vars`, that should remain unchanged; and
- initializes and empty tibble.

Observe that we are asking your `cast()` to accept an optional parameter, `join_how`, that may take the values `'outer'` or `'inner'` (with `'outer'` as the default). Why do you need such a parameter?


```python
def cast(df, key, value, join_how='outer'):
    """Casts the input data frame into a tibble,
    given the key column and value column.
    """
    assert type(df) is pd.DataFrame
    assert key in df.columns and value in df.columns
    assert join_how in ['outer', 'inner']
    
    fixed_vars = df.columns.difference([key, value])

    tibble = pd.DataFrame(columns = fixed_vars) # empty frame
    
    new_vars = df[key].unique()
    
    for v in new_vars:
        df_v = df[df[key] == v]
        del df_v[key]
        df_v = df_v.rename(columns = {value : v}) 
        tibble = tibble.merge(df_v, on = list(fixed_vars), how = join_how)
    
    return tibble
```


```python
# Test: `cast_test`

table2 = pd.read_csv(local_data['table2.csv'])
print('=== table2 ===')
display(table2)

print('\n=== tibble2 = cast (table2, "type", "count") ===')
tibble2 = cast(table2, 'type', 'count')
display(tibble2)

assert tibbles_are_equivalent(table1, tibble2)
print('\n(Passed.)')
```

    === table2 ===
    


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
      <th>country</th>
      <th>year</th>
      <th>type</th>
      <th>count</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>Afghanistan</td>
      <td>1999</td>
      <td>cases</td>
      <td>745</td>
    </tr>
    <tr>
      <th>1</th>
      <td>Afghanistan</td>
      <td>1999</td>
      <td>population</td>
      <td>19987071</td>
    </tr>
    <tr>
      <th>2</th>
      <td>Afghanistan</td>
      <td>2000</td>
      <td>cases</td>
      <td>2666</td>
    </tr>
    <tr>
      <th>3</th>
      <td>Afghanistan</td>
      <td>2000</td>
      <td>population</td>
      <td>20595360</td>
    </tr>
    <tr>
      <th>4</th>
      <td>Brazil</td>
      <td>1999</td>
      <td>cases</td>
      <td>37737</td>
    </tr>
    <tr>
      <th>5</th>
      <td>Brazil</td>
      <td>1999</td>
      <td>population</td>
      <td>172006362</td>
    </tr>
    <tr>
      <th>6</th>
      <td>Brazil</td>
      <td>2000</td>
      <td>cases</td>
      <td>80488</td>
    </tr>
    <tr>
      <th>7</th>
      <td>Brazil</td>
      <td>2000</td>
      <td>population</td>
      <td>174504898</td>
    </tr>
    <tr>
      <th>8</th>
      <td>China</td>
      <td>1999</td>
      <td>cases</td>
      <td>212258</td>
    </tr>
    <tr>
      <th>9</th>
      <td>China</td>
      <td>1999</td>
      <td>population</td>
      <td>1272915272</td>
    </tr>
    <tr>
      <th>10</th>
      <td>China</td>
      <td>2000</td>
      <td>cases</td>
      <td>213766</td>
    </tr>
    <tr>
      <th>11</th>
      <td>China</td>
      <td>2000</td>
      <td>population</td>
      <td>1280428583</td>
    </tr>
  </tbody>
</table>
</div>


    
    === tibble2 = cast (table2, "type", "count") ===
    


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
      <th>country</th>
      <th>year</th>
      <th>cases</th>
      <th>population</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>Afghanistan</td>
      <td>1999</td>
      <td>745</td>
      <td>19987071</td>
    </tr>
    <tr>
      <th>1</th>
      <td>Afghanistan</td>
      <td>2000</td>
      <td>2666</td>
      <td>20595360</td>
    </tr>
    <tr>
      <th>2</th>
      <td>Brazil</td>
      <td>1999</td>
      <td>37737</td>
      <td>172006362</td>
    </tr>
    <tr>
      <th>3</th>
      <td>Brazil</td>
      <td>2000</td>
      <td>80488</td>
      <td>174504898</td>
    </tr>
    <tr>
      <th>4</th>
      <td>China</td>
      <td>1999</td>
      <td>212258</td>
      <td>1272915272</td>
    </tr>
    <tr>
      <th>5</th>
      <td>China</td>
      <td>2000</td>
      <td>213766</td>
      <td>1280428583</td>
    </tr>
  </tbody>
</table>
</div>


    
    (Passed.)
    

## Separating variables

Consider the following table.


```python
table3 = pd.read_csv(local_data['table3.csv'])
display(table3)
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
      <th>country</th>
      <th>year</th>
      <th>rate</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>Afghanistan</td>
      <td>1999</td>
      <td>745/19987071</td>
    </tr>
    <tr>
      <th>1</th>
      <td>Afghanistan</td>
      <td>2000</td>
      <td>2666/20595360</td>
    </tr>
    <tr>
      <th>2</th>
      <td>Brazil</td>
      <td>1999</td>
      <td>37737/172006362</td>
    </tr>
    <tr>
      <th>3</th>
      <td>Brazil</td>
      <td>2000</td>
      <td>80488/174504898</td>
    </tr>
    <tr>
      <th>4</th>
      <td>China</td>
      <td>1999</td>
      <td>212258/1272915272</td>
    </tr>
    <tr>
      <th>5</th>
      <td>China</td>
      <td>2000</td>
      <td>213766/1280428583</td>
    </tr>
  </tbody>
</table>
</div>


In this table, the `rate` variable combines what had previously been the `cases` and `population` data. This example is an instance in which we might want to _separate_ a column into two variables.

**Exercise 7A** (3 points). Write a function that takes a data frame (`df`) and separates an existing column (`key`) into new variables (given by the list of new variable names, `into`).

How will the separation happen? The caller should provide a function, `splitter(x)`, that given a value returns a _list_ containing the components. Observe that the partial solution below defines a default splitter, which uses the regular expression, `(\d+\.?\d+)`, to find all integer or floating-point values in a string input `x`.


```python
import re

def default_splitter(text):
    """Searches the given spring for all integer and floating-point
    values, returning them as a list _of strings_.
    
    E.g., the call
    
      default_splitter('Give me $10.52 in exchange for 91 kitten stickers.')
      
    will return ['10.52', '91'].
    """
    fields = re.findall('(\d+\.?\d+)', text)
    return fields

def separate(df, key, into, splitter=default_splitter):
    """Given a data frame, separates one of its columns, the key,
    into new variables.
    """
    assert type(df) is pd.DataFrame
    assert key in df.columns
    
    # Hint: http://stackoverflow.com/questions/16236684/apply-pandas-function-to-column-to-create-multiple-new-columns
    def apply_splitter(text):
        fields = splitter(text)
        return pd.Series({into[i]: f for i, f in enumerate (fields)})
    
    fixed_vars = df.columns.difference([key])
    tibble = df[fixed_vars].copy()
    tibble_extra = df[key].apply(apply_splitter)
    return pd.concat([tibble, tibble_extra], axis=1)
```


```python
# Test: `separate_test`

print("=== Recall: table3 ===")
display(table3)

tibble3 = separate(table3, key='rate', into=['cases', 'population'])
print("\n=== tibble3 = separate (table3, ...) ===")
display(tibble3)

assert 'cases' in tibble3.columns
assert 'population' in tibble3.columns
assert 'rate' not in tibble3.columns

tibble3['cases'] = tibble3['cases'].apply(int)
tibble3['population'] = tibble3['population'].apply(int)

assert tibbles_are_equivalent(tibble3, table1)
print("\n(Passed.)")
```

    === Recall: table3 ===
    


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
      <th>country</th>
      <th>year</th>
      <th>rate</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>Afghanistan</td>
      <td>1999</td>
      <td>745/19987071</td>
    </tr>
    <tr>
      <th>1</th>
      <td>Afghanistan</td>
      <td>2000</td>
      <td>2666/20595360</td>
    </tr>
    <tr>
      <th>2</th>
      <td>Brazil</td>
      <td>1999</td>
      <td>37737/172006362</td>
    </tr>
    <tr>
      <th>3</th>
      <td>Brazil</td>
      <td>2000</td>
      <td>80488/174504898</td>
    </tr>
    <tr>
      <th>4</th>
      <td>China</td>
      <td>1999</td>
      <td>212258/1272915272</td>
    </tr>
    <tr>
      <th>5</th>
      <td>China</td>
      <td>2000</td>
      <td>213766/1280428583</td>
    </tr>
  </tbody>
</table>
</div>


    
    === tibble3 = separate (table3, ...) ===
    


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
      <th>country</th>
      <th>year</th>
      <th>cases</th>
      <th>population</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>Afghanistan</td>
      <td>1999</td>
      <td>745</td>
      <td>19987071</td>
    </tr>
    <tr>
      <th>1</th>
      <td>Afghanistan</td>
      <td>2000</td>
      <td>2666</td>
      <td>20595360</td>
    </tr>
    <tr>
      <th>2</th>
      <td>Brazil</td>
      <td>1999</td>
      <td>37737</td>
      <td>172006362</td>
    </tr>
    <tr>
      <th>3</th>
      <td>Brazil</td>
      <td>2000</td>
      <td>80488</td>
      <td>174504898</td>
    </tr>
    <tr>
      <th>4</th>
      <td>China</td>
      <td>1999</td>
      <td>212258</td>
      <td>1272915272</td>
    </tr>
    <tr>
      <th>5</th>
      <td>China</td>
      <td>2000</td>
      <td>213766</td>
      <td>1280428583</td>
    </tr>
  </tbody>
</table>
</div>


    
    (Passed.)
    

**Exercise 7B** (2 points). Implement the inverse of separate, which is `unite`. This function should take a data frame (`df`), the set of columns to combine (`cols`), the name of the new column (`new_var`), and a function that takes the subset of the `cols` variables from a single observation. It should return a new value for that observation.


```python
def str_join_elements(x, sep=""):
    assert type(sep) is str
    return sep.join([str(xi) for xi in x])

def unite(df, cols, new_var, combine=str_join_elements):
    # Hint: http://stackoverflow.com/questions/13331698/how-to-apply-a-function-to-two-columns-of-pandas-dataframe
    fixed_vars = df.columns.difference(cols)
    table = df[fixed_vars].copy()
    table[new_var] = df[cols].apply(combine, axis=1)
    return table
```


```python
# Test: `unite_test`

table3_again = unite(tibble3, ['cases', 'population'], 'rate',
                     combine=lambda x: str_join_elements(x, "/"))
display(table3_again)
assert tibbles_are_equivalent(table3, table3_again)

print("\n(Passed.)")
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
      <th>country</th>
      <th>year</th>
      <th>rate</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>Afghanistan</td>
      <td>1999</td>
      <td>745/19987071</td>
    </tr>
    <tr>
      <th>1</th>
      <td>Afghanistan</td>
      <td>2000</td>
      <td>2666/20595360</td>
    </tr>
    <tr>
      <th>2</th>
      <td>Brazil</td>
      <td>1999</td>
      <td>37737/172006362</td>
    </tr>
    <tr>
      <th>3</th>
      <td>Brazil</td>
      <td>2000</td>
      <td>80488/174504898</td>
    </tr>
    <tr>
      <th>4</th>
      <td>China</td>
      <td>1999</td>
      <td>212258/1272915272</td>
    </tr>
    <tr>
      <th>5</th>
      <td>China</td>
      <td>2000</td>
      <td>213766/1280428583</td>
    </tr>
  </tbody>
</table>
</div>


    
    (Passed.)
    


# Putting it all together #

Let's use primitives to tidy up the original WHO TB data set. First, here is the raw data.



```python
who_raw = pd.read_csv(local_data['who.csv'])

print("=== WHO TB data set: {} rows x {} columns ===".format(who_raw.shape[0],
                                                              who_raw.shape[1]))
print("Column names:", who_raw.columns)

print("\n=== A few randomly selected rows ===")
import random
row_sample = sorted(random.sample(range(len(who_raw)), 5))
display(who_raw.iloc[row_sample])
```

    === WHO TB data set: 7240 rows x 60 columns ===
    Column names: Index(['country', 'iso2', 'iso3', 'year', 'new_sp_m014', 'new_sp_m1524',
           'new_sp_m2534', 'new_sp_m3544', 'new_sp_m4554', 'new_sp_m5564',
           'new_sp_m65', 'new_sp_f014', 'new_sp_f1524', 'new_sp_f2534',
           'new_sp_f3544', 'new_sp_f4554', 'new_sp_f5564', 'new_sp_f65',
           'new_sn_m014', 'new_sn_m1524', 'new_sn_m2534', 'new_sn_m3544',
           'new_sn_m4554', 'new_sn_m5564', 'new_sn_m65', 'new_sn_f014',
           'new_sn_f1524', 'new_sn_f2534', 'new_sn_f3544', 'new_sn_f4554',
           'new_sn_f5564', 'new_sn_f65', 'new_ep_m014', 'new_ep_m1524',
           'new_ep_m2534', 'new_ep_m3544', 'new_ep_m4554', 'new_ep_m5564',
           'new_ep_m65', 'new_ep_f014', 'new_ep_f1524', 'new_ep_f2534',
           'new_ep_f3544', 'new_ep_f4554', 'new_ep_f5564', 'new_ep_f65',
           'newrel_m014', 'newrel_m1524', 'newrel_m2534', 'newrel_m3544',
           'newrel_m4554', 'newrel_m5564', 'newrel_m65', 'newrel_f014',
           'newrel_f1524', 'newrel_f2534', 'newrel_f3544', 'newrel_f4554',
           'newrel_f5564', 'newrel_f65'],
          dtype='object')
    
    === A few randomly selected rows ===
    


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
      <th>country</th>
      <th>iso2</th>
      <th>iso3</th>
      <th>year</th>
      <th>new_sp_m014</th>
      <th>new_sp_m1524</th>
      <th>new_sp_m2534</th>
      <th>new_sp_m3544</th>
      <th>new_sp_m4554</th>
      <th>new_sp_m5564</th>
      <th>...</th>
      <th>newrel_m4554</th>
      <th>newrel_m5564</th>
      <th>newrel_m65</th>
      <th>newrel_f014</th>
      <th>newrel_f1524</th>
      <th>newrel_f2534</th>
      <th>newrel_f3544</th>
      <th>newrel_f4554</th>
      <th>newrel_f5564</th>
      <th>newrel_f65</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>138</th>
      <td>Andorra</td>
      <td>AD</td>
      <td>AND</td>
      <td>1982</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>...</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
    </tr>
    <tr>
      <th>503</th>
      <td>Bahamas</td>
      <td>BS</td>
      <td>BHS</td>
      <td>2007</td>
      <td>0.0</td>
      <td>3.0</td>
      <td>3.0</td>
      <td>9.0</td>
      <td>4.0</td>
      <td>1.0</td>
      <td>...</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
    </tr>
    <tr>
      <th>2234</th>
      <td>Estonia</td>
      <td>EE</td>
      <td>EST</td>
      <td>1996</td>
      <td>1.0</td>
      <td>7.0</td>
      <td>34.0</td>
      <td>53.0</td>
      <td>39.0</td>
      <td>28.0</td>
      <td>...</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
    </tr>
    <tr>
      <th>3985</th>
      <td>Malta</td>
      <td>MT</td>
      <td>MLT</td>
      <td>2013</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>...</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>3.0</td>
      <td>0.0</td>
      <td>2.0</td>
      <td>8.0</td>
      <td>1.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>4.0</td>
    </tr>
    <tr>
      <th>3994</th>
      <td>Marshall Islands</td>
      <td>MH</td>
      <td>MHL</td>
      <td>1988</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>...</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
    </tr>
  </tbody>
</table>
<p>5 rows × 60 columns</p>
</div>


The data set has 7,240 rows and 60 columns. Here is how to decode the columns.
- Columns `'country'`, `'iso2'`, and `'iso3'` are different ways to designate the country and redundant, meaning you only really need to keep one of them.
- Column `'year'` is the year of the report and is a natural variable.
- Among columns `'new_sp_m014'` through `'newrel_f65'`, the `'new...'` prefix indicates that the column's values count new cases of TB. In this particular data set, all the data are for new cases.
- The short codes, `rel`, `ep`, `sn`, and `sp` describe the type of TB case. They stand for relapse, extrapulmonary, pulmonary not detectable by a pulmonary smear test ("smear negative"), and pulmonary detectable by such a test ("smear positive"), respectively.
- The codes `'m'` and `'f'` indicate the gender (male and female, respectively).
- The trailing numeric code indicates the age group: `014` is 0-14 years of age, `1524` for 15-24 years, `2534` for 25-34 years, etc., and `65` stands for 65 years or older.

In other words, it looks like you are likely to want to treat all the columns as values of multiple variables!

**Exercise 8** (3 points). As a first step, start with `who_raw` and create a new data frame, `who2`, with the following properties:

- All the `'new...'` columns of `who_raw` become values of a _single_ variable, `case_type`. Store the counts associated with each `case_type` value as a new variable called `'count'`.
- Remove the `iso2` and `iso3` columns, since they are redundant with `country` (which you should keep!).
- Keep the `year` column as a variable.
- Remove all not-a-number (`NaN`) counts. _Hint_: You can test for a `NaN` using Python's [`math.isnan()`](https://docs.python.org/3/library/math.html).
- Convert the counts to integers. (Because of the presence of NaNs, the counts will be otherwise be treated as floating-point values, which is undesirable since you do not expect to see non-integer counts.)


```python
from math import isnan

col_vals = who_raw.columns.difference(['country', 'iso2', 'iso3', 'year'])
who2 = melt(who_raw, col_vals, 'case_type', 'count')

del who2['iso2']
del who2['iso3']

who2 = who2[who2['count'].apply(lambda x: not isnan(x))]

who2['count'] = who2['count'].apply(lambda x: int(x))
```


```python
# Test: `who2_test`

print("=== First few rows of your solution ===")
display(who2.head())

print ("=== First few rows of the instructor's solution ===")
who2_soln = pd.read_csv(local_data['who2_soln.csv'])
display(who2_soln.head())

# Check it
assert tibbles_are_equivalent(who2, who2_soln)
print ("\n(Passed.)")
```

    === First few rows of your solution ===
    


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
      <th>country</th>
      <th>year</th>
      <th>case_type</th>
      <th>count</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>60</th>
      <td>Albania</td>
      <td>2006</td>
      <td>new_ep_f014</td>
      <td>7</td>
    </tr>
    <tr>
      <th>61</th>
      <td>Albania</td>
      <td>2007</td>
      <td>new_ep_f014</td>
      <td>1</td>
    </tr>
    <tr>
      <th>62</th>
      <td>Albania</td>
      <td>2008</td>
      <td>new_ep_f014</td>
      <td>3</td>
    </tr>
    <tr>
      <th>63</th>
      <td>Albania</td>
      <td>2009</td>
      <td>new_ep_f014</td>
      <td>2</td>
    </tr>
    <tr>
      <th>64</th>
      <td>Albania</td>
      <td>2010</td>
      <td>new_ep_f014</td>
      <td>2</td>
    </tr>
  </tbody>
</table>
</div>


    === First few rows of the instructor's solution ===
    


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
      <th>country</th>
      <th>year</th>
      <th>case_type</th>
      <th>count</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>Albania</td>
      <td>2006</td>
      <td>new_ep_f014</td>
      <td>7</td>
    </tr>
    <tr>
      <th>1</th>
      <td>Albania</td>
      <td>2007</td>
      <td>new_ep_f014</td>
      <td>1</td>
    </tr>
    <tr>
      <th>2</th>
      <td>Albania</td>
      <td>2008</td>
      <td>new_ep_f014</td>
      <td>3</td>
    </tr>
    <tr>
      <th>3</th>
      <td>Albania</td>
      <td>2009</td>
      <td>new_ep_f014</td>
      <td>2</td>
    </tr>
    <tr>
      <th>4</th>
      <td>Albania</td>
      <td>2010</td>
      <td>new_ep_f014</td>
      <td>2</td>
    </tr>
  </tbody>
</table>
</div>


    
    (Passed.)
    

**Exercise 9** (5 points). Starting from your `who2` data frame, create a new tibble, `who3`, for which each `'key'` value is split into three new variables:
- `'type'`, to hold the TB type, having possible values of `rel`, `ep`, `sn`, and `sp`;
- `'gender'`, to hold the gender as a string having possible values of `female` and `male`; and
- `'age_group'`, to hold the age group as a string having possible values of `0-14`, `15-24`, `25-34`, `35-44`, `45-54`, `55-64`, and `65+`.

> The input data file is large enough that your solution might take a minute to run. But if it appears to be taking much more than that, you may want to revisit your approach.


```python
import re
def who_splitter(text):
    m = re.match("^new_?(rel|ep|sn|sp)_(f|m)(\\d{2,4})$", text)
    if m is None or len(m.groups()) != 3: # no match?
        return ['', '', '']
    fields = list(m.groups())
    if fields[1] == 'f':
        fields[1] = 'female'
    elif fields[1] == 'm':
        fields[1] = 'male'
    if fields[2] == '014':
        fields[2] = '0-14'
    elif fields[2] == '65':
        fields[2] = '65+'
    elif len(fields[2]) == 4 and fields[2].isdigit():
        fields[2] = fields[2][0:2] + '-' + fields[2][2:4]
    return fields
who3 = separate(
    who2,
    key = 'case_type',
    into = ['type', 'gender', 'age_group'],
    splitter = who_splitter
)
```


```python
# Test: `who3_test`

print("=== First few rows of your solution ===")
display(who3.head())

who3_soln = pd.read_csv(local_data['who3_soln.csv'])
print("\n=== First few rows of the instructor's solution ===")
display(who3_soln.head())

assert tibbles_are_equivalent(who3, who3_soln)
print("\n(Passed.)")
```

    === First few rows of your solution ===
    


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
      <th>count</th>
      <th>country</th>
      <th>year</th>
      <th>type</th>
      <th>gender</th>
      <th>age_group</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>60</th>
      <td>7</td>
      <td>Albania</td>
      <td>2006</td>
      <td>ep</td>
      <td>female</td>
      <td>0-14</td>
    </tr>
    <tr>
      <th>61</th>
      <td>1</td>
      <td>Albania</td>
      <td>2007</td>
      <td>ep</td>
      <td>female</td>
      <td>0-14</td>
    </tr>
    <tr>
      <th>62</th>
      <td>3</td>
      <td>Albania</td>
      <td>2008</td>
      <td>ep</td>
      <td>female</td>
      <td>0-14</td>
    </tr>
    <tr>
      <th>63</th>
      <td>2</td>
      <td>Albania</td>
      <td>2009</td>
      <td>ep</td>
      <td>female</td>
      <td>0-14</td>
    </tr>
    <tr>
      <th>64</th>
      <td>2</td>
      <td>Albania</td>
      <td>2010</td>
      <td>ep</td>
      <td>female</td>
      <td>0-14</td>
    </tr>
  </tbody>
</table>
</div>


    
    === First few rows of the instructor's solution ===
    


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
      <th>count</th>
      <th>country</th>
      <th>year</th>
      <th>age_group</th>
      <th>gender</th>
      <th>type</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>7</td>
      <td>Albania</td>
      <td>2006</td>
      <td>0-14</td>
      <td>female</td>
      <td>ep</td>
    </tr>
    <tr>
      <th>1</th>
      <td>1</td>
      <td>Albania</td>
      <td>2007</td>
      <td>0-14</td>
      <td>female</td>
      <td>ep</td>
    </tr>
    <tr>
      <th>2</th>
      <td>3</td>
      <td>Albania</td>
      <td>2008</td>
      <td>0-14</td>
      <td>female</td>
      <td>ep</td>
    </tr>
    <tr>
      <th>3</th>
      <td>2</td>
      <td>Albania</td>
      <td>2009</td>
      <td>0-14</td>
      <td>female</td>
      <td>ep</td>
    </tr>
    <tr>
      <th>4</th>
      <td>2</td>
      <td>Albania</td>
      <td>2010</td>
      <td>0-14</td>
      <td>female</td>
      <td>ep</td>
    </tr>
  </tbody>
</table>
</div>


    
    (Passed.)
    

**Fin!** That's the end of this assignment. Don't forget to restart and run this notebook from the beginning to verify that it works top-to-bottom before submitting.
