# Problem 1: Drug Trial

In this problem, you'll analyze some data from a medical drug trial. There are three exercises worth a total of ten points.

Two of the exercises allow you to use **either** Pandas **or** SQL to solve it. Choose the method that feels is more natural to you.

## Setup

Run the following few code cells, which will load the modules and sample data you'll need for this problem.


```python
import sys
import pandas as pd
import numpy as np
import sqlite3 as db

print("Python version: {}".format(sys.version))
print("pandas version: {}".format(pd.__version__))
print("numpy version: {}".format(np.__version__))
print("sqlite3 version: {}".format(db.version))

from IPython.display import display
from cse6040utils import canonicalize_tibble, tibbles_are_equivalent
```

    Python version: 3.7.4 (tags/v3.7.4:e09359112e, Jul  8 2019, 20:34:20) [MSC v.1916 64 bit (AMD64)]
    pandas version: 0.25.3
    numpy version: 1.18.1
    sqlite3 version: 2.6.0
    


```python
from cse6040utils import download_all

datasets = {'Drugs_soln.csv': '6df060bde8dea48986dc12650a4fbef5',
            'avg_dose_soln.csv': 'f604e3da488d489792fec220ada738f8',
            'drugs.csv': '33bb1fa5068069a483a6e05fafde40d0',
            'nst_count_soln.csv': '7519ad4764eb238a9120fa7cd1f006de',
            'nst_count_soln--corrected.csv': '81f801cd20775a51f92a1b28593c0915',
            'swt_count_soln.csv': 'fbbb7368d31856665c3e5e1b19d93d65'}

DATA_SUFFIX = "drug-trials/"
data_paths = download_all(datasets, local_suffix=DATA_SUFFIX, url_suffix=DATA_SUFFIX)
    
print("\n(All data appears to be ready.)")
```

    'Drugs_soln.csv' is ready!
    'avg_dose_soln.csv' is ready!
    'drugs.csv' is ready!
    'nst_count_soln.csv' is ready!
    'nst_count_soln--corrected.csv' is ready!
    'swt_count_soln.csv' is ready!
    
    (All data appears to be ready.)
    

## The data

Company XYZ currently uses Medication A to treat all its patients and is considering a switch to Medication B. A critical part of the evaluation of Medication B is how much of it would be used among XYZ’s patients.

The company did a trial of Medication B. The data in the accompanying CSV file, `Drugs.csv`, is data taken from roughly 130 patients at least 2 months before switching medications and up to 3 months while on the new medication.

A patient can be taking medication A or medication B, but cannot be taking both at the same time.

The following code cell will read this data and store it in a dataframe named `Drugs`.


```python
Drugs = pd.read_csv(data_paths['drugs.csv'], header=0)
assert len(Drugs) == 2022
Drugs.head()
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
      <th>ID</th>
      <th>Med</th>
      <th>Admin Date</th>
      <th>Units</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>1</td>
      <td>Med A</td>
      <td>7/2/12</td>
      <td>1,500.00</td>
    </tr>
    <tr>
      <th>1</th>
      <td>1</td>
      <td>Med A</td>
      <td>7/6/12</td>
      <td>1,500.00</td>
    </tr>
    <tr>
      <th>2</th>
      <td>1</td>
      <td>Med A</td>
      <td>7/9/12</td>
      <td>1,500.00</td>
    </tr>
    <tr>
      <th>3</th>
      <td>1</td>
      <td>Med A</td>
      <td>7/11/12</td>
      <td>1,500.00</td>
    </tr>
    <tr>
      <th>4</th>
      <td>1</td>
      <td>Med A</td>
      <td>7/13/12</td>
      <td>1,500.00</td>
    </tr>
  </tbody>
</table>
</div>



Each row indicates that a patient (identified by his or her `'ID'`) took one **dose** of a particular drug on a particular day. The size of the dose was `'Units'`.

## Exercises

**Exercise 0** (1 points). All you have to do is read the code in the following code cell and run it. You should observe the following.

First, the `'Med'`, `'Admin Date'`, and `'Units'` columns are stored as strings initially.

Secondly, there are some handy functions in Pandas to change the `'Admin Date'` and '`Units`' columns into more "natural" native Python types, namely, a floating-point type and a Python `datetime` type, respectively. Indeed, once in this form, it is easy to use Pandas to, say, extract the month as its own column.


```python
# Observe types:
for col in ['Med', 'Admin Date', 'Units']:
    print("Column '{}' has type {}.".format(col, type(Drugs[col].iloc[0])))
    
# Convert strings to "natural" types:
Drugs = pd.read_csv(data_paths['drugs.csv'], header=0)
Drugs['Units'] = pd.to_numeric(Drugs['Units'].str.replace(',',''), errors='coerce')
Drugs['Admin Date'] = pd.to_datetime(Drugs['Admin Date'], infer_datetime_format=True)
Drugs['Month'] = Drugs['Admin Date'].dt.month

print ("\nFive random records from the `Drugs` table:")
display(Drugs.iloc[np.random.choice (len (Drugs), 5)])

assert Drugs['Units'].dtype == 'float64'
assert Drugs['Month'].dtype == 'int64'
```

    Column 'Med' has type <class 'str'>.
    Column 'Admin Date' has type <class 'str'>.
    Column 'Units' has type <class 'str'>.
    
    Five random records from the `Drugs` table:
    


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
      <th>ID</th>
      <th>Med</th>
      <th>Admin Date</th>
      <th>Units</th>
      <th>Month</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>1974</th>
      <td>126</td>
      <td>Med A</td>
      <td>2012-08-13</td>
      <td>7500.0</td>
      <td>8</td>
    </tr>
    <tr>
      <th>1692</th>
      <td>111</td>
      <td>Med A</td>
      <td>2012-07-07</td>
      <td>700.0</td>
      <td>7</td>
    </tr>
    <tr>
      <th>267</th>
      <td>18</td>
      <td>Med A</td>
      <td>2012-07-16</td>
      <td>3300.0</td>
      <td>7</td>
    </tr>
    <tr>
      <th>16</th>
      <td>1</td>
      <td>Med A</td>
      <td>2012-08-13</td>
      <td>1700.0</td>
      <td>8</td>
    </tr>
    <tr>
      <th>880</th>
      <td>60</td>
      <td>Med A</td>
      <td>2012-08-31</td>
      <td>2500.0</td>
      <td>8</td>
    </tr>
  </tbody>
</table>
</div>


**Exercise 1** (1 point). Again, all you need to do is read and run the following code cell. It creates an SQLite database file named `drug_trial.db` and copies the Pandas dataframe from above into it as a table named `Drugs`.

The `conn` variable holds a live connection to this data.


```python
# Import Drugs_soln dataframe above to sqlite database
# Connect to a database (or create one if it doesn't exist)
conn = db.connect('drug_trial.db')
Drugs.to_sql('Drugs', conn, if_exists='replace', index=False)
pd.read_sql_query('SELECT * FROM Drugs LIMIT 5', conn)
```

    c:\program files\python37\lib\site-packages\pandas\core\generic.py:2712: UserWarning: The spaces in these column names will not be changed. In pandas versions < 0.14, spaces were converted to underscores.
      method=method,
    




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
      <th>ID</th>
      <th>Med</th>
      <th>Admin Date</th>
      <th>Units</th>
      <th>Month</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>1</td>
      <td>Med A</td>
      <td>2012-07-02 00:00:00</td>
      <td>1500.0</td>
      <td>7</td>
    </tr>
    <tr>
      <th>1</th>
      <td>1</td>
      <td>Med A</td>
      <td>2012-07-06 00:00:00</td>
      <td>1500.0</td>
      <td>7</td>
    </tr>
    <tr>
      <th>2</th>
      <td>1</td>
      <td>Med A</td>
      <td>2012-07-09 00:00:00</td>
      <td>1500.0</td>
      <td>7</td>
    </tr>
    <tr>
      <th>3</th>
      <td>1</td>
      <td>Med A</td>
      <td>2012-07-11 00:00:00</td>
      <td>1500.0</td>
      <td>7</td>
    </tr>
    <tr>
      <th>4</th>
      <td>1</td>
      <td>Med A</td>
      <td>2012-07-13 00:00:00</td>
      <td>1500.0</td>
      <td>7</td>
    </tr>
  </tbody>
</table>
</div>



**Exercise 2** (2 points). **Suppose you want to know the average dose, for each medication (A and B) and month ranging from July to November.**

For example, it will turn out that in July the average dose of drug A was 5,129.56 units (rounded to two decimal places), and in September the average dose of drug B was 7.04.

Write some code to perform this calculation. Store your results in a Pandas data frame named `avg_dose` having the following three columns:
- `'Month'`: The month;
- `'Med'`: The medication, either `'Med A'` and `'Med B'`;
- `'Units'`: The average dose, **rounded to 2 decimal digits**.

> You can write either Pandas code or SQL code. If using Pandas, the data exists in the `Drugs` dataframe; if using SQL, the `conn` database connection holds a table named `Drugs`.


```python
query = '''
    
    select month, med, round(avg(units), 2) as 'Units'
    from drugs
    where month > 6 and month < 12
    group by 1, 2
    order by 2, 1
    
    '''

avg_dose = pd.read_sql_query(query, conn)

# Show your solution:
display(avg_dose)
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
      <th>Month</th>
      <th>Med</th>
      <th>Units</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>7</td>
      <td>Med A</td>
      <td>5129.56</td>
    </tr>
    <tr>
      <th>1</th>
      <td>8</td>
      <td>Med A</td>
      <td>5645.78</td>
    </tr>
    <tr>
      <th>2</th>
      <td>9</td>
      <td>Med A</td>
      <td>5311.88</td>
    </tr>
    <tr>
      <th>3</th>
      <td>11</td>
      <td>Med A</td>
      <td>10757.14</td>
    </tr>
    <tr>
      <th>4</th>
      <td>9</td>
      <td>Med B</td>
      <td>7.04</td>
    </tr>
    <tr>
      <th>5</th>
      <td>10</td>
      <td>Med B</td>
      <td>5.78</td>
    </tr>
    <tr>
      <th>6</th>
      <td>11</td>
      <td>Med B</td>
      <td>5.60</td>
    </tr>
  </tbody>
</table>
</div>



```python
# Test code
# Read what we believe is the exact result (up to permutations)
avg_dose_soln = pd.read_csv(data_paths['avg_dose_soln.csv'])

# Check that we got a data frame of the expected shape:
assert 'avg_dose' in globals(), "You need to store your results in a dataframe named `avg_dose`."
assert type(avg_dose) is type(pd.DataFrame()), "`avg_dose` does not appear to be a Pandas dataframe."
assert len(avg_dose) == len(avg_dose_soln), "The number of rows of `avg_dose` does not match our solution."
assert set(avg_dose.columns) == set(['Month', 'Med', 'Units']), "Your table does not have the right set of columns."

assert tibbles_are_equivalent(avg_dose, avg_dose_soln)
print("\n(Passed!)")
```

    
    (Passed!)
    

**Exercise 3** (6 points). For each month, write some code to calculate the following:
- (3 points) How many patients switched from medication A to medication B? Store youre results in a Pandas dataframe named `swt_count`.
- (3 points) How many patients started on medication B, having never been on medication A before? Store your results in a Pandas dataframe named `nst_count`.

The two dataframes should have two columns: `Month` and `Count`. Again, you can choose to use SQL queries or Pandas directly to generate these dataframes.

> If it's helpful, recall that patients can only be switched from medication A to medication B, but not from B back to A.


```python
# Write your solution to compute `swt_count` in this code cell.

query = '''
    
    select t2.start_month_b as Month, count(*) as Count
    from (
        select id, med as med_a, min(month) as start_month_a
        from drugs
        where med = "Med A"
        group by 1
    ) as t1
    join (
        select id, med as med_b, min(month) as start_month_b
        from drugs
        where med = "Med B"
        group by 1
    ) as t2 on t2.id = t1.id
    group by 1
    
    --select 
    --    first_month as Month,
    --    count(distinct id) as Count
    --from (
    --    select 
    --        d.id,
    --        min(month) as first_month
    --    from (
    --        select
    --            id,
    --            count(distinct case when med = 'Med A' then id end) as a,
    --            count(distinct case when med = 'Med B' then id end) as b
    --        from drugs
    --        group by 1
    --        having a > 0 and b > 0
    --    ) as a
    --    join drugs d on a.id = d.id
    --    where med = 'Med B'
    --    group by 1
    --) as b
    --group by 1
    
'''

swt_count = pd.read_sql_query(query, conn)
```


```python
Drugs
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
      <th>ID</th>
      <th>Med</th>
      <th>Admin Date</th>
      <th>Units</th>
      <th>Month</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>1</td>
      <td>Med A</td>
      <td>2012-07-02</td>
      <td>1500.0</td>
      <td>7</td>
    </tr>
    <tr>
      <th>1</th>
      <td>1</td>
      <td>Med A</td>
      <td>2012-07-06</td>
      <td>1500.0</td>
      <td>7</td>
    </tr>
    <tr>
      <th>2</th>
      <td>1</td>
      <td>Med A</td>
      <td>2012-07-09</td>
      <td>1500.0</td>
      <td>7</td>
    </tr>
    <tr>
      <th>3</th>
      <td>1</td>
      <td>Med A</td>
      <td>2012-07-11</td>
      <td>1500.0</td>
      <td>7</td>
    </tr>
    <tr>
      <th>4</th>
      <td>1</td>
      <td>Med A</td>
      <td>2012-07-13</td>
      <td>1500.0</td>
      <td>7</td>
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
      <th>2017</th>
      <td>129</td>
      <td>Med A</td>
      <td>2012-08-27</td>
      <td>5200.0</td>
      <td>8</td>
    </tr>
    <tr>
      <th>2018</th>
      <td>129</td>
      <td>Med A</td>
      <td>2012-08-30</td>
      <td>5200.0</td>
      <td>8</td>
    </tr>
    <tr>
      <th>2019</th>
      <td>129</td>
      <td>Med A</td>
      <td>2012-09-04</td>
      <td>5200.0</td>
      <td>9</td>
    </tr>
    <tr>
      <th>2020</th>
      <td>129</td>
      <td>Med A</td>
      <td>2012-09-06</td>
      <td>6500.0</td>
      <td>9</td>
    </tr>
    <tr>
      <th>2021</th>
      <td>129</td>
      <td>Med B</td>
      <td>2012-09-13</td>
      <td>10.0</td>
      <td>9</td>
    </tr>
  </tbody>
</table>
<p>2022 rows × 5 columns</p>
</div>




```python
# Test code for exercise_a
# Read what we believe is the exact result
swt_count_soln = pd.read_csv(data_paths['swt_count_soln.csv'])

# Check that we got a data frame of the expected shape:
assert 'swt_count' in globals ()
assert type (swt_count) is type (pd.DataFrame ())
assert len (swt_count) == len (swt_count_soln)
assert set (swt_count.columns) == set (['Month', 'Count'])

print ("Number of patients who switched from Med A to Med B each month:")
display (swt_count)

assert tibbles_are_equivalent (swt_count, swt_count_soln)
print ("\n(Passed!)")
```

    Number of patients who switched from Med A to Med B each month:
    


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
      <th>Month</th>
      <th>Count</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>9</td>
      <td>71</td>
    </tr>
    <tr>
      <th>1</th>
      <td>10</td>
      <td>10</td>
    </tr>
  </tbody>
</table>
</div>


    
    (Passed!)
    


```python
# Write your solution to compute `nst_count` in this code cell.

query = '''

    select t2.start_month_b as Month, count(*) as Count
    from (
        select id, med as med_b, min(month) as start_month_b
        from drugs
        where med = "Med B"
        group by 1
    ) as t2
    left join (
        select id, med as med_a, min(month) as start_month_a
        from drugs
        where med = "Med A"
        group by 1
    ) as t1 on t1.id = t2.id
    where t1.start_month_a is null
    group by 1    
    
    --select 
    --    Month,
    --    count(distinct d.id) as Count
    --from drugs d 
    --left join (
    --    select
    --        id,
    --        count(distinct case when med = 'Med A' then id end) as a,
    --        count(distinct case when med = 'Med B' then id end) as b
    --    from drugs
    --    group by 1
    --    having a > 0 and b > 0
    --) as a on a.id = d.id
    --where a.id is null and med = 'Med B'
    --group by 1
    
'''
nst_count = pd.read_sql_query(query, conn)
```


```python
# Test code for exercise_b
# Read what we believe is the exact result
nst_count_soln_corrected = pd.read_csv(data_paths['nst_count_soln--corrected.csv'])
nst_count_soln_ok = pd.read_csv(data_paths['nst_count_soln.csv'])

# Check that we got a data frame of the expected shape:
assert 'nst_count' in globals ()
assert type (nst_count) is type (pd.DataFrame ())
assert (len (nst_count) == len (nst_count_soln_corrected)) or (len (nst_count) == len (nst_count_soln_ok))
assert set (nst_count.columns) == set (['Month', 'Count'])

print ("Number of patients who newly start Med B each month:")
display (nst_count)

assert tibbles_are_equivalent(nst_count, nst_count_soln_ok) \
       or tibbles_are_equivalent(nst_count, nst_count_soln_corrected)
print ("\n(Passed!)")
```

    Number of patients who newly start Med B each month:
    


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
      <th>Month</th>
      <th>Count</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>9</td>
      <td>5</td>
    </tr>
    <tr>
      <th>1</th>
      <td>10</td>
      <td>5</td>
    </tr>
    <tr>
      <th>2</th>
      <td>11</td>
      <td>6</td>
    </tr>
  </tbody>
</table>
</div>


    
    (Passed!)
    


```python
# Some cleanup code
conn.close()
```

**Fin!** Well done! If you have successfully completed this problem, move on to the next one. Good luck!
