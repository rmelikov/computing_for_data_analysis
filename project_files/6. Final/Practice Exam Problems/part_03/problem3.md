# Problem 3

This problem checks that you can perform some basic data cleaning and analysis. You'll work with what we think is a pretty interesting dataset, which can tell us something about how people move within the United States.

This problem has five (5) exercises (numbered 0-4) and is worth a total of ten (10) points.

## Setup: IRS Tax Migration Data

The data for this problem comes from the IRS, which can tell where many households move from or to in any given year based on their tax returns.

For your convenience, we've place the data you'll need on Vocareum. If you wish to work in your local environment, you'll need the following copy of these files, which are split by year among four consecutive years (2011-2015). 

- 2011-2012 data: https://cse6040.gatech.edu/datasets/stateoutflow1112.csv
- 2012-2013 data: https://cse6040.gatech.edu/datasets/stateoutflow1213.csv
- 2013-2014 data: https://cse6040.gatech.edu/datasets/stateoutflow1314.csv
- 2014-2015 data: https://cse6040.gatech.edu/datasets/stateoutflow1415.csv

These data files reference states by their FIPS codes. So, you'll need some additional data to translate state FIPS numbers to "friendly" names. (Again, this file is pre-loaded into the Vocareum environment and the link below is for your use if you are working locally.)

- FIPS data: https://cse6040.gatech.edu/datasets/fips-state-2010-census.txt

> These are state-level data though county-level data also exist elsewhere. If you ever need that, you'll find it at the IRS website: https://www.irs.gov/uac/soi-tax-stats-migration-data. And if you ever need the original FIPS codes data, see the Census Bureau website: https://www.census.gov/geo/reference/codes/cou.html.

Beyond the data, you'll also need the following Python modules.


```python
from IPython.display import display
import pandas as pd

def tbc (X):
    var_names = sorted (X.columns)
    Y = X[var_names].copy ()
    Y.sort_values (by=var_names, inplace=True)
    Y.set_index ([list (range (0, len (Y)))], inplace=True)
    return Y

def tbeq(A, B):
    A_c = tbc(A)
    B_c = tbc(B)
    return A_c.eq(B_c).all().all()
```

Here is a sneak peek of what one of the data files looks like. Note the encoding specification, which may be needed to get Pandas to parse it.

> The cell below defines a function called `fn(fn_base, dirname)`, which you can use to form a fully qualified file path for accessing a data file. The argument `fn_base` is the name of the file (e.g., `'foo.csv'`) and `dirname` is the subdirectory path containing that file. It has a default value that is appropriate for the Vocareum platform. In some of the exercises below, you'll need to load files and you should, therefore, use the `fn` function to generate the names of the files you need to load. Read the code cells below to see how `fn()` is used.


```python
def fn(fn_base, dirname='./'):
    return '{}{}'.format(dirname, fn_base)

# Demo of `fn()`, which we'll use to look at the first few rows of one of the input files.
print ("First few rows...")
display (pd.read_csv (fn('stateoutflow1112.csv'), encoding='latin-1').head (3))

print ("\n...and some from the middle somewhere...")
display (pd.read_csv (fn('stateoutflow1112.csv'), encoding='latin-1').head (1000).tail (3))
```

    First few rows...
    


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
      <th>y1_statefips</th>
      <th>y2_statefips</th>
      <th>y2_state</th>
      <th>y2_state_name</th>
      <th>n1</th>
      <th>n2</th>
      <th>AGI</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>1</td>
      <td>96</td>
      <td>AL</td>
      <td>AL Total Migration US and Foreign</td>
      <td>51971</td>
      <td>107304</td>
      <td>2109108</td>
    </tr>
    <tr>
      <th>1</th>
      <td>1</td>
      <td>97</td>
      <td>AL</td>
      <td>AL Total Migration US</td>
      <td>50940</td>
      <td>105006</td>
      <td>2059642</td>
    </tr>
    <tr>
      <th>2</th>
      <td>1</td>
      <td>98</td>
      <td>AL</td>
      <td>AL Total Migration Foreign</td>
      <td>1031</td>
      <td>2298</td>
      <td>49465</td>
    </tr>
  </tbody>
</table>
</div>


    
    ...and some from the middle somewhere...
    


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
      <th>y1_statefips</th>
      <th>y2_statefips</th>
      <th>y2_state</th>
      <th>y2_state_name</th>
      <th>n1</th>
      <th>n2</th>
      <th>AGI</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>997</th>
      <td>22</td>
      <td>13</td>
      <td>GA</td>
      <td>GEORGIA</td>
      <td>2526</td>
      <td>4984</td>
      <td>83544</td>
    </tr>
    <tr>
      <th>998</th>
      <td>22</td>
      <td>6</td>
      <td>CA</td>
      <td>CALIFORNIA</td>
      <td>2267</td>
      <td>3974</td>
      <td>89566</td>
    </tr>
    <tr>
      <th>999</th>
      <td>22</td>
      <td>5</td>
      <td>AR</td>
      <td>ARKANSAS</td>
      <td>1355</td>
      <td>2851</td>
      <td>52356</td>
    </tr>
  </tbody>
</table>
</div>


The `y1_.*` fields describe the state in which the household originated (the "source" vertices) and the `y2_.*` fields describe the state into which the household moved (the "destination"). Column `n1` is the number of such households for the given (source, destination) locations. Notice that there are some special FIPS designators as well, e.g., in the first three rows. These show total outflows, which you can use to normalize counts.

**Exercise 0** (2 points). The data files are separated by year. Write some code to merge all of the data into a single Pandas data frame called `StateOutFlows`. It should have the same columns as the original data (e.g., `y1_statefips`, `y2_statefips`), plus an additional `year` column to hold the year.

> Represent the year by a 4-digit value, e.g., `2011` rather than just `11`. Also, use the starting year for the file. That is, if the file is the `1314` file, use `2013` as the year.

Here is my solution.


```python
from glob import glob
from re import search

StateOutFlows = pd.concat(
    [
        pd.read_csv(f, encoding = 'latin-1')
            .assign(year = '20' + search('[a-z]+(?P<year>[0-9]{2})', f).group('year'))
            .astype({'year' : 'int32'})
        for f in glob('stateoutflow*[0-9].csv')
    ],
    ignore_index = True
)
```

And here is a solution that was provided by the school.

```Python
all_df = []
for yy in range (11, 15):
    filename = "stateoutflow{}{}.csv".format (yy, yy+1)
    df = pd.read_csv (fn(filename), encoding='latin-1')
    df['year'] = 2000 + yy
    all_df.append (df)   
StateOutFlows = pd.concat (all_df)
```


```python
assert 'StateOutFlows' in globals ()
assert type (StateOutFlows) is type (pd.DataFrame ())

print ("Found {} outflow records between 2011-2015.".format (len (StateOutFlows)))
print ("First few rows...")
display (StateOutFlows.head ())

StateOutFlows_soln = pd.read_csv (fn('StateOutFlows_soln.csv'))
assert tbeq (StateOutFlows, StateOutFlows_soln)

print ("\n(Passed!)")
```

    Found 11320 outflow records between 2011-2015.
    First few rows...
    


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
      <th>y1_statefips</th>
      <th>y2_statefips</th>
      <th>y2_state</th>
      <th>y2_state_name</th>
      <th>n1</th>
      <th>n2</th>
      <th>AGI</th>
      <th>year</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>1</td>
      <td>96</td>
      <td>AL</td>
      <td>AL Total Migration US and Foreign</td>
      <td>51971</td>
      <td>107304</td>
      <td>2109108</td>
      <td>2011</td>
    </tr>
    <tr>
      <th>1</th>
      <td>1</td>
      <td>97</td>
      <td>AL</td>
      <td>AL Total Migration US</td>
      <td>50940</td>
      <td>105006</td>
      <td>2059642</td>
      <td>2011</td>
    </tr>
    <tr>
      <th>2</th>
      <td>1</td>
      <td>98</td>
      <td>AL</td>
      <td>AL Total Migration Foreign</td>
      <td>1031</td>
      <td>2298</td>
      <td>49465</td>
      <td>2011</td>
    </tr>
    <tr>
      <th>3</th>
      <td>1</td>
      <td>1</td>
      <td>AL</td>
      <td>AL Non-migrants</td>
      <td>1584665</td>
      <td>3603439</td>
      <td>87222478</td>
      <td>2011</td>
    </tr>
    <tr>
      <th>4</th>
      <td>1</td>
      <td>13</td>
      <td>GA</td>
      <td>GEORGIA</td>
      <td>9920</td>
      <td>19470</td>
      <td>329213</td>
      <td>2011</td>
    </tr>
  </tbody>
</table>
</div>


    
    (Passed!)
    

Observe that the `y2_state_name` column has some special values.

For instance, suppose you want to know the _total_ number of households that filed returns within the state of Alabama. Evidently, there is a row in each year with `AL Total Migration US and Foreign` as well as an `AL Non-migrants`, the sum of which is presumably the total number of returns.

**Exercise 1** (4 points). Create a new Pandas data frame named `Totals` with one row for each state and the following five (5) columns:

- `st`: The two-letter state abbreviation
- `year`: The year of the observation
- `migrated`: The state's `Total Migration US and Foreign` value during that year
- `stayed`: The state's `Non-migrants` value that year
- `all`: The sum of `migrated` and `stayed` columns

> _Hint:_ Before proceeding, run the cell below and observe how the strings marking total migrations appear.


```python
print ("=== HINT! Observe this hint before proceeding with your solution... ===\n")
print (list (StateOutFlows[StateOutFlows['y2_state'] == 'GA']['y2_state_name'].unique ()))
```

    === HINT! Observe this hint before proceeding with your solution... ===
    
    ['GEORGIA', 'GA Total Migration US and Foreign', 'GA Total Migration US', 'GA Total Migration Foreign', 'GA Non-migrants', 'Georgia', 'GA Total Migration-US and Foreign', 'GA Total Migration-US', 'GA Total Migration-Foreign', 'GA Total Migration-Same State']
    

**Solution**

Here is a solution that was provided by the school and below that is my solution.

```Python
def ends_in (pattern, s):
    import re
    return re.match ("^.*{}$".format (pattern), s) is not None

def ends_in_total_migration (s):
    return ends_in ('Total Migration[ -]US and Foreign', s)

def ends_in_non_migrants (s):
    return ends_in ('Non-migrants', s)

migrants = StateOutFlows['y2_state_name'].apply (ends_in_total_migration)
stayed = StateOutFlows['y2_state_name'].apply (ends_in_non_migrants)

Migrated = StateOutFlows[migrants][['y2_state', 'year', 'n1']] \
           .rename (columns={'y2_state': 'st', 'n1': 'migrated'})

Stayed = StateOutFlows[stayed][['y2_state', 'year', 'n1']] \
         .rename (columns={'y2_state': 'st', 'n1': 'stayed'})

Totals = pd.merge (Migrated, Stayed, on=['st', 'year'])
Totals['all'] = Totals['migrated'] + Totals['stayed']
```


```python
migrated = (
    StateOutFlows
        .loc[lambda df: df['y2_state_name'].str.contains('Total Migration.US and Foreign', regex = True)]
        .filter(['y2_state', 'n1', 'year'])
)

stayed = (
    StateOutFlows
        .loc[lambda df: df['y2_state_name'].str.contains('Non-migrants', regex = True)]
        .filter(['y2_state', 'n1', 'year'])
)

Totals = (
    pd.merge(
        migrated, stayed, on = ['y2_state', 'year'], suffixes = ('_mg', '_st')
    )
        .assign(all = lambda df: df['n1_mg'] + df['n1_st'])
        .rename(columns = {'y2_state' : 'st', 'n1_mg' : 'migrated', 'n1_st' : 'stayed'})
        .filter(['st', 'year', 'migrated', 'stayed', 'all'])
        .sort_values(by = ['st', 'year'])
        .reset_index(drop = True)
)
```


```python
Totals_soln = pd.read_csv (fn('Totals_soln.csv'))

assert 'Totals' in globals ()
assert type (Totals) is type (Totals_soln)
assert set (Totals.columns) == set (['st', 'year', 'migrated', 'stayed', 'all'])

print ("Some rows of Totals:")
print (Totals.head ())
print ("...")
print (Totals.tail ())

print ("\n({} rows total.)".format (len (Totals)))

assert tbeq (Totals, Totals_soln)
```

    Some rows of Totals:
       st  year  migrated   stayed      all
    0  AK  2011     19446   258223   277669
    1  AK  2012     20763   257450   278213
    2  AK  2013     19096   259665   278761
    3  AK  2014     13405   265963   279368
    4  AL  2011     51971  1584665  1636636
    ...
         st  year  migrated  stayed     all
    199  WV  2014     14869  631644  646513
    200  WY  2011     14651  209678  224329
    201  WY  2012     16277  210979  227256
    202  WY  2013     13960  212527  226487
    203  WY  2014      9834  216928  226762
    
    (204 rows total.)
    

**Exercise 2** (1 points). Load the FIPS codes from `fips-state-2010-census.txt`. Store them in a Pandas data frame named `FIPS`. Use the original column names from the input file: `STATE`, `STUSAB`, `STATE_NAME`, `STATENS`.

> Hint: You can use Pandas's [`read_csv()`](http://pandas.pydata.org/pandas-docs/stable/generated/pandas.read_csv.html) function to read the file. However, be sure to take a look at the file before you try to load it, so you know how to parse by setting the arguments of `read_csv()` appropriately.


```python
FIPS = pd.read_csv('fips-state-2010-census.txt', sep = '|')
```


```python
assert 'FIPS' in globals ()
assert type (FIPS) is type (pd.DataFrame ())
assert len (FIPS) == 57

print ("FIPS data frame, at location 10:\n")
print (FIPS.loc[10])
assert FIPS.loc[10, 'STATE_NAME'] == 'Georgia'

print ("\n(Passed!)")
```

    FIPS data frame, at location 10:
    
    STATE              13
    STUSAB             GA
    STATE_NAME    Georgia
    STATENS       1705317
    Name: 10, dtype: object
    
    (Passed!)
    

Inspect the test code above. Notice that the FIPS code for Georgia is 13, which is located at index position 10 of the data frame (i.e., at `FIPS.loc[10]`).

It would help if the index of the data frame were also the same as the FIPS state code (`STATE`). That way, you could use `FIPS.loc[13]` to get the state code for Georgia; in effect, converting the data frame into something similar to a Python dictionary.

**Exercise 3** (1 points). Convert the `STATE` column into an index. To do so, use the Pandas method, [`FIPS.set_index()`](http://pandas.pydata.org/pandas-docs/stable/generated/pandas.DataFrame.set_index.html). Set the arguments to `set_index()` so that the change is made in-place.


```python
FIPS = FIPS.set_index('STATE')
```


```python
display (FIPS[10:15])

assert set (FIPS.columns) == set (['STUSAB', 'STATE_NAME', 'STATENS'])
assert FIPS.loc[13, 'STATE_NAME'] == 'Georgia'
assert FIPS.loc[15, 'STATE_NAME'] == 'Hawaii'
print ("\n(Passed!)")
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
      <th>STUSAB</th>
      <th>STATE_NAME</th>
      <th>STATENS</th>
    </tr>
    <tr>
      <th>STATE</th>
      <th></th>
      <th></th>
      <th></th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>13</th>
      <td>GA</td>
      <td>Georgia</td>
      <td>1705317</td>
    </tr>
    <tr>
      <th>15</th>
      <td>HI</td>
      <td>Hawaii</td>
      <td>1779782</td>
    </tr>
    <tr>
      <th>16</th>
      <td>ID</td>
      <td>Idaho</td>
      <td>1779783</td>
    </tr>
    <tr>
      <th>17</th>
      <td>IL</td>
      <td>Illinois</td>
      <td>1779784</td>
    </tr>
    <tr>
      <th>18</th>
      <td>IN</td>
      <td>Indiana</td>
      <td>448508</td>
    </tr>
  </tbody>
</table>
</div>


    
    (Passed!)
    

## Migration edges

Using the code you've set up above, we can build a table of _migration edges_, that is, a succinct summary of the number of households that moved from one state to another, broken down by year. The following code cell does that, leaving the result in a Pandas data frame called `MigrationEdges`.


```python
pd.merge (StateOutFlows[['y1_statefips', 'y2_state', 'year', 'n1']], FIPS[['STUSAB']],
                  left_on='y1_statefips', right_index=True)
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
      <th>y1_statefips</th>
      <th>y2_state</th>
      <th>year</th>
      <th>n1</th>
      <th>STUSAB</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>1</td>
      <td>AL</td>
      <td>2011</td>
      <td>51971</td>
      <td>AL</td>
    </tr>
    <tr>
      <th>1</th>
      <td>1</td>
      <td>AL</td>
      <td>2011</td>
      <td>50940</td>
      <td>AL</td>
    </tr>
    <tr>
      <th>2</th>
      <td>1</td>
      <td>AL</td>
      <td>2011</td>
      <td>1031</td>
      <td>AL</td>
    </tr>
    <tr>
      <th>3</th>
      <td>1</td>
      <td>AL</td>
      <td>2011</td>
      <td>1584665</td>
      <td>AL</td>
    </tr>
    <tr>
      <th>4</th>
      <td>1</td>
      <td>GA</td>
      <td>2011</td>
      <td>9920</td>
      <td>AL</td>
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
      <th>11315</th>
      <td>56</td>
      <td>VT</td>
      <td>2014</td>
      <td>19</td>
      <td>WY</td>
    </tr>
    <tr>
      <th>11316</th>
      <td>56</td>
      <td>NH</td>
      <td>2014</td>
      <td>-1</td>
      <td>WY</td>
    </tr>
    <tr>
      <th>11317</th>
      <td>56</td>
      <td>DE</td>
      <td>2014</td>
      <td>-1</td>
      <td>WY</td>
    </tr>
    <tr>
      <th>11318</th>
      <td>56</td>
      <td>DC</td>
      <td>2014</td>
      <td>-1</td>
      <td>WY</td>
    </tr>
    <tr>
      <th>11319</th>
      <td>56</td>
      <td>RI</td>
      <td>2014</td>
      <td>-1</td>
      <td>WY</td>
    </tr>
  </tbody>
</table>
<p>11320 rows × 5 columns</p>
</div>




```python
Edges = StateOutFlows[['y1_statefips', 'y2_state', 'year', 'n1']]
Edges = pd.merge (Edges, FIPS[['STUSAB']],
                  left_on='y1_statefips', right_index=True)
Edges.rename (columns={'STUSAB': 'from', 'y2_state': 'to', 'n1': 'moved'}, inplace=True)
del Edges['y1_statefips']

MigrationEdges = Edges[Edges['from'] != Edges['to']]
MigrationEdges.head ()
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
      <th>to</th>
      <th>year</th>
      <th>moved</th>
      <th>from</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>4</th>
      <td>GA</td>
      <td>2011</td>
      <td>9920</td>
      <td>AL</td>
    </tr>
    <tr>
      <th>5</th>
      <td>FL</td>
      <td>2011</td>
      <td>7550</td>
      <td>AL</td>
    </tr>
    <tr>
      <th>6</th>
      <td>TN</td>
      <td>2011</td>
      <td>4237</td>
      <td>AL</td>
    </tr>
    <tr>
      <th>7</th>
      <td>TX</td>
      <td>2011</td>
      <td>4121</td>
      <td>AL</td>
    </tr>
    <tr>
      <th>8</th>
      <td>MS</td>
      <td>2011</td>
      <td>2868</td>
      <td>AL</td>
    </tr>
  </tbody>
</table>
</div>



Using the `MigrationEdges` data frame, we can (relatively) easily determine the top 5 states whose households moved to the state of Georgia over all years. Here is one way to do so:

1. Filter rows keeping only those containing `'GA'` as the destination.
2. Group the results by originating state.
3. Sum the results over all years.
4. Sort these results in descending order.
5. Emit just the top 5 results.


```python
# Steps 1 and 2
ToGA = (MigrationEdges['to'] == 'GA')
MovedToGA = MigrationEdges[ToGA].groupby ('from')

# Step 3
MovedToGA_counts_by_state = MovedToGA['moved'].sum ()
MovedToGA_counts_by_state[:10]
```




    from
    AK     1978
    AL    34691
    AR     3321
    AZ     5808
    CA    25857
    CO     6674
    CT     4713
    DC     1794
    DE     1619
    FL    89736
    Name: moved, dtype: int64




```python
# Steps 4 and 5: Sort and report the top 5
MovedToGA_counts_by_state.sort_values (ascending=False)[:5]
```




    from
    FL    89736
    TX    36614
    AL    34691
    NC    29759
    SC    27938
    Name: moved, dtype: int64



**Exercise 4** (2 points). Following a similar procedure, determine the top 5 states that Georgians moved to. Store the resulting names and counts in a variable named `GAExodus`.

**Solution**

Here is the school solution and below that is my solution.

```Python
FromGA = (MigrationEdges['from'] == 'GA')
MovedFromGA = MigrationEdges[FromGA].groupby ('to')
MovedFromGA_counts_by_state = MovedFromGA['moved'].sum ()
GAExodus = MovedFromGA_counts_by_state.sort_values (ascending=False)[:5]
```


```python
GAExodus = (
    MigrationEdges
        .loc[lambda df: df['from'].str.contains('GA')]
        .groupby('to')['moved'].sum()
        .sort_values(ascending = False)
        .head()
)
```


```python
assert 'GAExodus' in globals ()
assert type (GAExodus) is type (pd.Series ())
assert len (GAExodus) == 5

print ("=== The exodus from Georgia ===")
assert set (GAExodus.index) == set (['FL', 'TX', 'AL', 'NC', 'SC'])
assert (GAExodus.values == [86178, 50467, 32970, 30352, 30141]).all ()
print (GAExodus)

print ("\n(Passed!)")
```

    === The exodus from Georgia ===
    to
    FL    86178
    TX    50467
    AL    32970
    NC    30352
    SC    30141
    Name: moved, dtype: int64
    
    (Passed!)
    

**Fin!** If you've reached this point and all tests above pass, you are ready to submit your solution to this problem. Don't forget to save you work prior to submitting.
