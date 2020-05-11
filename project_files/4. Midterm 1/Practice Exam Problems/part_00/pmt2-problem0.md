# Problem 0: Graph search

This problem tests your familiarity with Pandas data frames. As such, you'll need this import:


```python
import sys
print(sys.version)

import pandas as pd
print(pd.__version__)
```

    3.7.4 (tags/v3.7.4:e09359112e, Jul  8 2019, 20:34:20) [MSC v.1916 64 bit (AMD64)]
    0.25.3
    

This problem has four exercises worth a total of ten (10) points.

## Dataset: (simplified) airport segments

The dataset for this problem is a simplified version of the airport segments dataset from Notebook 11. Start by getting and inspecting the data, so you know what you will be working with.


```python
from cse6040utils import on_vocareum, download_all

datasets = {'L_AIRPORT_ID.csv': 'e9f250e3c93d625cce92d08648c4bbf0',
            'segments.csv': 'b5e8ce736bc36a9dd89c3ae0f6eeb491',
            'two_away_solns.csv': '7421b3eead7b5107c7fbd565228e50c7'}

DATA_SUFFIX = "us-flights/"
data_paths = download_all(datasets, local_suffix=DATA_SUFFIX, url_suffix=DATA_SUFFIX)

print("\n(All data appears to be ready.)")
```

    'L_AIRPORT_ID.csv' is ready!
    'segments.csv' is ready!
    'two_away_solns.csv' is ready!
    
    (All data appears to be ready.)
    

The first bit of data you'll need is a list of airports, each of which has a code and a string description.


```python
airports = pd.read_csv(data_paths['L_AIRPORT_ID.csv'])
airports.head()
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
      <th>Code</th>
      <th>Description</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>10001</td>
      <td>Afognak Lake, AK: Afognak Lake Airport</td>
    </tr>
    <tr>
      <th>1</th>
      <td>10003</td>
      <td>Granite Mountain, AK: Bear Creek Mining Strip</td>
    </tr>
    <tr>
      <th>2</th>
      <td>10004</td>
      <td>Lik, AK: Lik Mining Camp</td>
    </tr>
    <tr>
      <th>3</th>
      <td>10005</td>
      <td>Little Squaw, AK: Little Squaw Airport</td>
    </tr>
    <tr>
      <th>4</th>
      <td>10006</td>
      <td>Kizhuyak, AK: Kizhuyak Bay</td>
    </tr>
  </tbody>
</table>
</div>



The other bit of data you'll need is a list of available direct connections.


```python
segments = pd.read_csv(data_paths['segments.csv'])
print("There are {} direct flight segments.".format(len(segments)))
segments.head()
```

    There are 4191 direct flight segments.
    




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
      <th>ORIGIN_AIRPORT_ID</th>
      <th>DEST_AIRPORT_ID</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>10135</td>
      <td>10397</td>
    </tr>
    <tr>
      <th>1</th>
      <td>10135</td>
      <td>11433</td>
    </tr>
    <tr>
      <th>2</th>
      <td>10135</td>
      <td>13930</td>
    </tr>
    <tr>
      <th>3</th>
      <td>10140</td>
      <td>10397</td>
    </tr>
    <tr>
      <th>4</th>
      <td>10140</td>
      <td>10423</td>
    </tr>
  </tbody>
</table>
</div>



## Exercises

Complete the following exercises.

**Exercise 0** (1 point). Given an airport code, implement the function, `get_description(code, airports)`, so that it returns the row of `airports` having that code.

For example,

```python
    get_description(10397, airports)
```

would return the dataframe,

| | Code | Description |
|:-:|:-:|:-:|
| **373** | 10397 | Atlanta, GA: Hartsfield-Jackson Atlanta Intern... |


```python
def get_description(code, airports):
    return airports[airports['Code'] == code]
# Demo:
get_description(10397, airports)
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
      <th>Code</th>
      <th>Description</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>373</th>
      <td>10397</td>
      <td>Atlanta, GA: Hartsfield-Jackson Atlanta Intern...</td>
    </tr>
  </tbody>
</table>
</div>




```python
# Test cell: `get_description_test`

from numpy.random import choice
for offset in choice(len(airports), size=10):
    code = airports.iloc[offset]['Code']
    df = get_description(code, airports)
    assert type(df) is pd.DataFrame
    assert len(df) == 1
    assert (df['Code'] == code).all()
    
print("\n(Passed!)")
```

    
    (Passed!)
    

**Exercise 1** (2 points). Suppose that, instead of one code, you are given a Python set of codes. Implement the function, `get_all_descriptions(codes, airports)`, so that it returns a dataframe whose rows consist of all rows from `airports` that match one of the codes in `codes`.

For example,

```python
    get_all_descriptions({10397, 12892, 14057}, airports)
```

would return,

| | Code | Description |
|:-:|:-:|:-:|
| **373** | 10397 | Atlanta, GA: Hartsfield-Jackson Atlanta Intern... |
| **2765** | 12892 | Los Angeles, CA: Los Angeles International |
| **3892** | 14057 | Portland, OR: Portland International |


```python
def get_all_descriptions(codes, airports):
    assert type(codes) is set
    return airports[airports['Code'].isin(codes)]
    #return airports[airports['Code'].apply(lambda x : x in codes)]
    
get_all_descriptions({10397, 12892, 14057}, airports)
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
      <th>Code</th>
      <th>Description</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>373</th>
      <td>10397</td>
      <td>Atlanta, GA: Hartsfield-Jackson Atlanta Intern...</td>
    </tr>
    <tr>
      <th>2765</th>
      <td>12892</td>
      <td>Los Angeles, CA: Los Angeles International</td>
    </tr>
    <tr>
      <th>3892</th>
      <td>14057</td>
      <td>Portland, OR: Portland International</td>
    </tr>
  </tbody>
</table>
</div>




```python
# Test cell: `get_all_descriptions_test`

from numpy.random import choice
offsets = choice(len(airports), size=10)
codes = set(airports.iloc[offsets]['Code'])
df = get_all_descriptions(codes, airports)
assert type(df) is pd.DataFrame
assert len(df) == len(codes)
assert set(df['Code']) == codes

print("\n(Passed!)")
```

    
    (Passed!)
    

**Exercise 2** (2 points). Implement the function, `find_description(desc, airports)`, so that it returns the subset of rows of the dataframe `airports` whose `Description` string contains `desc`, where `desc` is a string.

For example,

```python
    find_description('Atlanta', airports)
```

should return a dataframe with these rows:

| Code  | Description                                       |
|:-----:|:-------------------------------------------------:|
| 10397	| Atlanta, GA: Hartsfield-Jackson Atlanta Intern... |
| 11790	| Atlanta, GA: Fulton County Airport-Brown Field    |
| 11838	| Atlanta, GA: Newnan Coweta County                 |
| 12445	| Atlanta, GA: Perimeter Mall Helipad               |
| 12449	| Atlanta, GA: Beaver Ruin                          |
| 12485	| Atlanta, GA: Galleria                             |
| 14050	| Atlanta, GA: Dekalb Peachtree                     |
| 14430	| Peachtree City, GA: Atlanta Regional Falcon Field |

Notice that the last row of this dataframe has "Atlanta" in the middle of the description.

> _Hint_: The easiest way to do this problem is to apply a neat feature of Pandas, which is that there are functions that help do string searches within a column (i.e., within a Series): https://pandas.pydata.org/pandas-docs/stable/text.html


```python
def find_description(desc, airports):
    return airports[airports.Description.str.contains(desc, case = False)]
    
find_description('Atlanta', airports)
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
      <th>Code</th>
      <th>Description</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>373</th>
      <td>10397</td>
      <td>Atlanta, GA: Hartsfield-Jackson Atlanta Intern...</td>
    </tr>
    <tr>
      <th>1717</th>
      <td>11790</td>
      <td>Atlanta, GA: Fulton County Airport-Brown Field</td>
    </tr>
    <tr>
      <th>1762</th>
      <td>11838</td>
      <td>Atlanta, GA: Newnan Coweta County</td>
    </tr>
    <tr>
      <th>2350</th>
      <td>12445</td>
      <td>Atlanta, GA: Perimeter Mall Helipad</td>
    </tr>
    <tr>
      <th>2354</th>
      <td>12449</td>
      <td>Atlanta, GA: Beaver Ruin</td>
    </tr>
    <tr>
      <th>2387</th>
      <td>12485</td>
      <td>Atlanta, GA: Galleria</td>
    </tr>
    <tr>
      <th>3885</th>
      <td>14050</td>
      <td>Atlanta, GA: Dekalb Peachtree</td>
    </tr>
    <tr>
      <th>4222</th>
      <td>14430</td>
      <td>Peachtree City, GA: Atlanta Regional Falcon Field</td>
    </tr>
  </tbody>
</table>
</div>




```python
# Test cell: `lookup_description_test`

assert len(find_description('Los Angeles', airports)) == 4
assert len(find_description('Washington', airports)) == 12
assert len(find_description('Arizona', airports)) == 0
assert len(find_description('Warsaw', airports)) == 2

print("\n(Passed!)")
```

    
    (Passed!)
    

**Exercise 3** (4 points). Suppose you are given an airport code. Implement a function, `find_two_away(code, segments)`, so that it finds all airports that are **two hops** away. It should return this result as a Python set.

For example, the `segments` table happens to include these two rows:

|    | ORIGIN_AIRPORT_ID | DEST_AIRPORT_ID |
|:-:|:-:|:-:|
| ... | ... | ... |
| **178** | 10397 | 12892 |
| ... | ... | ... |
| **2155** | 12892 | 14057 |
| ... | ... | ... |

We say that 14057 is "two hops away" because there is one segment from 10397 to 12892, followed by a second segment from 12892 to 14057. Thus, the set returned by `find_two_away(code, segments)` should include 14057, i.e.,

```python
    assert 14057 in find_two_away(10397, segments)
```

Your function may assume that the given `code` is valid, that is, appears in the `segments` data frame and has at least one outgoing segment.


```python
def find_two_away(code, segments):
    return set(segments[segments['ORIGIN_AIRPORT_ID'] == code].merge(segments, how = 'left', left_on = 'DEST_AIRPORT_ID', right_on = 'ORIGIN_AIRPORT_ID')['DEST_AIRPORT_ID_y'])
    
atl_two_hops = find_two_away(10397, segments)
atl_desc = get_description(10397, airports)['Description'].iloc[0]
print("Your solution found {} airports that are two hops from '{}'.".format(len(atl_two_hops), atl_desc))
```

    Your solution found 277 airports that are two hops from 'Atlanta, GA: Hartsfield-Jackson Atlanta International'.
    


```python
# Test cell: `find_two_away_test`

assert 14057 in find_two_away(10397, segments)
assert len(atl_two_hops) == 277

print("\n(Passed first test.)")
```

    
    (Passed first test.)
    


```python
# Test cell: `find_two_away_test2`
if False:
    solns = {}
    for code in airports['Code']:
        two_away = find_two_away(code, segments)
        if code not in solns:
            solns[code] = len(two_away)
    with open('{}two_away_solns.csv'.format(DATA_SUFFIX), 'w') as fp:
        fp.write('Code,TwoAway\n')
        for code, num_two_away in solns.items():
            fp.write('{},{}\n'.format(code, num_two_away))
            
two_away_solns = pd.read_csv(data_paths['two_away_solns.csv'])
for row in range(len(two_away_solns)):
    code = two_away_solns['Code'].iloc[row]
    count = two_away_solns['TwoAway'].iloc[row]
    your_count = len(find_two_away(code, segments))
    msg = "Expected {} airports two-away from {}, but your code found {} instead.".format(count, code, your_count)
    assert your_count == count, msg
print("\n(Passed!)")
```

    
    (Passed!)
    

**Fin!** If you've reached this point and all tests above pass, you are ready to submit your solution to this problem. Don't forget to save you work prior to submitting.
