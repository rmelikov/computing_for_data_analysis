# Part 1: NYC 311 calls

This notebook derives from a [demo by the makers of plot.ly](https://plot.ly/ipython-notebooks/big-data-analytics-with-pandas-and-sqlite/). We've adapted it to use [Bokeh (and HoloViews)](http://bokeh.pydata.org/en/latest/).

You will start with a large database of complaints filed by residents of New York City since 2010 via 311 calls. The full dataset is available at the [NYC open data portal](https://nycopendata.socrata.com/data). At about 6 GB and 10 million complaints, you can infer that a) you might not want to read it all into memory at once, and b) NYC residents have a lot to complain about. (Maybe only conclusion "a" is valid.) The notebook then combines the use of `sqlite`, `pandas`, and `bokeh`.

## Module setup

Before diving in, run the following cells to preload some functions you'll need later. These include a few functions from Notebook 7.


```python
import sys
print(sys.version) # Print Python version -- On Vocareum, it should be 3.7+

from IPython.display import display
import pandas as pd

from nb7utils import canonicalize_tibble, tibbles_are_equivalent, cast
```

    3.7.4 (tags/v3.7.4:e09359112e, Jul  8 2019, 20:34:20) [MSC v.1916 64 bit (AMD64)]
    

Lastly, some of the test cells will need some auxiliary files, which the following code cell will check for and, if they are missing, download.


```python
from nb9utils import download, get_path, auxfiles

for filename, checksum in auxfiles.items():
    download(filename, checksum=checksum, url_suffix="lab9-sql/")
    
print("(Auxiliary files appear to be ready.)")
```



<div class="bk-root">
    <a href="https://bokeh.org" target="_blank" class="bk-logo bk-logo-small bk-logo-notebook"></a>
    <span id="1001">Loading BokehJS ...</span>
</div>




    [https://cse6040.gatech.edu/datasets/lab9-sql/df_complaints_by_city_soln.csv]
    ==> 'resource/asnlib/publicdata/df_complaints_by_city_soln.csv' is already available.
    ==> Checksum test passes: b07d65c208bd791ea21679a3551ae265
    ==> 'resource/asnlib/publicdata/df_complaints_by_city_soln.csv' is ready!
    
    [https://cse6040.gatech.edu/datasets/lab9-sql/df_complaints_by_hour_soln.csv]
    ==> 'resource/asnlib/publicdata/df_complaints_by_hour_soln.csv' is already available.
    ==> Checksum test passes: f06fcd917876d51ad52ddc13b2fee69e
    ==> 'resource/asnlib/publicdata/df_complaints_by_hour_soln.csv' is ready!
    
    [https://cse6040.gatech.edu/datasets/lab9-sql/df_noisy_by_hour_soln.csv]
    ==> 'resource/asnlib/publicdata/df_noisy_by_hour_soln.csv' is already available.
    ==> Checksum test passes: 30f3fa7c753d4d3f4b3edfa1f6d05bcc
    ==> 'resource/asnlib/publicdata/df_noisy_by_hour_soln.csv' is ready!
    
    [https://cse6040.gatech.edu/datasets/lab9-sql/df_plot_stacked_fraction_soln.csv]
    ==> 'resource/asnlib/publicdata/df_plot_stacked_fraction_soln.csv' is already available.
    ==> Checksum test passes: ab46e3f514824529edf65767771d4622
    ==> 'resource/asnlib/publicdata/df_plot_stacked_fraction_soln.csv' is ready!
    
    (Auxiliary files appear to be ready.)
    

## Viz setup

This notebook includes some simple visualizations. This section just ensures you have the right software setup to follow along.


```python
from nb9utils import make_barchart, make_stacked_barchart
from bokeh.io import show
```


```python
def demo_bar():
    from bokeh.plotting import figure
    from bokeh.models import ColumnDataSource
    data = [
        ['201720', 'cat1', 20],
        ['201720', 'cat2', 30],
        ['201720', 'cat3', 40],
        ['201721', 'cat1', 20],
        ['201721', 'cat2', 0],
        ['201721', 'cat3', 40],
        ['201722', 'cat1', 50],
        ['201722', 'cat2', 60],
        ['201722', 'cat3', 10],
    ]
    df = pd.DataFrame(data, columns=['week', 'category', 'count'])
    pt = df.pivot('week', 'category', 'count')
    pt.cumsum(axis=1)
    return df, pt

df_demo, pt_demo = demo_bar()
pt_demo
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
      <th>category</th>
      <th>cat1</th>
      <th>cat2</th>
      <th>cat3</th>
    </tr>
    <tr>
      <th>week</th>
      <th></th>
      <th></th>
      <th></th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>201720</th>
      <td>20</td>
      <td>30</td>
      <td>40</td>
    </tr>
    <tr>
      <th>201721</th>
      <td>20</td>
      <td>0</td>
      <td>40</td>
    </tr>
    <tr>
      <th>201722</th>
      <td>50</td>
      <td>60</td>
      <td>10</td>
    </tr>
  </tbody>
</table>
</div>




```python
def demo_stacked_bar(pt):
    from bokeh.models.ranges import FactorRange
    from bokeh.io import show
    from bokeh.plotting import figure
    p = figure(title="count",
               x_axis_label='week', y_axis_label='category',
               x_range = FactorRange(factors=list(pt.index)),
               plot_height=300, plot_width=500)
    p.vbar(x=pt.index, bottom=0, top=pt.cat1, width=0.2, color='red', legend='cat1')
    p.vbar(x=pt.index, bottom=pt.cat1, top=pt.cat2, width=0.2, color='blue', legend='cat2')
    p.vbar(x=pt.index, bottom=pt.cat2, top=pt.cat3, width=0.2, color='green', legend='cat3')
    return p
    
show(demo_stacked_bar(pt_demo))
```

    BokehDeprecationWarning: 'legend' keyword is deprecated, use explicit 'legend_label', 'legend_field', or 'legend_group' keywords instead
    BokehDeprecationWarning: 'legend' keyword is deprecated, use explicit 'legend_label', 'legend_field', or 'legend_group' keywords instead
    BokehDeprecationWarning: 'legend' keyword is deprecated, use explicit 'legend_label', 'legend_field', or 'legend_group' keywords instead
    








<div class="bk-root" id="afd5fb03-1205-47eb-90ec-01ef979c8129" data-root-id="1003"></div>






```python
# Build a Pandas data frame
names = ['Bob','Jessica','Mary','John','Mel']
births = [968, 155, 77, 578, 973]
name_birth_pairs = list(zip(names, births))
baby_names = pd.DataFrame(data=name_birth_pairs, columns=['Names', 'Births'])
display(baby_names)
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
      <th>Names</th>
      <th>Births</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>Bob</td>
      <td>968</td>
    </tr>
    <tr>
      <th>1</th>
      <td>Jessica</td>
      <td>155</td>
    </tr>
    <tr>
      <th>2</th>
      <td>Mary</td>
      <td>77</td>
    </tr>
    <tr>
      <th>3</th>
      <td>John</td>
      <td>578</td>
    </tr>
    <tr>
      <th>4</th>
      <td>Mel</td>
      <td>973</td>
    </tr>
  </tbody>
</table>
</div>



```python
p = make_barchart(baby_names, 'Names', 'Births', kwargs_figure={'plot_width': 640, 'plot_height': 320})
show(p)
```








<div class="bk-root" id="02f06691-8ce2-4e27-8623-c680730c0333" data-root-id="1162"></div>





## Data setup

You'll also need the NYC 311 calls dataset. What we've provided is actually a small subset (about 250+ MiB) of the full data as of 2015.

> If you are not running on Vocareum, you will need to download this file manually from the following link and place it locally in a (nested) subdirectory or folder named `resource/asnlib/publicdata`.
>
> [Link to the pre-constructed NYC 311 Database on MS OneDrive](https://onedrive.live.com/download?cid=FD520DDC6BE92730&resid=FD520DDC6BE92730%21616&authkey=AEeP_4E1uh-vyDE)


```python
from nb9utils import download_nyc311db
DB_FILENAME = download_nyc311db()
```

    [https://onedrive.live.com/download?cid=FD520DDC6BE92730&resid=FD520DDC6BE92730%21616&authkey=AEeP_4E1uh-vyDENYC-311-2M.db]
    ==> 'resource/asnlib/publicdata/NYC-311-2M.db' is already available.
    ==> Checksum test passes: f48eba2fb06e8ece7479461ea8c6dee9
    ==> 'resource/asnlib/publicdata/NYC-311-2M.db' is ready!
    
    

**Connecting.** Let's open up a connection to this dataset.


```python
# Connect
import sqlite3 as db
disk_engine = db.connect('file:{}?mode=ro'.format(DB_FILENAME), uri=True)
```

**Preview the data.** This sample database has just a single table, named `data`. Let's query it and see how long it takes to read. To carry out the query, we will use the SQL reader built into `pandas`.


```python
import time

print ("Reading ...")
start_time = time.time ()

# Perform SQL query through the disk_engine connection.
# The return value is a pandas data frame.
df = pd.read_sql_query ('select * from data', disk_engine)

elapsed_time = time.time () - start_time
print ("==> Took %g seconds." % elapsed_time)

# Dump the first few rows
df.head()
```

    Reading ...
    ==> Took 8.34658 seconds.
    




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
      <th>index</th>
      <th>CreatedDate</th>
      <th>ClosedDate</th>
      <th>Agency</th>
      <th>ComplaintType</th>
      <th>Descriptor</th>
      <th>City</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>1</td>
      <td>2015-09-15 02:14:04.000000</td>
      <td>None</td>
      <td>NYPD</td>
      <td>Illegal Parking</td>
      <td>Blocked Hydrant</td>
      <td>None</td>
    </tr>
    <tr>
      <th>1</th>
      <td>2</td>
      <td>2015-09-15 02:12:49.000000</td>
      <td>None</td>
      <td>NYPD</td>
      <td>Noise - Street/Sidewalk</td>
      <td>Loud Talking</td>
      <td>NEW YORK</td>
    </tr>
    <tr>
      <th>2</th>
      <td>3</td>
      <td>2015-09-15 02:11:19.000000</td>
      <td>None</td>
      <td>NYPD</td>
      <td>Noise - Street/Sidewalk</td>
      <td>Loud Talking</td>
      <td>NEW YORK</td>
    </tr>
    <tr>
      <th>3</th>
      <td>4</td>
      <td>2015-09-15 02:09:46.000000</td>
      <td>None</td>
      <td>NYPD</td>
      <td>Noise - Commercial</td>
      <td>Loud Talking</td>
      <td>BRONX</td>
    </tr>
    <tr>
      <th>4</th>
      <td>5</td>
      <td>2015-09-15 02:08:01.000000</td>
      <td>2015-09-15 02:08:18.000000</td>
      <td>DHS</td>
      <td>Homeless Person Assistance</td>
      <td>Status Call</td>
      <td>NEW YORK</td>
    </tr>
  </tbody>
</table>
</div>



**Partial queries: `LIMIT` clause.** The preceding command was overkill for what we wanted, which was just to preview the table. Instead, we could have used the `LIMIT` option to ask for just a few results.


```python
query = '''
  SELECT *
    FROM data
    LIMIT 5
'''
start_time = time.time ()
df = pd.read_sql_query (query, disk_engine)
elapsed_time = time.time () - start_time
print ("==> LIMIT version took %g seconds." % elapsed_time)

df
```

    ==> LIMIT version took 0.00300288 seconds.
    




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
      <th>index</th>
      <th>CreatedDate</th>
      <th>ClosedDate</th>
      <th>Agency</th>
      <th>ComplaintType</th>
      <th>Descriptor</th>
      <th>City</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>1</td>
      <td>2015-09-15 02:14:04.000000</td>
      <td>None</td>
      <td>NYPD</td>
      <td>Illegal Parking</td>
      <td>Blocked Hydrant</td>
      <td>None</td>
    </tr>
    <tr>
      <th>1</th>
      <td>2</td>
      <td>2015-09-15 02:12:49.000000</td>
      <td>None</td>
      <td>NYPD</td>
      <td>Noise - Street/Sidewalk</td>
      <td>Loud Talking</td>
      <td>NEW YORK</td>
    </tr>
    <tr>
      <th>2</th>
      <td>3</td>
      <td>2015-09-15 02:11:19.000000</td>
      <td>None</td>
      <td>NYPD</td>
      <td>Noise - Street/Sidewalk</td>
      <td>Loud Talking</td>
      <td>NEW YORK</td>
    </tr>
    <tr>
      <th>3</th>
      <td>4</td>
      <td>2015-09-15 02:09:46.000000</td>
      <td>None</td>
      <td>NYPD</td>
      <td>Noise - Commercial</td>
      <td>Loud Talking</td>
      <td>BRONX</td>
    </tr>
    <tr>
      <th>4</th>
      <td>5</td>
      <td>2015-09-15 02:08:01.000000</td>
      <td>2015-09-15 02:08:18.000000</td>
      <td>DHS</td>
      <td>Homeless Person Assistance</td>
      <td>Status Call</td>
      <td>NEW YORK</td>
    </tr>
  </tbody>
</table>
</div>



**Finding unique values: `DISTINCT` qualifier.** Another common idiom is to ask for the unique values of some attribute, for which you can use the `DISTINCT` qualifier.


```python
query = 'SELECT DISTINCT City FROM data'
df = pd.read_sql_query(query, disk_engine)

print("Found {} unique cities. The first few are:".format(len(df)))
df.head()
```

    Found 547 unique cities. The first few are:
    




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
      <th>City</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>None</td>
    </tr>
    <tr>
      <th>1</th>
      <td>NEW YORK</td>
    </tr>
    <tr>
      <th>2</th>
      <td>BRONX</td>
    </tr>
    <tr>
      <th>3</th>
      <td>STATEN ISLAND</td>
    </tr>
    <tr>
      <th>4</th>
      <td>ELMHURST</td>
    </tr>
  </tbody>
</table>
</div>



However, `DISTINCT` applied to strings is case-sensitive. We'll deal with that momentarily.

**Grouping Information: GROUP BY operator.** The GROUP BY operator lets you group information using a particular column or multiple columns of the table. The output generated is more of a pivot table.


```python
query = '''
  SELECT ComplaintType, Descriptor, Agency
    FROM data
    GROUP BY ComplaintType
    LIMIT 10
'''

df = pd.read_sql_query(query, disk_engine)
df.head()
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
      <th>ComplaintType</th>
      <th>Descriptor</th>
      <th>Agency</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>AGENCY</td>
      <td>HOUSING QUALITY STANDARDS</td>
      <td>HPD</td>
    </tr>
    <tr>
      <th>1</th>
      <td>APPLIANCE</td>
      <td>ELECTRIC/GAS RANGE</td>
      <td>HPD</td>
    </tr>
    <tr>
      <th>2</th>
      <td>Adopt-A-Basket</td>
      <td>10A Adopt-A-Basket</td>
      <td>DSNY</td>
    </tr>
    <tr>
      <th>3</th>
      <td>Agency Issues</td>
      <td>Bike Share</td>
      <td>DOT</td>
    </tr>
    <tr>
      <th>4</th>
      <td>Air Quality</td>
      <td>Air: Odor/Fumes, Vehicle Idling (AD3)</td>
      <td>DEP</td>
    </tr>
  </tbody>
</table>
</div>



**`GROUP BY` aggregations.** A common pattern is to combine grouping with aggregation. For example, suppose we want to count how many times each complaint occurs. Here is one way to do it.


```python
query = '''
  SELECT ComplaintType, COUNT(*)
    FROM data
    GROUP BY ComplaintType
    LIMIT 10
'''

df = pd.read_sql_query(query, disk_engine)
df.head()
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
      <th>ComplaintType</th>
      <th>COUNT(*)</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>AGENCY</td>
      <td>2</td>
    </tr>
    <tr>
      <th>1</th>
      <td>APPLIANCE</td>
      <td>11263</td>
    </tr>
    <tr>
      <th>2</th>
      <td>Adopt-A-Basket</td>
      <td>50</td>
    </tr>
    <tr>
      <th>3</th>
      <td>Agency Issues</td>
      <td>7428</td>
    </tr>
    <tr>
      <th>4</th>
      <td>Air Quality</td>
      <td>8151</td>
    </tr>
  </tbody>
</table>
</div>



**Character-case conversions.** From the two preceding examples, observe that the strings employ a mix of case conventions (i.e., lowercase vs. uppercase vs. mixed case). A convenient way to query and "normalize" case is to apply SQL's `UPPER()` and `LOWER()` functions. Here is an example:


```python
query = '''
  SELECT LOWER(ComplaintType), LOWER(Descriptor), LOWER(Agency)
    FROM data
    GROUP BY LOWER(ComplaintType)
    LIMIT 10
'''

df = pd.read_sql_query(query, disk_engine)
df.head()
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
      <th>LOWER(ComplaintType)</th>
      <th>LOWER(Descriptor)</th>
      <th>LOWER(Agency)</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>adopt-a-basket</td>
      <td>10a adopt-a-basket</td>
      <td>dsny</td>
    </tr>
    <tr>
      <th>1</th>
      <td>agency</td>
      <td>housing quality standards</td>
      <td>hpd</td>
    </tr>
    <tr>
      <th>2</th>
      <td>agency issues</td>
      <td>bike share</td>
      <td>dot</td>
    </tr>
    <tr>
      <th>3</th>
      <td>air quality</td>
      <td>air: odor/fumes, vehicle idling (ad3)</td>
      <td>dep</td>
    </tr>
    <tr>
      <th>4</th>
      <td>animal abuse</td>
      <td>other (complaint details)</td>
      <td>nypd</td>
    </tr>
  </tbody>
</table>
</div>



**Filtered aggregations: `HAVING` clauses.** A common pattern for aggregation queries (e.g., `GROUP BY` plus `COUNT()`) is to filter the grouped results. You cannot do that with a `WHERE` clause alone, because `WHERE` is applied *before* grouping.

As an example, recall that some `ComplaintType` values are in all uppercase whereas some use mixed case. Since we didn't inspect all of them, there might even be some are all lowercase. Worse, you would expect some inconsistencies. For instance, it turns out that both `"Plumbing"` (mixed case) and `"PLUMBING"` (all caps) appear. Here is a pair of queries that makes this point.


```python
query0 = "SELECT DISTINCT ComplaintType FROM data"
df0 = pd.read_sql_query(query0, disk_engine)
print("Found {} unique `ComplaintType` strings.".format(len(df0)))
display(df0.head())

query1 = "SELECT DISTINCT LOWER(ComplaintType) FROM data"
df1 = pd.read_sql_query(query1, disk_engine)
print("\nFound {} unique `LOWER(ComplaintType)` strings.".format(len(df1)))
display(df1.head())

print("\n==> Therefore, there are {} cases that are duplicated. Which ones?".format(len(df0) - len(df1)))
```

    Found 200 unique `ComplaintType` strings.
    


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
      <th>ComplaintType</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>Illegal Parking</td>
    </tr>
    <tr>
      <th>1</th>
      <td>Noise - Street/Sidewalk</td>
    </tr>
    <tr>
      <th>2</th>
      <td>Noise - Commercial</td>
    </tr>
    <tr>
      <th>3</th>
      <td>Homeless Person Assistance</td>
    </tr>
    <tr>
      <th>4</th>
      <td>Highway Condition</td>
    </tr>
  </tbody>
</table>
</div>


    
    Found 198 unique `LOWER(ComplaintType)` strings.
    


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
      <th>LOWER(ComplaintType)</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>illegal parking</td>
    </tr>
    <tr>
      <th>1</th>
      <td>noise - street/sidewalk</td>
    </tr>
    <tr>
      <th>2</th>
      <td>noise - commercial</td>
    </tr>
    <tr>
      <th>3</th>
      <td>homeless person assistance</td>
    </tr>
    <tr>
      <th>4</th>
      <td>highway condition</td>
    </tr>
  </tbody>
</table>
</div>


    
    ==> Therefore, there are 2 cases that are duplicated. Which ones?
    

What if we wanted a query that identifies these inconsistent capitalizations? Here is one way to do it, which demonstrates the `HAVING` clause. (It also uses a **nested query**, that is, it performs one query and then selects immediately from that result.) Can you read it and figure out what it is doing and why it works?


```python
query2 = '''
    SELECT ComplaintType, COUNT(*)
      FROM (SELECT DISTINCT ComplaintType FROM data)
      GROUP BY LOWER(ComplaintType)
      HAVING COUNT(*) >= 2
'''
df2 = pd.read_sql_query(query2, disk_engine)
df2
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
      <th>ComplaintType</th>
      <th>COUNT(*)</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>Elevator</td>
      <td>2</td>
    </tr>
    <tr>
      <th>1</th>
      <td>PLUMBING</td>
      <td>2</td>
    </tr>
  </tbody>
</table>
</div>



You should see that "elevator" and "plumbing" complaints use inconsistent case, which we can then verify directly using the next technique, the `IN` operator.

**Set membership: `IN` operator.** Another common idiom is to ask for rows whose attributes fall within a set, for which you can use the `IN` operator. Let's use it to see the two inconsistent-capitalization complaint types from above.


```python
query = '''
    SELECT DISTINCT ComplaintType
      FROM data
      WHERE LOWER(ComplaintType) IN ("plumbing", "elevator")
'''
df = pd.read_sql_query(query, disk_engine)
df.head()
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
      <th>ComplaintType</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>PLUMBING</td>
    </tr>
    <tr>
      <th>1</th>
      <td>Elevator</td>
    </tr>
    <tr>
      <th>2</th>
      <td>Plumbing</td>
    </tr>
    <tr>
      <th>3</th>
      <td>ELEVATOR</td>
    </tr>
  </tbody>
</table>
</div>



**Renaming columns: `AS` operator.** Sometimes you might want to rename a result column. For instance, the following query counts the number of complaints by "Agency," using the `COUNT(*)` function and `GROUP BY` clause, which we discussed in an earlier lab. If you wish to refer to the counts column of the resulting data frame, you can give it a more "friendly" name using the `AS` operator.


```python
query = '''
  SELECT Agency, COUNT(*) AS NumComplaints
    FROM data
    GROUP BY Agency
'''
df = pd.read_sql_query(query, disk_engine)
df.head()
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
      <th>Agency</th>
      <th>NumComplaints</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>3-1-1</td>
      <td>1289</td>
    </tr>
    <tr>
      <th>1</th>
      <td>ACS</td>
      <td>3</td>
    </tr>
    <tr>
      <th>2</th>
      <td>AJC</td>
      <td>6</td>
    </tr>
    <tr>
      <th>3</th>
      <td>CAU</td>
      <td>1</td>
    </tr>
    <tr>
      <th>4</th>
      <td>CCRB</td>
      <td>1</td>
    </tr>
  </tbody>
</table>
</div>



**Ordering results: `ORDER BY` clause.** You can also order the results. For instance, suppose we want to execute the previous query by number of complaints.


```python
query = '''
  SELECT Agency, COUNT(*) AS NumComplaints
    FROM data
    GROUP BY UPPER(Agency)
    ORDER BY NumComplaints
'''
df = pd.read_sql_query(query, disk_engine)
df.tail()
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
      <th>Agency</th>
      <th>NumComplaints</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>45</th>
      <td>DSNY</td>
      <td>152004</td>
    </tr>
    <tr>
      <th>46</th>
      <td>DEP</td>
      <td>181121</td>
    </tr>
    <tr>
      <th>47</th>
      <td>DOT</td>
      <td>322969</td>
    </tr>
    <tr>
      <th>48</th>
      <td>NYPD</td>
      <td>340694</td>
    </tr>
    <tr>
      <th>49</th>
      <td>HPD</td>
      <td>640096</td>
    </tr>
  </tbody>
</table>
</div>



Note that the above example prints the bottom (tail) of the data frame. You could have also asked for the query results in reverse (descending) order, by prefixing the `ORDER BY` attribute with a `-` (minus) symbol. Alternatively, you can use `DESC` to achieve the same result.


```python
query = '''
  SELECT Agency, COUNT(*) AS NumComplaints
    FROM data
    GROUP BY UPPER(Agency)
    ORDER BY -NumComplaints
'''

# Alternative: query =
'''
SELECT Agency, COUNT(*) AS NumComplaints 
    FROM data 
    GROUP BY UPPER(Agency)
    ORDER BY NumComplaints DESC 
'''

df = pd.read_sql_query(query, disk_engine)
df.head()
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
      <th>Agency</th>
      <th>NumComplaints</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>HPD</td>
      <td>640096</td>
    </tr>
    <tr>
      <th>1</th>
      <td>NYPD</td>
      <td>340694</td>
    </tr>
    <tr>
      <th>2</th>
      <td>DOT</td>
      <td>322969</td>
    </tr>
    <tr>
      <th>3</th>
      <td>DEP</td>
      <td>181121</td>
    </tr>
    <tr>
      <th>4</th>
      <td>DSNY</td>
      <td>152004</td>
    </tr>
  </tbody>
</table>
</div>



And of course we can plot all of this data!

**Exercise 0** (ungraded). Run the following code cell, which will create an interactive bar chart from the data in the previous query.


```python
p = make_barchart(df[:20], 'Agency', 'NumComplaints',
                  {'title': 'Top 20 agencies by number of complaints',
                   'plot_width': 800, 'plot_height': 320})
p.xaxis.major_label_orientation = 0.66
show(p)
```








<div class="bk-root" id="ffbb8248-cbac-486b-b373-ace137faa2fa" data-root-id="1273"></div>





**Exercise 1** (2 points). Create a string, `query`, containing an SQL query that will return the number of complaints by type. The columns should be named `type` and `freq`, and the results should be sorted in descending order by `freq`. Also, since we know some complaints use an inconsistent case, for your function convert complaints to lowercase.

> What is the most common type of complaint? What, if anything, does it tell you about NYC?


```python
del query # clears any existing `query` variable; you should define it, below!

# Define a variable named `query` containing your solution
query = '''
    select
        lower(complainttype) as type,
        count(*) as freq
    from data 
    group by 1
    order by 2 desc  
'''

# Runs your `query`:
df_complaint_freq = pd.read_sql_query(query, disk_engine)
df_complaint_freq.head()
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
      <th>type</th>
      <th>freq</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>heat/hot water</td>
      <td>241430</td>
    </tr>
    <tr>
      <th>1</th>
      <td>street condition</td>
      <td>124347</td>
    </tr>
    <tr>
      <th>2</th>
      <td>street light condition</td>
      <td>98577</td>
    </tr>
    <tr>
      <th>3</th>
      <td>blocked driveway</td>
      <td>95080</td>
    </tr>
    <tr>
      <th>4</th>
      <td>illegal parking</td>
      <td>83961</td>
    </tr>
  </tbody>
</table>
</div>




```python
# Test cell: `complaints_test`

print("Top 10 complaints:")
display(df_complaint_freq.head(10))

assert set(df_complaint_freq.columns) == {'type', 'freq'}, "Output columns should be named 'type' and 'freq', not {}".format(set(df_complaint_freq.columns))

soln = ['heat/hot water', 'street condition', 'street light condition', 'blocked driveway', 'illegal parking', 'unsanitary condition', 'paint/plaster', 'water system', 'plumbing', 'noise', 'noise - street/sidewalk', 'traffic signal condition', 'noise - commercial', 'door/window', 'water leak', 'dirty conditions', 'sewer', 'sanitation condition', 'dof literature request', 'electric', 'rodent', 'flooring/stairs', 'general construction/plumbing', 'building/use', 'broken muni meter', 'general', 'missed collection (all materials)', 'benefit card replacement', 'derelict vehicle', 'noise - vehicle', 'damaged tree', 'consumer complaint', 'derelict vehicles', 'taxi complaint', 'overgrown tree/branches', 'graffiti', 'snow', 'opinion for the mayor', 'appliance', 'maintenance or facility', 'animal abuse', 'dead tree', 'elevator', 'hpd literature request', 'root/sewer/sidewalk condition', 'safety', 'food establishment', 'scrie', 'air quality', 'agency issues', 'construction', 'highway condition', 'other enforcement', 'water conservation', 'sidewalk condition', 'indoor air quality', 'street sign - damaged', 'traffic', 'fire safety director - f58', 'homeless person assistance', 'homeless encampment', 'special enforcement', 'street sign - missing', 'noise - park', 'vending', 'for hire vehicle complaint', 'food poisoning', 'special projects inspection team (spit)', 'hazardous materials', 'electrical', 'dot literature request', 'litter basket / request', 'taxi report', 'illegal tree damage', 'dof property - reduction issue', 'unsanitary animal pvt property', 'asbestos', 'lead', 'vacant lot', 'dca / doh new license application request', 'street sign - dangling', 'smoking', 'violation of park rules', 'outside building', 'animal in a park', 'noise - helicopter', 'school maintenance', 'dpr internal', 'boilers', 'industrial waste', 'sweeping/missed', 'overflowing litter baskets', 'non-residential heat', 'curb condition', 'drinking', 'standing water', 'indoor sewage', 'water quality', 'eap inspection - f59', 'derelict bicycle', 'noise - house of worship', 'dca literature request', 'recycling enforcement', 'dof parking - tax exemption', 'broken parking meter', 'request for information', 'taxi compliment', 'unleashed dog', 'urinating in public', 'unsanitary pigeon condition', 'investigations and discipline (iad)', 'bridge condition', 'ferry inquiry', 'bike/roller/skate chronic', 'public payphone complaint', 'vector', 'best/site safety', 'sweeping/inadequate', 'disorderly youth', 'found property', 'mold', 'senior center complaint', 'fire alarm - reinspection', 'for hire vehicle report', 'city vehicle placard complaint', 'cranes and derricks', 'ferry complaint', 'illegal animal kept as pet', 'posting advertisement', 'harboring bees/wasps', 'panhandling', 'scaffold safety', 'oem literature request', 'plant', 'bus stop shelter placement', 'collection truck noise', 'beach/pool/sauna complaint', 'complaint', 'compliment', 'illegal fireworks', 'fire alarm - modification', 'dep literature request', 'drinking water', 'fire alarm - new system', 'poison ivy', 'bike rack condition', 'emergency response team (ert)', 'municipal parking facility', 'tattooing', 'unsanitary animal facility', 'animal facility - no permit', 'miscellaneous categories', 'misc. comments', 'literature request', 'special natural area district (snad)', 'highway sign - damaged', 'public toilet', 'adopt-a-basket', 'ferry permit', 'invitation', 'window guard', 'parking card', 'illegal animal sold', 'stalled sites', 'open flame permit', 'overflowing recycling baskets', 'highway sign - missing', 'public assembly', 'dpr literature request', 'fire alarm - addition', 'lifeguard', 'transportation provider complaint', 'dfta literature request', 'bottled water', 'highway sign - dangling', 'dhs income savings requirement', 'legal services provider complaint', 'foam ban enforcement', 'tunnel condition', 'calorie labeling', 'fire alarm - replacement', 'x-ray machine/equipment', 'sprinkler - mechanical', 'hazmat storage/use', 'tanning', 'radioactive material', 'rangehood', 'squeegee', 'srde', 'building condition', 'sg-98', 'standpipe - mechanical', 'agency', 'forensic engineering', 'public assembly - temporary', 'vacant apartment', 'laboratory', 'sg-99']
assert all(soln[:25] == df_complaint_freq['type'].iloc[:25])

print("\n(Passed.)")
```

    Top 10 complaints:
    


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
      <th>type</th>
      <th>freq</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>heat/hot water</td>
      <td>241430</td>
    </tr>
    <tr>
      <th>1</th>
      <td>street condition</td>
      <td>124347</td>
    </tr>
    <tr>
      <th>2</th>
      <td>street light condition</td>
      <td>98577</td>
    </tr>
    <tr>
      <th>3</th>
      <td>blocked driveway</td>
      <td>95080</td>
    </tr>
    <tr>
      <th>4</th>
      <td>illegal parking</td>
      <td>83961</td>
    </tr>
    <tr>
      <th>5</th>
      <td>unsanitary condition</td>
      <td>81394</td>
    </tr>
    <tr>
      <th>6</th>
      <td>paint/plaster</td>
      <td>69929</td>
    </tr>
    <tr>
      <th>7</th>
      <td>water system</td>
      <td>69209</td>
    </tr>
    <tr>
      <th>8</th>
      <td>plumbing</td>
      <td>60105</td>
    </tr>
    <tr>
      <th>9</th>
      <td>noise</td>
      <td>54165</td>
    </tr>
  </tbody>
</table>
</div>


    
    (Passed.)
    

Let's also visualize the result, as a bar chart showing complaint types on the x-axis and the number of complaints on the y-axis.


```python
p = make_barchart(df_complaint_freq[:25], 'type', 'freq',
                  {'title': 'Top 25 complaints by type',
                   'plot_width': 800, 'plot_height': 320})
p.xaxis.major_label_orientation = 0.66
show(p)
```








<div class="bk-root" id="eb035295-581b-47fe-9495-b0de08be14f6" data-root-id="1388"></div>





# Lesson 3: More SQL stuff

**Simple substring matching: the `LIKE` operator.** Suppose we just want to look at the counts for all complaints that have the word `noise` in them. You can use the `LIKE` operator combined with the string wildcard, `%`, to look for case-insensitive substring matches.


```python
query = '''
  SELECT LOWER(ComplaintType) AS type, COUNT(*) AS freq
    FROM data
    WHERE LOWER(ComplaintType) LIKE '%noise%'
    GROUP BY type
    ORDER BY -freq
'''

df_noisy = pd.read_sql_query(query, disk_engine)
print("Found {} queries with 'noise' in them.".format(len(df_noisy)))
df_noisy
```

    Found 8 queries with 'noise' in them.
    




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
      <th>type</th>
      <th>freq</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>noise</td>
      <td>54165</td>
    </tr>
    <tr>
      <th>1</th>
      <td>noise - street/sidewalk</td>
      <td>48436</td>
    </tr>
    <tr>
      <th>2</th>
      <td>noise - commercial</td>
      <td>42422</td>
    </tr>
    <tr>
      <th>3</th>
      <td>noise - vehicle</td>
      <td>18370</td>
    </tr>
    <tr>
      <th>4</th>
      <td>noise - park</td>
      <td>4020</td>
    </tr>
    <tr>
      <th>5</th>
      <td>noise - helicopter</td>
      <td>1715</td>
    </tr>
    <tr>
      <th>6</th>
      <td>noise - house of worship</td>
      <td>1143</td>
    </tr>
    <tr>
      <th>7</th>
      <td>collection truck noise</td>
      <td>184</td>
    </tr>
  </tbody>
</table>
</div>



**Exercise 2** (2 points). Create a string variable, `query`, that contains an SQL query that will return the top 10 cities with the largest number of complaints, in descending order. It should return a table with two columns, one named `name` holding the name of the city, and one named `freq` holding the number of complaints by that city. 

Like complaint types, cities are not capitalized consistently. Therefore, standardize the city names by converting them to **uppercase**.


```python
del query # define a new `query` variable, below

query = '''
    select
        upper(city) as name,
        count(*) as freq
    from data 
    group by 1
    order by 2 desc
    limit 10
'''

# Runs your `query`:
df_whiny_cities = pd.read_sql_query(query, disk_engine)
df_whiny_cities
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
      <th>name</th>
      <th>freq</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>BROOKLYN</td>
      <td>579363</td>
    </tr>
    <tr>
      <th>1</th>
      <td>NEW YORK</td>
      <td>385655</td>
    </tr>
    <tr>
      <th>2</th>
      <td>BRONX</td>
      <td>342533</td>
    </tr>
    <tr>
      <th>3</th>
      <td>None</td>
      <td>168692</td>
    </tr>
    <tr>
      <th>4</th>
      <td>STATEN ISLAND</td>
      <td>92509</td>
    </tr>
    <tr>
      <th>5</th>
      <td>JAMAICA</td>
      <td>46683</td>
    </tr>
    <tr>
      <th>6</th>
      <td>FLUSHING</td>
      <td>35504</td>
    </tr>
    <tr>
      <th>7</th>
      <td>ASTORIA</td>
      <td>31873</td>
    </tr>
    <tr>
      <th>8</th>
      <td>RIDGEWOOD</td>
      <td>21618</td>
    </tr>
    <tr>
      <th>9</th>
      <td>WOODSIDE</td>
      <td>15932</td>
    </tr>
  </tbody>
</table>
</div>



Brooklynites are "vocal" about their issues, evidently.


```python
# Test cell: `whiny_cities__test`

assert df_whiny_cities['name'][0] == 'BROOKLYN'
assert df_whiny_cities['name'][1] == 'NEW YORK'
assert df_whiny_cities['name'][2] == 'BRONX'
assert df_whiny_cities['name'][3] is None
assert df_whiny_cities['name'][4] == 'STATEN ISLAND'

print ("\n(Passed partial test.)")
```

    
    (Passed partial test.)
    

**Case-insensitive grouping: `COLLATE NOCASE`.** Another way to carry out the preceding query in a case-insensitive way is to add a `COLLATE NOCASE` qualifier to the `GROUP BY` clause.

The next example demonstrates this clause. Note that it also filters out the 'None' cases, where the `<>` operator denotes "not equal to." Lastly, this query ensures that the returned city names are uppercase.

> The `COLLATE NOCASE` clause modifies the column next to which it appears. So if you are grouping by more than one key and want to be case-insensitive, you need to write, `... GROUP BY ColumnA COLLATE NOCASE, ColumnB COLLATE NOCASE ...`.


```python
query = '''
  SELECT UPPER(City) AS name, COUNT(*) AS freq
    FROM data
    WHERE name <> 'None'
    GROUP BY City COLLATE NOCASE
    ORDER BY -freq
    LIMIT 10
'''
df_whiny_cities2 = pd.read_sql_query(query, disk_engine)
df_whiny_cities2
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
      <th>name</th>
      <th>freq</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>BROOKLYN</td>
      <td>579363</td>
    </tr>
    <tr>
      <th>1</th>
      <td>NEW YORK</td>
      <td>385655</td>
    </tr>
    <tr>
      <th>2</th>
      <td>BRONX</td>
      <td>342533</td>
    </tr>
    <tr>
      <th>3</th>
      <td>STATEN ISLAND</td>
      <td>92509</td>
    </tr>
    <tr>
      <th>4</th>
      <td>JAMAICA</td>
      <td>46683</td>
    </tr>
    <tr>
      <th>5</th>
      <td>FLUSHING</td>
      <td>35504</td>
    </tr>
    <tr>
      <th>6</th>
      <td>ASTORIA</td>
      <td>31873</td>
    </tr>
    <tr>
      <th>7</th>
      <td>RIDGEWOOD</td>
      <td>21618</td>
    </tr>
    <tr>
      <th>8</th>
      <td>WOODSIDE</td>
      <td>15932</td>
    </tr>
    <tr>
      <th>9</th>
      <td>CORONA</td>
      <td>15740</td>
    </tr>
  </tbody>
</table>
</div>



Lastly, for later use, let's save the names of just the top seven (7) cities by numbers of complaints.


```python
TOP_CITIES = list(df_whiny_cities2.head(7)['name'])
TOP_CITIES
```




    ['BROOKLYN',
     'NEW YORK',
     'BRONX',
     'STATEN ISLAND',
     'JAMAICA',
     'FLUSHING',
     'ASTORIA']



**Exercise 3** (1 point). Implement a function that takes a list of strings, `str_list`, and returns a single string consisting of each value, `str_list[i]`, enclosed by double-quotes and separated by a comma-space delimiters. For example, if

```python
   assert str_list == ['a', 'b', 'c', 'd']
```

then

```python
   assert strs_to_args(str_list) == '"a", "b", "c", "d"'
```

> **Tip.** Try to avoid manipulating the input `str_list` directly and returning the updated `str_list`. This may result in your function adding `""` to the strings in your list each time the function is used (which will be more than once in this notebook!)


```python
def strs_to_args(str_list):
    assert type (str_list) is list
    assert all ([type (s) is str for s in str_list])
    return ', '.join(['"{}"'.format(c) for c in str_list])

```


```python
# Test cell: `strs_to_args__test`

print ("Your solution, applied to TOP_CITIES:", strs_to_args(TOP_CITIES))

TOP_CITIES_as_args = strs_to_args(TOP_CITIES)
assert TOP_CITIES_as_args == \
       '"BROOKLYN", "NEW YORK", "BRONX", "STATEN ISLAND", "Jamaica", "Flushing", "ASTORIA"'.upper()
assert TOP_CITIES == list(df_whiny_cities2.head(7)['name']), \
       "Does your implementation cause the `TOP_CITIES` variable to change? If so, you need to fix that."
    
print ("\n(Passed.)")
```

    Your solution, applied to TOP_CITIES: "BROOKLYN", "NEW YORK", "BRONX", "STATEN ISLAND", "JAMAICA", "FLUSHING", "ASTORIA"
    
    (Passed.)
    

**Exercise 4** (3 points). Suppose we want to look at the number of complaints by type _and_ by city **for only the top cities**, i.e., those in the list `TOP_CITIES` computed above. Execute an SQL query to produce a tibble named `df_complaints_by_city` with the variables {`complaint_type`, `city_name`, `complaint_count`}.

In your output `DataFrame`, convert all city names to uppercase and convert all complaint types to lowercase.


```python
del query # define a new `query` variable, below

query = '''
    select
        lower(complainttype) as complaint_type,
        upper(city) as city_name,
        count(*) as complaint_count
    from data
    where city_name in ({})
    group by 1, 2
    order by 2, 1
'''.format(strs_to_args(TOP_CITIES))

# Runs your `query`:
df_complaints_by_city = pd.read_sql_query(query, disk_engine)
df_complaints_by_city

# Previews the results of your query:
print("Found {} records.".format(len(df_complaints_by_city)))
display(df_complaints_by_city.head(10))
```

    Found 1042 records.
    


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
      <th>complaint_type</th>
      <th>city_name</th>
      <th>complaint_count</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>air quality</td>
      <td>ASTORIA</td>
      <td>142</td>
    </tr>
    <tr>
      <th>1</th>
      <td>animal abuse</td>
      <td>ASTORIA</td>
      <td>174</td>
    </tr>
    <tr>
      <th>2</th>
      <td>animal facility - no permit</td>
      <td>ASTORIA</td>
      <td>3</td>
    </tr>
    <tr>
      <th>3</th>
      <td>animal in a park</td>
      <td>ASTORIA</td>
      <td>29</td>
    </tr>
    <tr>
      <th>4</th>
      <td>appliance</td>
      <td>ASTORIA</td>
      <td>70</td>
    </tr>
    <tr>
      <th>5</th>
      <td>asbestos</td>
      <td>ASTORIA</td>
      <td>36</td>
    </tr>
    <tr>
      <th>6</th>
      <td>beach/pool/sauna complaint</td>
      <td>ASTORIA</td>
      <td>2</td>
    </tr>
    <tr>
      <th>7</th>
      <td>best/site safety</td>
      <td>ASTORIA</td>
      <td>18</td>
    </tr>
    <tr>
      <th>8</th>
      <td>bike rack condition</td>
      <td>ASTORIA</td>
      <td>3</td>
    </tr>
    <tr>
      <th>9</th>
      <td>bike/roller/skate chronic</td>
      <td>ASTORIA</td>
      <td>7</td>
    </tr>
  </tbody>
</table>
</div>



```python
# Test cell: `df_complaints_by_city__test`

print("Reading instructor's solution...")
if False:
    df_complaints_by_city.to_csv(get_path('df_complaints_by_city_soln.csv'), index=False)
df_complaints_by_city_soln = pd.read_csv(get_path('df_complaints_by_city_soln.csv'))

print("Checking...")
assert tibbles_are_equivalent(df_complaints_by_city,
                              df_complaints_by_city_soln)

print("\n(Passed.)")
del df_complaints_by_city_soln
```

    Reading instructor's solution...
    Checking...
    
    (Passed.)
    

Let's use Bokeh to visualize the results as a stacked bar chart.


```python
# Let's consider only the top 25 complaints (by total)
top_complaints = df_complaint_freq[:25]
print("Top complaints:")
display(top_complaints)
```

    Top complaints:
    


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
      <th>type</th>
      <th>freq</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>heat/hot water</td>
      <td>241430</td>
    </tr>
    <tr>
      <th>1</th>
      <td>street condition</td>
      <td>124347</td>
    </tr>
    <tr>
      <th>2</th>
      <td>street light condition</td>
      <td>98577</td>
    </tr>
    <tr>
      <th>3</th>
      <td>blocked driveway</td>
      <td>95080</td>
    </tr>
    <tr>
      <th>4</th>
      <td>illegal parking</td>
      <td>83961</td>
    </tr>
    <tr>
      <th>5</th>
      <td>unsanitary condition</td>
      <td>81394</td>
    </tr>
    <tr>
      <th>6</th>
      <td>paint/plaster</td>
      <td>69929</td>
    </tr>
    <tr>
      <th>7</th>
      <td>water system</td>
      <td>69209</td>
    </tr>
    <tr>
      <th>8</th>
      <td>plumbing</td>
      <td>60105</td>
    </tr>
    <tr>
      <th>9</th>
      <td>noise</td>
      <td>54165</td>
    </tr>
    <tr>
      <th>10</th>
      <td>noise - street/sidewalk</td>
      <td>48436</td>
    </tr>
    <tr>
      <th>11</th>
      <td>traffic signal condition</td>
      <td>44229</td>
    </tr>
    <tr>
      <th>12</th>
      <td>noise - commercial</td>
      <td>42422</td>
    </tr>
    <tr>
      <th>13</th>
      <td>door/window</td>
      <td>39695</td>
    </tr>
    <tr>
      <th>14</th>
      <td>water leak</td>
      <td>36149</td>
    </tr>
    <tr>
      <th>15</th>
      <td>dirty conditions</td>
      <td>35122</td>
    </tr>
    <tr>
      <th>16</th>
      <td>sewer</td>
      <td>33628</td>
    </tr>
    <tr>
      <th>17</th>
      <td>sanitation condition</td>
      <td>31260</td>
    </tr>
    <tr>
      <th>18</th>
      <td>dof literature request</td>
      <td>30326</td>
    </tr>
    <tr>
      <th>19</th>
      <td>electric</td>
      <td>30248</td>
    </tr>
    <tr>
      <th>20</th>
      <td>rodent</td>
      <td>28454</td>
    </tr>
    <tr>
      <th>21</th>
      <td>flooring/stairs</td>
      <td>27007</td>
    </tr>
    <tr>
      <th>22</th>
      <td>general construction/plumbing</td>
      <td>26861</td>
    </tr>
    <tr>
      <th>23</th>
      <td>building/use</td>
      <td>25807</td>
    </tr>
    <tr>
      <th>24</th>
      <td>broken muni meter</td>
      <td>25428</td>
    </tr>
  </tbody>
</table>
</div>



```python
# Plot subset of data corresponding to the top complaints
df_plot = top_complaints.merge(df_complaints_by_city,
                               left_on=['type'],
                               right_on=['complaint_type'],
                               how='left')
df_plot.dropna(inplace=True)
print("Data to plot (first few rows):")
display(df_plot.head())
print("...")
```

    Data to plot (first few rows):
    


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
      <th>type</th>
      <th>freq</th>
      <th>complaint_type</th>
      <th>city_name</th>
      <th>complaint_count</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>heat/hot water</td>
      <td>241430</td>
      <td>heat/hot water</td>
      <td>ASTORIA</td>
      <td>3396.0</td>
    </tr>
    <tr>
      <th>1</th>
      <td>heat/hot water</td>
      <td>241430</td>
      <td>heat/hot water</td>
      <td>BRONX</td>
      <td>79690.0</td>
    </tr>
    <tr>
      <th>2</th>
      <td>heat/hot water</td>
      <td>241430</td>
      <td>heat/hot water</td>
      <td>BROOKLYN</td>
      <td>72410.0</td>
    </tr>
    <tr>
      <th>3</th>
      <td>heat/hot water</td>
      <td>241430</td>
      <td>heat/hot water</td>
      <td>FLUSHING</td>
      <td>2741.0</td>
    </tr>
    <tr>
      <th>4</th>
      <td>heat/hot water</td>
      <td>241430</td>
      <td>heat/hot water</td>
      <td>JAMAICA</td>
      <td>3376.0</td>
    </tr>
  </tbody>
</table>
</div>


    ...
    


```python
# Some code to render a Bokeh stacked bar chart

kwargs_figure = {'title': "Distribution of the top 25 complaints among top 7 cities with the most complaints",
                 'width': 800,
                 'height': 400,
                 'tools': "hover,crosshair,pan,box_zoom,wheel_zoom,save,reset,help"}

def plot_complaints_stacked_by_city(df, y='complaint_count'):
    p = make_stacked_barchart(df, 'complaint_type', 'city_name', y,
                              x_labels=list(top_complaints['type']), bar_labels=TOP_CITIES,
                              kwargs_figure=kwargs_figure)
    p.xaxis.major_label_orientation = 0.66
    from bokeh.models import HoverTool
    hover_tool = p.select(dict(type=HoverTool))
    hover_tool.tooltips = [("y", "$y{int}")]
    return p

show(plot_complaints_stacked_by_city(df_plot))
```

    BokehDeprecationWarning: 'legend' keyword is deprecated, use explicit 'legend_label', 'legend_field', or 'legend_group' keywords instead
    BokehDeprecationWarning: 'legend' keyword is deprecated, use explicit 'legend_label', 'legend_field', or 'legend_group' keywords instead
    BokehDeprecationWarning: 'legend' keyword is deprecated, use explicit 'legend_label', 'legend_field', or 'legend_group' keywords instead
    BokehDeprecationWarning: 'legend' keyword is deprecated, use explicit 'legend_label', 'legend_field', or 'legend_group' keywords instead
    BokehDeprecationWarning: 'legend' keyword is deprecated, use explicit 'legend_label', 'legend_field', or 'legend_group' keywords instead
    BokehDeprecationWarning: 'legend' keyword is deprecated, use explicit 'legend_label', 'legend_field', or 'legend_group' keywords instead
    BokehDeprecationWarning: 'legend' keyword is deprecated, use explicit 'legend_label', 'legend_field', or 'legend_group' keywords instead
    








<div class="bk-root" id="92908617-7999-4f37-8a02-231504b54df4" data-root-id="1512"></div>





**Exercise 5** (2 points). Suppose we want to create a different stacked bar plot that shows, for each complaint type $t$ and city $c$, the fraction of all complaints of type $t$ (across all cities, not just the top ones) that occurred in city $c$. Store your result in a dataframe named `df_plot_fraction`. It should have the same columns as `df_plot`, **except** that the `complaint_count` column should be replaced by one named `complaint_frac`, which holds the fractional values.

> **Hint.** Everything you need is already in `df_plot`.
>
> **Note.** The test cell will create the chart in addition to checking your result. Note that the normalized bars will not necessarily add up to 1; why not?


```python
df_plot_fraction = df_plot.copy()
df_plot_fraction['complaint_frac'] = df_plot_fraction['complaint_count'] / df_plot_fraction['freq']
del df_plot_fraction['complaint_count']
df_plot_fraction.head()
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
      <th>type</th>
      <th>freq</th>
      <th>complaint_type</th>
      <th>city_name</th>
      <th>complaint_frac</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>heat/hot water</td>
      <td>241430</td>
      <td>heat/hot water</td>
      <td>ASTORIA</td>
      <td>0.014066</td>
    </tr>
    <tr>
      <th>1</th>
      <td>heat/hot water</td>
      <td>241430</td>
      <td>heat/hot water</td>
      <td>BRONX</td>
      <td>0.330075</td>
    </tr>
    <tr>
      <th>2</th>
      <td>heat/hot water</td>
      <td>241430</td>
      <td>heat/hot water</td>
      <td>BROOKLYN</td>
      <td>0.299921</td>
    </tr>
    <tr>
      <th>3</th>
      <td>heat/hot water</td>
      <td>241430</td>
      <td>heat/hot water</td>
      <td>FLUSHING</td>
      <td>0.011353</td>
    </tr>
    <tr>
      <th>4</th>
      <td>heat/hot water</td>
      <td>241430</td>
      <td>heat/hot water</td>
      <td>JAMAICA</td>
      <td>0.013983</td>
    </tr>
  </tbody>
</table>
</div>




```python
# Test cell: `norm_above_test`

df_plot_stacked_fraction = cast(df_plot_fraction, key='city_name', value='complaint_frac')

if False:
    df_plot_stacked_fraction.to_csv(get_path('df_plot_stacked_fraction_soln.csv'), index=False)

show(plot_complaints_stacked_by_city(df_plot_fraction, y='complaint_frac'))

def all_tol(x, tol=1e-14):
    return all([abs(i) <= tol for i in x])

df_plot_fraction_soln = canonicalize_tibble(pd.read_csv(get_path('df_plot_stacked_fraction_soln.csv')))
df_plot_fraction_yours = canonicalize_tibble(df_plot_stacked_fraction)

nonfloat_cols = df_plot_stacked_fraction.columns.difference(TOP_CITIES)
assert tibbles_are_equivalent(df_plot_fraction_yours[nonfloat_cols],
                              df_plot_fraction_soln[nonfloat_cols])
for c in TOP_CITIES:
    assert all(abs(df_plot_fraction_yours[c] - df_plot_fraction_soln[c]) <= 1e-13), \
           "Fractions for city {} do not match the values we are expecting.".format(c)

print("\n(Passed!)")
```

    BokehDeprecationWarning: 'legend' keyword is deprecated, use explicit 'legend_label', 'legend_field', or 'legend_group' keywords instead
    BokehDeprecationWarning: 'legend' keyword is deprecated, use explicit 'legend_label', 'legend_field', or 'legend_group' keywords instead
    BokehDeprecationWarning: 'legend' keyword is deprecated, use explicit 'legend_label', 'legend_field', or 'legend_group' keywords instead
    BokehDeprecationWarning: 'legend' keyword is deprecated, use explicit 'legend_label', 'legend_field', or 'legend_group' keywords instead
    BokehDeprecationWarning: 'legend' keyword is deprecated, use explicit 'legend_label', 'legend_field', or 'legend_group' keywords instead
    BokehDeprecationWarning: 'legend' keyword is deprecated, use explicit 'legend_label', 'legend_field', or 'legend_group' keywords instead
    BokehDeprecationWarning: 'legend' keyword is deprecated, use explicit 'legend_label', 'legend_field', or 'legend_group' keywords instead
    








<div class="bk-root" id="e732d80e-3ef9-4df7-aeb5-bebc65ac4075" data-root-id="1874"></div>





    
    (Passed!)
    

## Dates and times in SQL

Recall that the input data had a column with timestamps corresponding to when someone submitted a complaint. Let's quickly summarize some of the features in SQL and Python for reasoning about these timestamps.

The `CreatedDate` column is actually a specially formatted date and time stamp, where you can query against by comparing to strings of the form, `YYYY-MM-DD hh:mm:ss`.

For example, let's look for all complaints on September 15, 2015.


```python
query = '''
  SELECT LOWER(ComplaintType), CreatedDate, UPPER(City)
    from data
    where CreatedDate >= "2015-09-15 00:00:00.0"
      and CreatedDate < "2015-09-16 00:00:00.0"
    order by CreatedDate
'''
df = pd.read_sql_query (query, disk_engine)
df
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
      <th>LOWER(ComplaintType)</th>
      <th>CreatedDate</th>
      <th>UPPER(City)</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>illegal parking</td>
      <td>2015-09-15 00:01:23.000000</td>
      <td>None</td>
    </tr>
    <tr>
      <th>1</th>
      <td>blocked driveway</td>
      <td>2015-09-15 00:02:29.000000</td>
      <td>REGO PARK</td>
    </tr>
    <tr>
      <th>2</th>
      <td>taxi complaint</td>
      <td>2015-09-15 00:02:34.000000</td>
      <td>NEW YORK</td>
    </tr>
    <tr>
      <th>3</th>
      <td>opinion for the mayor</td>
      <td>2015-09-15 00:03:07.000000</td>
      <td>None</td>
    </tr>
    <tr>
      <th>4</th>
      <td>opinion for the mayor</td>
      <td>2015-09-15 00:03:07.000000</td>
      <td>None</td>
    </tr>
    <tr>
      <th>...</th>
      <td>...</td>
      <td>...</td>
      <td>...</td>
    </tr>
    <tr>
      <th>113</th>
      <td>homeless person assistance</td>
      <td>2015-09-15 02:08:01.000000</td>
      <td>NEW YORK</td>
    </tr>
    <tr>
      <th>114</th>
      <td>noise - commercial</td>
      <td>2015-09-15 02:09:46.000000</td>
      <td>BRONX</td>
    </tr>
    <tr>
      <th>115</th>
      <td>noise - street/sidewalk</td>
      <td>2015-09-15 02:11:19.000000</td>
      <td>NEW YORK</td>
    </tr>
    <tr>
      <th>116</th>
      <td>noise - street/sidewalk</td>
      <td>2015-09-15 02:12:49.000000</td>
      <td>NEW YORK</td>
    </tr>
    <tr>
      <th>117</th>
      <td>illegal parking</td>
      <td>2015-09-15 02:14:04.000000</td>
      <td>None</td>
    </tr>
  </tbody>
</table>
<p>118 rows × 3 columns</p>
</div>



This next example shows how to extract just the hour from the time stamp, using SQL's `strftime()`.


```python
query = '''
  SELECT CreatedDate, STRFTIME('%H', CreatedDate) AS Hour, LOWER(ComplaintType)
    FROM data
    LIMIT 5
'''
df = pd.read_sql_query (query, disk_engine)
df
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
      <th>CreatedDate</th>
      <th>Hour</th>
      <th>LOWER(ComplaintType)</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>2015-09-15 02:14:04.000000</td>
      <td>02</td>
      <td>illegal parking</td>
    </tr>
    <tr>
      <th>1</th>
      <td>2015-09-15 02:12:49.000000</td>
      <td>02</td>
      <td>noise - street/sidewalk</td>
    </tr>
    <tr>
      <th>2</th>
      <td>2015-09-15 02:11:19.000000</td>
      <td>02</td>
      <td>noise - street/sidewalk</td>
    </tr>
    <tr>
      <th>3</th>
      <td>2015-09-15 02:09:46.000000</td>
      <td>02</td>
      <td>noise - commercial</td>
    </tr>
    <tr>
      <th>4</th>
      <td>2015-09-15 02:08:01.000000</td>
      <td>02</td>
      <td>homeless person assistance</td>
    </tr>
  </tbody>
</table>
</div>



**Exercise 6** (3 points). Construct a tibble called `df_complaints_by_hour`, which contains the total number of complaints during a given hour of the day. That is, the variables or column names should be {`hour`, `count`} where each observation is the total number of complaints (`count`) that occurred during a given `hour`.

> Interpret `hour` as follows: when `hour` is `02`, that corresponds to the open time interval [`02:00:00`, `03:00:00.0`).


```python
query = '''
    select
        strftime('%H', createddate) as hour,
        count(*) as count
    from data
    group by 1
'''
df_complaints_by_hour = pd.read_sql_query (query, disk_engine)

# Displays your answer:
display(df_complaints_by_hour)
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
      <th>hour</th>
      <th>count</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>00</td>
      <td>564703</td>
    </tr>
    <tr>
      <th>1</th>
      <td>01</td>
      <td>23489</td>
    </tr>
    <tr>
      <th>2</th>
      <td>02</td>
      <td>15226</td>
    </tr>
    <tr>
      <th>3</th>
      <td>03</td>
      <td>10164</td>
    </tr>
    <tr>
      <th>4</th>
      <td>04</td>
      <td>8692</td>
    </tr>
    <tr>
      <th>5</th>
      <td>05</td>
      <td>10224</td>
    </tr>
    <tr>
      <th>6</th>
      <td>06</td>
      <td>23051</td>
    </tr>
    <tr>
      <th>7</th>
      <td>07</td>
      <td>42273</td>
    </tr>
    <tr>
      <th>8</th>
      <td>08</td>
      <td>73811</td>
    </tr>
    <tr>
      <th>9</th>
      <td>09</td>
      <td>100077</td>
    </tr>
    <tr>
      <th>10</th>
      <td>10</td>
      <td>114079</td>
    </tr>
    <tr>
      <th>11</th>
      <td>11</td>
      <td>115849</td>
    </tr>
    <tr>
      <th>12</th>
      <td>12</td>
      <td>102392</td>
    </tr>
    <tr>
      <th>13</th>
      <td>13</td>
      <td>100970</td>
    </tr>
    <tr>
      <th>14</th>
      <td>14</td>
      <td>105425</td>
    </tr>
    <tr>
      <th>15</th>
      <td>15</td>
      <td>100271</td>
    </tr>
    <tr>
      <th>16</th>
      <td>16</td>
      <td>86968</td>
    </tr>
    <tr>
      <th>17</th>
      <td>17</td>
      <td>69920</td>
    </tr>
    <tr>
      <th>18</th>
      <td>18</td>
      <td>67467</td>
    </tr>
    <tr>
      <th>19</th>
      <td>19</td>
      <td>57637</td>
    </tr>
    <tr>
      <th>20</th>
      <td>20</td>
      <td>54997</td>
    </tr>
    <tr>
      <th>21</th>
      <td>21</td>
      <td>53126</td>
    </tr>
    <tr>
      <th>22</th>
      <td>22</td>
      <td>52076</td>
    </tr>
    <tr>
      <th>23</th>
      <td>23</td>
      <td>47113</td>
    </tr>
  </tbody>
</table>
</div>



```python
# Test cell: `df_complaints_by_hour_test`
    
print ("Reading instructor's solution...")
if False:
    df_complaints_by_hour_soln.to_csv(get_path('df_complaints_by_hour_soln.csv'), index=False)
df_complaints_by_hour_soln = pd.read_csv (get_path('df_complaints_by_hour_soln.csv'))
display (df_complaints_by_hour_soln)

df_complaints_by_hour_norm = df_complaints_by_hour.copy ()
df_complaints_by_hour_norm['hour'] = \
    df_complaints_by_hour_norm['hour'].apply (int)
assert tibbles_are_equivalent (df_complaints_by_hour_norm,
                               df_complaints_by_hour_soln)
print ("\n(Passed.)")
```

    Reading instructor's solution...
    


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
      <th>hour</th>
      <th>count</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>0</td>
      <td>564703</td>
    </tr>
    <tr>
      <th>1</th>
      <td>1</td>
      <td>23489</td>
    </tr>
    <tr>
      <th>2</th>
      <td>2</td>
      <td>15226</td>
    </tr>
    <tr>
      <th>3</th>
      <td>3</td>
      <td>10164</td>
    </tr>
    <tr>
      <th>4</th>
      <td>4</td>
      <td>8692</td>
    </tr>
    <tr>
      <th>5</th>
      <td>5</td>
      <td>10224</td>
    </tr>
    <tr>
      <th>6</th>
      <td>6</td>
      <td>23051</td>
    </tr>
    <tr>
      <th>7</th>
      <td>7</td>
      <td>42273</td>
    </tr>
    <tr>
      <th>8</th>
      <td>8</td>
      <td>73811</td>
    </tr>
    <tr>
      <th>9</th>
      <td>9</td>
      <td>100077</td>
    </tr>
    <tr>
      <th>10</th>
      <td>10</td>
      <td>114079</td>
    </tr>
    <tr>
      <th>11</th>
      <td>11</td>
      <td>115849</td>
    </tr>
    <tr>
      <th>12</th>
      <td>12</td>
      <td>102392</td>
    </tr>
    <tr>
      <th>13</th>
      <td>13</td>
      <td>100970</td>
    </tr>
    <tr>
      <th>14</th>
      <td>14</td>
      <td>105425</td>
    </tr>
    <tr>
      <th>15</th>
      <td>15</td>
      <td>100271</td>
    </tr>
    <tr>
      <th>16</th>
      <td>16</td>
      <td>86968</td>
    </tr>
    <tr>
      <th>17</th>
      <td>17</td>
      <td>69920</td>
    </tr>
    <tr>
      <th>18</th>
      <td>18</td>
      <td>67467</td>
    </tr>
    <tr>
      <th>19</th>
      <td>19</td>
      <td>57637</td>
    </tr>
    <tr>
      <th>20</th>
      <td>20</td>
      <td>54997</td>
    </tr>
    <tr>
      <th>21</th>
      <td>21</td>
      <td>53126</td>
    </tr>
    <tr>
      <th>22</th>
      <td>22</td>
      <td>52076</td>
    </tr>
    <tr>
      <th>23</th>
      <td>23</td>
      <td>47113</td>
    </tr>
  </tbody>
</table>
</div>


    
    (Passed.)
    

Let's take a quick look at the hour-by-hour breakdown above.


```python
p = make_barchart(df_complaints_by_hour, 'hour', 'count',
                  {'title': 'Complaints by hour',
                   'plot_width': 800, 'plot_height': 320})
show(p)
```








<div class="bk-root" id="5aba8cfd-7ed9-459a-a9ab-155685c5d5aa" data-root-id="2255"></div>





An unusual aspect of these data are the excessively large number of reports associated with hour 0 (midnight up to but excluding 1 am), which would probably strike you as suspicious. Indeed, the reason is that there are some complaints that are dated but with no associated time, which was recorded in the data as exactly `00:00:00.000`.


```python
query = '''
  SELECT COUNT(*)
    FROM data
    WHERE STRFTIME('%H:%M:%f', CreatedDate) = '00:00:00.000'
'''

pd.read_sql_query(query, disk_engine)
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
      <th>COUNT(*)</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>532285</td>
    </tr>
  </tbody>
</table>
</div>



**Exercise 7** (2 points). What is the most common hour for noise complaints? Compute a tibble called `df_noisy_by_hour` whose variables are {`hour`, `count`} and whose observations are the number of noise complaints that occurred during a given `hour`. Consider a "noise complaint" to be any complaint string containing the word `noise`. Be sure to filter out any dates _without_ an associated time, i.e., a timestamp of `00:00:00.000`.


```python
query = '''
    select
        strftime('%H', createddate) as hour,
        count(*) as count
    from data
    where complainttype like '%noise%'
        and strftime('%H:%M:%f', CreatedDate) != '00:00:00.000'
    group by 1
'''

df_noisy_by_hour = pd.read_sql_query(query, disk_engine)

display(df_noisy_by_hour)
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
      <th>hour</th>
      <th>count</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>00</td>
      <td>15349</td>
    </tr>
    <tr>
      <th>1</th>
      <td>01</td>
      <td>11284</td>
    </tr>
    <tr>
      <th>2</th>
      <td>02</td>
      <td>7170</td>
    </tr>
    <tr>
      <th>3</th>
      <td>03</td>
      <td>4241</td>
    </tr>
    <tr>
      <th>4</th>
      <td>04</td>
      <td>3083</td>
    </tr>
    <tr>
      <th>5</th>
      <td>05</td>
      <td>2084</td>
    </tr>
    <tr>
      <th>6</th>
      <td>06</td>
      <td>2832</td>
    </tr>
    <tr>
      <th>7</th>
      <td>07</td>
      <td>3708</td>
    </tr>
    <tr>
      <th>8</th>
      <td>08</td>
      <td>4553</td>
    </tr>
    <tr>
      <th>9</th>
      <td>09</td>
      <td>5122</td>
    </tr>
    <tr>
      <th>10</th>
      <td>10</td>
      <td>4672</td>
    </tr>
    <tr>
      <th>11</th>
      <td>11</td>
      <td>4745</td>
    </tr>
    <tr>
      <th>12</th>
      <td>12</td>
      <td>4316</td>
    </tr>
    <tr>
      <th>13</th>
      <td>13</td>
      <td>4364</td>
    </tr>
    <tr>
      <th>14</th>
      <td>14</td>
      <td>4505</td>
    </tr>
    <tr>
      <th>15</th>
      <td>15</td>
      <td>4576</td>
    </tr>
    <tr>
      <th>16</th>
      <td>16</td>
      <td>4957</td>
    </tr>
    <tr>
      <th>17</th>
      <td>17</td>
      <td>5126</td>
    </tr>
    <tr>
      <th>18</th>
      <td>18</td>
      <td>6797</td>
    </tr>
    <tr>
      <th>19</th>
      <td>19</td>
      <td>7958</td>
    </tr>
    <tr>
      <th>20</th>
      <td>20</td>
      <td>9790</td>
    </tr>
    <tr>
      <th>21</th>
      <td>21</td>
      <td>12659</td>
    </tr>
    <tr>
      <th>22</th>
      <td>22</td>
      <td>17155</td>
    </tr>
    <tr>
      <th>23</th>
      <td>23</td>
      <td>19343</td>
    </tr>
  </tbody>
</table>
</div>



```python
# Test cell: `df_noisy_by_hour_test`

print ("Reading instructor's solution...")
if False:
    df_noisy_by_hour.to_csv(get_path('df_noisy_by_hour_soln.csv'), index=False)
df_noisy_by_hour_soln = pd.read_csv (get_path('df_noisy_by_hour_soln.csv'))
display(df_noisy_by_hour_soln)

df_noisy_by_hour_norm = df_noisy_by_hour.copy()
df_noisy_by_hour_norm['hour'] = \
    df_noisy_by_hour_norm['hour'].apply(int)
assert tibbles_are_equivalent (df_noisy_by_hour_norm,
                               df_noisy_by_hour_soln)
print ("\n(Passed.)")
```

    Reading instructor's solution...
    


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
      <th>hour</th>
      <th>count</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>0</td>
      <td>15349</td>
    </tr>
    <tr>
      <th>1</th>
      <td>1</td>
      <td>11284</td>
    </tr>
    <tr>
      <th>2</th>
      <td>2</td>
      <td>7170</td>
    </tr>
    <tr>
      <th>3</th>
      <td>3</td>
      <td>4241</td>
    </tr>
    <tr>
      <th>4</th>
      <td>4</td>
      <td>3083</td>
    </tr>
    <tr>
      <th>5</th>
      <td>5</td>
      <td>2084</td>
    </tr>
    <tr>
      <th>6</th>
      <td>6</td>
      <td>2832</td>
    </tr>
    <tr>
      <th>7</th>
      <td>7</td>
      <td>3708</td>
    </tr>
    <tr>
      <th>8</th>
      <td>8</td>
      <td>4553</td>
    </tr>
    <tr>
      <th>9</th>
      <td>9</td>
      <td>5122</td>
    </tr>
    <tr>
      <th>10</th>
      <td>10</td>
      <td>4672</td>
    </tr>
    <tr>
      <th>11</th>
      <td>11</td>
      <td>4745</td>
    </tr>
    <tr>
      <th>12</th>
      <td>12</td>
      <td>4316</td>
    </tr>
    <tr>
      <th>13</th>
      <td>13</td>
      <td>4364</td>
    </tr>
    <tr>
      <th>14</th>
      <td>14</td>
      <td>4505</td>
    </tr>
    <tr>
      <th>15</th>
      <td>15</td>
      <td>4576</td>
    </tr>
    <tr>
      <th>16</th>
      <td>16</td>
      <td>4957</td>
    </tr>
    <tr>
      <th>17</th>
      <td>17</td>
      <td>5126</td>
    </tr>
    <tr>
      <th>18</th>
      <td>18</td>
      <td>6797</td>
    </tr>
    <tr>
      <th>19</th>
      <td>19</td>
      <td>7958</td>
    </tr>
    <tr>
      <th>20</th>
      <td>20</td>
      <td>9790</td>
    </tr>
    <tr>
      <th>21</th>
      <td>21</td>
      <td>12659</td>
    </tr>
    <tr>
      <th>22</th>
      <td>22</td>
      <td>17155</td>
    </tr>
    <tr>
      <th>23</th>
      <td>23</td>
      <td>19343</td>
    </tr>
  </tbody>
</table>
</div>


    
    (Passed.)
    


```python
p = make_barchart(df_noisy_by_hour, 'hour', 'count',
                  {'title': 'Noise complaints by hour',
                   'plot_width': 800, 'plot_height': 320})
show(p)
```








<div class="bk-root" id="d69c503f-244f-468e-99e7-d5d3c5f707d1" data-root-id="2426"></div>





**Exercise 8** (ungraded). Create a line chart to show the fraction of complaints (y-axis) associated with each hour of the day (x-axis), with each complaint type shown as a differently colored line. Show just the top 5 complaints (`top_complaints[:5]`). Remember to exclude complaints with a zero-timestamp (i.e., `00:00:00.000`).

> **Note.** This exercise is ungraded but if your time permits, please give it a try! Feel free to discuss your approaches to this problem on the discussion forums (but do try to do it yourself first). One library you may find useful to try out is holoviews (http://holoviews.org/index.html)


```python
import holoviews as hv
hv.extension('bokeh')
from holoviews import Bars

query1 = '''
    SELECT strftime ('%H', CreatedDate) AS hour, ComplaintType, COUNT(*) AS count
    FROM data
    WHERE CreatedDate NOT LIKE "%00:00:00.000%"
    GROUP BY hour, ComplaintType
'''
query2 = '''
    SELECT COUNT(*) AS freq, strftime ('%H', CreatedDate) AS hour
    FROM data
    WHERE CreatedDate NOT LIKE "%00:00:00.000%"
    GROUP BY hour
'''
query3 = '''
    SELECT ComplaintType, count(*) AS num
    FROM data
    GROUP BY ComplaintType
    ORDER BY -count(*)
    LIMIT 5
'''

df_query1 = pd.read_sql_query(query1, disk_engine)
df_query2 = pd.read_sql_query(query2, disk_engine)
df_query3 = pd.read_sql_query(query3, disk_engine)

A = df_query1.merge(df_query3, on =['ComplaintType'],how ='inner')
B = A.merge(df_query2, on = ['hour'], how = 'inner')
B = B[['freq','hour','ComplaintType','count']]

df_cast = cast(B, key = 'ComplaintType', value = 'count')

df_new = df_cast.copy()

for i in df_new.columns[2:]:
    df_new[i] = df_new[i]/df_new['freq']

df_top5_frac = df_new.copy()
display(df_top5_frac.head(10))
del df_top5_frac['freq']

%opts Overlay [width=800 height=600 legend_position='top_right'] Curve

hv.Curve((df_top5_frac['Blocked Driveway'])      , label = 'Blocked Driveway') *\
hv.Curve((df_top5_frac['HEAT/HOT WATER'])        , label = 'HEAT/HOT WATER') *\
hv.Curve((df_top5_frac['Illegal Parking'])       , label = 'Illegal Parking') *\
hv.Curve((df_top5_frac['Street Condition'])      , label = 'Street Condition') *\
hv.Curve((df_top5_frac['Street Light Condition']), label = 'Street Light Condition')
```





<link rel="stylesheet" href="https://code.jquery.com/ui/1.10.4/themes/smoothness/jquery-ui.css">
<style>div.hololayout {
  display: flex;
  align-items: center;
  margin: 0;
}

div.holoframe {
  width: 75%;
}

div.holowell {
  display: flex;
  align-items: center;
}

form.holoform {
  background-color: #fafafa;
  border-radius: 5px;
  overflow: hidden;
  padding-left: 0.8em;
  padding-right: 0.8em;
  padding-top: 0.4em;
  padding-bottom: 0.4em;
  box-shadow: inset 0 1px 1px rgba(0, 0, 0, 0.05);
  margin-bottom: 20px;
  border: 1px solid #e3e3e3;
}

div.holowidgets {
  padding-right: 0;
  width: 25%;
}

div.holoslider {
  min-height: 0 !important;
  height: 0.8em;
  width: 100%;
}

div.holoformgroup {
  padding-top: 0.5em;
  margin-bottom: 0.5em;
}

div.hologroup {
  padding-left: 0;
  padding-right: 0.8em;
  width: 100%;
}

.holoselect {
  width: 92%;
  margin-left: 0;
  margin-right: 0;
}

.holotext {
  padding-left:  0.5em;
  padding-right: 0;
  width: 100%;
}

.holowidgets .ui-resizable-se {
  visibility: hidden
}

.holoframe > .ui-resizable-se {
  visibility: hidden
}

.holowidgets .ui-resizable-s {
  visibility: hidden
}


/* CSS rules for noUISlider based slider used by JupyterLab extension  */

.noUi-handle {
  width: 20px !important;
  height: 20px !important;
  left: -5px !important;
  top: -5px !important;
}

.noUi-handle:before, .noUi-handle:after {
  visibility: hidden;
  height: 0px;
}

.noUi-target {
  margin-left: 0.5em;
  margin-right: 0.5em;
}

div.bk-hbox {
    display: flex;
    justify-content: center;
}

div.bk-hbox div.bk-plot {
    padding: 8px;
}

div.bk-hbox div.bk-data-table {
    padding: 20px;
}
</style>


<div class="logo-block">
<img src='data:image/png;base64,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
roTBs2RqAAAAAElFTkSuQmCC'
     style='height:25px; border-radius:12px; display: inline-block; float: left; vertical-align: middle'></img>


  <img src='data:image/png;base64,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'
       style='height:15px; border-radius:12px; display: inline-block; float: left'></img>





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
      <th>freq</th>
      <th>hour</th>
      <th>Blocked Driveway</th>
      <th>HEAT/HOT WATER</th>
      <th>Illegal Parking</th>
      <th>Street Condition</th>
      <th>Street Light Condition</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>32418</td>
      <td>00</td>
      <td>0.093467</td>
      <td>0.004288</td>
      <td>0.079369</td>
      <td>0.040039</td>
      <td>0.027485</td>
    </tr>
    <tr>
      <th>1</th>
      <td>23489</td>
      <td>01</td>
      <td>0.090936</td>
      <td>0.003789</td>
      <td>0.071012</td>
      <td>0.046532</td>
      <td>0.026523</td>
    </tr>
    <tr>
      <th>2</th>
      <td>15226</td>
      <td>02</td>
      <td>0.090569</td>
      <td>0.004466</td>
      <td>0.070997</td>
      <td>0.043413</td>
      <td>0.025286</td>
    </tr>
    <tr>
      <th>3</th>
      <td>10164</td>
      <td>03</td>
      <td>0.102322</td>
      <td>0.004427</td>
      <td>0.077135</td>
      <td>0.034337</td>
      <td>0.028532</td>
    </tr>
    <tr>
      <th>4</th>
      <td>8692</td>
      <td>04</td>
      <td>0.110216</td>
      <td>0.005983</td>
      <td>0.083065</td>
      <td>0.030143</td>
      <td>0.030488</td>
    </tr>
    <tr>
      <th>5</th>
      <td>10224</td>
      <td>05</td>
      <td>0.139769</td>
      <td>0.005966</td>
      <td>0.109057</td>
      <td>0.071694</td>
      <td>0.039808</td>
    </tr>
    <tr>
      <th>6</th>
      <td>23051</td>
      <td>06</td>
      <td>0.118129</td>
      <td>0.008329</td>
      <td>0.088803</td>
      <td>0.102035</td>
      <td>0.026680</td>
    </tr>
    <tr>
      <th>7</th>
      <td>42273</td>
      <td>07</td>
      <td>0.108769</td>
      <td>0.008611</td>
      <td>0.089442</td>
      <td>0.158683</td>
      <td>0.025051</td>
    </tr>
    <tr>
      <th>8</th>
      <td>73811</td>
      <td>08</td>
      <td>0.080245</td>
      <td>0.006232</td>
      <td>0.067822</td>
      <td>0.102735</td>
      <td>0.053976</td>
    </tr>
    <tr>
      <th>9</th>
      <td>100077</td>
      <td>09</td>
      <td>0.057935</td>
      <td>0.005426</td>
      <td>0.054468</td>
      <td>0.090550</td>
      <td>0.063131</td>
    </tr>
  </tbody>
</table>
</div>





<div id='2607' style='display: table; margin: 0 auto;'>





  <div class="bk-root" id="8fe61a48-a81b-4952-b2a9-7729607284b5" data-root-id="2607"></div>
</div>
<script type="application/javascript">(function(root) {
  function embed_document(root) {

  var docs_json = {"03a80c7b-8be8-4855-bfda-52b6ee4598f8":{"roots":{"references":[{"attributes":{"align":null,"below":[{"id":"2616","type":"LinearAxis"}],"center":[{"id":"2620","type":"Grid"},{"id":"2625","type":"Grid"},{"id":"2655","type":"Legend"}],"left":[{"id":"2621","type":"LinearAxis"}],"margin":null,"min_border_bottom":10,"min_border_left":10,"min_border_right":10,"min_border_top":10,"plot_width":800,"renderers":[{"id":"2647","type":"GlyphRenderer"},{"id":"2663","type":"GlyphRenderer"},{"id":"2680","type":"GlyphRenderer"},{"id":"2699","type":"GlyphRenderer"},{"id":"2720","type":"GlyphRenderer"}],"sizing_mode":"fixed","title":{"id":"2608","type":"Title"},"toolbar":{"id":"2631","type":"Toolbar"},"x_range":{"id":"2605","type":"Range1d"},"x_scale":{"id":"2612","type":"LinearScale"},"y_range":{"id":"2606","type":"Range1d"},"y_scale":{"id":"2614","type":"LinearScale"}},"id":"2607","subtype":"Figure","type":"Plot"},{"attributes":{"callback":null,"end":23.0,"reset_end":23.0,"reset_start":0.0,"tags":[[["index","index",null]]]},"id":"2605","type":"Range1d"},{"attributes":{"grid_line_color":null,"ticker":{"id":"2617","type":"BasicTicker"}},"id":"2620","type":"Grid"},{"attributes":{"line_alpha":0.2,"line_color":"#fc4f30","line_width":2,"x":{"field":"index"},"y":{"field":"HEAT/HOT WATER"}},"id":"2662","type":"Line"},{"attributes":{"data_source":{"id":"2674","type":"ColumnDataSource"},"glyph":{"id":"2677","type":"Line"},"hover_glyph":null,"muted_glyph":{"id":"2679","type":"Line"},"nonselection_glyph":{"id":"2678","type":"Line"},"selection_glyph":null,"view":{"id":"2681","type":"CDSView"}},"id":"2680","type":"GlyphRenderer"},{"attributes":{"axis_label":"index","bounds":"auto","formatter":{"id":"2638","type":"BasicTickFormatter"},"major_label_orientation":"horizontal","ticker":{"id":"2617","type":"BasicTicker"}},"id":"2616","type":"LinearAxis"},{"attributes":{"source":{"id":"2693","type":"ColumnDataSource"}},"id":"2700","type":"CDSView"},{"attributes":{},"id":"2715","type":"Selection"},{"attributes":{"data_source":{"id":"2714","type":"ColumnDataSource"},"glyph":{"id":"2717","type":"Line"},"hover_glyph":null,"muted_glyph":{"id":"2719","type":"Line"},"nonselection_glyph":{"id":"2718","type":"Line"},"selection_glyph":null,"view":{"id":"2721","type":"CDSView"}},"id":"2720","type":"GlyphRenderer"},{"attributes":{"click_policy":"mute","items":[{"id":"2656","type":"LegendItem"},{"id":"2673","type":"LegendItem"},{"id":"2692","type":"LegendItem"},{"id":"2713","type":"LegendItem"},{"id":"2736","type":"LegendItem"}]},"id":"2655","type":"Legend"},{"attributes":{"callback":null,"data":{"Street Condition":{"__ndarray__":"MM6xJg6ApD+ufFDnGNOnP5w2yJYsOqY/DN50WJqUoT81FTyqt92ePxi92J2KWrI/sKcv1/Aeuj8Cl5gzuE/EP7LITWfdTLo/pCtMi00utz9eOH/0OSK0P0IMbVwurLI/sscAJ57jtD+gG9gugru2PzN4ZJGNBL0/cStHpIkYuz9RRtpNGjW3P5kLdRFTo7Y/6qjUmUkRvD+/keqkPsi0P5P4zhDRVrE/nbvncYfeqj8C4IAS6QmoP45dhzB1+aM/","dtype":"float64","shape":[24]},"index":[0,1,2,3,4,5,6,7,8,9,10,11,12,13,14,15,16,17,18,19,20,21,22,23]},"selected":{"id":"2694","type":"Selection"},"selection_policy":{"id":"2733","type":"UnionRenderers"}},"id":"2693","type":"ColumnDataSource"},{"attributes":{"line_alpha":0.1,"line_color":"#6d904f","line_width":2,"x":{"field":"index"},"y":{"field":"Street Condition"}},"id":"2697","type":"Line"},{"attributes":{"overlay":{"id":"2654","type":"BoxAnnotation"}},"id":"2629","type":"BoxZoomTool"},{"attributes":{"text":"","text_color":{"value":"black"},"text_font_size":{"value":"12pt"}},"id":"2608","type":"Title"},{"attributes":{},"id":"2670","type":"UnionRenderers"},{"attributes":{},"id":"2689","type":"UnionRenderers"},{"attributes":{},"id":"2614","type":"LinearScale"},{"attributes":{"line_alpha":0.2,"line_color":"#6d904f","line_width":2,"x":{"field":"index"},"y":{"field":"Street Condition"}},"id":"2698","type":"Line"},{"attributes":{"active_drag":"auto","active_inspect":"auto","active_multi":null,"active_scroll":"auto","active_tap":"auto","tools":[{"id":"2626","type":"SaveTool"},{"id":"2627","type":"PanTool"},{"id":"2628","type":"WheelZoomTool"},{"id":"2629","type":"BoxZoomTool"},{"id":"2630","type":"ResetTool"}]},"id":"2631","type":"Toolbar"},{"attributes":{"axis_label":"Blocked Driveway","bounds":"auto","formatter":{"id":"2640","type":"BasicTickFormatter"},"major_label_orientation":"horizontal","ticker":{"id":"2622","type":"BasicTicker"}},"id":"2621","type":"LinearAxis"},{"attributes":{"source":{"id":"2714","type":"ColumnDataSource"}},"id":"2721","type":"CDSView"},{"attributes":{},"id":"2612","type":"LinearScale"},{"attributes":{"dimension":1,"grid_line_color":null,"ticker":{"id":"2622","type":"BasicTicker"}},"id":"2625","type":"Grid"},{"attributes":{},"id":"2622","type":"BasicTicker"},{"attributes":{"line_color":"#8b8b8b","line_width":2,"x":{"field":"index"},"y":{"field":"Street Light Condition"}},"id":"2717","type":"Line"},{"attributes":{"source":{"id":"2674","type":"ColumnDataSource"}},"id":"2681","type":"CDSView"},{"attributes":{"label":{"value":"Illegal Parking"},"renderers":[{"id":"2680","type":"GlyphRenderer"}]},"id":"2692","type":"LegendItem"},{"attributes":{"label":{"value":"Street Light Condition"},"renderers":[{"id":"2720","type":"GlyphRenderer"}]},"id":"2736","type":"LegendItem"},{"attributes":{},"id":"2638","type":"BasicTickFormatter"},{"attributes":{},"id":"2733","type":"UnionRenderers"},{"attributes":{"line_alpha":0.2,"line_color":"#e5ae38","line_width":2,"x":{"field":"index"},"y":{"field":"Illegal Parking"}},"id":"2679","type":"Line"},{"attributes":{"line_color":"#6d904f","line_width":2,"x":{"field":"index"},"y":{"field":"Street Condition"}},"id":"2696","type":"Line"},{"attributes":{"bottom_units":"screen","fill_alpha":{"value":0.5},"fill_color":{"value":"lightgrey"},"left_units":"screen","level":"overlay","line_alpha":{"value":1.0},"line_color":{"value":"black"},"line_dash":[4,4],"line_width":{"value":2},"render_mode":"css","right_units":"screen","top_units":"screen"},"id":"2654","type":"BoxAnnotation"},{"attributes":{},"id":"2642","type":"Selection"},{"attributes":{},"id":"2627","type":"PanTool"},{"attributes":{},"id":"2694","type":"Selection"},{"attributes":{},"id":"2750","type":"UnionRenderers"},{"attributes":{"line_alpha":0.1,"line_color":"#e5ae38","line_width":2,"x":{"field":"index"},"y":{"field":"Illegal Parking"}},"id":"2678","type":"Line"},{"attributes":{},"id":"2710","type":"UnionRenderers"},{"attributes":{"source":{"id":"2641","type":"ColumnDataSource"}},"id":"2648","type":"CDSView"},{"attributes":{"source":{"id":"2657","type":"ColumnDataSource"}},"id":"2664","type":"CDSView"},{"attributes":{"data_source":{"id":"2641","type":"ColumnDataSource"},"glyph":{"id":"2644","type":"Line"},"hover_glyph":null,"muted_glyph":{"id":"2646","type":"Line"},"nonselection_glyph":{"id":"2645","type":"Line"},"selection_glyph":null,"view":{"id":"2648","type":"CDSView"}},"id":"2647","type":"GlyphRenderer"},{"attributes":{"data_source":{"id":"2693","type":"ColumnDataSource"},"glyph":{"id":"2696","type":"Line"},"hover_glyph":null,"muted_glyph":{"id":"2698","type":"Line"},"nonselection_glyph":{"id":"2697","type":"Line"},"selection_glyph":null,"view":{"id":"2700","type":"CDSView"}},"id":"2699","type":"GlyphRenderer"},{"attributes":{},"id":"2628","type":"WheelZoomTool"},{"attributes":{"callback":null,"data":{"Blocked Driveway":{"__ndarray__":"RmGhNm3ttz+Rpo37l0e3P+JZO7eDL7c/cTs69sQxuj8aSqeEIje8Pw0gUMj048E/kY2/4Lk9vj8tdgY1TNi7P9B2NfP3irQ/lnuDGLWprT8apmK4thCoPzKqvWrQ9KU/RrBtbC3fpj+Q1F3YZx2lP8uwegKTE6M/IitP0Y+boz839QXVJqmnP37erZTzma4/xlZhwUSosj/keJbiBZe2Pz8MVZ0Y+Lk/b9Yyt1WWuz9i4FcRCgC6P7olGYxBGbg/","dtype":"float64","shape":[24]},"index":[0,1,2,3,4,5,6,7,8,9,10,11,12,13,14,15,16,17,18,19,20,21,22,23]},"selected":{"id":"2642","type":"Selection"},"selection_policy":{"id":"2670","type":"UnionRenderers"}},"id":"2641","type":"ColumnDataSource"},{"attributes":{},"id":"2630","type":"ResetTool"},{"attributes":{"line_alpha":0.1,"line_color":"#8b8b8b","line_width":2,"x":{"field":"index"},"y":{"field":"Street Light Condition"}},"id":"2718","type":"Line"},{"attributes":{"label":{"value":"HEAT/HOT WATER"},"renderers":[{"id":"2663","type":"GlyphRenderer"}]},"id":"2673","type":"LegendItem"},{"attributes":{"line_color":"#30a2da","line_width":2,"x":{"field":"index"},"y":{"field":"Blocked Driveway"}},"id":"2644","type":"Line"},{"attributes":{"line_alpha":0.2,"line_color":"#8b8b8b","line_width":2,"x":{"field":"index"},"y":{"field":"Street Light Condition"}},"id":"2719","type":"Line"},{"attributes":{},"id":"2617","type":"BasicTicker"},{"attributes":{},"id":"2658","type":"Selection"},{"attributes":{"callback":null,"end":0.15868284720743736,"reset_end":0.15868284720743736,"reset_start":0.0037890076205883607,"start":0.0037890076205883607,"tags":[[["Blocked Driveway","Blocked Driveway",null]]]},"id":"2606","type":"Range1d"},{"attributes":{"callback":null,"data":{"HEAT/HOT WATER":{"__ndarray__":"r3TkzwWQcT9sMxL6HwpvPyvpMs78SnI/dhV3qnQicj/1YD2DHoF4P81/v94scHg/vHIPU/sOgT8KDLw0fKKBP3vEspPdhnk/Y6S9CGM5dj/EVSvP/4p1P1dqndzk8HM/8R898OsUdj8ekG+zbvVzP6wNYgSDrXE/i2RFbCgJcz/uLNQiz3x0P8l6qNKqXng/bp2Kh0XEdj/bLyKDPCh5P6jhGi2O8H0/pHjenDligD9I4PZv5nt5P7ZxeaJbIHU/","dtype":"float64","shape":[24]},"index":[0,1,2,3,4,5,6,7,8,9,10,11,12,13,14,15,16,17,18,19,20,21,22,23]},"selected":{"id":"2658","type":"Selection"},"selection_policy":{"id":"2689","type":"UnionRenderers"}},"id":"2657","type":"ColumnDataSource"},{"attributes":{"line_color":"#e5ae38","line_width":2,"x":{"field":"index"},"y":{"field":"Illegal Parking"}},"id":"2677","type":"Line"},{"attributes":{"label":{"value":"Blocked Driveway"},"renderers":[{"id":"2647","type":"GlyphRenderer"}]},"id":"2656","type":"LegendItem"},{"attributes":{},"id":"2640","type":"BasicTickFormatter"},{"attributes":{"callback":null,"data":{"Illegal Parking":{"__ndarray__":"ulUMA49RtD+t3OgK1y2yP2+zSKbbLLI/qLY3Ux6/sz/VLzKOvUO1P/Rg8t0q67s/vWsxmsy7tj/u7qIHs+W2P4LK5CbGXLE/5j5EzTzjqz97bvEjMcqlP9usqZBPCqY/mzLiLfYUqD8oJz6DDsqlP5PZAAIXbaM/w+bB7N7Uoj+TQvMKCYClPyBFoh/iRqs/XzMa9adIrD9Z2aVU9iyxP2lyiQh+TLM/YeDaA9CQuj+MWYLBXSC2P3f40iYfg7U/","dtype":"float64","shape":[24]},"index":[0,1,2,3,4,5,6,7,8,9,10,11,12,13,14,15,16,17,18,19,20,21,22,23]},"selected":{"id":"2675","type":"Selection"},"selection_policy":{"id":"2710","type":"UnionRenderers"}},"id":"2674","type":"ColumnDataSource"},{"attributes":{"line_alpha":0.2,"line_color":"#30a2da","line_width":2,"x":{"field":"index"},"y":{"field":"Blocked Driveway"}},"id":"2646","type":"Line"},{"attributes":{"label":{"value":"Street Condition"},"renderers":[{"id":"2699","type":"GlyphRenderer"}]},"id":"2713","type":"LegendItem"},{"attributes":{"line_color":"#fc4f30","line_width":2,"x":{"field":"index"},"y":{"field":"HEAT/HOT WATER"}},"id":"2660","type":"Line"},{"attributes":{},"id":"2675","type":"Selection"},{"attributes":{"line_alpha":0.1,"line_color":"#30a2da","line_width":2,"x":{"field":"index"},"y":{"field":"Blocked Driveway"}},"id":"2645","type":"Line"},{"attributes":{"line_alpha":0.1,"line_color":"#fc4f30","line_width":2,"x":{"field":"index"},"y":{"field":"HEAT/HOT WATER"}},"id":"2661","type":"Line"},{"attributes":{"callback":null,"data":{"Street Light Condition":{"__ndarray__":"h3oSDvUknD/+7M/62yibP80tRk1+5Jk/r2mjEoM3nT84H4PzMTifP//9erPAYaQ/wb0Gg/5Rmz/PUOBvFqeZP6Ai28azoqs/oORg8mApsD8eT1k2e7q8P3wRn1Ccvr4/l26Z91v2rT8JkD8FwfOyP+r1Ga1qVbg/aVetBw9Ptj95kuzO+12vP/COXBpnEKQ/R6TmsAnSoj8qPN5JFfumP+LkuQDUO68/+pD/TmlbsT+T7FM8TAGqP+yEYgVZOaY/","dtype":"float64","shape":[24]},"index":[0,1,2,3,4,5,6,7,8,9,10,11,12,13,14,15,16,17,18,19,20,21,22,23]},"selected":{"id":"2715","type":"Selection"},"selection_policy":{"id":"2750","type":"UnionRenderers"}},"id":"2714","type":"ColumnDataSource"},{"attributes":{},"id":"2626","type":"SaveTool"},{"attributes":{"data_source":{"id":"2657","type":"ColumnDataSource"},"glyph":{"id":"2660","type":"Line"},"hover_glyph":null,"muted_glyph":{"id":"2662","type":"Line"},"nonselection_glyph":{"id":"2661","type":"Line"},"selection_glyph":null,"view":{"id":"2664","type":"CDSView"}},"id":"2663","type":"GlyphRenderer"}],"root_ids":["2607"]},"title":"Bokeh Application","version":"1.4.0"}};
  var render_items = [{"docid":"03a80c7b-8be8-4855-bfda-52b6ee4598f8","roots":{"2607":"8fe61a48-a81b-4952-b2a9-7729607284b5"}}];
  root.Bokeh.embed.embed_items_notebook(docs_json, render_items);

  }
  if (root.Bokeh !== undefined) {
    embed_document(root);
  } else {
    var attempts = 0;
    var timer = setInterval(function(root) {
      if (root.Bokeh !== undefined) {
        clearInterval(timer);
        embed_document(root);
      } else {
        attempts++;
        if (attempts > 100) {
          clearInterval(timer);
          console.log("Bokeh: ERROR: Unable to run BokehJS code because BokehJS library is missing");
        }
      }
    }, 10, root)
  }
})(window);</script>



### Learn more

- Find more open data sets on [Data.gov](https://data.gov) and [NYC Open Data](https://nycopendata.socrata.com)
- Learn how to setup [MySql with Pandas and Plotly](http://moderndata.plot.ly/graph-data-from-mysql-database-in-python/)
- Big data workflows with [HDF5 and Pandas](http://stackoverflow.com/questions/14262433/large-data-work-flows-using-pandas)
