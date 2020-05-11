**Important note**! Before you turn in this lab notebook, make sure everything runs as expected:

- First, restart the kernel -- in the menubar, select Kernel → Restart.
- Then run all cells -- in the menubar, select Cell → Run All.

Make sure you fill in any place that says YOUR CODE HERE or "YOUR ANSWER HERE."

## UK Traffic Accidents

In this problem, you will work with and analyze some data about accidents in the UK from 2009 to 2011. This data was derived from Kaggle.  The original dataset can be found here: https://www.kaggle.com/daveianhickey/2000-16-traffic-flow-england-scotland-wales/data.

This problem has 4 exercises worth a total of 10 points.

## Setup

Run the following code cell, which will load the modules you'll need for this problem.

> **Note.** This problem involves SQLite and the `sqlite3` module. Since that module is not supported in Vocareum when using the Python 3.6 kernel, we have set this notebook to use Python 3.5. If you do any testing or prototyping on your local machine, keep in mind that you are still responsible for making your code work when submitted through the autograder on Vocareum, so be mindful of potential versioning differences.


```python
import sys
import pandas as pd
import numpy as np
import sqlite3 as db

print("Python version: {}".format(sys.version))
print("Pandas version: {}".format(pd.__version__))
print("Numpy version: {}".format(np.__version__))
print("SQLite3 version: {}".format(db.version))

from IPython.display import display
from cse6040utils import download_all, canonicalize_tibble, tibbles_are_equivalent
```

    Python version: 3.7.4 (tags/v3.7.4:e09359112e, Jul  8 2019, 20:34:20) [MSC v.1916 64 bit (AMD64)]
    Pandas version: 0.25.3
    Numpy version: 1.18.1
    SQLite3 version: 2.6.0
    

## The Dataset

To help with your analysis, we will first drop any record that has missing value. We will also transform the column, `Date`, to have the structure yyyy-mm-dd.  Using this transformed Date column, we will then add a `Month` column to the dataset, which you will use in the exercises.


```python
print('Loading dataset...This may take a while...')

datasets = {'accident_by_hour_soln.csv': '46ae91224473fc2d15794716d10231ce',
            'accidents_2009_to_2011.csv': '530ce2d51394f77a21fdd741a8ac9f0b',
            'max_dayofweek_soln.csv': '54f0f74c9ac05880e6a5b23d5d34f11b',
            'top10_location_soln.csv': '5b67bcf14fd719afe8444a00a3390c80'}
datapaths = download_all(datasets, suffix='accidents/')

#let's read the data into our environment
Accidents = pd.read_csv(datapaths["accidents_2009_to_2011.csv"])

#we will remove any rows that has missing values
Accidents = Accidents.dropna() 

#transform the Date column
Accidents['Date'] = pd.to_datetime(Accidents['Date'], dayfirst=True, infer_datetime_format=True).dt.date

#add the Month column
Accidents['Month'] = pd.to_datetime(Accidents['Date'], dayfirst=True, infer_datetime_format=True).dt.month


assert len(Accidents)==281765 # number of records
assert len(Accidents.columns) == 18 # number of columns

print('\nAfter preprocessing, Accidents has {} records and {} columns'.format(len(Accidents), len(Accidents.columns)))
print('\nFirst 5 records of Accidents')
Accidents.head()
```

    Loading dataset...This may take a while...
    'accident_by_hour_soln.csv' is ready!
    'accidents_2009_to_2011.csv' is ready!
    'max_dayofweek_soln.csv' is ready!
    'top10_location_soln.csv' is ready!
    
    After preprocessing, Accidents has 281765 records and 18 columns
    
    First 5 records of Accidents
    




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
      <th>Accident_Index</th>
      <th>Location_Easting_OSGR</th>
      <th>Location_Northing_OSGR</th>
      <th>Accident_Severity</th>
      <th>Number_of_Vehicles</th>
      <th>Date</th>
      <th>Day_of_Week</th>
      <th>Time</th>
      <th>Road_Type</th>
      <th>Speed_limit</th>
      <th>Junction_Control</th>
      <th>Pedestrian_Crossing-Human_Control</th>
      <th>Light_Conditions</th>
      <th>Weather_Conditions</th>
      <th>Road_Surface_Conditions</th>
      <th>Special_Conditions_at_Site</th>
      <th>Year</th>
      <th>Month</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>200901BS70001</td>
      <td>524910</td>
      <td>180800</td>
      <td>2</td>
      <td>2</td>
      <td>2009-01-01</td>
      <td>5</td>
      <td>15:11</td>
      <td>One way street</td>
      <td>30</td>
      <td>Giveway or uncontrolled</td>
      <td>None within 50 metres</td>
      <td>Daylight: Street light present</td>
      <td>Fine without high winds</td>
      <td>Dry</td>
      <td>None</td>
      <td>2009</td>
      <td>1</td>
    </tr>
    <tr>
      <th>1</th>
      <td>200901BS70002</td>
      <td>525050</td>
      <td>181040</td>
      <td>2</td>
      <td>2</td>
      <td>2009-01-05</td>
      <td>2</td>
      <td>10:59</td>
      <td>Single carriageway</td>
      <td>30</td>
      <td>Giveway or uncontrolled</td>
      <td>None within 50 metres</td>
      <td>Daylight: Street light present</td>
      <td>Fine without high winds</td>
      <td>Wet/Damp</td>
      <td>None</td>
      <td>2009</td>
      <td>1</td>
    </tr>
    <tr>
      <th>2</th>
      <td>200901BS70003</td>
      <td>526490</td>
      <td>177990</td>
      <td>3</td>
      <td>2</td>
      <td>2009-01-04</td>
      <td>1</td>
      <td>14:19</td>
      <td>Single carriageway</td>
      <td>30</td>
      <td>Giveway or uncontrolled</td>
      <td>None within 50 metres</td>
      <td>Daylight: Street light present</td>
      <td>Fine without high winds</td>
      <td>Dry</td>
      <td>None</td>
      <td>2009</td>
      <td>1</td>
    </tr>
    <tr>
      <th>3</th>
      <td>200901BS70004</td>
      <td>524800</td>
      <td>180300</td>
      <td>2</td>
      <td>2</td>
      <td>2009-01-05</td>
      <td>2</td>
      <td>8:10</td>
      <td>Single carriageway</td>
      <td>30</td>
      <td>Automatic traffic signal</td>
      <td>None within 50 metres</td>
      <td>Daylight: Street light present</td>
      <td>Other</td>
      <td>Frost/Ice</td>
      <td>None</td>
      <td>2009</td>
      <td>1</td>
    </tr>
    <tr>
      <th>4</th>
      <td>200901BS70005</td>
      <td>526930</td>
      <td>177490</td>
      <td>2</td>
      <td>2</td>
      <td>2009-01-06</td>
      <td>3</td>
      <td>17:25</td>
      <td>Single carriageway</td>
      <td>30</td>
      <td>Automatic traffic signal</td>
      <td>None within 50 metres</td>
      <td>Darkness: Street lights present and lit</td>
      <td>Fine without high winds</td>
      <td>Dry</td>
      <td>None</td>
      <td>2009</td>
      <td>1</td>
    </tr>
  </tbody>
</table>
</div>



Let's take a look the column names for our dataset.


```python
print('\nA list of column names')
list(Accidents) # a list of column names
```

    
    A list of column names
    




    ['Accident_Index',
     'Location_Easting_OSGR',
     'Location_Northing_OSGR',
     'Accident_Severity',
     'Number_of_Vehicles',
     'Date',
     'Day_of_Week',
     'Time',
     'Road_Type',
     'Speed_limit',
     'Junction_Control',
     'Pedestrian_Crossing-Human_Control',
     'Light_Conditions',
     'Weather_Conditions',
     'Road_Surface_Conditions',
     'Special_Conditions_at_Site',
     'Year',
     'Month']



**Exercise 0** (2 points) Using the column, `Time`, which can be of the form HH:MM or H:MM, add a new column to the dataset called `Hour`.  We will use this new column in future exercises.



```python
Accidents['Hour'] = Accidents['Time'].str.split(':').str[0]

# Or this
# Accidents['Hour'] = Accidents['Time'].apply(lambda x: x.split(':')[0])
```


```python
## Test Cell: exercise0 ##
assert len(Accidents['Hour'])== 281765
assert Accidents.iloc[0]['Hour']=='15'
assert Accidents.iloc[100]['Hour']=='7'
assert Accidents.iloc[1000]['Hour']=='12'
assert Accidents.iloc[10000]['Hour']=='13'
assert Accidents.iloc[100000]['Hour']=='15'
assert Accidents.iloc[200000]['Hour']=='14'
assert Accidents.iloc[281764]['Hour']=='18'

print("\n(Passed!)")
```

    
    (Passed!)
    

The following code cell creates an SQLite database file named `accident.db` and copies the Pandas dataframe that we had above into the database as a table named `Accidents`.

> For the exercises in this problem, you can either use the Pandas representation or the SQL representation, whichever helps you best solve the problem.


```python
# Import Accidents dataframe to sqlite database
# Connect to a database (or create one if it doesn't exist)

conn = db.connect('accident.db')
Accidents.to_sql('Accidents', conn, if_exists='replace', index=False)
```

Using SQL we can see the first 5 records of `Accidents`.


```python
pd.read_sql_query('SELECT * FROM Accidents LIMIT 5', conn)
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
      <th>Accident_Index</th>
      <th>Location_Easting_OSGR</th>
      <th>Location_Northing_OSGR</th>
      <th>Accident_Severity</th>
      <th>Number_of_Vehicles</th>
      <th>Date</th>
      <th>Day_of_Week</th>
      <th>Time</th>
      <th>Road_Type</th>
      <th>Speed_limit</th>
      <th>Junction_Control</th>
      <th>Pedestrian_Crossing-Human_Control</th>
      <th>Light_Conditions</th>
      <th>Weather_Conditions</th>
      <th>Road_Surface_Conditions</th>
      <th>Special_Conditions_at_Site</th>
      <th>Year</th>
      <th>Month</th>
      <th>Hour</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>200901BS70001</td>
      <td>524910</td>
      <td>180800</td>
      <td>2</td>
      <td>2</td>
      <td>2009-01-01</td>
      <td>5</td>
      <td>15:11</td>
      <td>One way street</td>
      <td>30</td>
      <td>Giveway or uncontrolled</td>
      <td>None within 50 metres</td>
      <td>Daylight: Street light present</td>
      <td>Fine without high winds</td>
      <td>Dry</td>
      <td>None</td>
      <td>2009</td>
      <td>1</td>
      <td>15</td>
    </tr>
    <tr>
      <th>1</th>
      <td>200901BS70002</td>
      <td>525050</td>
      <td>181040</td>
      <td>2</td>
      <td>2</td>
      <td>2009-01-05</td>
      <td>2</td>
      <td>10:59</td>
      <td>Single carriageway</td>
      <td>30</td>
      <td>Giveway or uncontrolled</td>
      <td>None within 50 metres</td>
      <td>Daylight: Street light present</td>
      <td>Fine without high winds</td>
      <td>Wet/Damp</td>
      <td>None</td>
      <td>2009</td>
      <td>1</td>
      <td>10</td>
    </tr>
    <tr>
      <th>2</th>
      <td>200901BS70003</td>
      <td>526490</td>
      <td>177990</td>
      <td>3</td>
      <td>2</td>
      <td>2009-01-04</td>
      <td>1</td>
      <td>14:19</td>
      <td>Single carriageway</td>
      <td>30</td>
      <td>Giveway or uncontrolled</td>
      <td>None within 50 metres</td>
      <td>Daylight: Street light present</td>
      <td>Fine without high winds</td>
      <td>Dry</td>
      <td>None</td>
      <td>2009</td>
      <td>1</td>
      <td>14</td>
    </tr>
    <tr>
      <th>3</th>
      <td>200901BS70004</td>
      <td>524800</td>
      <td>180300</td>
      <td>2</td>
      <td>2</td>
      <td>2009-01-05</td>
      <td>2</td>
      <td>8:10</td>
      <td>Single carriageway</td>
      <td>30</td>
      <td>Automatic traffic signal</td>
      <td>None within 50 metres</td>
      <td>Daylight: Street light present</td>
      <td>Other</td>
      <td>Frost/Ice</td>
      <td>None</td>
      <td>2009</td>
      <td>1</td>
      <td>8</td>
    </tr>
    <tr>
      <th>4</th>
      <td>200901BS70005</td>
      <td>526930</td>
      <td>177490</td>
      <td>2</td>
      <td>2</td>
      <td>2009-01-06</td>
      <td>3</td>
      <td>17:25</td>
      <td>Single carriageway</td>
      <td>30</td>
      <td>Automatic traffic signal</td>
      <td>None within 50 metres</td>
      <td>Darkness: Street lights present and lit</td>
      <td>Fine without high winds</td>
      <td>Dry</td>
      <td>None</td>
      <td>2009</td>
      <td>1</td>
      <td>17</td>
    </tr>
  </tbody>
</table>
</div>



**Exercise 1** (1 point) Determine the number of accidents that occur for each hour of the day.  Order the number of accidents in descending order. Save your result in a table named **`accident_by_hour`** with the columns named **`Hour`** and **`Num_of_Accidents`**, which is the number of accidents during that hour. 


```python
query = '''
    select Hour, count(*) as Num_of_Accidents
    from accidents
    group by 1
'''

accident_by_hour = pd.read_sql_query(query, conn)

# Show your solution:
display(accident_by_hour)
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
      <th>Hour</th>
      <th>Num_of_Accidents</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>0</td>
      <td>3486</td>
    </tr>
    <tr>
      <th>1</th>
      <td>1</td>
      <td>2468</td>
    </tr>
    <tr>
      <th>2</th>
      <td>10</td>
      <td>13003</td>
    </tr>
    <tr>
      <th>3</th>
      <td>11</td>
      <td>14898</td>
    </tr>
    <tr>
      <th>4</th>
      <td>12</td>
      <td>17243</td>
    </tr>
    <tr>
      <th>5</th>
      <td>13</td>
      <td>18002</td>
    </tr>
    <tr>
      <th>6</th>
      <td>14</td>
      <td>17775</td>
    </tr>
    <tr>
      <th>7</th>
      <td>15</td>
      <td>22025</td>
    </tr>
    <tr>
      <th>8</th>
      <td>16</td>
      <td>23591</td>
    </tr>
    <tr>
      <th>9</th>
      <td>17</td>
      <td>25615</td>
    </tr>
    <tr>
      <th>10</th>
      <td>18</td>
      <td>20155</td>
    </tr>
    <tr>
      <th>11</th>
      <td>19</td>
      <td>14943</td>
    </tr>
    <tr>
      <th>12</th>
      <td>2</td>
      <td>1894</td>
    </tr>
    <tr>
      <th>13</th>
      <td>20</td>
      <td>10415</td>
    </tr>
    <tr>
      <th>14</th>
      <td>21</td>
      <td>8303</td>
    </tr>
    <tr>
      <th>15</th>
      <td>22</td>
      <td>6804</td>
    </tr>
    <tr>
      <th>16</th>
      <td>23</td>
      <td>4985</td>
    </tr>
    <tr>
      <th>17</th>
      <td>3</td>
      <td>1705</td>
    </tr>
    <tr>
      <th>18</th>
      <td>4</td>
      <td>1182</td>
    </tr>
    <tr>
      <th>19</th>
      <td>5</td>
      <td>1837</td>
    </tr>
    <tr>
      <th>20</th>
      <td>6</td>
      <td>4449</td>
    </tr>
    <tr>
      <th>21</th>
      <td>7</td>
      <td>11436</td>
    </tr>
    <tr>
      <th>22</th>
      <td>8</td>
      <td>21016</td>
    </tr>
    <tr>
      <th>23</th>
      <td>9</td>
      <td>14535</td>
    </tr>
  </tbody>
</table>
</div>



```python
## Test Cell: exercise1 ##
# Read what we believe is the exact result
accident_by_hour_soln = pd.read_csv(datapaths['accident_by_hour_soln.csv'])

# Check that we got a data frame of the expected shape:
assert 'accident_by_hour' in globals(), "You need to store your results in a dataframe named `accident_by_hour`."
assert type(accident_by_hour) is type(pd.DataFrame()), "`accident_by_hour` does not appear to be a Pandas dataframe."
assert len(accident_by_hour) == len(accident_by_hour_soln), "The number of rows of `accident_by_hour` does not match our solution."
assert set(accident_by_hour.columns) == set(['Hour', 'Num_of_Accidents']), "Your table does not have the right set of columns."

assert tibbles_are_equivalent(accident_by_hour.astype('int64'), accident_by_hour_soln)
print("\n(Passed!)")
```

    
    (Passed!)
    

**Exercise 2** (3 points) Suppose we are interested in determining which day of the week had the most accidents in a particular year-month pair.

For each year and month, report the day of the week that had the largest number of accidents. Your result should be in ascending order by years then months, i.e., 2009-2011 for the year and 1-12 for the month. Save your result in a table called **`max_dayofweek`**.

Your table should contain the following columns: {`'Year'`, `'Month'`, `'Day_of_Week'`, `'Num_of_Accidents'`}.

For example, a row of this table might be `{2009, 1, 6, XXXX}`, where `XXXX` is the number of accidents observed in January 2009 on Friday. (In this data, days of the week are numbered starting at Sunday equals one.)


```python
# interesting sql problem
query = '''
    select Year, Month, Day_of_Week, max(count) as Num_of_Accidents
    from (
        select Year, Month, Day_of_Week, count(*) as count
        from accidents
        group by 1, 2, 3
    )
    group by 1, 2
'''

max_dayofweek = pd.read_sql_query(query, conn)

# Show your solution:
display(max_dayofweek)
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
      <th>Year</th>
      <th>Month</th>
      <th>Day_of_Week</th>
      <th>Num_of_Accidents</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>2009</td>
      <td>1</td>
      <td>6</td>
      <td>1404</td>
    </tr>
    <tr>
      <th>1</th>
      <td>2009</td>
      <td>2</td>
      <td>6</td>
      <td>1032</td>
    </tr>
    <tr>
      <th>2</th>
      <td>2009</td>
      <td>3</td>
      <td>3</td>
      <td>1387</td>
    </tr>
    <tr>
      <th>3</th>
      <td>2009</td>
      <td>4</td>
      <td>4</td>
      <td>1426</td>
    </tr>
    <tr>
      <th>4</th>
      <td>2009</td>
      <td>5</td>
      <td>6</td>
      <td>1707</td>
    </tr>
    <tr>
      <th>5</th>
      <td>2009</td>
      <td>6</td>
      <td>3</td>
      <td>1579</td>
    </tr>
    <tr>
      <th>6</th>
      <td>2009</td>
      <td>7</td>
      <td>6</td>
      <td>1573</td>
    </tr>
    <tr>
      <th>7</th>
      <td>2009</td>
      <td>8</td>
      <td>7</td>
      <td>1244</td>
    </tr>
    <tr>
      <th>8</th>
      <td>2009</td>
      <td>9</td>
      <td>3</td>
      <td>1541</td>
    </tr>
    <tr>
      <th>9</th>
      <td>2009</td>
      <td>10</td>
      <td>6</td>
      <td>1693</td>
    </tr>
    <tr>
      <th>10</th>
      <td>2009</td>
      <td>11</td>
      <td>6</td>
      <td>1553</td>
    </tr>
    <tr>
      <th>11</th>
      <td>2009</td>
      <td>12</td>
      <td>4</td>
      <td>1456</td>
    </tr>
    <tr>
      <th>12</th>
      <td>2010</td>
      <td>1</td>
      <td>6</td>
      <td>1107</td>
    </tr>
    <tr>
      <th>13</th>
      <td>2010</td>
      <td>2</td>
      <td>6</td>
      <td>1141</td>
    </tr>
    <tr>
      <th>14</th>
      <td>2010</td>
      <td>3</td>
      <td>3</td>
      <td>1364</td>
    </tr>
    <tr>
      <th>15</th>
      <td>2010</td>
      <td>4</td>
      <td>5</td>
      <td>1413</td>
    </tr>
    <tr>
      <th>16</th>
      <td>2010</td>
      <td>5</td>
      <td>7</td>
      <td>1262</td>
    </tr>
    <tr>
      <th>17</th>
      <td>2010</td>
      <td>6</td>
      <td>4</td>
      <td>1550</td>
    </tr>
    <tr>
      <th>18</th>
      <td>2010</td>
      <td>7</td>
      <td>6</td>
      <td>1528</td>
    </tr>
    <tr>
      <th>19</th>
      <td>2010</td>
      <td>8</td>
      <td>3</td>
      <td>1401</td>
    </tr>
    <tr>
      <th>20</th>
      <td>2010</td>
      <td>9</td>
      <td>4</td>
      <td>1607</td>
    </tr>
    <tr>
      <th>21</th>
      <td>2010</td>
      <td>10</td>
      <td>6</td>
      <td>1590</td>
    </tr>
    <tr>
      <th>22</th>
      <td>2010</td>
      <td>11</td>
      <td>3</td>
      <td>1522</td>
    </tr>
    <tr>
      <th>23</th>
      <td>2010</td>
      <td>12</td>
      <td>6</td>
      <td>1048</td>
    </tr>
    <tr>
      <th>24</th>
      <td>2011</td>
      <td>1</td>
      <td>2</td>
      <td>1119</td>
    </tr>
    <tr>
      <th>25</th>
      <td>2011</td>
      <td>2</td>
      <td>4</td>
      <td>1119</td>
    </tr>
    <tr>
      <th>26</th>
      <td>2011</td>
      <td>3</td>
      <td>5</td>
      <td>1379</td>
    </tr>
    <tr>
      <th>27</th>
      <td>2011</td>
      <td>4</td>
      <td>6</td>
      <td>1281</td>
    </tr>
    <tr>
      <th>28</th>
      <td>2011</td>
      <td>5</td>
      <td>3</td>
      <td>1425</td>
    </tr>
    <tr>
      <th>29</th>
      <td>2011</td>
      <td>6</td>
      <td>5</td>
      <td>1407</td>
    </tr>
    <tr>
      <th>30</th>
      <td>2011</td>
      <td>7</td>
      <td>6</td>
      <td>1574</td>
    </tr>
    <tr>
      <th>31</th>
      <td>2011</td>
      <td>8</td>
      <td>4</td>
      <td>1229</td>
    </tr>
    <tr>
      <th>32</th>
      <td>2011</td>
      <td>9</td>
      <td>6</td>
      <td>1529</td>
    </tr>
    <tr>
      <th>33</th>
      <td>2011</td>
      <td>10</td>
      <td>2</td>
      <td>1354</td>
    </tr>
    <tr>
      <th>34</th>
      <td>2011</td>
      <td>11</td>
      <td>3</td>
      <td>1501</td>
    </tr>
    <tr>
      <th>35</th>
      <td>2011</td>
      <td>12</td>
      <td>6</td>
      <td>1598</td>
    </tr>
  </tbody>
</table>
</div>



```python
## Test Cell: exercise2 ##
# Read what we believe is the exact result
max_dayofweek_soln = pd.read_csv(datapaths['max_dayofweek_soln.csv'])

# Check that we got a data frame of the expected shape:
assert 'max_dayofweek' in globals(), "You need to store your results in a dataframe named `max_dayofweek`."
assert type(max_dayofweek) is type(pd.DataFrame()), "`max_dayofweek` does not appear to be a Pandas dataframe."
assert len(max_dayofweek) == len(max_dayofweek_soln), "The number of rows of `max_dayofweek` does not match our solution."
assert set(max_dayofweek.columns) == set(['Year', 'Month', 'Day_of_Week', 'Num_of_Accidents']), "Your table does not have the right set of columns."

assert tibbles_are_equivalent(max_dayofweek, max_dayofweek_soln)
print("\n(Passed!)")
```

    
    (Passed!)
    

**Exercise 3** (4 points). Find the top 9 locations that had the most accidents. Report the **`Road_Surface_Conditions`** and the count of accidents under that condition. Store your result in table, **`top9_locations`**, which should contain the following columns:

{`'Location_Easting_OSGR'`, `'Location_Northing_OSGR'`, `'Road_Surface_Conditions'`, `'Num_of_Accidents'`}

> **Note.** We define a location by (`Location_Easting_OSGR`, `Location_Northing_OSGR`), i.e., Local British coordinates x-value, Local British coordinates y-value.


```python
# interesting sql problem
query = '''
    select
        ag.Location_Easting_OSGR,
        ag.Location_Northing_OSGR,
        a.Road_Surface_Conditions,
        count(*) as Num_of_Accidents
    from (
        select Location_Easting_OSGR, Location_Northing_OSGR
        from accidents
        group by 1, 2
        order by count(*) desc
        limit 9
    ) as ag, accidents a
    where ag.Location_Easting_OSGR = a.Location_Easting_OSGR 
        and ag.Location_Northing_OSGR = a.Location_Northing_OSGR
    group by 1, 2, 3
'''

top9_locations = pd.read_sql_query(query, conn)

# Show your solution:
display(top9_locations)
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
      <th>Location_Easting_OSGR</th>
      <th>Location_Northing_OSGR</th>
      <th>Road_Surface_Conditions</th>
      <th>Num_of_Accidents</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>408860</td>
      <td>284550</td>
      <td>Dry</td>
      <td>13</td>
    </tr>
    <tr>
      <th>1</th>
      <td>408860</td>
      <td>284550</td>
      <td>Wet/Damp</td>
      <td>3</td>
    </tr>
    <tr>
      <th>2</th>
      <td>452740</td>
      <td>338230</td>
      <td>Dry</td>
      <td>11</td>
    </tr>
    <tr>
      <th>3</th>
      <td>452740</td>
      <td>338230</td>
      <td>Wet/Damp</td>
      <td>4</td>
    </tr>
    <tr>
      <th>4</th>
      <td>454440</td>
      <td>341450</td>
      <td>Dry</td>
      <td>9</td>
    </tr>
    <tr>
      <th>5</th>
      <td>454440</td>
      <td>341450</td>
      <td>Frost/Ice</td>
      <td>1</td>
    </tr>
    <tr>
      <th>6</th>
      <td>454440</td>
      <td>341450</td>
      <td>Wet/Damp</td>
      <td>6</td>
    </tr>
    <tr>
      <th>7</th>
      <td>468790</td>
      <td>339640</td>
      <td>Dry</td>
      <td>15</td>
    </tr>
    <tr>
      <th>8</th>
      <td>468790</td>
      <td>339640</td>
      <td>Wet/Damp</td>
      <td>1</td>
    </tr>
    <tr>
      <th>9</th>
      <td>522430</td>
      <td>180080</td>
      <td>Dry</td>
      <td>12</td>
    </tr>
    <tr>
      <th>10</th>
      <td>522430</td>
      <td>180080</td>
      <td>Wet/Damp</td>
      <td>4</td>
    </tr>
    <tr>
      <th>11</th>
      <td>526930</td>
      <td>177490</td>
      <td>Dry</td>
      <td>15</td>
    </tr>
    <tr>
      <th>12</th>
      <td>526930</td>
      <td>177490</td>
      <td>Wet/Damp</td>
      <td>2</td>
    </tr>
    <tr>
      <th>13</th>
      <td>530760</td>
      <td>186270</td>
      <td>Dry</td>
      <td>21</td>
    </tr>
    <tr>
      <th>14</th>
      <td>530760</td>
      <td>186270</td>
      <td>Wet/Damp</td>
      <td>2</td>
    </tr>
    <tr>
      <th>15</th>
      <td>534940</td>
      <td>181890</td>
      <td>Dry</td>
      <td>19</td>
    </tr>
    <tr>
      <th>16</th>
      <td>534940</td>
      <td>181890</td>
      <td>Wet/Damp</td>
      <td>2</td>
    </tr>
    <tr>
      <th>17</th>
      <td>540870</td>
      <td>182730</td>
      <td>Dry</td>
      <td>17</td>
    </tr>
    <tr>
      <th>18</th>
      <td>540870</td>
      <td>182730</td>
      <td>Wet/Damp</td>
      <td>2</td>
    </tr>
  </tbody>
</table>
</div>



```python
## Test Cell: exercise3 ##
# Read what we believe is the exact result
top9_locations_soln = pd.read_csv(datapaths['top10_location_soln.csv'])

# Check that we got a data frame of the expected shape:
assert 'top9_locations' in globals(), "You need to store your results in a dataframe named `top9_locations`."
assert type(top9_locations) is type(pd.DataFrame()), "`top9_locations` does not appear to be a Pandas dataframe."
assert len(top9_locations) == len(top9_locations_soln), "The number of rows of `top9_locations` does not match our solution."
assert set(top9_locations.columns) == set(['Location_Easting_OSGR', 'Location_Northing_OSGR', 'Road_Surface_Conditions', 'Num_of_Accidents']), "Your table does not have the right set of columns."

assert tibbles_are_equivalent(top9_locations, top9_locations_soln)
print("\n(Passed!)")

```

    
    (Passed!)
    


```python
# Some cleanup code
conn.close()
```

** Fin ** You've reached the end of this problem. Don't forget to restart the kernel and run the entire notebook from top-to-bottom to make sure you did everything correctly. If that is working, try submitting this problem. (Recall that you *must* submit and pass the autograder to get credit for your work.)
