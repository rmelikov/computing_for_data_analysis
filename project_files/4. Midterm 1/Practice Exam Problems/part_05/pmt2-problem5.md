**Important note**! Before you turn in this lab notebook, make sure everything runs as expected:

- First, restart the kernel -- in the menubar, select Kernel → Restart.
- Then run all cells -- in the menubar, select Cell → Run All.

Make sure you fill in any place that says YOUR CODE HERE or "YOUR ANSWER HERE."

# Air and Ground Travel Times

In this problem, you analyze travel times between cities in the United States. You will use several datasets, including a subset of data from a [Kaggle dataset](https://www.kaggle.com/giovamata/airlinedelaycauses/data).

The goals of this problem are to find the average flight time between two cities and to find the roundtrip time between two cities using ground travel.

The first set of code cells sets up the problem and loads the Kaggle dataset into a Pandas dataframe (`flighttimes`). The `flighttimes` table contains the flight departure and arrival times of over 1 million flights. There are four columns in the table: **DepTime**, which is the departure time of the flight (in Army time); **ArrTime**, which is the arrival time of the flight (in Army time); **Origin**, which is the three-letter airport code for the origin (or departure) airport; and **Dest**, which is the three-letter airport code for the destination airport.  


```python
# Load flighttimes dataset
from cse6040utils import download_all, canonicalize_tibble, tibbles_are_equivalent
import pandas as pd
import numpy as np

datasets = {'FlightInfo.csv': '64ac75c61dc09a3a7bb2a856a27f9584',
            'airports.csv': '07349facc5ac5e73a34f084f1a261148',
            'city_average_times_soln.csv': 'fccce0d257ba51d9518469e67963696b',
            'city_ids.csv': 'b78508ea9768a41fc2bcfa3f10056a6d',
            'city_travel_times_soln.csv': '104213b86e5082c22176d3b811cbf094',
            'flight_times_soln.csv': '967dd2f4e66999c76889c6e159be0169',
            'flights.csv': 'd9313f61c4689f20184bccb9e89afd6d',
            'ground_distances_cities.csv': 'ac64e84c460ea41244d7b4f254311b1c'}
datapaths = download_all(datasets, local_suffix='flight-paths/', url_suffix='flight-paths/')

print('Loading flighttimes dataset as DataFrame...')
# Data preprocessing on flighttimes dataset
FlightInfo = pd.read_csv(datapaths['FlightInfo.csv'])
d = FlightInfo['DepTime'].tolist()
a = FlightInfo['ArrTime'].tolist()

for i,j in enumerate(d):
    if len(str(int(j))) == 3:
        d[i] = '0{}:{}'.format(str(int(j))[0], str(int(j))[1:])
    elif len(str(int(j))) == 4:
        d[i] = '{}:{}'.format(str(int(j))[:2], str(int(j))[2:])
    else:
        d[i] = np.nan

for i,j in enumerate(a):
    if len(str(int(j))) == 3:
        a[i] = '0{}:{}'.format(str(int(j))[0], str(int(j))[1:])
    elif len(str(int(j))) == 4:
        a[i] = '{}:{}'.format(str(int(j))[:2], str(int(j))[2:])
    else:
        a[i] = np.nan
        
FlightInfo['DepTime'] = d
FlightInfo['ArrTime'] = a

flighttimes = FlightInfo.dropna()

print('flighttimes dataset successfully loaded as Pandas DataFrame!')
print('The First 5 Lines of the flighttimes Dataset: ')
print(flighttimes.head())
```

    'FlightInfo.csv' is ready!
    'airports.csv' is ready!
    'city_average_times_soln.csv' is ready!
    'city_ids.csv' is ready!
    'city_travel_times_soln.csv' is ready!
    'flight_times_soln.csv' is ready!
    'flights.csv' is ready!
    'ground_distances_cities.csv' is ready!
    Loading flighttimes dataset as DataFrame...
    flighttimes dataset successfully loaded as Pandas DataFrame!
    The First 5 Lines of the flighttimes Dataset: 
      DepTime ArrTime Origin Dest
    0   20:03   22:11    IAD  TPA
    1   07:54   10:02    IAD  TPA
    2   06:28   08:04    IND  BWI
    3   18:29   19:59    IND  BWI
    4   19:40   21:21    IND  JAX
    

To find the average time between unique flights, we must first compute the time between each of the flights in the data. 

**Exercise 0** (2 points). Create a dataframe, `flight_times`, that includes the time in minutes between the `ArrTime` and `DepTime` of each flight in the `flighttimes` dataset. The final result should have three columns:

* **`'Origin'`**: the origin airport three-letter code;
* **`'Dest'`**: the destination airport three-letter code; and
* **`'Time'`**: the time between `ArrTime` and `DepTime` in minutes. 

Note that some of the **Time** values may be negative, or even zero. In such cases, the most likely explanation is a "wraparound" effect, where `ArrTime` appears to occur before `DepTime`. **For simplicity, any such negative (or even any zero) values should be removed from the final dataFrame.**


```python
flight_times = flighttimes.copy()

# This is what we could actually do instead of creating lists below. However,
# to match the instructor's solution, we are sticking to the solution below.
# We can't convert to datetime since `00:00` is shown as `24:00`. We would
# have to replace these values first. However, our solution then doesn't match
# instructor's solution.
#flight_times['ArrTime'].replace(inplace = True, to_replace = '24:00', value = '00:00')
#flight_times['DepTime'].replace(inplace = True, to_replace = '24:00', value = '00:00')
#flight_times['ArrTime'] = pd.to_datetime(flight_times['ArrTime'], format='%H:%M')
#flight_times['DepTime'] = pd.to_datetime(flight_times['DepTime'], format='%H:%M')

#flight_times['Time']=(pd.to_timedelta(flight_times['ArrTime'] + ":00") - pd.to_timedelta(flight_times['DepTime'] + ":00")) / np.timedelta64(1,'m')

flight_times['ArrTime'] = [x + ':00' for x in flight_times['ArrTime']]
flight_times['DepTime'] = [x + ':00' for x in flight_times['DepTime']]
flight_times['Time'] = pd.to_timedelta(flight_times['ArrTime']) - pd.to_timedelta(flight_times['DepTime'])
flight_times['Time'] = flight_times['Time'].apply(lambda x : int(x / np.timedelta64(1, 'm')))
flight_times = flight_times[flight_times['Time'] > 0]
del flight_times['ArrTime']
del flight_times['DepTime']

# school solution
#flight_times["arr_min"] = flight_times["ArrTime"].apply(lambda x: int(x.split(":")[0])*60 + int(x.split(":")[1]))
#flight_times["dep_min"] = flight_times["DepTime"].apply(lambda x: int(x.split(":")[0])*60 + int(x.split(":")[1]))
#flight_times["Time"] = flight_times["arr_min"] - flight_times["dep_min"]
#flight_times.drop(["arr_min", "dep_min", "DepTime", "ArrTime"], axis=1, inplace=True)
#flight_times = flight_times[flight_times["Time"] > 0]
```


```python
## TEST CODE EXERCISE 0 - flight_times

flight_times_soln = pd.read_csv(datapaths['flight_times_soln.csv'])

print('===== First 5 Lines of Your Solution =====')
print(flight_times.head())

print('\n')
print('===== First 5 Lines of Instructor Solution =====')
print(flight_times_soln.head())

print('\n Checking if DataFrames Match...')
assert tibbles_are_equivalent(flight_times, flight_times_soln) == True, print('\n DataFrames do not match')
print("\n(Passed!)")
```

    ===== First 5 Lines of Your Solution =====
      Origin Dest  Time
    0    IAD  TPA   128
    1    IAD  TPA   128
    2    IND  BWI    96
    3    IND  BWI    90
    4    IND  JAX   101
    
    
    ===== First 5 Lines of Instructor Solution =====
      Origin Dest  Time
    0    IAD  TPA   128
    1    IAD  TPA   128
    2    IND  BWI    96
    3    IND  BWI    90
    4    IND  JAX   101
    
     Checking if DataFrames Match...
    
    (Passed!)
    

For the next part of this problem, we will load in a dataset containing the city names corresponding to each of the airports.


```python
# Load airports dataset into Pandas DataFrame
airports = pd.read_csv(datapaths['airports.csv'])
print(airports.head())
```

      Code                                 Name         City State
    0  ABE  Lehigh Valley International Airport    Allentown    PA
    1  ABI             Abilene Regional Airport      Abilene    TX
    2  ABQ    Albuquerque International Sunport  Albuquerque    NM
    3  ABR            Aberdeen Regional Airport     Aberdeen    SD
    4  ABY   Southwest Georgia Regional Airport       Albany    GA
    

**Exercise 1** (2 points). Replace the airport codes from execise 0 with their city and state using the airports dataset. Store the result in a DataFrame named `city_travel_times`. The final result should have three columns:

* **`'origin_city'`**: the origin city of the flight, in the form "City, State";
* **`'destination_city'`**: the destination city of the flight, in the form "City, State";
* and **`'Time'`**: same column and values as the previous exercise, i.e., the time between `'ArrTime'` and `'DepTime'` in minutes.

This final dataframe should only have the rows in which the origin and destination are present in your `flight_times` dataframe as well as the `airports` dataframe.

Note that some airports have the same city and state. For the purposes of this problem, you do NOT have to differentiate between those airports. For example, `IAD` and `DCA` will both have the same `origin_city`, `"Washington, DC"`. 


```python
# Concatenate two columns
airports['city_state'] = airports[['City', 'State']].apply(lambda x: ', '.join(x), axis = 1)

new_codes = airports[['Code', 'city_state']]

# Merge tables
city_travel_times = flight_times.merge(new_codes, left_on = 'Origin', right_on = 'Code', how = 'inner')
city_travel_times = city_travel_times.merge(new_codes, left_on = 'Dest', right_on = 'Code', how = 'inner')

# Rename columns
city_travel_times.rename(columns = {'city_state_x': 'origin_city', 'city_state_y': 'destination_city'}, inplace = True)

# Drop columns
city_travel_times.drop(['Origin', 'Dest', 'Code_x', 'Code_y'], axis = 1, inplace = True)

# Reorder columns
city_travel_times = city_travel_times[['origin_city', 'destination_city', 'Time']]
```


```python
## TEST CODE EXERCISE 1 - city_travel_times
city_travel_times_soln = pd.read_csv(datapaths['city_travel_times_soln.csv'])

print('===== First 5 Lines of Your Solution =====')
print(city_travel_times.head())

print('\n')
print('====== First 5 Lines of Instructor Solution =====')
print(city_travel_times_soln.head())

print('\n Checking if DataFrames Match...')
assert tibbles_are_equivalent(city_travel_times, city_travel_times_soln) == True, print("\n DataFrames do not match")
print("\n(Passed!)")
```

    ===== First 5 Lines of Your Solution =====
          origin_city destination_city  Time
    0  Washington, DC        Tampa, FL   128
    1  Washington, DC        Tampa, FL   128
    2  Washington, DC        Tampa, FL   126
    3  Washington, DC        Tampa, FL   137
    4  Washington, DC        Tampa, FL   133
    
    
    ====== First 5 Lines of Instructor Solution =====
          origin_city destination_city  Time
    0  Washington, DC        Tampa, FL   128
    1  Washington, DC        Tampa, FL   128
    2  Washington, DC        Tampa, FL   126
    3  Washington, DC        Tampa, FL   137
    4  Washington, DC        Tampa, FL   133
    
     Checking if DataFrames Match...
    
    (Passed!)
    

Finally, we will get the average flight time for each unique city to city flight.

**Exercise 2** (2 points). Create a new DataFrame, `city_average_times`, which lists the average flight time of each unique city to city flight in the `flighttimes` dataset. The final result should be a DataFrame with three columns:  **origin_city**: the origin city of the flight, in the form "City, State" ; **destination_city**: the destination city of the flight, in the form "City, State" ; and **average_time**: the average flight time between the origin and destination city. Round the results to the nearest two decimal places.


```python
city_average_times = city_travel_times.groupby(['origin_city', 'destination_city'], as_index = False).mean().round(2)
city_average_times.rename(columns = {'Time' : 'average_time'}, inplace = True)
```


```python
## TEST CODE EXERCISE 2 - city_average_times
city_average_times_soln = pd.read_csv(datapaths['city_average_times_soln.csv'])

print('===== First 5 Lines of Your Solution =====')
print(city_average_times.head())

print('\n')
print('====== First 5 Lines of Instructor Solution =====')
print(city_average_times_soln.head())

print('\n Checking if DataFrames Match...')
assert city_average_times.shape == city_average_times_soln.shape, print("Dimensions of your solution do not match the instructor's solution")
soln = pd.merge(city_average_times, city_average_times_soln, how="right", on=["origin_city", "destination_city"])
soln_time = soln["average_time_y"] - soln["average_time_x"]
tolerance = 1
assert max(abs(soln_time)) <=tolerance, print("Your average time is beyond the tolerances provided")
print("\n(Passed!)")
```

    ===== First 5 Lines of Your Solution =====
           origin_city destination_city  average_time
    0      Abilene, TX       Dallas, TX         59.84
    1  Adak Island, AK    Anchorage, AK        220.04
    2    Aguadilla, PR     New York, NY        200.63
    3    Aguadilla, PR       Newark, NJ        220.54
    4    Aguadilla, PR      Orlando, FL        143.21
    
    
    ====== First 5 Lines of Instructor Solution =====
           origin_city destination_city  average_time
    0      Abilene, TX       Dallas, TX         59.84
    1  Adak Island, AK    Anchorage, AK        220.04
    2    Aguadilla, PR     New York, NY        200.63
    3    Aguadilla, PR       Newark, NJ        220.54
    4    Aguadilla, PR      Orlando, FL        143.21
    
     Checking if DataFrames Match...
    
    (Passed!)
    

Next, let's look at ground travel times. In the test cell below, we generate a DataFrame, `ground_distances_cities`, that shows the average travel times (in hours) from one city to another if you did not take a plane. Note that the travel time from `A -> B` may not be the same as the travel time from `B -> A` because of traffic/waiting for trains/etc. 

(Also note: these are not true ground travel times. They are made up based on the distances between cities in terms of latitude/longitude. If you use these times to plan your next road trip, you may be in for a rude surprise!)


```python
ground_distances_cities = pd.read_csv(datapaths['ground_distances_cities.csv'])
ground_distances_cities["Average_Travel_Time"] = ground_distances_cities["Average_Travel_Time"].round(2)
print(ground_distances_cities.head())
```

      Starting_City      Ending_City  Average_Travel_Time
    0   Abilene, TX  Adak Island, AK                80.39
    1   Abilene, TX    Aguadilla, PR                43.40
    2   Abilene, TX       Albany, GA                19.51
    3   Abilene, TX       Albany, NY                29.87
    4   Abilene, TX  Albuquerque, NM                12.41
    

Next, we will assign each city a unique id and make a new DataFrame that shows the starting and ending cities in terms of their ids:


```python
# Read city ids
city_ids = pd.read_csv(datapaths['city_ids.csv'], index_col="City")
city_codes_dict = city_ids.to_dict()["id"]
print('The First 5 Lines of the city_ids Dataset: ')
print(city_ids.head())
```

    The First 5 Lines of the city_ids Dataset: 
                     id
    City               
    Abilene, TX       0
    Adak Island, AK   1
    Aguadilla, PR     2
    Albany, GA        3
    Albany, NY        4
    

The name of the new DataFrame being generated is `gnd_travel_ids`. The first five lines of the DataFrame can be seen by running the code cell below:


```python
# Generate gnd_travel_ids DataFrame

gnd_travel_ids = ground_distances_cities.copy()
gnd_travel_ids['Starting_City'] = gnd_travel_ids['Starting_City'].map(city_codes_dict)
gnd_travel_ids['Ending_City'] = gnd_travel_ids['Ending_City'].map(city_codes_dict)
print('The First 5 Lines of gnd_travel_ids: ')
print(gnd_travel_ids.head())
```

    The First 5 Lines of gnd_travel_ids: 
       Starting_City  Ending_City  Average_Travel_Time
    0              0            1                80.39
    1              0            2                43.40
    2              0            3                19.51
    3              0            4                29.87
    4              0            5                12.41
    

Now, we will put the values in the `gnd_travel_ids` DataFrame into a square table, with equal number of rows and columns which represent the origins and destinations. 

**Exercise 3** (1 point). Create a **pandas `DataFrame`**, named `travel_matrix`, where each element in `travel_matrix`, `[origin_id, destination_id]`, is the average_travel_time for that origin_id, destination_id combination. For instance, the value for `travel_matrix[0][1]` should be 80.387829. It should be noted that there are 293 distinct city ids (ranging from 0 to 292). 

(Note: The function `pivot_table()` in pandas may be helpful here. Also, the diagonal entries in the table represent the same origin and destination. Such entries must be equal to zero since the direct travel time from the origin to itself is 0. In the square table, any missing values must be filled by zero as well.)


```python
# my solution (it is actually slower than the school solution)
#n = 293
#travel_matrix = pd.DataFrame(0, index = range(n), columns = range(n))
#for index, row in gnd_travel_ids.iterrows():
#    travel_matrix[row['Ending_City']][row['Starting_City']] = row['Average_Travel_Time']
    
# school solution
travel_matrix = pd.pivot_table(gnd_travel_ids, index = "Starting_City", columns = "Ending_City", values = "Average_Travel_Time")
travel_matrix.fillna(0, inplace = True)
```


```python
## TEST CODE PART 1, EXERCISE 3 - travel_matrix_1

assert type(travel_matrix) is pd.DataFrame, "`type(travel_matrix) == {}` instead of `pd.DataFrame`.".format(type(travel_matrix))

# Test 1 - All Diagonals in the Matrix are 0
print('Test 1: Are all Diagonals 0?')
travel_mat = np.array(travel_matrix)
assert np.all(np.diag(travel_mat) == 0) == True
print('Yes, all Diagonals are 0! \n')
# Test 2 - Dimensions
print('Test 2: Are the dimensions correct?')
row, col = travel_matrix.shape
assert row == col == 293
print('Yes, dimensions are correct! \n')


# Test 3 - Select Values in Matrix are the same
tol = 1
print('Test 3: Checking if Select Values are the Same...')
assert abs(travel_matrix[1][0] - 80.38) < tol
assert abs(travel_matrix[0][1] - 82.38) < tol
assert abs(travel_matrix[30][50] - 24.47) < tol
assert abs(travel_matrix[50][30] - 29.47) < tol
assert abs(travel_matrix[260][118] - 96.85) < tol
assert abs(travel_matrix[118][260] - 95.85) < tol
assert abs(travel_matrix[3][292] - 36.43) < tol
assert abs(travel_matrix[292][3] - 32.43) < tol
assert abs(travel_matrix[279][256] - 15.82) < tol
assert abs(travel_matrix[256][279] - 18.82) < tol
print('Great! Select Values are the Same!')

print('\n(Passed!)')
```

    Test 1: Are all Diagonals 0?
    Yes, all Diagonals are 0! 
    
    Test 2: Are the dimensions correct?
    Yes, dimensions are correct! 
    
    Test 3: Checking if Select Values are the Same...
    Great! Select Values are the Same!
    
    (Passed!)
    

**Exercise 4** (3 points) Now write some code to compute a **2-D Numpy array** named `round_trip`, which contains the amount of time taken to complete a round trip between all possible origins and destinations as appear in the table `gnd_travel_ids`. Your table should be a square matrix. Any entry `(i, j)` in the matrix must contain the total time to go from `i` to `j` and back to `i`.


```python
travel_matrix = np.array(travel_matrix)
round_trip = travel_matrix + travel_matrix.T
```


```python
## TEST CODE PART 2 OF 2, EXERCISE 4 - travel_matrix_2
import random
n_test = 1000
for _ in range(n_test):
    origin = random.randint(0, 292)
    dest = random.randint(0, 292)
    round_travel_time = round_trip[origin, dest]
    o1 = gnd_travel_ids["Starting_City"] == origin
    d1 = gnd_travel_ids["Ending_City"] == dest
    d2 = gnd_travel_ids["Starting_City"] == dest
    o2 = gnd_travel_ids["Ending_City"] == origin
    if origin != dest:
        time = gnd_travel_ids[o1 & d1]["Average_Travel_Time"].values[0] + gnd_travel_ids[o2 & d2]["Average_Travel_Time"].values[0]
        assert time == round_travel_time

print('\n(Passed!)')
```

    
    (Passed!)
    

**Fin!** You've reached the end of this problem. Don't forget to restart the
kernel and run the entire notebook from top-to-bottom to make sure you did
everything correctly. If that is working, try submitting this problem. (Recall
that you *must* submit and pass the autograder to get credit for your work!)
