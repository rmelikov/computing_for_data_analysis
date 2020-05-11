# Problem 11: Who's the fastest of 'em all!?

This problem will test your mastery of basic Python data structures. It consists of five (5) exercises, numbered 0 through 4, worth a total of ten (10) points.

For this problem, you will be dealing with Formula1 race results of a particular year. The Formula1 season is divided into several races, or [Grands Prix](https://en.wiktionary.org/wiki/Grands_Prix#English), each held in a unique country. 
For each Grand Prix, the Top 10 drivers receive points according to their finish positions for that race. Accumulation of points from the race results determines the end-of-season Rank of the Driver.

For example - when Australia hosted the Grand Prix, the results were as follows from 1st to 10th position: 

| Grand Prix | 1st | 2nd | 3rd | 4th | 5th | 6th | 7th | 8th | 9th | 10th |
|------------|------------------|----------------|----------------|------------------|-----------------|----------------|-----------------|-----------------|-------------------|--------------|
| Australia | Sebastian Vettel | Lewis Hamilton | Kimi Räikkönen | Daniel Ricciardo | Fernando Alonso | Max Verstappen | Nico Hulkenberg | Valtteri Bottas | Stoffel Vandoorne | Carlos Sainz |
| Points | 25 | 18 | 15 | 12 | 10 | 8 | 6 | 4 | 2 | 1 |

*Alternatively, if the driver is not in Top 10, he will receive 0 points*



### Loading the Data
First, run the following code cell to set up the problem. The provided output shows the data you will be working with. The data is a dictionary, **`race_results`**, that maps the country that hosted that particular Grand Prix to a list of the top 10 drivers in order of their finish in the race. That is, the name of the top driver is first in the list, and driver with rank 10 is last in the list.

> This cell uses Pandas (Topic 7) to simplify reading in the data. However, you do not need Pandas to solve the problem, as this exam covers only Topics 1-5.


```python
# Some modules you'll need in this part
import pandas as pd
from pprint import pprint 

df = pd.read_csv('Formula1.csv')
race_results = df.to_dict('list')
pprint(race_results)
```

    {'Australia': ['Sebastian Vettel',
                   'Lewis Hamilton',
                   'Kimi Raikkonen',
                   'Daniel Ricciardo',
                   'Fernando Alonso',
                   'Max Verstappen',
                   'Nico Hulkenberg',
                   'Valtteri Bottas',
                   'Stoffel Vandoorne',
                   'Carlos Sainz'],
     'Austria': ['Max Verstappen',
                 'Kimi Raikkonen',
                 'Sebastian Vettel',
                 'Romain Grosjean',
                 'Kevin Magnussen',
                 'Esteban Ocon',
                 'Sergio Perez',
                 'Fernando Alonso',
                 'Charles Leclerc',
                 'Marcus Ericsson'],
     'Azerbaijan': ['Lewis Hamilton',
                    'Kimi Raikkonen',
                    'Sergio Perez',
                    'Sebastian Vettel',
                    'Carlos Sainz',
                    'Charles Leclerc',
                    'Fernando Alonso',
                    'Lance Stroll',
                    'Stoffel Vandoorne',
                    'Brendon Hartley'],
     'Bahrain': ['Sebastian Vettel',
                 'Valtteri Bottas',
                 'Lewis Hamilton',
                 'Pierre Gasly',
                 'Kevin Magnussen',
                 'Nico Hulkenberg',
                 'Fernando Alonso',
                 'Stoffel Vandoorne',
                 'Marcus Ericsson',
                 'Esteban Ocon'],
     'Belgium': ['Sebastian Vettel',
                 'Lewis Hamilton',
                 'Max Verstappen',
                 'Valtteri Bottas',
                 'Sergio Perez',
                 'Esteban Ocon',
                 'Romain Grosjean',
                 'Kevin Magnussen',
                 'Pierre Gasly',
                 'Marcus Ericsson'],
     'Canada': ['Sebastian Vettel',
                'Valtteri Bottas',
                'Max Verstappen',
                'Daniel Ricciardo',
                'Lewis Hamilton',
                'Kimi Raikkonen',
                'Nico Hulkenberg',
                'Carlos Sainz',
                'Esteban Ocon',
                'Charles Leclerc'],
     'China': ['Daniel Ricciardo',
               'Valtteri Bottas',
               'Kimi Raikkonen',
               'Lewis Hamilton',
               'Max Verstappen',
               'Nico Hulkenberg',
               'Fernando Alonso',
               'Sebastian Vettel',
               'Carlos Sainz',
               'Kevin Magnussen'],
     'France': ['Lewis Hamilton',
                'Max Verstappen',
                'Kimi Raikkonen',
                'Daniel Ricciardo',
                'Sebastian Vettel',
                'Kevin Magnussen',
                'Valtteri Bottas',
                'Carlos Sainz',
                'Nico Hulkenberg',
                'Charles Leclerc'],
     'Germany': ['Lewis Hamilton',
                 'Valtteri Bottas',
                 'Kimi Raikkonen',
                 'Max Verstappen',
                 'Nico Hulkenberg',
                 'Romain Grosjean',
                 'Sergio Perez',
                 'Esteban Ocon',
                 'Marcus Ericsson',
                 'Brendon Hartley'],
     'Great Britain': ['Sebastian Vettel',
                       'Lewis Hamilton',
                       'Kimi Raikkonen',
                       'Valtteri Bottas',
                       'Daniel Ricciardo',
                       'Nico Hulkenberg',
                       'Esteban Ocon',
                       'Fernando Alonso',
                       'Kevin Magnussen',
                       'Sergio Perez'],
     'Hungary': ['Lewis Hamilton',
                 'Sebastian Vettel',
                 'Kimi Raikkonen',
                 'Daniel Ricciardo',
                 'Valtteri Bottas',
                 'Pierre Gasly',
                 'Kevin Magnussen',
                 'Fernando Alonso',
                 'Carlos Sainz',
                 'Romain Grosjean'],
     'Italy': ['Lewis Hamilton',
               'Kimi Raikkonen',
               'Valtteri Bottas',
               'Sebastian Vettel',
               'Max Verstappen',
               'Esteban Ocon',
               'Sergio Perez',
               'Carlos Sainz',
               'Lance Stroll',
               'Sergey Sirotkin'],
     'Monaco': ['Daniel Ricciardo',
                'Sebastian Vettel',
                'Lewis Hamilton',
                'Kimi Raikkonen',
                'Valtteri Bottas',
                'Esteban Ocon',
                'Pierre Gasly',
                'Nico Hulkenberg',
                'Max Verstappen',
                'Carlos Sainz'],
     'Spain': ['Lewis Hamilton',
               'Valtteri Bottas',
               'Max Verstappen',
               'Sebastian Vettel',
               'Daniel Ricciardo',
               'Kevin Magnussen',
               'Carlos Sainz',
               'Fernando Alonso',
               'Sergio Perez',
               'Charles Leclerc']}
    

#### You're all set to start!



**Exercise 0** (2 points). Write a function, **`find_country(driver_name, rank)`** that, given the name of a driver and a rank, returns a list of all the countries where the driver received that specific rank.  
   
For example:

```python
       find_country ("Lewis Hamilton", 1) == ["Azerbaijan", "Spain", "France", "Germany", "Hungary", "Italy"]
       find_country ("Sebastian Vettel", 6) == []
```


```python
def find_country(driver_name, rank):
    return [
        country
        for country, driver_list in race_results.items()
        if driver_list[rank - 1] == driver_name
    ]
```


```python
# `find_country_test`: Test cell
assert type(find_country("Lewis Hamilton" ,1)) is list,  "Did not create a list."
assert set(find_country("Lewis Hamilton" ,1)) == {'Azerbaijan', 'Spain', 'France', 'Germany', 'Hungary', 'Italy'}
assert set(find_country("Kimi Raikkonen" ,2)) == {'Azerbaijan', 'Austria', 'Italy'}
assert set(find_country("Valtteri Bottas" ,8)) == {'Australia'}
assert set(find_country("Carlos Sainz" ,10)) == {'Australia', 'Monaco'}
assert set(find_country("Sebastian Vettel" ,2)) == {'Monaco', 'Hungary'}
assert set(find_country("Sebastian Vettel" ,6)) == set()
print("\n(Passed!)")
```

    
    (Passed!)
    

**Exercise 1** (3 points). Implement a `function`, **`three_top_driver(points)`**, that returns a dictionary mapping each of the top 3 drivers (in end-of-season standings) to their full-season point totals, as explained below.

First, the input, `points`, tells you how many points to award to a given finishing place in a race. That is, the first place driver should receive `points[0]` points, the second place driver should receive `points[1]` points, ..., and the tenth place driver should receive `points[9]` points.

The return value is a dictionary with the following properties:

- It should have three entries, one for each of the top 3 drivers.
- Each key is a driver's name.
- Each value is the corresponding driver's full-season point total.

Your function needs to calculate the total points for each driver across all race results, find the three drivers with the highest overall points, and return these as the dictionary described above.


```python
def three_top_driver(points):
    
    def my_solution():
    
        races_driver_points = [
            [
                (driver_name, points[rank])
                for rank, driver_name in enumerate(driver_list)
            ]
            for country, driver_list in race_results.items()
        ]
        
        from collections import defaultdict
        
        driver_points = defaultdict(list)
        
        for race in races_driver_points:
            for driver, race_points in race:
                driver_points[driver].append(race_points)
        
        standings = {
            driver_name : sum(race_points)
            for driver_name, race_points in driver_points.items()
        }
        
        return dict(
            sorted(
                standings.items(),
                key = lambda v : v[1],
                reverse = True
            )[:3]
        )
    
    def school_solution():
        
        standing={}
        
        for country, driver_list in race_results.items():
            
            for rank, driver in enumerate(driver_list):
                
                if driver in standing:
                    
                    standing[driver] += points[rank]
                
                else:
                    
                    standing[driver] = points[rank]
        
        return {k:standing[k] for k in sorted(standing, key=standing.get, reverse=True)[:3]}
    
    return school_solution()
```


```python
# `three_top_driver`: Test cell
points1=[25 , 18 , 15 , 12 , 10 , 8 , 6 , 4 , 2 , 1]
standing1 = three_top_driver(points1)
assert type(standing1) is dict,  "Did not create a dictionary."
assert len(standing1) == 3, "Dictionary has the wrong number of entries."
assert {'Lewis Hamilton', 'Sebastian Vettel', 'Kimi Raikkonen'} == set(standing1.keys()), "Dictionary has the wrong keys."
assert standing1['Lewis Hamilton'] == 256, 'Wrong points for: Lewis Hamilton'
assert standing1['Sebastian Vettel'] == 226, 'Wrong points for: Sebastian Vettel'
assert standing1['Kimi Raikkonen'] == 164, 'Wrong points for: Kimi Raikkonen'
       
points2=[10 , 9 , 8 , 7 , 6 , 5 , 4 , 3 , 2 , 1]
standing2 = three_top_driver(points2)
assert type(standing2) is dict,  "Did not create a dictionary."
assert len(standing2) == 3, "Dictionary has the wrong number of entries."
assert {'Lewis Hamilton', 'Sebastian Vettel', 'Kimi Raikkonen'} == set(standing2.keys()), "Dictionary has the wrong keys."
assert standing2['Lewis Hamilton'] == 116, 'Wrong points for: Lewis Hamilton'
assert standing2['Sebastian Vettel'] == 106, 'Wrong points for: Sebastian Vettel'
assert standing2['Kimi Raikkonen'] == 87, 'Wrong points for: Kimi Raikkonen'

points3=[1, 2, 4, 6, 8, 10, 12, 15, 18, 25]
standing3 = three_top_driver(points3)
assert type(standing3) is dict,  "Did not create a dictionary."
assert len(standing3) == 3, "Dictionary has the wrong number of entries."
assert {'Carlos Sainz', 'Esteban Ocon', 'Kevin Magnussen'} == set(standing3.keys()), "Dictionary has the wrong keys."
assert standing3['Carlos Sainz'] == 151, 'Wrong points for: Carlos Sainz'
assert standing3['Esteban Ocon'] == 110, 'Wrong points for: Esteban Ocon'
assert standing3['Kevin Magnussen'] == 106, 'Wrong points for: Kevin Magnussen'

print("\n(Passed!)")
```

    
    (Passed!)
    

**Exercise 2** (2 points). A driver finishes a race on the podium if he finishes in **first**, **second** or **third** place. Write a function, **`podium_count(driver_name)`**, that given the name of the driver as input returns the number of times that the driver has finished a race on the podium that season. 

For example:

```python
    podium_count("Sebastian Vettel") == 8
    podium_count("Esteban Ocon") == 0
```


```python
def podium_count(driver_name):

    def my_solution():
        
        from collections import Counter
        
        podiums = [driver_list[:3] for country, driver_list in race_results.items()]
        
        flat_list = [driver for race in podiums for driver in race]
        
        return Counter(flat_list)[driver_name]

    
    def school_solution():
        
        count = 0
        
        for country, driver_list in race_results.items():
        
            if driver_name in driver_list[:3]:
                
                count += 1
        
        return count
    
    return my_solution()
```


```python
# `podium_count`: Test cell
assert podium_count("Lewis Hamilton") == 11 , 'Wrong count'
assert podium_count("Sebastian Vettel") == 8, 'Wrong count'
assert podium_count("Kimi Raikkonen") == 9, 'Wrong count'
assert podium_count("Valtteri Bottas") == 6, 'Wrong count'
assert podium_count("Esteban Ocon") == 0, 'Wrong count'
assert podium_count("Kevin Magnussen") == 0, 'Wrong count'
assert podium_count("Max Verstappen") == 5, 'Wrong count'
print("\n(Passed!)")
```

    
    (Passed!)
    

**Exercise 3** (1 points). Write a function, **`drivers_on_podium()`**, that returns a list of **all** of the drivers that have finished a race on the podium that season.


```python
def drivers_on_podium():
    
    def my_solution():
        podiums = [driver_list[:3] for country, driver_list in race_results.items()]
        flat_list = [driver for race in podiums for driver in race]
        return(list(set(flat_list)))
    
    def school_solution():
        return list({d for country, driver_list in race_results.items() for d in driver_list[:3]})
    
    return school_solution()
```


```python
# `drivers_on_podium`: Test cell
assert type(drivers_on_podium()) is list, "Did not create a list."
assert len(drivers_on_podium()) == 7 , "List has the wrong number of entries."
assert set(drivers_on_podium()) == set(['Sebastian Vettel', 'Max Verstappen', 'Lewis Hamilton', 'Kimi Raikkonen', 'Sergio Perez', 'Daniel Ricciardo', 'Valtteri Bottas'])
print("\n(Passed!)")
```

    
    (Passed!)
    

**Exercise 4** (2 points). Implement the following function, **`podium_proba()`**. It should return a dictionary that contains the top three drivers with the highest probability to be on the podium in the next race. 

The dictionary should be in the following form:

```python
    {'driver1': probabilty1, 'driver2': probabilty2, 'driver3': probabilty3}
```

For each driver, this probabilty should, for each driver, be estimated as

$$\mathrm{probability\ of\ a\ podium\ finish} \equiv \frac{\mathrm{number\, of\, podium\, finishes}}{\mathrm{number\, of\, races}}.$$    


```python
def podium_proba():
    
    def my_solution():
        
        from collections import Counter
        
        number_races = len(race_results)
        
        podiums = [driver_list[:3] for country, driver_list in race_results.items()]
                
        flat_list = [driver for race in podiums for driver in race]
                
        podium_count = dict(Counter(flat_list))
        
        probabilities = {k : v/number_races for k, v in podium_count.items()}
        
        return dict(
            sorted(
                probabilities.items(),
                key = lambda v : v[1],
                reverse = True
                )[:3]
        )
    
    def school_solution():
        
        probablity = {
            driver : podium_count(driver) / len(race_results) 
            for driver in drivers_on_podium()
        }
        
        return {
            driver : probablity[driver] 
            for driver in sorted(
                probablity,
                key = probablity.get,
                reverse = True
            )[:3]
        }
    
    return school_solution()
```


```python
# `podium_proba`: Test cell
def check_podium_proba(s, v, v_true, tol=1e-2):
    assert type(v) is float, "Your function did not return a `float`"
    delta_v = v - v_true
    msg = "[Podium proba for {} is {} : You computed {}, which differs by {}.".format(s, v_true, v, delta_v)
    print(msg)
    assert abs(delta_v) <= tol, "Difference exceeds expected tolerance."

podium_porba1 = podium_proba()
assert type(podium_porba1) is dict, "Did not create a dictionary."
assert len(podium_porba1) == 3 , "Dictionary has the wrong number of entries."
assert {'Lewis Hamilton', 'Sebastian Vettel', 'Kimi Raikkonen'} == set(podium_porba1.keys()), "Dictionary has the wrong keys."
check_podium_proba('Lewis Hamilton', podium_porba1['Lewis Hamilton'],  0.785, tol=1e-3)
check_podium_proba('Sebastian Vettel', podium_porba1['Sebastian Vettel'],  0.571, tol=1e-3)
check_podium_proba('Kimi Raikkonen', podium_porba1['Kimi Raikkonen'],  0.642, tol=1e-3)

print("\n(Passed!)")
```

    [Podium proba for Lewis Hamilton is 0.785 : You computed 0.7857142857142857, which differs by 0.0007142857142856673.
    [Podium proba for Sebastian Vettel is 0.571 : You computed 0.5714285714285714, which differs by 0.0004285714285714448.
    [Podium proba for Kimi Raikkonen is 0.642 : You computed 0.6428571428571429, which differs by 0.0008571428571428896.
    
    (Passed!)
    

**Fin!** Remember to test your solutions by running them as the autograder will: restart the kernel and run all cells from "top-to-bottom." Also remember to submit to the autograder; otherwise, you will **not** get credit for your hard work!
