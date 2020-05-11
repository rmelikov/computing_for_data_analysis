# Problem 9: SQL Operations

This problem will test your ability to manipulate two simple SQL tables. You may find a problem easier to complete using Pandas, or you may find a problem easier to complete in SQL. We will provide you will a SQLite database containing two tables, and two Pandas Dataframes that are identical to the SQLite tables. 

Recall that on the current version of the Vocareum platform, the `sqlite3` module only works with their Python 3.5 build, rather than the 3.6 build we usually use. The cell below imports the necessary modules and prints their versions. As such, **if** you are prototyping on your local machine with a version of Python greater than 3.5 or using versions of the pandas and SQLite modules that differ from what is on Vocareum, you may need to make adjustments to pass the autograder. **Doing so is your responsibility so budget your time accordingly.**


```python
import sys
import pandas as pd
import sqlite3 as db
from IPython.display import display

def get_data_path(filebase):
    return f"movies/{filebase}"

print(f"* Python version: {sys.version}")
print(f"* pandas version: {pd.__version__}")
print(f"* sqlite3 version: {db.version}")
```

    * Python version: 3.7.4 (tags/v3.7.4:e09359112e, Jul  8 2019, 20:34:20) [MSC v.1916 64 bit (AMD64)]
    * pandas version: 0.25.3
    * sqlite3 version: 2.6.0
    

## The Movies and Cast Dataset

The data consists of two tables. The first is a table of movies along with (random) audience scores from 1-100. The second is a table of cast members for those movies. There are some interesting cast members in here that you might stumble upon!

Let's read in the database file and show the table descriptions.


```python
disk_engine = db.connect(get_data_path('movieDB.db'))
c = disk_engine.cursor()

c.execute('SELECT type, name, sql FROM sqlite_master')
results = c.fetchall()
for table in results:
    print(table)
```

    ('table', 'movies', 'CREATE TABLE movies (id integer, name text, score integer)')
    ('table', 'cast', 'CREATE TABLE cast (movie_id integer, cast_id integer, cast_name text)')
    


```python
movies = pd.read_table(get_data_path('movie-name-score.txt'), sep=',', header=None, names=['id', 'name', 'score'])
cast = pd.read_table(get_data_path('movie-cast.txt'), sep=',', header=None, names=['movie_id', 'cast_id', 'cast_name'])

print('Movies Dataframe:')
print('-------------------')
display(movies.head())
print('\n\n')
print('Cast Dataframe:')
print('-------------------')
display(cast.head())
```

    Movies Dataframe:
    -------------------
    


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
      <th>id</th>
      <th>name</th>
      <th>score</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>9</td>
      <td>Star Wars: Episode III - Revenge of the Sith 3D</td>
      <td>61</td>
    </tr>
    <tr>
      <th>1</th>
      <td>24214</td>
      <td>The Chronicles of Narnia: The Lion, The Witch ...</td>
      <td>46</td>
    </tr>
    <tr>
      <th>2</th>
      <td>1789</td>
      <td>War of the Worlds</td>
      <td>94</td>
    </tr>
    <tr>
      <th>3</th>
      <td>10009</td>
      <td>Star Wars: Episode II - Attack of the Clones 3D</td>
      <td>28</td>
    </tr>
    <tr>
      <th>4</th>
      <td>771238285</td>
      <td>Warm Bodies</td>
      <td>3</td>
    </tr>
  </tbody>
</table>
</div>


    
    
    
    Cast Dataframe:
    -------------------
    


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
      <th>movie_id</th>
      <th>cast_id</th>
      <th>cast_name</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>9</td>
      <td>162652153</td>
      <td>Hayden Christensen</td>
    </tr>
    <tr>
      <th>1</th>
      <td>9</td>
      <td>162652152</td>
      <td>Ewan McGregor</td>
    </tr>
    <tr>
      <th>2</th>
      <td>9</td>
      <td>418638213</td>
      <td>Kenny Baker</td>
    </tr>
    <tr>
      <th>3</th>
      <td>9</td>
      <td>548155708</td>
      <td>Graeme Blundell</td>
    </tr>
    <tr>
      <th>4</th>
      <td>9</td>
      <td>358317901</td>
      <td>Jeremy Bulloch</td>
    </tr>
  </tbody>
</table>
</div>


In terms of Database structures, the **`cast`** table's **`movie_id`** column is a foreign key to the **`movie`** table's **`id`** column. 

This means you can perform any SQL joins or Pandas merges between the two tables on this column. 

One final code cell to get you started - implement the all-too-familiar `canonicalize_tibble` and `tibbles_are_equivalent` functions. 


```python
def canonicalize_tibble(X):
    var_names = sorted(X.columns)
    Y = X[var_names].copy()
    Y.sort_values(by=var_names, inplace=True)
    Y.reset_index(drop=True, inplace=True)
    return Y

def tibbles_are_equivalent (A, B):
    A_canonical = canonicalize_tibble(A)
    B_canonical = canonicalize_tibble(B)
    equal = (A_canonical == B_canonical)
    return equal.all().all()
```

Let's start with two warm-up exercises. 

**Exercise 0** (2 points): Create a dataframe, ***`cast_size`***, that contains the number of distinct cast members per movie. Your table will have two columns, *`movie_name`*, the name of each film, and *`cast_count`*, the number of unique cast members for the film. 

Order the result by *`cast_count`* from highest to lowest.


```python
# interesting sql problem

query = '''
    select
        m.name as movie_name, 
        count(cast_id) as cast_count
    from movies m, cast c
    where m.id = c.movie_id
    group by m.id
    order by 2 desc
'''

cast_size = pd.read_sql_query(query, disk_engine)
```


```python
# Test cell : `test_cast_size`

print("Reading instructor's solution...")

cast_size_solution = pd.read_csv(get_data_path('cast_size_solution.csv'))

print("Checking...")

assert set(cast_size.columns) == {'movie_name', 'cast_count'}
assert tibbles_are_equivalent(cast_size, cast_size_solution), "Your Dataframe is incorrect"
assert all(cast_size['cast_count'] == cast_size_solution['cast_count'])


print("\n(Passed!.)")

del cast_size_solution
```

    Reading instructor's solution...
    Checking...
    
    (Passed!.)
    

**Exercise 1** (2 point): Create a dataframe, **`cast_score`**, that contains the average movie score for each cast member. Your table will have two columns, *`cast_name`*, the name of each cast member, and *`avg_score`*, the average movie review score for each movie that the cast member appears in. 

Order this result by `avg_score` from highest to lowest, and round your result for `avg_score` to two (2) decimal places. 

Break any ties in your sorting by cast name in alphabetical order from A-Z. 


```python
# interesting sql problem

query = '''
    select
        c.cast_name,
        round(avg(m.score), 2) as avg_score
    from movies m, cast c
    where m.id = c.movie_id
    group by c.cast_id
    order by 2 desc, 1
'''

cast_score = pd.read_sql_query(query, disk_engine)
```


```python
# Test cell : `test_cast_score`
print("Reading instructor's solution...")

cast_score_solution = pd.read_csv(get_data_path('cast_score_solution.csv'))

print("Checking...")

assert set(cast_score.columns) == {'cast_name', 'avg_score'}
assert tibbles_are_equivalent(cast_score, cast_score_solution), "Your Dataframe is incorrect"
assert all(cast_score['avg_score'] == cast_score_solution['avg_score'])


print("\n(Passed!)")

del cast_score_solution


```

    Reading instructor's solution...
    Checking...
    
    (Passed!)
    

**Exercise 2** (3 points): You will now create a dataframe, **`one_hit_wonders`**, that contains actors and actresses that appear in **exactly** one movie, with a movie score == 100. Your result will have three columns, *`cast_name`,* the name of each cast member that meets the criteria, *`movie_name`*, the name of the movie that cast member appears in, and *`movie_score`*, which for the purposes of this Exercise is always == 100. 

Order your result by `cast_name` in alphabetical order from A-Z.


```python
# interesting sql problem

query = '''
    select
        c.cast_name,
        m.name as movie_name,
        m.score as movie_score
    from movies m, cast c
    where m.id = c.movie_id
    group by c.cast_id
    having count(m.id) = 1 and m.score = 100
    order by c.cast_name
'''

one_hit_wonders = pd.read_sql_query(query, disk_engine)
```


```python
# Test cell : `one_hit_wonders_score`

print("Reading instructor's solution...")

one_hit_wonders_solution = pd.read_csv(get_data_path('one_hit_wonders_solution.csv'))

print("Checking...")

assert set(one_hit_wonders.columns) == {'cast_name','movie_name', 'movie_score'}
assert tibbles_are_equivalent(one_hit_wonders, one_hit_wonders_solution)
assert all(one_hit_wonders['movie_score'] == one_hit_wonders_solution['movie_score'])

print("\n(Passed!)")

del one_hit_wonders_solution
```

    Reading instructor's solution...
    Checking...
    
    (Passed!)
    

**Exercise 3** (3 points): For this problem, you will find cast members that work well together. We define this as two cast members being in **>= 3** movies together, with the **average movie score being >= 50**. 

You will create a dataframe called **`good_teamwork`** that contains four columns:
- *`cast_member_1`* and *`cast_member_2`*, the names of each pair of cast members that appear in the same movie;
- *`num_movies`*, the number of movies that each pair of cast members appears in; and
- *`avg_score`*, the average review score for each of those movies containing the two cast members. 

Order the results by `cast_member_1` alphabetically from A-Z, and break any ties by sorting by `cast_member_2` alphabetically from A-Z. Round the result for `avg_score` to two (2) decimal places.

One more wrinkle: your solution will likely create several duplicate pairs of cast members: rows such as:

cast_member_1     |cast_member_2  |num_movies  |avg_score
------------------|---------------|------------|---------
 Anthony Daniels  |Frank Oz       |5           |50.60
 Frank Oz         |Anthony Daniels|5           |50.60
 
Remove all duplicate pairs, keeping all cases where `cast_member_1`'s name comes before `cast_member_2`'s name in the alphabet. In the example above, you will keep **only** the first row in your final solution. Make sure to also remove matches where `cast_member_1` == `cast_member_2`.


```python
# interesting sql problem

query = '''
    with main as (
        select *
        from movies m, cast c
        where m.id = c.movie_id
    )
    select    
        m1.cast_name as cast_member_1, 
        m2.cast_name as cast_member_2,
        count(*) as num_movies,
        round(avg(m1.score), 2) as avg_score
    from main m1, main m2
    where m1.movie_id = m2.movie_id and m1.cast_name < m2.cast_name
    group by m1.cast_id, m2.cast_id
    having num_movies >= 3 and avg_score >= 50
    order by 1, 2
'''

good_teamwork = pd.read_sql_query(query, disk_engine)
```


```python
# Test cell : `good_teamwork_score`
print("Reading instructor's solution...")

good_teamwork_solution = pd.read_csv(get_data_path('good_teamwork_solution.csv'))
print(good_teamwork_solution)

print("Checking...")

assert set(good_teamwork.columns) == {'cast_member_1','cast_member_2', 'num_movies', 'avg_score'}
assert tibbles_are_equivalent(good_teamwork, good_teamwork_solution)
assert all(good_teamwork['num_movies'] == good_teamwork_solution['num_movies'])
assert all(good_teamwork['avg_score'] == good_teamwork_solution['avg_score'])

print("\n(Passed!)")

del good_teamwork_solution
```

    Reading instructor's solution...
             cast_member_1      cast_member_2  num_movies  avg_score
    0           Ahmed Best    Anthony Daniels           3      54.67
    1           Ahmed Best      Ewan McGregor           3      54.67
    2           Ahmed Best           Frank Oz           3      54.67
    3           Ahmed Best      Ian McDiarmid           3      54.67
    4           Ahmed Best        Kenny Baker           3      54.67
    ..                 ...                ...         ...        ...
    63     Natalie Portman  Samuel L. Jackson           3      54.67
    64     Natalie Portman       Silas Carson           3      54.67
    65  Oliver Ford Davies  Samuel L. Jackson           3      54.67
    66  Oliver Ford Davies       Silas Carson           3      54.67
    67   Samuel L. Jackson       Silas Carson           3      54.67
    
    [68 rows x 4 columns]
    Checking...
    
    (Passed!)
    


```python
c.close()
disk_engine.close()
```

**Fin!** Remember to test your solutions by running them as the autograder will: restart the kernel and run all cells from "top-to-bottom." Also remember to submit to the autograder; otherwise, you will **not** get credit for your hard work!
