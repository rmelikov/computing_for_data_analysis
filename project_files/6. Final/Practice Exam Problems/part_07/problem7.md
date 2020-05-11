## Problem 7

Millions of searches happen on modern search engines like Google. Advertisers want to know about search interests in order to target consumers effectively. In this notebook, we will look at "search interest scores" for the 2016 Olympics obtained from [Google Trends](https://trends.google.com/trends/).

This problem is divided into four (4) exercises, numbered 0-3. They are worth a total of ten (10) points.

> **Note 0.** By way of background, a search interest score is computed by region and normalized by population size, in order to account for differences in populations between different regions. You can read more about search interest here. https://medium.com/google-news-lab/what-is-google-trends-data-and-what-does-it-mean-b48f07342ee8
>
> **Note 1.** We have pre-loaded the dataset you'll need on Vocareum. For this problem, a copy of this data is also available at the following URL. However, if you choose to work on this problem outside of Vocareum, you may need to adapt the `fn(f)` function, below, which is set up for use within Vocareum. (You should be able to figure out how to change it for your local environment.) https://cse6040.gatech.edu/datasets/olympics.zip


```python
# Some modules and functions we'll need
import sys
import pandas as pd
import sqlite3
from IPython.display import display

print("=== Python version ===\n{}\n".format(sys.version))
print("\n=== SQLite version ===\n{}\n".format(sqlite3.version))
print("\n=== Pandas version ===\n{}\n".format(pd.__version__))
```

    === Python version ===
    3.7.4 (tags/v3.7.4:e09359112e, Jul  8 2019, 20:34:20) [MSC v.1916 64 bit (AMD64)]
    
    
    === SQLite version ===
    2.6.0
    
    
    === Pandas version ===
    0.25.3
    
    


```python
def fn(fn_base, dirname='./'):
    return "{}{}".format(dirname, fn_base)

# Demo:
fn('olympics_data.dat')
```




    './olympics_data.dat'




```python
def canonicalize_tibble(X):
    var_names = sorted(X.columns)
    Y = X[var_names].copy()
    Y.sort_values(by=var_names, inplace=True)
    Y.set_index([list(range(0, len(Y)))], inplace=True)
    return Y

def tibbles_are_equivalent(A, B):
    A_canonical = canonicalize_tibble(A)
    B_canonical = canonicalize_tibble(B)
    cmp = A_canonical.eq(B_canonical)
    return cmp.all().all()
```

## The data

We will be working with two sources of data.

The first is the [search interest data taken from Google Trends](https://raw.githubusercontent.com/googletrends/data/master/20160819_OlympicSportsByCountries.csv).

The second is [world population data taken from the U.S. Census Bureau](https://www.census.gov/population/international/data/idb/).

For your convenience, these data are stored in two tables in a SQLite database stored in a file named `olympics/sports.db`. We will need to read the data into dataframes before proceeding.

**Exercise 0** (2 points). The SQLite database has two tables in it, one named `search_interest` and the other named `countries`. Implement the function, **`read_data(conn)`** below, to read these tables into a pair of Pandas dataframes.

In particular, assume that **`conn`** is an open SQLite database connection object. Your function should return a pair of dataframes, `(search_interest, countries)`, corresponding to these tables. (See the `# Demo code` below.)


```python
def read_data(conn):
    
    #search_interest_query = '''
    #    select *
    #    from search_interest
    #'''
    #
    #countries_query = '''
    #    select *
    #    from countries
    #'''
    #
    #search_interest = pd.read_sql_query(search_interest_query, conn)
    #countries = pd.read_sql_query(countries_query, conn)
    #
    #return search_interest, countries
    
    df1 = pd.read_sql('select * from search_interest', conn)
    df2 = pd.read_sql('select * from countries', conn)
    return df1, df2

# Demo code:
conn = sqlite3.connect(fn('sports.db'))
search_interest, countries = read_data(conn)
conn.close()

print("=== search_interest ===")
display(search_interest.head())

print("=== countries ===")
display(countries.head())
```

    === search_interest ===
    


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
      <th>Country</th>
      <th>Search_Interest</th>
      <th>Sport</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>0</td>
      <td>Iran</td>
      <td>1</td>
      <td>Archery</td>
    </tr>
    <tr>
      <th>1</th>
      <td>1</td>
      <td>South Korea</td>
      <td>2</td>
      <td>Archery</td>
    </tr>
    <tr>
      <th>2</th>
      <td>2</td>
      <td>Mexico</td>
      <td>1</td>
      <td>Archery</td>
    </tr>
    <tr>
      <th>3</th>
      <td>3</td>
      <td>Netherlands</td>
      <td>1</td>
      <td>Archery</td>
    </tr>
    <tr>
      <th>4</th>
      <td>4</td>
      <td>Aruba</td>
      <td>16</td>
      <td>Artistic gymnastics</td>
    </tr>
  </tbody>
</table>
</div>


    === countries ===
    


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
      <th>Country</th>
      <th>Year</th>
      <th>Population</th>
      <th>Area_sq_km</th>
      <th>Density</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>0</td>
      <td>Reunion</td>
      <td>2016</td>
      <td>850996</td>
      <td>2511</td>
      <td>340.0</td>
    </tr>
    <tr>
      <th>1</th>
      <td>1</td>
      <td>Martinique</td>
      <td>2016</td>
      <td>385551</td>
      <td>128</td>
      <td>340.0</td>
    </tr>
    <tr>
      <th>2</th>
      <td>2</td>
      <td>Guadeloupe</td>
      <td>2016</td>
      <td>402119</td>
      <td>1628</td>
      <td>250.0</td>
    </tr>
    <tr>
      <th>3</th>
      <td>3</td>
      <td>Myanmar</td>
      <td>2016</td>
      <td>54616716</td>
      <td>653508</td>
      <td>83.6</td>
    </tr>
    <tr>
      <th>4</th>
      <td>4</td>
      <td>CzechRepublic</td>
      <td>2016</td>
      <td>10660932</td>
      <td>77247</td>
      <td>138.0</td>
    </tr>
  </tbody>
</table>
</div>



```python
# Test cell: `read_data_test`

df1 = pd.read_csv(fn("OlympicSportsByCountries_2016.csv"))
df2 = pd.read_csv(fn("census_data_2016.csv"))

try:
    ref = pd.read_csv
    del pd.read_csv
    conn = sqlite3.connect(fn('sports.db'))
    search_interest, countries = read_data(conn)
    conn.close()
except AttributeError as e:
    raise RuntimeError("Were you using read_csv to read the csv solution ?")
finally:
    pd.read_csv = ref

print("\n(Passed!)")
```

    
    (Passed!)
    

**Exercise 1** (3 points). In this exercise, compute the answers to the following three questions about the `search_interests` data.

1. Which country has the "most varied" interest in Olympic sports? That is, in the dataframe of search interests, which country appears most often? Store the result in the variable named **`top_country`**.
2. Which Olympic sport generates interest in the largest number of countries? Store the result in the variable **`top_sport`**.
3. How many sports are listed in the table? Store the result in the variable **`sport_count`**.


```python
search_interest['Country'].value_counts().rename_axis('Country').to_frame('Count').reset_index()[['Country']][0:1]
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
      <th>Country</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>Croatia</td>
    </tr>
  </tbody>
</table>
</div>




```python
pd.Series(search_interest.groupby('Country')['Country'].count().idxmax()).to_frame('Country')
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
      <th>Country</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>Croatia</td>
    </tr>
  </tbody>
</table>
</div>




```python
search_interest.loc[search_interest['Country'] == search_interest.groupby('Country')['Country'].count().idxmax()]
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
      <th>index</th>
      <th>Country</th>
      <th>Search_Interest</th>
      <th>Sport</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>69</th>
      <td>69</td>
      <td>Croatia</td>
      <td>4</td>
      <td>Artistic gymnastics</td>
    </tr>
    <tr>
      <th>243</th>
      <td>243</td>
      <td>Croatia</td>
      <td>9</td>
      <td>Athletics (Track &amp; Field)</td>
    </tr>
    <tr>
      <th>428</th>
      <td>428</td>
      <td>Croatia</td>
      <td>6</td>
      <td>Basketball</td>
    </tr>
    <tr>
      <th>497</th>
      <td>497</td>
      <td>Croatia</td>
      <td>1</td>
      <td>Boxing</td>
    </tr>
    <tr>
      <th>536</th>
      <td>536</td>
      <td>Croatia</td>
      <td>1</td>
      <td>Cycling</td>
    </tr>
    <tr>
      <th>589</th>
      <td>589</td>
      <td>Croatia</td>
      <td>1</td>
      <td>Diving</td>
    </tr>
    <tr>
      <th>726</th>
      <td>726</td>
      <td>Croatia</td>
      <td>1</td>
      <td>Football (Soccer)</td>
    </tr>
    <tr>
      <th>833</th>
      <td>833</td>
      <td>Croatia</td>
      <td>2</td>
      <td>Handball</td>
    </tr>
    <tr>
      <th>923</th>
      <td>923</td>
      <td>Croatia</td>
      <td>3</td>
      <td>Rowing</td>
    </tr>
    <tr>
      <th>940</th>
      <td>940</td>
      <td>Croatia</td>
      <td>2</td>
      <td>Sailing</td>
    </tr>
    <tr>
      <th>953</th>
      <td>953</td>
      <td>Croatia</td>
      <td>1</td>
      <td>Shooting</td>
    </tr>
    <tr>
      <th>1043</th>
      <td>1043</td>
      <td>Croatia</td>
      <td>5</td>
      <td>Swimming</td>
    </tr>
    <tr>
      <th>1171</th>
      <td>1171</td>
      <td>Croatia</td>
      <td>1</td>
      <td>Taekwondo</td>
    </tr>
    <tr>
      <th>1238</th>
      <td>1238</td>
      <td>Croatia</td>
      <td>6</td>
      <td>Tennis</td>
    </tr>
    <tr>
      <th>1329</th>
      <td>1329</td>
      <td>Croatia</td>
      <td>1</td>
      <td>Water polo</td>
    </tr>
    <tr>
      <th>1366</th>
      <td>1366</td>
      <td>Croatia</td>
      <td>1</td>
      <td>Wrestling</td>
    </tr>
  </tbody>
</table>
</div>




```python
top_country = None
top_sport = None
sport_count = None

def compute_basic_stats():
    
    conn = sqlite3.connect(fn('sports.db'))
    
    
    ## USING PANDAS
    
    search_interest, countries = read_data(conn)
    
    top_country = (
        search_interest
            .groupby('Country')['Country']
            .count()
            .nlargest(1)
            .to_frame('Count')
            .reset_index()
            .filter(['Country'])
    )
    
    top_sport = (
        search_interest
            .groupby('Sport')['Sport']
            .count()
            .nlargest(1)
            .to_frame('Count')
            .reset_index()
            .filter(['Sport'])
    )
    
    sport_count = (
        search_interest
            .groupby('Sport')
            .ngroups
    )
    
    
    ## USING SQLITE3
    
    #top_country_query = '''
    #    
    #    --SQLITE SPECIFIC SQL QUERY
    #    select Country
    #    from search_interest
    #    group by Country
    #    order by count(*) desc
    #    limit 1
    #    
    #    ----ANSI COMPLIANT SQL QUERY WHICH WILL WORK IN MOST DATABASES
    #    --select tc.Country
    #    --from (
    #    --    select
    #    --        si.Country,
    #    --        row_number() over (order by count(*) desc) as Rank
    #    --    from search_interest as si
    #    --    group by Country
    #    --) as tc
    #    --where Rank = 1
    #    --;
    #    
    #'''
    #
    #top_sport_query = '''
    #
    #    --SQLITE SPECIFIC SQL QUERY
    #    select Sport
    #    from search_interest
    #    group by Sport
    #    order by count(*) desc
    #    limit 1
    #
    #    ----ANSI COMPLIANT SQL QUERY WHICH WILL WORK IN MOST DATABASES
    #    --select ts.Sport
    #    --from (
    #    --    select
    #    --        si.Sport,
    #    --        row_number() over (order by count(*) desc) as Rank
    #    --    from search_interest as si
    #    --    group by si.Sport
    #    --) as ts
    #    --where Rank = 1
    #    --;
    #    
    #'''
    #
    #sport_count_query = '''
    #
    #    --ANSI COMPLIANT SQL QUERY WHICH WILL WORK IN MOST DATABASES
    #    select count(distinct si.Sport) as Count
    #    from search_interest as si
    #
    #'''
    #
    #top_country = pd.read_sql_query(top_country_query, conn)
    #top_sport = pd.read_sql_query(top_sport_query, conn)
    #sport_count = pd.read_sql_query(sport_count_query, conn)
    
    conn.close()
    
    return top_country, top_sport, sport_count
    
top_country, top_sport, sport_count = compute_basic_stats()
```


```python
# Test code
try:
    ref = search_interest
    del search_interest
    top_country, top_sport, sport_count = compute_basic_stats()
except NameError:
    search_interest = ref
    top_country, top_sport, sport_count = compute_basic_stats()
    assert top_country == 'Croatia' or top_country == 'New Zealand'
    assert top_sport == 'Athletics (Track & Field)'
    assert sport_count == 34
except Exception as e:
    print(e)
    print("Were you not using the search_interest dataframe to compute the stats ?")
finally:
    search_interest = ref

print("\n(Passed!)")
```

    
    (Passed!)
    

## Worldwide popularity of a sport

To estimate the popularity of a sport, it is not good enough to get only a count of the countries where the sport generated enough search interest. We might get a better estimate of popularity by computing a weighted average of search interest that accounts for differences in search interests and populations among countries.

**Exercise 2** (2 points). Before we can perform a weighted average, we need to find the weights for each country. To do that, we need the population for each of the countries in the search interest table, which we can obtain by querying the census population table.

Complete the function **`join_pop(si, c)`** below to perform this task. That is, given the dataframe of search interests, **`si`**, and the census data, **`c`**, this function should join the `Population` column from `c` to `si` and return the result.

The returned value of `join_pop(si, c)` should be a copy of `si` with one additional column named `'Population'` that holds the corresponding population value from `c`.

> To match the country names between the `si` and `c` dataframes, note that the `si` dataframe's `'Country'` column includes spaces whereas `c` does not. You'll want to account for that by, for instance, stripping out the spaces from `si` before merging or joining with `c`.


```python
def translate_country(country):
    """
    Removes spaces from country names
    """
    return country.replace(' ', '')

def join_pop(si, c):
    return (
        si
            .assign(CountryTemp = lambda df: df['Country'])
            .assign(Country = lambda df: df['Country'].apply(lambda x: translate_country(x)))
            .merge(c[['Country', 'Population']], how = 'left', on = 'Country')
            .assign(Country = lambda df: df['CountryTemp'])
            .drop(['CountryTemp'], axis = 1)
    )

total_world_population = sum(countries["Population"])
join_df = join_pop(search_interest, countries)

display(join_df.head())
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
      <th>index</th>
      <th>Country</th>
      <th>Search_Interest</th>
      <th>Sport</th>
      <th>Population</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>0</td>
      <td>Iran</td>
      <td>1</td>
      <td>Archery</td>
      <td>80987449</td>
    </tr>
    <tr>
      <th>1</th>
      <td>1</td>
      <td>South Korea</td>
      <td>2</td>
      <td>Archery</td>
      <td>50924172</td>
    </tr>
    <tr>
      <th>2</th>
      <td>2</td>
      <td>Mexico</td>
      <td>1</td>
      <td>Archery</td>
      <td>123166749</td>
    </tr>
    <tr>
      <th>3</th>
      <td>3</td>
      <td>Netherlands</td>
      <td>1</td>
      <td>Archery</td>
      <td>17016967</td>
    </tr>
    <tr>
      <th>4</th>
      <td>4</td>
      <td>Aruba</td>
      <td>16</td>
      <td>Artistic gymnastics</td>
      <td>113648</td>
    </tr>
  </tbody>
</table>
</div>



```python
# Test cell: `join_tables_test`

join_df_ref = pd.read_csv(fn("joined_df.csv"))

try:
    ref = pd.read_csv
    del pd.read_csv
    join_df = join_pop(search_interest, countries)
    assert tibbles_are_equivalent(join_df, join_df_ref), "Solution is incorrect"
except AttributeError as e:
    raise RuntimeError("Were you using read_csv to read the csv solution ?")
finally:
    pd.read_csv = ref

print("\n(Passed!)")
```

    
    (Passed!)
    

**Weighing search interest by population.** Suppose that to compare different Olympic sports by global popularity, we want to account for each country's population.

For instance, suppose we are looking at the global search interest in volleyball. If volleyball's search interest equals `1` in both China and the Netherlands, we might weigh China's search interest more since it is the more populous contry.

To determine the weights for each country, let's just use each country's fraction of the global population. Recall that an earlier code cell computed the variable, `total_world_population`, which is the global population. Let the weight of a given country be its population divided by the global population. (For instance, if the global population is 6 billion people and the population of India is 1 billion, then India's "weight" would be one-sixth.)

**Exercise 3** (3 points). Create a dataframe named `ranking` with two columns, `'Sport'` and `'weighted_interest'`, where there is one row per sport, and each sport's `'weighted_interest'` is the overall weighted interest across countries, using the population weights for each country as described above.

> **Hint**: Consider using [groupby()](https://pandas.pydata.org/pandas-docs/stable/generated/pandas.DataFrame.groupby.html) for Pandas DataFrames. It is very similar to `GROUP BY` in SQL.


```python
ranking = (
    join_df
        .assign(weighted_interest = lambda df: df['Search_Interest'] * df['Population'] / total_world_population)
        .groupby('Sport')['weighted_interest']
        .sum()
        .sort_values(ascending = False)
        .to_frame()
        .reset_index()
)

# top 10 sports
display(ranking[:10])
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
      <th>Sport</th>
      <th>weighted_interest</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>Swimming</td>
      <td>5.983388</td>
    </tr>
    <tr>
      <th>1</th>
      <td>Athletics (Track &amp; Field)</td>
      <td>4.273728</td>
    </tr>
    <tr>
      <th>2</th>
      <td>Badminton</td>
      <td>3.051064</td>
    </tr>
    <tr>
      <th>3</th>
      <td>Artistic gymnastics</td>
      <td>2.337363</td>
    </tr>
    <tr>
      <th>4</th>
      <td>Tennis</td>
      <td>2.119308</td>
    </tr>
    <tr>
      <th>5</th>
      <td>Football (Soccer)</td>
      <td>1.345433</td>
    </tr>
    <tr>
      <th>6</th>
      <td>Table tennis</td>
      <td>0.929301</td>
    </tr>
    <tr>
      <th>7</th>
      <td>Wrestling</td>
      <td>0.845934</td>
    </tr>
    <tr>
      <th>8</th>
      <td>Diving</td>
      <td>0.727840</td>
    </tr>
    <tr>
      <th>9</th>
      <td>Basketball</td>
      <td>0.462788</td>
    </tr>
  </tbody>
</table>
</div>



```python
# Test cell: `ranking_test`

ranking_ref = pd.read_csv(fn("rankings_ref.csv"))
assert (ranking_ref["Sport"] == ranking["Sport"]).all()

print("\n(Passed!)")
```

    
    (Passed!)
    

**Fin!** You have reached the end of this problem. Be sure to submit it before moving on.
