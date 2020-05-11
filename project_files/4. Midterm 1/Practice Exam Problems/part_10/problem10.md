# Problem 10: Political Network Connections

In this problem, you will analyze the network connections and strength between all persons and organizations in the *Trump World* using a combination of hash tables (i.e., dictionaries) and pandas dataframe.  

## The dataset

The dataset for this problem is built from public records, news reports, and other sources on the Trump family, his Cabinet picks, and top advisers - more than 1,500 people and organizations altogether. 

Each row represents a connection between a person and an organization (e.g., The Trump Organization Inc. and Donald J. Trump), a person and another person (e.g., Donald J. Trump and Linda McMahon), or two organizations (e.g., Bedford Hills Corp. and Seven Springs LLC).

Source: https://www.buzzfeednews.com/article/johntemplon/help-us-map-trumpworld

Before starting, please run the following cell to set up the environment and import the data to `Network`.


```python
import math
import pandas as pd
import numpy as np
from collections import defaultdict

Network = pd.read_csv("./network/network.csv", encoding='latin-1' )
assert len(Network) == 3380
Network.head()
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
      <th>Entity A Type</th>
      <th>Entity A</th>
      <th>Entity B Type</th>
      <th>Entity B</th>
      <th>Connection_Strength</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>Organization</td>
      <td>4 SHADOW TREE LANE MEMBER CORP.</td>
      <td>Organization</td>
      <td>4 SHADOW TREE LANE LLC</td>
      <td>0.469155</td>
    </tr>
    <tr>
      <th>1</th>
      <td>Organization</td>
      <td>40 WALL DEVELOPMENT ASSOCIATES LLC</td>
      <td>Organization</td>
      <td>40 WALL STREET LLC</td>
      <td>0.035480</td>
    </tr>
    <tr>
      <th>2</th>
      <td>Organization</td>
      <td>40 WALL STREET LLC</td>
      <td>Organization</td>
      <td>40 WALL STREET COMMERCIAL LLC</td>
      <td>0.177874</td>
    </tr>
    <tr>
      <th>3</th>
      <td>Organization</td>
      <td>40 WALL STREET MEMBER CORP.</td>
      <td>Organization</td>
      <td>40 WALL STREET LLC</td>
      <td>0.236508</td>
    </tr>
    <tr>
      <th>4</th>
      <td>Organization</td>
      <td>401 MEZZ VENTURE LLC</td>
      <td>Organization</td>
      <td>401 NORTH WABASH VENTURE LLC</td>
      <td>0.169532</td>
    </tr>
  </tbody>
</table>
</div>



**Exercise 0** (1 points). Create a subset of the data frame named `Network_sub`, keeping only records where `Entity B` contains the keyword "TRUMP" (not case sensitive).


```python
Network_sub = Network[Network['Entity B'].str.contains('TRUMP')]
```


```python
# Test cell: `test_subset`

assert type(Network_sub)==pd.DataFrame, "Your subset is not a panda dataframe"
assert list(Network_sub)==['Entity A Type','Entity A','Entity B Type','Entity B','Connection_Strength'], "Your subset columns are not consistent with the master dataset"
assert len(Network_sub)==648, "The length of your subset is not correct"

test = Network_sub.sort_values(by='Connection_Strength')
test.reset_index(drop=True, inplace=True)
assert test.loc[0,'Connection_Strength']==0.001315204
assert test.loc[200,'Connection_Strength']==0.312599997
assert test.loc[400,'Connection_Strength']==0.610184514
assert test.loc[647,'Connection_Strength']==0.996641965

print("\n(Passed.)")
```

    
    (Passed.)
    

Now, let's take a look at part of the `Network_sub` data.


```python
Network_sub.iloc[25:36]
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
      <th>Entity A Type</th>
      <th>Entity A</th>
      <th>Entity B Type</th>
      <th>Entity B</th>
      <th>Connection_Strength</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>232</th>
      <td>Person</td>
      <td>BRIAN BAUDREAU</td>
      <td>Organization</td>
      <td>THE TRUMP ORGANIZATION, INC.</td>
      <td>0.249506</td>
    </tr>
    <tr>
      <th>237</th>
      <td>Organization</td>
      <td>BRIARCLIFF PROPERTIES, INC.</td>
      <td>Organization</td>
      <td>TRUMP BRIARCLIFF MANOR DEVELOPMENT LLC</td>
      <td>0.102998</td>
    </tr>
    <tr>
      <th>238</th>
      <td>Person</td>
      <td>BRITTANY HEBERT</td>
      <td>Organization</td>
      <td>THE ERIC TRUMP FOUNDATION</td>
      <td>0.724913</td>
    </tr>
    <tr>
      <th>257</th>
      <td>Person</td>
      <td>CARTER PAGE</td>
      <td>Person</td>
      <td>DONALD J. TRUMP</td>
      <td>0.694884</td>
    </tr>
    <tr>
      <th>280</th>
      <td>Person</td>
      <td>CHARLES P. REISS</td>
      <td>Organization</td>
      <td>THE TRUMP ORGANIZATION, INC.</td>
      <td>0.937458</td>
    </tr>
    <tr>
      <th>283</th>
      <td>Person</td>
      <td>CHEN SITING AKA CHARLYNE CHEN</td>
      <td>Organization</td>
      <td>TRUMP ORGANIZATION LLC</td>
      <td>0.137199</td>
    </tr>
    <tr>
      <th>284</th>
      <td>Organization</td>
      <td>CHEVY CHASE TRUST HOLDINGS, INC.</td>
      <td>Organization</td>
      <td>TRUMP NATIONAL GOLF CLUB WASHINGTON DC LLC</td>
      <td>0.925422</td>
    </tr>
    <tr>
      <th>286</th>
      <td>Person</td>
      <td>CHLOE MURDOCH</td>
      <td>Person</td>
      <td>IVANKA TRUMP</td>
      <td>0.805567</td>
    </tr>
    <tr>
      <th>294</th>
      <td>Person</td>
      <td>CHRISTL MAHFOUZ</td>
      <td>Organization</td>
      <td>THE ERIC TRUMP FOUNDATION</td>
      <td>0.426780</td>
    </tr>
    <tr>
      <th>326</th>
      <td>Organization</td>
      <td>DAEWOO AMERICA DEVELOPMENT (NEW YORK) CORP</td>
      <td>Organization</td>
      <td>TRUMP KOREA LLC</td>
      <td>0.994785</td>
    </tr>
    <tr>
      <th>327</th>
      <td>Organization</td>
      <td>DAEWOO AMERICA DEVELOPMENT (NEW YORK) CORP.</td>
      <td>Organization</td>
      <td>TRUMP KOREA LLC</td>
      <td>0.618116</td>
    </tr>
  </tbody>
</table>
</div>




```python
Network_sub[Network_sub['Entity B Type'] == 'Person']
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
      <th>Entity A Type</th>
      <th>Entity A</th>
      <th>Entity B Type</th>
      <th>Entity B</th>
      <th>Connection_Strength</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>257</th>
      <td>Person</td>
      <td>CARTER PAGE</td>
      <td>Person</td>
      <td>DONALD J. TRUMP</td>
      <td>0.694884</td>
    </tr>
    <tr>
      <th>286</th>
      <td>Person</td>
      <td>CHLOE MURDOCH</td>
      <td>Person</td>
      <td>IVANKA TRUMP</td>
      <td>0.805567</td>
    </tr>
    <tr>
      <th>584</th>
      <td>Person</td>
      <td>DONALD J. TRUMP</td>
      <td>Person</td>
      <td>DONALD TRUMP JR.</td>
      <td>0.453991</td>
    </tr>
    <tr>
      <th>674</th>
      <td>Person</td>
      <td>DONALD J. TRUMP</td>
      <td>Person</td>
      <td>ERIC TRUMP</td>
      <td>0.468002</td>
    </tr>
    <tr>
      <th>709</th>
      <td>Person</td>
      <td>DONALD J. TRUMP</td>
      <td>Person</td>
      <td>IVANKA TRUMP</td>
      <td>0.773875</td>
    </tr>
    <tr>
      <th>758</th>
      <td>Person</td>
      <td>DONALD J. TRUMP</td>
      <td>Person</td>
      <td>MARYANNE TRUMP BARRY</td>
      <td>0.330120</td>
    </tr>
    <tr>
      <th>761</th>
      <td>Person</td>
      <td>DONALD J. TRUMP</td>
      <td>Person</td>
      <td>MELANIA TRUMP</td>
      <td>0.517144</td>
    </tr>
    <tr>
      <th>1248</th>
      <td>Organization</td>
      <td>DONALD J. TRUMP FOR PRESIDENT, INC.</td>
      <td>Person</td>
      <td>DONALD J. TRUMP</td>
      <td>0.377887</td>
    </tr>
    <tr>
      <th>1271</th>
      <td>Person</td>
      <td>DONALD TRUMP JR.</td>
      <td>Person</td>
      <td>ERIC TRUMP</td>
      <td>0.405052</td>
    </tr>
    <tr>
      <th>1293</th>
      <td>Person</td>
      <td>DONALD TRUMP JR.</td>
      <td>Person</td>
      <td>VANESSA TRUMP</td>
      <td>0.025757</td>
    </tr>
    <tr>
      <th>1504</th>
      <td>Person</td>
      <td>GRACE MURDOCH</td>
      <td>Person</td>
      <td>IVANKA TRUMP</td>
      <td>0.966638</td>
    </tr>
    <tr>
      <th>1582</th>
      <td>Organization</td>
      <td>IVANKA M. TRUMP BUSINESS TRUST</td>
      <td>Person</td>
      <td>IVANKA TRUMP</td>
      <td>0.141786</td>
    </tr>
    <tr>
      <th>1585</th>
      <td>Person</td>
      <td>IVANKA TRUMP</td>
      <td>Person</td>
      <td>DONALD TRUMP JR.</td>
      <td>0.030932</td>
    </tr>
    <tr>
      <th>1586</th>
      <td>Person</td>
      <td>IVANKA TRUMP</td>
      <td>Person</td>
      <td>ERIC TRUMP</td>
      <td>0.034109</td>
    </tr>
    <tr>
      <th>2022</th>
      <td>Person</td>
      <td>MICHAEL FASCITELLI</td>
      <td>Person</td>
      <td>DONALD J. TRUMP</td>
      <td>0.924708</td>
    </tr>
    <tr>
      <th>2095</th>
      <td>Person</td>
      <td>MIKE PENCE</td>
      <td>Person</td>
      <td>DONALD J. TRUMP</td>
      <td>0.828195</td>
    </tr>
    <tr>
      <th>2333</th>
      <td>Organization</td>
      <td>REPUBLICAN NATIONAL COMMITTEE</td>
      <td>Person</td>
      <td>DONALD J. TRUMP</td>
      <td>0.563547</td>
    </tr>
    <tr>
      <th>2370</th>
      <td>Person</td>
      <td>RHONA GRAFF RICCIO</td>
      <td>Person</td>
      <td>DONALD J. TRUMP</td>
      <td>0.365323</td>
    </tr>
    <tr>
      <th>2508</th>
      <td>Person</td>
      <td>SERGEI MILLIAN</td>
      <td>Person</td>
      <td>DONALD J. TRUMP</td>
      <td>0.199488</td>
    </tr>
    <tr>
      <th>2721</th>
      <td>Organization</td>
      <td>T INTERNATIONAL REALTY LLC</td>
      <td>Person</td>
      <td>IVANKA TRUMP</td>
      <td>0.207326</td>
    </tr>
    <tr>
      <th>3142</th>
      <td>Organization</td>
      <td>TTT CONSULTING LLC</td>
      <td>Person</td>
      <td>IVANKA TRUMP</td>
      <td>0.563544</td>
    </tr>
    <tr>
      <th>3149</th>
      <td>Organization</td>
      <td>TTTT VENTURE LLC</td>
      <td>Person</td>
      <td>IVANKA TRUMP</td>
      <td>0.600213</td>
    </tr>
    <tr>
      <th>3194</th>
      <td>Person</td>
      <td>WENDI DENG MURDOCH</td>
      <td>Person</td>
      <td>IVANKA TRUMP</td>
      <td>0.669636</td>
    </tr>
  </tbody>
</table>
</div>



**Exercise 1** (4 points). Write a function 

```python
def Connection_Strength(Network_sub, Entity_B_Type)
```

that takes two inputs

1. `Network_sub` is the dataset you get from exercise 0
2. `Entity_B_Type` can take two values: either `Person` or `Organization`

and for every entity A that is connected to entity B, based on the type of entity B, returns a nested dictionary (i.e. dictionary of dictionaries) of the form:

```python 
{Entity A: {Entity B: Connection_Strength, Entity B: Connection_Strength}, ... }```

For example: for entity A that is connected to entity B of type person, the function will return something like the following: 

```python
{'DONALD J. TRUMP': {'DONALD TRUMP JR.': 0.453990548,
  'ERIC TRUMP': 0.468002101,
  'IVANKA TRUMP': 0.773874808,
  'MARYANNE TRUMP BARRY': 0.330120053,
  'MELANIA TRUMP': 0.5171444000000001},
 'DONALD J. TRUMP FOR PRESIDENT, INC.': {'DONALD J. TRUMP': 0.377887355},
 'DONALD TRUMP JR.': {'ERIC TRUMP': 0.405052388, 'VANESSA TRUMP': 0.025756815},
 'GRACE MURDOCH': {'IVANKA TRUMP': 0.966637541},
 'IVANKA M. TRUMP BUSINESS TRUST': {'IVANKA TRUMP': 0.141785871}, ...}```


```python
def Connection_Strength(Network_sub, Entity_B_Type):
    assert type(Entity_B_Type) == str
    assert Entity_B_Type in ['Person', 'Organization']
    df = Network_sub[Network_sub['Entity B Type'] == Entity_B_Type]
    d = defaultdict(lambda : defaultdict(float))
    for index, row in df.iterrows():
        d[row['Entity A']][row['Entity B']] = row['Connection_Strength']
    return d
```


```python
# Test Cell: `Connection_Strength`

# Create a dictonary 'Person' for entity B of type person
Person = Connection_Strength(Network_sub, 'Person')
# Create a dictionary 'Organization' for entity B of type organization
Organization = Connection_Strength(Network_sub, 'Organization')

assert type(Person)==dict or defaultdict, "Your function does not return a dictionary"
assert len(Person)==17, "Your result is wrong for entity B of type person"
assert len(Organization)==296, "Your result is wrong for entity B of type organization"

assert Person['DONALD J. TRUMP']=={'DONALD TRUMP JR.': 0.453990548,'ERIC TRUMP': 0.468002101,'IVANKA TRUMP': 0.773874808,
  'MARYANNE TRUMP BARRY': 0.330120053,'MELANIA TRUMP': 0.5171444000000001}, "Wrong result"
assert Person['DONALD J. TRUMP FOR PRESIDENT, INC.']=={'DONALD J. TRUMP': 0.377887355}, "Wrong result"
assert Person['WENDI DENG MURDOCH']=={'IVANKA TRUMP': 0.669636181}, "Wrong result"

assert Organization['401 MEZZ VENTURE LLC']=={'TRUMP CHICAGO RETAIL LLC': 0.85298544}, "Wrong result"
assert Organization['ACE ENTERTAINMENT HOLDINGS INC']=={'TRUMP CASINOS INC.': 0.202484568,'TRUMP TAJ MAHAL INC.': 0.48784823299999996}, "Wrong result"
assert Organization['ANDREW JOBLON']=={'THE ERIC TRUMP FOUNDATION': 0.629688777}, "Wrong result"

print("\n(Passed.)")
```

    
    (Passed.)
    

**Exercise 2** (1 point). For the dictionary `Organization` **created in the above test cell**, create another dictionary `Organization_avg` which for every entity A gives the average connection strength (i.e., the average of nested dictionary values). `Organization_avg` should be in the following form:
```python
{Entity A: avg_Connection_Strength, Entity A: avg_Connection_Strength, ... }```



```python
Organization_avg = defaultdict(float)
for k, v in Organization.items():
    Organization_avg[k] = sum(v.values()) / len(v)
```


```python
# Test Cell: `Organization_avg`
assert type(Organization_avg)==dict or defaultdict, "Organization_avg is not a dictionary"
assert len(Organization_avg)==len(Organization)

for k_, v_ in {'401 MEZZ VENTURE LLC': 0.85298544,
               'DJT HOLDINGS LLC': 0.5855800477222223,
               'DONALD J. TRUMP': 0.4878277050144927,
               'JAMES BURNHAM': 0.187474088}.items():
    print(k_, Organization_avg[k_], v_)
    assert math.isclose(Organization_avg[k_], v_, rel_tol=4e-15*len(Organization[k_])), \
           "Wrong result for '{}': Expected {}, got {}".format(k_, v_, Organization_avg[k_])

print("\n(Passed.)")
```

    401 MEZZ VENTURE LLC 0.85298544 0.85298544
    DJT HOLDINGS LLC 0.5855800477222224 0.5855800477222223
    DONALD J. TRUMP 0.4878277050144922 0.4878277050144927
    JAMES BURNHAM 0.187474088 0.187474088
    
    (Passed.)
    

**Exercise 3** (4 points). Based on the `Organization_avg` dictionary you just created, determine which organizations have an average connection strength that is strictly greater than a given threshold, `THRESHOLD` (defined in the code cell below). Then, create a new data frame named `Network_strong` that has a subset of the rows of `Network_sub` whose `Entity A` values match these organizations **and** whose `"Entity B Type"` equals `"Organization"`.


```python
THRESHOLD = 0.5

# my solution
#filtered_dict = {k : v for k, v in Organization_avg.items() if v > THRESHOLD}
#filtered_df = pd.DataFrame(filtered_dict.items(), columns = ['Entity A', 'Avg Connection Strength'])
#Network_strong = filtered_df[['Entity A']].merge(Network_sub[Network_sub['Entity B Type'] == 'Organization'], on = 'Entity A', how = 'inner')
#Network_strong = Network_strong[['Entity A Type','Entity A','Entity B Type','Entity B','Connection_Strength']]

# school solution
names = [k for k, v in Organization_avg.items() if v > THRESHOLD]
organization_sub = Network_sub[Network_sub['Entity B Type'] == 'Organization']
Network_strong = organization_sub[organization_sub['Entity A'].isin(names)]
```


```python
# Test Cell: `Network_strong`
assert type(Network_strong)==pd.DataFrame, "Network_strong is not a panda dataframe"
assert list(Network_strong)==['Entity A Type','Entity A','Entity B Type','Entity B','Connection_Strength'], "Your Network_strong columns are not consistent with the master dataset"
assert len(Network_strong)==194, "The length of your Network_strong is not correct. Correct length should be 194."
test2 = Network_strong.sort_values(by='Connection_Strength')
test2.reset_index(drop=True, inplace=True)
assert math.isclose(test2.loc[0, 'Connection_Strength'], 0.039889119, rel_tol=1e-13)
assert math.isclose(test2.loc[100, 'Connection_Strength'], 0.744171895, rel_tol=1e-13)
assert math.isclose(test2.loc[193, 'Connection_Strength'], 0.996641965, rel_tol=1e-13)

print("\n(Passed.)")
```

    
    (Passed.)
    

**Fin!** Remember to test your solutions by running them as the autograder will: restart the kernel and run all cells from "top-to-bottom." Also remember to submit to the autograder; otherwise, you will **not** get credit for your hard work!
