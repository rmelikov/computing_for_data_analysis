# Problem 2: "But her emails..."

In this problem, you'll show your SQL and Pandas chops on the dataset consisting of Hilary Rodham Clinton's emails!

This problem has four (4) exercises (0-3) and is worth a total of ten (10) points.

> This problem also involves SQLite. However, at present, the `sqlite3` module is broken on Vocareum when using the Python 3.6 kernel; therefore, the kernel has been set to Python 3.5 for this notebook. If you are trying to work on a local copy, please be wary of potential differences in modules and versions.


```python
import sys
print("=== Python version info ===\n{}".format(sys.version))

import sqlite3 as db
print("\n=== sqlite3 version info: {} ===".format(db.version))
```

    === Python version info ===
    3.7.4 (tags/v3.7.4:e09359112e, Jul  8 2019, 20:34:20) [MSC v.1916 64 bit (AMD64)]
    
    === sqlite3 version info: 2.6.0 ===
    

## Setup

We downloaded [this database from Kaggle](https://www.kaggle.com/kaggle/hillary-clinton-emails).

> We are only making it available on Vocareum for use in this problem. If you wish to work on this problem in your local environment, you are on your own in figuring out how to get the data (or a subset of the data) to be able to do so.

Start by running the following setup code, which will load the modules you'll need for this problem


```python
from IPython.display import display
import pandas as pd
import numpy as np

def peek_table (db, name, num=5):
    """
    Given a database connection (`db`), prints both the number of
    records in the table as well as its first few entries.
    """
    count = '''select count (*) FROM {table}'''.format (table=name)
    peek = '''select * from {table} limit {limit}'''.format (table=name, limit=num)

    print ("Total number of records:", pd.read_sql_query (count, db)['count (*)'].iloc[0], "\n")

    print ("First {} entries:".format (num))
    display (pd.read_sql_query (peek, db))

def list_tables (conn):
    """Return the names of all visible tables, given a database connection."""
    query = """select name from sqlite_master where type = 'table';"""
    c = conn.cursor ()
    c.execute (query)
    table_names = [t[0] for t in c.fetchall ()]
    return table_names

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


```python
DATA_PATH = "./"
conn = db.connect ('{}hrc.db'.format(DATA_PATH))

print ("List of tables in the database:", list_tables (conn))
```

    List of tables in the database: ['Emails', 'Persons', 'Aliases', 'EmailReceivers']
    


```python
peek_table (conn, 'Emails')
peek_table (conn, 'EmailReceivers', num=3)
peek_table (conn, 'Persons')
```

    Total number of records: 7945 
    
    First 5 entries:
    


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
      <th>Id</th>
      <th>DocNumber</th>
      <th>MetadataSubject</th>
      <th>MetadataTo</th>
      <th>MetadataFrom</th>
      <th>SenderPersonId</th>
      <th>MetadataDateSent</th>
      <th>MetadataDateReleased</th>
      <th>MetadataPdfLink</th>
      <th>MetadataCaseNumber</th>
      <th>...</th>
      <th>ExtractedTo</th>
      <th>ExtractedFrom</th>
      <th>ExtractedCc</th>
      <th>ExtractedDateSent</th>
      <th>ExtractedCaseNumber</th>
      <th>ExtractedDocNumber</th>
      <th>ExtractedDateReleased</th>
      <th>ExtractedReleaseInPartOrFull</th>
      <th>ExtractedBodyText</th>
      <th>RawText</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>1</td>
      <td>C05739545</td>
      <td>WOW</td>
      <td>H</td>
      <td>Sullivan, Jacob J</td>
      <td>87</td>
      <td>2012-09-12T04:00:00+00:00</td>
      <td>2015-05-22T04:00:00+00:00</td>
      <td>DOCUMENTS/HRC_Email_1_296/HRCH2/DOC_0C05739545...</td>
      <td>F-2015-04841</td>
      <td>...</td>
      <td></td>
      <td>Sullivan, Jacob J &lt;Sullivan11@state.gov&gt;</td>
      <td></td>
      <td>Wednesday, September 12, 2012 10:16 AM</td>
      <td>F-2015-04841</td>
      <td>C05739545</td>
      <td>05/13/2015</td>
      <td>RELEASE IN FULL</td>
      <td></td>
      <td>UNCLASSIFIED\nU.S. Department of State\nCase N...</td>
    </tr>
    <tr>
      <th>1</th>
      <td>2</td>
      <td>C05739546</td>
      <td>H: LATEST: HOW SYRIA IS AIDING QADDAFI AND MOR...</td>
      <td>H</td>
      <td></td>
      <td></td>
      <td>2011-03-03T05:00:00+00:00</td>
      <td>2015-05-22T04:00:00+00:00</td>
      <td>DOCUMENTS/HRC_Email_1_296/HRCH1/DOC_0C05739546...</td>
      <td>F-2015-04841</td>
      <td>...</td>
      <td></td>
      <td></td>
      <td></td>
      <td></td>
      <td>F-2015-04841</td>
      <td>C05739546</td>
      <td>05/13/2015</td>
      <td>RELEASE IN PART</td>
      <td>B6\nThursday, March 3, 2011 9:45 PM\nH: Latest...</td>
      <td>UNCLASSIFIED\nU.S. Department of State\nCase N...</td>
    </tr>
    <tr>
      <th>2</th>
      <td>3</td>
      <td>C05739547</td>
      <td>CHRIS STEVENS</td>
      <td>;H</td>
      <td>Mills, Cheryl D</td>
      <td>32</td>
      <td>2012-09-12T04:00:00+00:00</td>
      <td>2015-05-22T04:00:00+00:00</td>
      <td>DOCUMENTS/HRC_Email_1_296/HRCH2/DOC_0C05739547...</td>
      <td>F-2015-04841</td>
      <td>...</td>
      <td>B6</td>
      <td>Mills, Cheryl D &lt;MillsCD@state.gov&gt;</td>
      <td>Abedin, Huma</td>
      <td>Wednesday, September 12, 2012 11:52 AM</td>
      <td>F-2015-04841</td>
      <td>C05739547</td>
      <td>05/14/2015</td>
      <td>RELEASE IN PART</td>
      <td>Thx</td>
      <td>UNCLASSIFIED\nU.S. Department of State\nCase N...</td>
    </tr>
    <tr>
      <th>3</th>
      <td>4</td>
      <td>C05739550</td>
      <td>CAIRO CONDEMNATION - FINAL</td>
      <td>H</td>
      <td>Mills, Cheryl D</td>
      <td>32</td>
      <td>2012-09-12T04:00:00+00:00</td>
      <td>2015-05-22T04:00:00+00:00</td>
      <td>DOCUMENTS/HRC_Email_1_296/HRCH2/DOC_0C05739550...</td>
      <td>F-2015-04841</td>
      <td>...</td>
      <td></td>
      <td>Mills, Cheryl D &lt;MillsCD@state.gov&gt;</td>
      <td>Mitchell, Andrew B</td>
      <td>Wednesday, September 12,2012 12:44 PM</td>
      <td>F-2015-04841</td>
      <td>C05739550</td>
      <td>05/13/2015</td>
      <td>RELEASE IN PART</td>
      <td></td>
      <td>UNCLASSIFIED\nU.S. Department of State\nCase N...</td>
    </tr>
    <tr>
      <th>4</th>
      <td>5</td>
      <td>C05739554</td>
      <td>H: LATEST: HOW SYRIA IS AIDING QADDAFI AND MOR...</td>
      <td>Abedin, Huma</td>
      <td>H</td>
      <td>80</td>
      <td>2011-03-11T05:00:00+00:00</td>
      <td>2015-05-22T04:00:00+00:00</td>
      <td>DOCUMENTS/HRC_Email_1_296/HRCH1/DOC_0C05739554...</td>
      <td>F-2015-04841</td>
      <td>...</td>
      <td></td>
      <td></td>
      <td></td>
      <td></td>
      <td>F-2015-04841</td>
      <td>C05739554</td>
      <td>05/13/2015</td>
      <td>RELEASE IN PART</td>
      <td>H &lt;hrod17@clintonemail.com&gt;\nFriday, March 11,...</td>
      <td>B6\nUNCLASSIFIED\nU.S. Department of State\nCa...</td>
    </tr>
  </tbody>
</table>
<p>5 rows × 22 columns</p>
</div>


    Total number of records: 9306 
    
    First 3 entries:
    


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
      <th>Id</th>
      <th>EmailId</th>
      <th>PersonId</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>1</td>
      <td>1</td>
      <td>80</td>
    </tr>
    <tr>
      <th>1</th>
      <td>2</td>
      <td>2</td>
      <td>80</td>
    </tr>
    <tr>
      <th>2</th>
      <td>3</td>
      <td>3</td>
      <td>228</td>
    </tr>
  </tbody>
</table>
</div>


    Total number of records: 513 
    
    First 5 entries:
    


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
      <th>Id</th>
      <th>Name</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>1</td>
      <td>111th Congress</td>
    </tr>
    <tr>
      <th>1</th>
      <td>2</td>
      <td>AGNA USEMB Kabul Afghanistan</td>
    </tr>
    <tr>
      <th>2</th>
      <td>3</td>
      <td>AP</td>
    </tr>
    <tr>
      <th>3</th>
      <td>4</td>
      <td>ASUNCION</td>
    </tr>
    <tr>
      <th>4</th>
      <td>5</td>
      <td>Alec</td>
    </tr>
  </tbody>
</table>
</div>


**Exercise 0** (1 point). Extract the `Persons` table from the database and store it as a Pandas data frame in a variable named `Persons` having two columns: `Id` and `Name`.


```python
query = '''
    select *
    from persons
'''
Persons = pd.read_sql_query(query, conn)
```


```python
assert 'Persons' in globals ()
assert type (Persons) is type (pd.DataFrame ())
assert len (Persons) == 513

print ("Five random people from the `Persons` table:")
display (Persons.iloc[np.random.choice (len (Persons), 5)])

print ("\n(Passed!)")
```

    Five random people from the `Persons` table:
    


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
      <th>Id</th>
      <th>Name</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>49</th>
      <td>50</td>
      <td>David Axelrod</td>
    </tr>
    <tr>
      <th>479</th>
      <td>480</td>
      <td>jake.sullivan h</td>
    </tr>
    <tr>
      <th>420</th>
      <td>421</td>
      <td>edwards christopher (jakarta/pro)</td>
    </tr>
    <tr>
      <th>327</th>
      <td>328</td>
      <td>.1ilotylc@state.gov.</td>
    </tr>
    <tr>
      <th>377</th>
      <td>378</td>
      <td>iewij@state.gov</td>
    </tr>
  </tbody>
</table>
</div>


    
    (Passed!)
    

**Exercise 1** (3 points). Query the database to determine how frequently particular pairs of people communicate. Store the results in a Pandas data frame named `CommEdges` having the following three columns:

- `Sender`: The ID of the sender (taken from the `Emails` table).
- `Receiver`: The ID of the receiver (taken from the `EmailReceivers` table).
- `Frequency`: The number of times this particular (`Sender`, `Receiver`) pair occurs.

Order the results in _descending_ order of `Frequency`.

There is one corner case that you should also handle: sometimes the `Sender` field is empty (unknown). You can filter these cases by checking that the sender ID is not the empty string.


```python
query = '''
    select
        e.SenderPersonId as Sender,
        er.PersonId as Receiver,
        count(*) as Frequency
    from EmailReceivers er, Emails e
    where e.Id = er.EmailId and e.SenderPersonId != ''
    group by Sender, Receiver
    order by -Frequency
    ;
'''

CommEdges = pd.read_sql_query(query, conn)
```


```python
# Read what we believe is the exact result (up to permutations)
CommEdges_soln = pd.read_csv ('{}CommEdges_soln.csv'.format(DATA_PATH))

# Check that we got a data frame of the expected shape:
assert 'CommEdges' in globals ()
assert type (CommEdges) is type (pd.DataFrame ())
assert len (CommEdges) == len (CommEdges_soln)
assert set (CommEdges.columns) == set (['Sender', 'Receiver', 'Frequency'])

# Check that the results are sorted:
non_increasing = (CommEdges['Frequency'].iloc[:-1].values >= CommEdges['Frequency'].iloc[1:].values)
assert non_increasing.all ()

print ("Top 5 communicating pairs:")
display (CommEdges.head ())

assert tbeq (CommEdges, CommEdges_soln)
print ("\n(Passed!)")
```

    Top 5 communicating pairs:
    


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
      <th>Sender</th>
      <th>Receiver</th>
      <th>Frequency</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>81</td>
      <td>80</td>
      <td>1406</td>
    </tr>
    <tr>
      <th>1</th>
      <td>32</td>
      <td>80</td>
      <td>1262</td>
    </tr>
    <tr>
      <th>2</th>
      <td>87</td>
      <td>80</td>
      <td>857</td>
    </tr>
    <tr>
      <th>3</th>
      <td>80</td>
      <td>81</td>
      <td>529</td>
    </tr>
    <tr>
      <th>4</th>
      <td>80</td>
      <td>32</td>
      <td>372</td>
    </tr>
  </tbody>
</table>
</div>


    
    (Passed!)
    

**Exercise 2** (3 points). Consider any pair of people, $a$ and $b$. Suppose we don't care whether person $a$ sends and person $b$ receives or whether person $b$ sends and person $a$ receives. Rather, we only care that $\{a, b\}$ have exchanged messages.

That is, the previous exercise computed a _directed_ graph, $G = \left(g_{a,b}\right)$, where $g_{a,b}$ is the number of times (or "frequency") that person $a$ was the sender and person $b$ was the receiver. Instead, suppose we wish to compute its _symmetrized_ or _undirected_ version, $H = G + G^T$.

Write some code that computes $H$ and stores it in a Pandas data frame named `CommPairs` with the columns, `A`, `B`, and `Frequency`. Per the definition of $H$, the `Frequency` column should combine frequencies from $G$ and $G^T$ accordingly.

**Solution**

This can be solved using `pandas` or `sqlite`. Perhaps here it is better to stay in `SQL`. However, here are a couple of methods to do it in `pandas` as well.

```Python
# Method 1
G = CommEdges.rename(columns = {'Sender': 'A', 'Receiver': 'B'})
GT = CommEdges.rename(columns = {'Sender': 'B', 'Receiver': 'A'})
H = pd.merge(G, GT, on = ['A', 'B'], suffixes = ('_G', '_GT'))
H['Frequency'] = H['Frequency_G'] + H['Frequency_GT']
del H['Frequency_G']
del H['Frequency_GT']
CommPairs = H
```

```Python
# Method 2
CommPairs = (
    CommEdges
        .merge(CommEdges, left_on = ['Sender', 'Receiver'], right_on = ['Receiver', 'Sender'])
        .assign(Frequency = lambda row: row['Frequency_x'] + row['Frequency_y'])
        .filter(['Sender_x', 'Receiver_x', 'Frequency'])
        .rename(columns = {'Sender_x' : 'A', 'Receiver_x' : 'B'})
)
```


```python
query = '''
    with g as (
        select
            e.SenderPersonId as Sender,
            er.PersonId as Receiver,
            count(*) as Frequency
        from EmailReceivers er, Emails e
        where e.Id = er.EmailId and e.SenderPersonId != ''
        group by Sender, Receiver
    )
    select
        g1.Sender as A,
        g1.Receiver as B,
        g1.Frequency + g2.Frequency as Frequency
    from g g1, g g2
    where g1.Sender = g2.Receiver 
        and  g1.Receiver = g2.Sender 
        --and A < B -- if you want to exclude duplicates
    order by Frequency desc
    ;
'''

CommPairs = pd.read_sql_query(query, conn)
```

```SQL
--Here is a more elegant way to count round trip communications
select
    min(e.SenderPersonId, er.PersonId) as A, -- In other databases, instead of min, you wuold have to use `least`
    max(e.SenderPersonId, er.PersonId) as B, -- In other databases, instead of max, you wuold have to use `greatest`
    count(*) Frequency
from Emails as e, EmailReceivers as er
where er.EmailId = e.Id
group by A, B
having min(e.SenderPersonId) <> max(e.SenderPersonId)
order by Frequency desc
;
```



```python
CommPairs_soln = pd.read_csv ('{}CommPairs_soln.csv'.format(DATA_PATH))

assert 'CommPairs' in globals ()
assert type (CommPairs) is type (pd.DataFrame ())
assert len (CommPairs) == len (CommPairs_soln)

print ("Most frequently communicating pairs:")
display (CommPairs.sort_values (by='Frequency', ascending=False).head (10))

assert tbeq (CommPairs, CommPairs_soln)
print ("\n(Passed!)")
```

    Most frequently communicating pairs:
    


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
      <th>A</th>
      <th>B</th>
      <th>Frequency</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>80</td>
      <td>81</td>
      <td>1935</td>
    </tr>
    <tr>
      <th>1</th>
      <td>81</td>
      <td>80</td>
      <td>1935</td>
    </tr>
    <tr>
      <th>2</th>
      <td>32</td>
      <td>80</td>
      <td>1634</td>
    </tr>
    <tr>
      <th>3</th>
      <td>80</td>
      <td>32</td>
      <td>1634</td>
    </tr>
    <tr>
      <th>4</th>
      <td>80</td>
      <td>87</td>
      <td>1206</td>
    </tr>
    <tr>
      <th>5</th>
      <td>87</td>
      <td>80</td>
      <td>1206</td>
    </tr>
    <tr>
      <th>6</th>
      <td>80</td>
      <td>116</td>
      <td>580</td>
    </tr>
    <tr>
      <th>7</th>
      <td>116</td>
      <td>80</td>
      <td>580</td>
    </tr>
    <tr>
      <th>8</th>
      <td>80</td>
      <td>194</td>
      <td>413</td>
    </tr>
    <tr>
      <th>9</th>
      <td>194</td>
      <td>80</td>
      <td>413</td>
    </tr>
  </tbody>
</table>
</div>


    
    (Passed!)
    

**Exercise 3** (3 points). Starting with a copy of `CommPairs`, named `CommPairsNamed`, add two additional columns that contain the names of the communicators. Place these values in columns named `A_name` and `B_name` in `CommPairsNamed`.

**Solution**

Again, this can be solved using `pandas` and `sqlite`. Here are a couple of methods to solve it using `pandas` as well.

```Python
# Methods 1
CommPairsNamed = CommPairs.copy ()

CommPairsNamed = pd.merge(CommPairsNamed, Persons, left_on = ['A'], right_on = ['Id'])
CommPairsNamed.rename(columns = {'Name' : 'A_name'}, inplace = True)
del CommPairsNamed['Id']

CommPairsNamed = pd.merge(CommPairsNamed, Persons, left_on = ['B'], right_on = ['Id'])
CommPairsNamed.rename(columns = {'Name' : 'B_name'}, inplace = True)
del CommPairsNamed['Id']
```

```Python
# Method 2
CommPairsNamed = (
    CommPairs
        .merge(Persons, left_on = 'A', right_on = 'Id')
        .merge(Persons, left_on = 'B', right_on = 'Id')
        .filter(['A', 'B', 'Frequency', 'Name_x', 'Name_y'])
        .rename(columns = {'Name_x' : 'A_name', 'Name_y' : 'B_name'})
)
```


```python
query = '''
    with g as (
        select
            e.SenderPersonId as Sender,
            er.PersonId as Receiver,
            count(*) as Frequency
        from EmailReceivers er, Emails e
        where e.Id = er.EmailId and e.SenderPersonId != ''
        group by Sender, Receiver
    )
    select
        g1.Sender as A,
        g1.Receiver as B,
        g1.Frequency + g2.Frequency as Frequency,
        p1.Name as A_name,
        p2.Name as B_name
    from g g1, g g2, Persons p1, Persons p2
    where g1.Sender = g2.Receiver
        and  g1.Receiver = g2.Sender
        and g1.Sender = p1.Id
        and g1.Receiver = p2.Id
        --and A < B -- if you want to exclude duplicates
    order by Frequency desc
    ;
'''

CommPairsNamed = pd.read_sql_query(query, conn)
```


```python
CommPairsNamed_soln = pd.read_csv ('{}CommPairsNamed_soln.csv'.format(DATA_PATH))

assert 'CommPairsNamed' in globals ()
assert type (CommPairsNamed) is type (pd.DataFrame ())
assert set (CommPairsNamed.columns) == set (['A', 'A_name', 'B', 'B_name', 'Frequency'])

print ("Top few entries:")
CommPairsNamed.sort_values (by=['Frequency', 'A', 'B'], ascending=False, inplace=True)
display (CommPairsNamed.head (10))

assert tbeq (CommPairsNamed, CommPairsNamed_soln)
print ("\n(Passed!)")
```

    Top few entries:
    


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
      <th>A</th>
      <th>B</th>
      <th>Frequency</th>
      <th>A_name</th>
      <th>B_name</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>1</th>
      <td>81</td>
      <td>80</td>
      <td>1935</td>
      <td>Huma Abedin</td>
      <td>Hillary Clinton</td>
    </tr>
    <tr>
      <th>0</th>
      <td>80</td>
      <td>81</td>
      <td>1935</td>
      <td>Hillary Clinton</td>
      <td>Huma Abedin</td>
    </tr>
    <tr>
      <th>3</th>
      <td>80</td>
      <td>32</td>
      <td>1634</td>
      <td>Hillary Clinton</td>
      <td>Cheryl Mills</td>
    </tr>
    <tr>
      <th>2</th>
      <td>32</td>
      <td>80</td>
      <td>1634</td>
      <td>Cheryl Mills</td>
      <td>Hillary Clinton</td>
    </tr>
    <tr>
      <th>5</th>
      <td>87</td>
      <td>80</td>
      <td>1206</td>
      <td>Jake Sullivan</td>
      <td>Hillary Clinton</td>
    </tr>
    <tr>
      <th>4</th>
      <td>80</td>
      <td>87</td>
      <td>1206</td>
      <td>Hillary Clinton</td>
      <td>Jake Sullivan</td>
    </tr>
    <tr>
      <th>7</th>
      <td>116</td>
      <td>80</td>
      <td>580</td>
      <td>Lauren Jiloty</td>
      <td>Hillary Clinton</td>
    </tr>
    <tr>
      <th>6</th>
      <td>80</td>
      <td>116</td>
      <td>580</td>
      <td>Hillary Clinton</td>
      <td>Lauren Jiloty</td>
    </tr>
    <tr>
      <th>9</th>
      <td>194</td>
      <td>80</td>
      <td>413</td>
      <td>Sidney Blumenthal</td>
      <td>Hillary Clinton</td>
    </tr>
    <tr>
      <th>8</th>
      <td>80</td>
      <td>194</td>
      <td>413</td>
      <td>Hillary Clinton</td>
      <td>Sidney Blumenthal</td>
    </tr>
  </tbody>
</table>
</div>


    
    (Passed!)
    

When you are all done, it's good practice to close the database. The following will do that for you.


```python
conn.close ()
```

**Fin!** If you've reached this point and all tests above pass, you are ready to submit your solution to this problem. Don't forget to save you work prior to submitting.
