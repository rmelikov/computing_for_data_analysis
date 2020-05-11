# Lesson 0: SQLite

The de facto language for managing relational databases is the Structured Query Language, or SQL ("sequel").

Many commerical and open-source relational data management systems (RDBMS) support SQL. The one we will consider in this class is the simplest, called [sqlite3](https://www.sqlite.org/). It stores the database in a simple file and can be run in a "standalone" mode from the command-line. However, we will, naturally, [invoke it from Python](https://docs.python.org/3/library/sqlite3.html). But all of the basic techniques apply to any commercial SQL backend.

With a little luck, you _might_ by the end of this class understand this [xkcd comic on SQL injection attacks](http://xkcd.com/327).

> **Important note.** Due to a limitation in Vocareum's software stack, this notebook is set to use the Python 3.5 kernel (rather than a more up-to-date 3.6 or 3.7 kernel). If you are developing on your local machine and are using a different version of Python, you may need to adapt your solution before submitting to the autograder.


```python
import sys
print("=== Python version info ===\n{}".format(sys.version))
```

    === Python version info ===
    3.7.4 (tags/v3.7.4:e09359112e, Jul  8 2019, 20:34:20) [MSC v.1916 64 bit (AMD64)]
    

## Getting started

In Python, you _connect_ to an `sqlite3` database by creating a _connection object_.

**Exercise 0** (ungraded). Run this code cell to get started.


```python
import sqlite3 as db

# Connect to a database (or create one if it doesn't exist)
conn = db.connect('example.db')
```

The `sqlite` engine maintains a database as a file; in this example, the name of that file is `example.db`.

> **Important usage note!** If the named file does **not** yet exist, this code creates it. However, if the database has been created before, this same code will open it. This fact can be important when you are debugging. For example, if your code depends on the database not existing initially, then you may need to remove the file first. You're likely to encounter this situation in this notebook.

You issue commands to the database through an object called a _cursor_.


```python
# Create a 'cursor' for executing commands
c = conn.cursor()
```

A cursor tracks the current state of the database, and you will mostly be using the cursor to issue commands that modify or query the database.

## Tables and Basic Queries

The central object of a relational database is a _table_. It's identical to what you called a "tibble" in the tidy data lab: observations as rows, variables as columns. In the relational database world, we sometimes refer to rows as _items_ or _records_ and columns as _attributes_. We'll use all of these terms interchangeably in this course.

Let's look at a concrete example. Suppose we wish to maintain a database of Georgia Tech students, whose attributes are their names and Georgia Tech-issued ID numbers. You might start by creating a table named `Students` to hold this data. You can create the table using the command, [`CREATE TABLE`](https://www.sqlite.org/lang_createtable.html).

> Note: If you try to create a table that already exists, it will **fail**. If you are trying to carry out these exercises from scratch, you may need to remove any existing `example.db` file or destroy any existing table; you can do the latter with the SQL command, `DROP TABLE IF EXISTS Students`.


```python
# If this is not the first time you run this cell, 
# you need to delete the existed "Students" table first
c.execute("DROP TABLE IF EXISTS Students")

# create a table named "Students" with 2 columns: "gtid" and "name".
# the type for column "gtid" is integer and for "name" is text. 
c.execute("CREATE TABLE Students (gtid INTEGER, name TEXT)")
```




    <sqlite3.Cursor at 0x2603348f420>



To populate the table with items, you can use the command, [`INSERT INTO`](https://www.sqlite.org/lang_insert.html).


```python
c.execute("INSERT INTO Students VALUES (123, 'Vuduc')")
c.execute("INSERT INTO Students VALUES (456, 'Chau')")
c.execute("INSERT INTO Students VALUES (381, 'Bader')")
c.execute("INSERT INTO Students VALUES (991, 'Sokol')")
```




    <sqlite3.Cursor at 0x2603348f420>



**Commitment issues.** The commands above modify the database. However, these are temporary modifications and aren't actually saved to the databases until you say so. (_Aside:_ Why would you want such behavior?) The way to do that is to issue a _commit_ operation from the _connection_ object.

> There are some subtleties related to when you actually need to commit, since the SQLite database engine does commit at certain points as discussed [here](https://stackoverflow.com/questions/13642956/commit-behavior-and-atomicity-in-python-sqlite3-module). However, it's probably simpler if you remember to encode commits when you intend for them to take effect.


```python
conn.commit()
```

Another common operation is to perform a bunch of insertions into a table from a list of tuples. In this case, you can use `executemany()`.


```python
# An important (and secure!) idiom
more_students = [(723, 'Rozga'),
                 (882, 'Zha'),
                 (401, 'Park'),
                 (377, 'Vetter'),
                 (904, 'Brown')]

# '?' question marks are placeholders for the two columns in Students table
c.executemany('INSERT INTO Students VALUES (?, ?)', more_students)
conn.commit()
```

Given a table, the most common operation is a _query_, which asks for some subset or transformation of the data. The simplest kind of query is called a [`SELECT`](https://www.sqlite.org/lang_select.html).

The following example selects all rows (items) from the `Students` table.


```python
c.execute("SELECT * FROM Students")
results = c.fetchall()
print("Your results:", len(results), "\nThe entries of Students:\n", results)
```

    Your results: 9 
    The entries of Students:
     [(123, 'Vuduc'), (456, 'Chau'), (381, 'Bader'), (991, 'Sokol'), (723, 'Rozga'), (882, 'Zha'), (401, 'Park'), (377, 'Vetter'), (904, 'Brown')]
    

**Exercise 1** (2 points). Suppose we wish to maintain a second table, called `Takes`, which records classes that students have taken and the grades they earn.

In particular, each row of `Takes` stores a student by his/her GT ID, the course he/she took, and the grade he/she earned in terms of GPA (i.e. 4.0, 3.0, etc). More formally, suppose this table is defined as follows:


```python
# Run this cell
c.execute('DROP TABLE IF EXISTS Takes')
c.execute('CREATE TABLE Takes (gtid INTEGER, course TEXT, grade REAL)')
```




    <sqlite3.Cursor at 0x2603348f420>



Write a command to insert the following records into the `Takes` table.

* Vuduc: CSE 6040 - A (4.0), ISYE 6644 - B (3.0), MGMT 8803 - D (1.0)
* Sokol: CSE 6040 - A (4.0), ISYE 6740 - A (4.0)
* Chau: CSE 6040 - A (4.0), ISYE 6740 - C (2.0), MGMT 8803 - B (3.0)

(Note: See `students` table above to get the GT IDs for Vuduc, Sokol, and Chau. You don't have to write any code to retrieve their GT IDs. You can just type them in manually. However, it would be a good and extra practice for you if you can use some sql commands to retrieve their IDs.) 


```python
classes_taken = [
    (123, 'CSE 6040', 4.0),
    (123, 'ISYE 6644', 3.0),
    (123, 'MGMT 8803', 1.0),
    (991, 'CSE 6040', 4.0),
    (991, 'ISYE 6740', 4.0),
    (456, 'CSE 6040', 4.0),
    (456, 'ISYE 6740', 2.0),
    (456, 'MGMT 8803', 3.0)
]

c.executemany(
    'INSERT INTO Takes VALUES (?, ?, ?)', 
    classes_taken
)
conn.commit()

# Displays the results of your code
c.execute('SELECT * FROM Takes')
results = c.fetchall()
print("Your results:", len(results), "\nThe entries of Takes:", results)
```

    Your results: 8 
    The entries of Takes: [(123, 'CSE 6040', 4.0), (123, 'ISYE 6644', 3.0), (123, 'MGMT 8803', 1.0), (991, 'CSE 6040', 4.0), (991, 'ISYE 6740', 4.0), (456, 'CSE 6040', 4.0), (456, 'ISYE 6740', 2.0), (456, 'MGMT 8803', 3.0)]
    


```python
# Test cell: `insert_many__test`

# Close the database and reopen it
conn.close()
conn = db.connect('example.db')
c = conn.cursor()
c.execute('SELECT * FROM Takes')
results = c.fetchall()

if len(results) == 0:
    print("*** No matching records. Did you remember to commit the results? ***")
assert len(results) == 8, "The `Takes` table has {} when it should have {}.".format(len(results), 8)

assert (123, 'CSE 6040', 4.0) in results
assert (123, 'ISYE 6644', 3.0) in results
assert (123, 'MGMT 8803', 1.0) in results
assert (991, 'CSE 6040', 4.0) in results
assert (991, 'ISYE 6740', 4.0) in results
assert (456, 'CSE 6040', 4.0) in results
assert (456, 'ISYE 6740', 2.0) in results
assert (456, 'MGMT 8803', 3.0) in results

print("\n(Passed.)")
```

    
    (Passed.)
    

# Lesson 1: Join queries

The main type of query that combines information from multiple tables is the _join query_. Recall from our discussion of tibbles these four types:

- `INNER JOIN(A, B)`: Keep rows of `A` and `B` only where `A` and `B` match
- `OUTER JOIN(A, B)`: Keep all rows of `A` and `B`, but merge matching rows and fill in missing values with some default (`NaN` in Pandas, `NULL` in SQL)
- `LEFT JOIN(A, B)`: Keep all rows of `A` but only merge matches from `B`.
- `RIGHT JOIN(A, B)`: Keep all rows of `B` but only merge matches from `A`.

If you are a visual person, see [this page](https://www.codeproject.com/Articles/33052/Visual-Representation-of-SQL-Joins) for illustrations of the different join types.

In SQL, you can use the `WHERE` clause of a `SELECT` statement to specify how to match rows from the tables being joined. For example, recall that the `Takes` table stores classes taken by each student. However, these classes are recorded by a student's GT ID. Suppose we want a report where we want each student's name rather than his/her ID. We can get the matching name from the `Students` table. Here is a query to accomplish this matching:


```python
# See all (name, course, grade) tuples
query = '''
        SELECT Students.name, Takes.course, Takes.grade
        FROM Students, Takes
        WHERE Students.gtid = Takes.gtid
'''

for match in c.execute(query): # Note this alternative idiom for iterating over query results
    print(match)
```

    ('Vuduc', 'CSE 6040', 4.0)
    ('Vuduc', 'ISYE 6644', 3.0)
    ('Vuduc', 'MGMT 8803', 1.0)
    ('Chau', 'CSE 6040', 4.0)
    ('Chau', 'ISYE 6740', 2.0)
    ('Chau', 'MGMT 8803', 3.0)
    ('Sokol', 'CSE 6040', 4.0)
    ('Sokol', 'ISYE 6740', 4.0)
    

**Exercise 2** (2 points). Define a query to select only the names and grades of students _who took CSE 6040_. The code below will execute your query and store the results in a list `results1` of tuples, where each tuple is a `(name, grade)` pair; thus, you should structure your query to match this format.


```python
# Define `query` with your query:
query = '''
    SELECT Students.name, Takes.grade
    FROM Students, Takes
    WHERE Students.gtid = Takes.gtid
        and takes.course = 'CSE 6040'
'''

c.execute(query)
results1 = c.fetchall()
results1
```




    [('Vuduc', 4.0), ('Sokol', 4.0), ('Chau', 4.0)]




```python
# Test cell: `join1__test`

print ("Your results:", results1)

assert type(results1) is list
assert len(results1) == 3, "Your query produced {} results instead of {}.".format(len(results1), 3)

assert set(results1) == {('Vuduc', 4.0), ('Sokol', 4.0), ('Chau', 4.0)}

print("\n(Passed.)")
```

    Your results: [('Vuduc', 4.0), ('Sokol', 4.0), ('Chau', 4.0)]
    
    (Passed.)
    

For contrast, let's do a quick exercise that executes a [left join](http://www.sqlitetutorial.net/sqlite-left-join/).

**Exercise 3** (2 points). Execute a `LEFT JOIN` that uses `Students` as the left table, `Takes` as the right table, and selects a student's name and course grade. Write your query as a string variable named `query`, which the subsequent code will execute.


```python
# Define `query` string here:
query = '''
    select s.name, t.grade
    from students s
    left join takes t on t.gtid = s.gtid
'''

# Executes your `query` string:
c.execute(query)
matches = c.fetchall()
for i, match in enumerate(matches):
    print(i, "->", match)
```

    0 -> ('Vuduc', 1.0)
    1 -> ('Vuduc', 3.0)
    2 -> ('Vuduc', 4.0)
    3 -> ('Chau', 2.0)
    4 -> ('Chau', 3.0)
    5 -> ('Chau', 4.0)
    6 -> ('Bader', None)
    7 -> ('Sokol', 4.0)
    8 -> ('Sokol', 4.0)
    9 -> ('Rozga', None)
    10 -> ('Zha', None)
    11 -> ('Park', None)
    12 -> ('Vetter', None)
    13 -> ('Brown', None)
    


```python
# Test cell: `left_join_test`

assert set(matches) == {('Vuduc', 4.0), ('Chau', 2.0), ('Park', None), ('Vuduc', 1.0), ('Chau', 3.0), ('Zha', None), ('Brown', None), ('Vetter', None), ('Vuduc', 3.0), ('Bader', None), ('Rozga', None), ('Chau', 4.0), ('Sokol', 4.0)}
print("\n(Passed!)")
```

    
    (Passed!)
    

## Aggregations

Another common style of query is an [_aggregation_](https://www.sqlite.org/lang_aggfunc.html), which is a summary of information across multiple records, rather than the raw records themselves.

For instance, suppose we want to compute the average GPA for each unique GT ID from the `Takes` table. Here is a query that does it using `AVG` aggregator:


```python
query = '''
        SELECT gtid, AVG(grade)
        FROM Takes 
        GROUP BY gtid
'''

for match in c.execute(query):
    print(match)
```

    (123, 2.6666666666666665)
    (456, 3.0)
    (991, 4.0)
    

Some other useful SQL aggregators include `MIN`, `MAX`, `SUM`, and `COUNT`.

## Cleanup

As one final bit of information, it's good practice to shutdown the cursor and connection, the same way you close files.


```python
c.close()
conn.close()
```

**What next?** It's now a good time to look at a different tutorial which reviews this material and introduces some additional topics: [A thorough guide to SQLite database operations in Python](http://sebastianraschka.com/Articles/2014_sqlite_in_python_tutorial.html).
