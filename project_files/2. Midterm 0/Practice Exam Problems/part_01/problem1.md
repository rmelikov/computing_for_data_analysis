# Problem 1: Boozy Containers

This problem is a review of Python's built-in _[container data structures](https://docs.python.org/3/tutorial/datastructures.html)_, or simply, _containers_. These include lists, sets, tuples, and dictionaries.

Below, there are four (4) exercises, numbered 0-3, which relate to basic principles of using containers. They are worth a total of ten (10) points.

## The dataset: Student alcohol consumption

The data files for this problem pertain to a study of [student alcohol consumption](https://www.kaggle.com/uciml/student-alcohol-consumption) and its effects on academic performance. The following cell downloads these files (if they aren't already available).


```python
import requests
import os
import hashlib
import io

def fn(file):
    if os.path.exists('.voc'):
        file = f'./resource/asnlib/publicdata/{file}'
    return file

def download(file, url_suffix=None, checksum=None):
    if url_suffix is None:
        url_suffix = file
        
    if not os.path.exists(file):
        url = 'https://cse6040.gatech.edu/datasets/{}'.format(url_suffix)
        print("Downloading: {} ...".format(url))
        r = requests.get(url)
        with open(file, 'w', encoding=r.encoding) as f:
            f.write(r.text)
            
    if checksum is not None:
        with io.open(file, 'r', encoding='utf-8', errors='replace') as f:
            body = f.read()
            body_checksum = hashlib.md5(body.encode('utf-8')).hexdigest()
            assert body_checksum == checksum, \
                "Downloaded file '{}' has incorrect checksum: '{}' instead of '{}'".format(file, body_checksum, checksum)
    
    print("'{}' is ready!".format(file))
    
datasets = {'student-mat.csv': '83dc97a218a3055f51cfca1e76b29036',
            'student-por.csv': 'c5fe725d1436c73e5bc16fe8c2618bf9'}

for filename, checksum in datasets.items():
    download(fn(filename), url_suffix='ksac/{}'.format(filename), checksum=checksum)
    
print("\n(All data appears to be ready.)")
```

    'student-mat.csv' is ready!
    'student-por.csv' is ready!
    
    (All data appears to be ready.)
    

Here is some code to show you the first few lines of each file:


```python
def dump_head(filename, max_lines=5):
    from sys import stdout
    from math import log10, floor
    lines = []
    with open(filename) as fp:
        for _ in range(max_lines):
            lines.append(fp.readline())
    stdout.write("\n=== First {} lines of: '{}' ===\n\n".format(max_lines, filename))
    for line_num, text in enumerate(lines):
        fmt = "[{{:0{}d}}] {{}}".format(floor(log10(max_lines))+1)
        stdout.write(fmt.format(line_num, text))
    stdout.write('.\n.\n.\n')

dump_head(fn('student-mat.csv'));
dump_head(fn('student-por.csv'));
```

    
    === First 5 lines of: 'student-mat.csv' ===
    
    [0] school,sex,age,address,famsize,Pstatus,Medu,Fedu,Mjob,Fjob,reason,guardian,traveltime,studytime,failures,schoolsup,famsup,paid,activities,nursery,higher,internet,romantic,famrel,freetime,goout,Dalc,Walc,health,absences,G1,G2,G3
    [1] GP,F,18,U,GT3,A,4,4,at_home,teacher,course,mother,2,2,0,yes,no,no,no,yes,yes,no,no,4,3,4,1,1,3,6,5,6,6
    [2] GP,F,17,U,GT3,T,1,1,at_home,other,course,father,1,2,0,no,yes,no,no,no,yes,yes,no,5,3,3,1,1,3,4,5,5,6
    [3] GP,F,15,U,LE3,T,1,1,at_home,other,other,mother,1,2,3,yes,no,yes,no,yes,yes,yes,no,4,3,2,2,3,3,10,7,8,10
    [4] GP,F,15,U,GT3,T,4,2,health,services,home,mother,1,3,0,no,yes,yes,yes,yes,yes,yes,yes,3,2,2,1,1,5,2,15,14,15
    .
    .
    .
    
    === First 5 lines of: 'student-por.csv' ===
    
    [0] school,sex,age,address,famsize,Pstatus,Medu,Fedu,Mjob,Fjob,reason,guardian,traveltime,studytime,failures,schoolsup,famsup,paid,activities,nursery,higher,internet,romantic,famrel,freetime,goout,Dalc,Walc,health,absences,G1,G2,G3
    [1] GP,F,18,U,GT3,A,4,4,at_home,teacher,course,mother,2,2,0,yes,no,no,no,yes,yes,no,no,4,3,4,1,1,3,4,0,11,11
    [2] GP,F,17,U,GT3,T,1,1,at_home,other,course,father,1,2,0,no,yes,no,no,no,yes,yes,no,5,3,3,1,1,3,2,9,11,11
    [3] GP,F,15,U,LE3,T,1,1,at_home,other,other,mother,1,2,0,yes,no,no,no,yes,yes,yes,no,4,3,2,2,3,3,6,12,13,12
    [4] GP,F,15,U,GT3,T,4,2,health,services,home,mother,1,3,0,no,yes,no,yes,yes,yes,yes,yes,3,2,2,1,1,5,0,14,14,14
    .
    .
    .
    

**What is this data?** Each of the two file fragments shown above is in _comma-separated values (CSV) format_. Each encodes a data table about students, their alcohol consumption, test scores, and other demographic attributes, as explained later.

The first row is a list of column headings, separated by commas. These are the attributes.

Each subsequent row corresponds to the data for a particular student.

The numbers in brackets (e.g., `[0]`, `[1]`) are only line numbers and do not exist in the actual file.

Of the two files, the first is data for a math class; the second file is for a Portuguese language class.

The `read_csv()` function, below, will read in these files using Python's [`csv` module](https://docs.python.org/3/library/csv.html).

The important detail is that this function returns two lists: a list of the column names, `header`, and a list of lists, named `data_rows`, which holds the rows. In particular, `data_rows[i]` is a list of the values that appear in the `i`-th data row, which you can see by comparing the sample output below to the raw file data above.


```python
def read_csv(filename):
    with open(filename) as fp:
        from csv import reader
        data_rows = list(reader(fp))
    header = data_rows.pop(0)
    return (header, data_rows)

# Read the math class data
math_header, math_data_rows = read_csv(fn('student-mat.csv'))

# Read the Portuguese class data
port_header, port_data_rows = read_csv(fn('student-por.csv'))

# Print a sample of the data
def print_sample(header, data, num_rows=5):
    from math import floor, log10
    fmt = "Row {{:0{}d}}: {{}}".format(floor(log10(num_rows))+1)
    string_rows = [fmt.format(i, str(r)) for i, r in enumerate(data[:num_rows])]
    print("--> Header ({} columns): {}".format(len(header), header))
    print("\n--> First {} of {} rows:\n{}".format(num_rows,
                                                  len(data),
                                                  '\n'.join(string_rows)))

print("=== Math data ===\n")
print_sample(math_header, math_data_rows)
```

    === Math data ===
    
    --> Header (33 columns): ['school', 'sex', 'age', 'address', 'famsize', 'Pstatus', 'Medu', 'Fedu', 'Mjob', 'Fjob', 'reason', 'guardian', 'traveltime', 'studytime', 'failures', 'schoolsup', 'famsup', 'paid', 'activities', 'nursery', 'higher', 'internet', 'romantic', 'famrel', 'freetime', 'goout', 'Dalc', 'Walc', 'health', 'absences', 'G1', 'G2', 'G3']
    
    --> First 5 of 395 rows:
    Row 0: ['GP', 'F', '18', 'U', 'GT3', 'A', '4', '4', 'at_home', 'teacher', 'course', 'mother', '2', '2', '0', 'yes', 'no', 'no', 'no', 'yes', 'yes', 'no', 'no', '4', '3', '4', '1', '1', '3', '6', '5', '6', '6']
    Row 1: ['GP', 'F', '17', 'U', 'GT3', 'T', '1', '1', 'at_home', 'other', 'course', 'father', '1', '2', '0', 'no', 'yes', 'no', 'no', 'no', 'yes', 'yes', 'no', '5', '3', '3', '1', '1', '3', '4', '5', '5', '6']
    Row 2: ['GP', 'F', '15', 'U', 'LE3', 'T', '1', '1', 'at_home', 'other', 'other', 'mother', '1', '2', '3', 'yes', 'no', 'yes', 'no', 'yes', 'yes', 'yes', 'no', '4', '3', '2', '2', '3', '3', '10', '7', '8', '10']
    Row 3: ['GP', 'F', '15', 'U', 'GT3', 'T', '4', '2', 'health', 'services', 'home', 'mother', '1', '3', '0', 'no', 'yes', 'yes', 'yes', 'yes', 'yes', 'yes', 'yes', '3', '2', '2', '1', '1', '5', '2', '15', '14', '15']
    Row 4: ['GP', 'F', '16', 'U', 'GT3', 'T', '3', '3', 'other', 'other', 'home', 'father', '1', '2', '0', 'no', 'yes', 'yes', 'no', 'yes', 'yes', 'no', 'no', '4', '3', '2', '1', '2', '5', '4', '6', '10', '10']
    

The function only separates the fields by comma; it doesn't do any additional postprocessing. So all the data elements are treated as strings, even though you can see that some are clearly numerical values. You'll need this fact in **Exercise 3**.

**Exercise 0** (2 points). Complete the function, `lookup_value(col_name, row_id, header, data_rows)`, to look up a particular value in the data when stored as shown above. In particular, the parameters of the function are

- `col_name`: Name of the column, e.g., `'school'`, `'address'`, `'freetime'`.
- `row_id`: The desired row number, starting at 0 (the first _data_ row).
- `header`, `data_rows`: The list of column names and data rows, respectively.

For example, consider the math data shown above. Then,

```python
    lookup_value('age', 0, math_header, math_data_rows) == '18'
    lookup_value('G2', 3, math_header, math_data_rows) == '14'
```

> **Hint.** Consider [`list.index()`](https://docs.python.org/3/tutorial/datastructures.html).


```python
def lookup_value(col_name, row_id, header, data_rows):
    assert col_name in header, "{} not in {}".format(col_name, header)
    assert 0 <= row_id < len(data_rows)
    #return [(h, d) for h, d in zip(header, data_rows[row_id]) if h == col_name][0][1]
    col_id = header.index(col_name)
    return data_rows[row_id][col_id]

```


```python
# Test cell: `exercise_0_test`

print("Checking examples from above...")
assert lookup_value('age', 0, math_header, math_data_rows) == '18'
assert lookup_value('G2', 3, math_header, math_data_rows) == '14'

print("Checking some random examples...")

# Generate random test cases
if False:
    for _ in range(5):
        from random import sample, randint
        col_name = sample(math_header, 1)[0]
        row_id = randint(0, len(math_data_rows)-1)
        value = lookup_value(col_name, row_id, math_header, math_data_rows)
        print("assert lookup_value('{}', {}, math_header, math_data_rows) == '{}'".format(col_name, row_id, value))                                                                                 

    for _ in range(5):
        from random import sample, randint
        col_name = sample(port_header, 1)[0]
        row_id = randint(0, len(port_data_rows)-1)
        value = lookup_value(col_name, row_id, port_header, port_data_rows)
        print("assert lookup_value('{}', {}, port_header, port_data_rows) == '{}'".format(col_name, row_id, value))                                                                                 

assert lookup_value('famsize', 143, math_header, math_data_rows) == 'LE3'
assert lookup_value('absences', 198, math_header, math_data_rows) == '24'
assert lookup_value('G3', 246, math_header, math_data_rows) == '13'
assert lookup_value('guardian', 175, math_header, math_data_rows) == 'mother'
assert lookup_value('paid', 362, math_header, math_data_rows) == 'no'
assert lookup_value('romantic', 87, port_header, port_data_rows) == 'no'
assert lookup_value('famsup', 246, port_header, port_data_rows) == 'yes'
assert lookup_value('Walc', 294, port_header, port_data_rows) == '1'
assert lookup_value('famsize', 431, port_header, port_data_rows) == 'GT3'
assert lookup_value('studytime', 224, port_header, port_data_rows) == '4'

print("\n(Passed!)")
```

    Checking examples from above...
    Checking some random examples...
    
    (Passed!)
    

**Exercise 1** (3 points). Suppose we wish to extract a list of all values stored in a given column of the table. Complete the function, `lookup_column_values(col, header, data_rows)`, which takes as input the column name `col`, list of column names `header`, and rows `data_rows`, and returns a list of all the values stored in that column.

For example, the first five entries of the returned list when reference the `'age'` column of the math class data should satisfy:

```python
  values = lookup_column_values('age', math_header, math_data_rows)
  assert values[:5] == ['18', '17', '15', '15', '16']
```


```python
def lookup_column_values(col, header, data_rows):
    assert col in header
    col_id = header.index(col)
    return [L[col_id] for L in data_rows]
```


```python
# Test cell: `exercise_1_test`

# The example:
values = lookup_column_values('age', math_header, math_data_rows)

print("First five values of 'age' column in the math data:")
print(values[:5])
assert values[:5] == ['18', '17', '15', '15', '16']

if False:
    from random import sample
    for col in sample(math_header, 5):
        values = lookup_column_values(col, math_header, math_data_rows)
        print("assert ''.join(lookup_column_values('{}', math_header, math_data_rows)) == '{}'".format(col, ''.join(values)))
    for col in sample(port_header, 5):
        values = lookup_column_values(col, port_header, port_data_rows)
        print("assert ''.join(lookup_column_values('{}', port_header, port_data_rows)) == '{}'".format(col, ''.join(values)))
        
print("\nSpot-checking some additional cases...")
assert ''.join(lookup_column_values('activities', math_header, math_data_rows)) == 'nononoyesnoyesnononoyesnoyesyesnononoyesyesyesyesnonoyesyesyesnononoyesyesnoyesyesyesnoyesyesyesyesyesyesnoyesnoyesyesnoyesnoyesnononononoyesyesyesyesnoyesyesyesyesyesyesyesnonononononoyesyesyesyesnoyesnoyesnonoyesyesnonoyesyesnonoyesnoyesyesyesyesnoyesnoyesyesyesnoyesnonoyesyesyesyesyesyesnoyesyesyesyesyesnonoyesyesyesnonoyesnoyesyesnoyesnononoyesnoyesnoyesnoyesyesnonononononononoyesyesnonoyesnoyesnonoyesnoyesnoyesyesnonononoyesyesyesyesyesyesyesyesyesyesyesyesyesyesnoyesyesyesnononoyesyesyesnoyesnoyesnonoyesyesnonoyesnonoyesnoyesyesnonoyesnononononoyesyesyesnonoyesyesyesnoyesyesyesyesyesyesnoyesyesnoyesnoyesnoyesyesnononononononononoyesnoyesyesnoyesyesnonoyesyesyesyesyesyesyesnoyesyesnoyesnononoyesyesyesyesnoyesyesnonononoyesyesyesnononoyesnoyesnonononononoyesyesyesyesnoyesyesyesnononononoyesyesyesnononoyesnonoyesyesnoyesnoyesnoyesyesyesnononononoyesyesyesnononononoyesnononononoyesnoyesnonononoyesnoyesnonononononoyesyesyesyesnononoyesnoyesyesyesyesnononoyesyesnoyesnonononono'
assert ''.join(lookup_column_values('reason', math_header, math_data_rows)) == 'coursecourseotherhomehomereputationhomehomehomehomereputationreputationcoursecoursehomehomereputationreputationcoursehomereputationothercoursereputationcoursehomehomeotherhomehomehomereputationcoursecoursehomeotherhomereputationcoursereputationhomehomecoursecoursecoursecoursehomereputationhomeothercourseotherothercourseotherotherreputationreputationhomecourseothercoursereputationhomereputationcoursereputationcoursereputationreputationreputationcoursereputationreputationhomehomecoursereputationhomecoursecoursehomereputationhomehomereputationcoursereputationreputationreputationhomereputationhomehomereputationhomereputationcoursereputationcourseotherothercoursehomecoursereputationcoursehomehomeothercoursereputationhomecoursereputationcoursereputationhomecoursereputationcoursehomecoursecoursehomehomehomecoursereputationcoursecoursecoursecoursecoursecoursecoursecoursecoursecoursecoursecoursereputationcoursecoursehomecoursehomecoursecoursecoursecoursecoursereputationhomecoursecoursereputationcoursecoursecoursecoursecoursecoursecoursecoursecoursecoursehomehomereputationcoursereputationreputationhomereputationcoursereputationreputationothercoursehomehomereputationreputationreputationotherothercoursereputationhomecoursecourseotherreputationhomecoursehomehomehomereputationhomereputationcoursereputationreputationhomecourseotherhomereputationreputationhomereputationhomeotherreputationreputationhomehomecoursereputationreputationotherhomehomereputationcoursereputationcoursecoursereputationcoursereputationreputationhomereputationhomehomecoursereputationcoursecoursecoursecoursecoursecoursecourseothercourseothercoursereputationothercoursecoursecoursereputationreputationhomecoursehomecoursecoursehomehomereputationotherreputationreputationreputationhomereputationhomehomereputationcoursehomehomereputationcoursehomehomereputationhomecoursereputationotherreputationreputationreputationhomereputationreputationreputationreputationhomereputationhomereputationhomehomehomereputationreputationhomereputationcoursereputationreputationreputationhomeothercoursereputationhomereputationcoursecoursecoursecoursecoursecoursecoursecoursehomecoursereputationcoursecoursecoursecoursecoursehomehomecoursecoursehomehomehomehomehomehomehomehomecourseothercoursecoursereputationcoursehomecoursecoursehomehomecourseotherreputationhomecoursecourseotherothercoursecoursecourseotherreputationcourseotherhomeotherhomecoursereputationhomecoursecoursehomereputationhomeotherhomeotherhomeotherreputationcoursecoursecoursecoursecoursecoursecoursecourse'
assert ''.join(lookup_column_values('guardian', math_header, math_data_rows)) == 'motherfathermothermotherfathermothermothermothermothermothermotherfatherfathermotherothermothermothermothermotherfathermotherfathermothermothermothermothermothermothermothermothermothermothermothermothermotherfathermothermothermothermothermotherotherfatherfatherfathermothermothermothermotherfathermothermotherfathermothermothermothermothermotherfathermothermotherfatherfathermotherfathermothermothermothermotherfatherfathermothermothermothermothermothermothermothermothermotherfathermothermothermotherfatherfathermothermotherfathermothermothermotherfathermothermothermothermothermothermothermothermotherfathermothermothermotherfathermotherfatherfathermothermotherfathermothermothermotherfatherfatherfatherfatherfathermotherfatherfathermothermotherfathermotherothermothermotherfathermotherfatherfathermothermothermotherothermothermotherfatherfathermothermothermotherfatherfathermothermothermothermothermotherothermothermothermothermothermothermotherfathermothermothermotherfathermothermotherfatherfathermothermotherfathermothermothermothermothermothermotherfathermothermothermothermotherfathermothermothermothermothermothermothermothermothermotherothermotherfatherfathermotherfathermothermothermothermothermothermothermothermothermothermothermothermotherothermothermothermothermothermothermotherfatherfathermothermothermothermothermothermothermothermotherfathermothermotherfatherfathermotherfathermothermotherfathermothermotherfathermothermothermothermotherothermotherfathermothermothermothermotherothermothermothermothermothermothermothermotherfatherfathermothermothermothermothermothermothermotherfathermotherotherfathermothermothermothermothermothermotherfathermothermothermothermothermothermotherfathermothermothermothermothermothermothermothermothermothermotherotherotherfathermothermotherfathermotherfatherotherotherotherotherfatherotherotherotherotherotherotherothermothermotherfathermothermothermothermotherfatherfathermothermothermothermothermothermotherfathermotherothermothermotherothermothermothermotherotherfathermotherfathermothermothermothermothermothermotherothermothermotherothermotherfatherfathermotherfatherfathermothermothermothermotherfathermothermothermotherfatherfatherotherfathermothermothermothermotherothermothermothermotherfathermotherfathermotherfathermothermothermothermothermotherothermotherothermotherfather'
assert ''.join(lookup_column_values('famsup', math_header, math_data_rows)) == 'noyesnoyesyesyesnoyesyesyesyesyesyesyesyesyesyesyesyesnonoyesnoyesyesyesyesnoyesyesyesyesyesnoyesyesyesyesyesyesyesyesyesyesnoyesyesnoyesyesyesyesnoyesnonoyesyesyesyesyesyesnoyesnoyesyesyesyesyesyesnoyesnoyesyesnonoyesyesyesnoyesnoyesnoyesyesnoyesyesnoyesyesyesyesyesyesnoyesyesyesyesyesyesnoyesyesyesyesnoyesnononoyesyesnoyesnonoyesyesyesnononoyesyesnoyesyesyesnoyesyesnoyesnonoyesyesyesnoyesyesyesyesyesnononoyesyesyesnonoyesnoyesnoyesnononoyesnonoyesyesyesyesyesnoyesnononoyesyesyesnoyesyesyesyesyesnoyesnonononoyesnononoyesyesyesyesyesyesyesyesyesyesnoyesyesyesyesyesyesyesnonoyesyesyesyesyesnoyesyesnonoyesyesnonoyesyesyesyesnonononononoyesnonoyesnononoyesnonoyesnononoyesyesyesnonoyesyesnonoyesyesnonoyesyesyesyesyesyesyesyesnonoyesnononoyesnoyesnoyesyesnoyesyesyesnoyesyesyesyesyesyesyesyesnoyesyesyesyesnoyesnoyesnoyesnoyesnonoyesyesyesyesyesyesyesyesnononoyesyesyesyesyesnononononoyesyesyesnoyesnoyesyesnonoyesyesyesnoyesnoyesyesyesyesnononononoyesyesnoyesnoyesyesyesnoyesnonononoyesnonoyesnonononononoyesnoyesnoyesnononono'
assert ''.join(lookup_column_values('nursery', math_header, math_data_rows)) == 'yesnoyesyesyesyesyesyesyesyesyesyesyesyesyesyesyesyesyesyesyesyesyesyesyesnoyesyesyesyesnoyesyesnonoyesyesyesyesyesnonoyesyesyesyesyesyesyesnoyesyesyesyesyesyesyesyesyesyesyesnoyesyesyesyesnoyesyesnoyesyesnoyesyesyesyesyesyesyesnonoyesyesnoyesyesyesnoyesyesyesyesyesyesyesnoyesnoyesyesyesnoyesyesyesyesyesyesyesyesyesyesyesyesyesyesyesyesyesnoyesyesyesyesyesyesnoyesyesyesnoyesyesyesyesyesnoyesyesyesyesyesyesyesyesyesnoyesyesyesyesyesyesyesyesyesnonoyesyesyesyesyesyesnoyesyesnoyesnoyesyesnoyesyesnonoyesyesyesyesyesyesyesyesyesyesnoyesyesyesnoyesyesyesyesyesyesyesyesyesnoyesnoyesyesyesyesyesyesyesyesyesyesyesyesyesyesyesyesyesyesyesnoyesnonoyesyesyesyesyesyesyesyesnoyesyesyesyesyesnoyesnoyesyesnoyesnoyesnonononoyesyesyesyesyesyesnononoyesyesyesyesyesyesyesyesnonoyesyesnoyesyesyesyesyesyesyesnonoyesyesyesyesyesyesyesyesyesyesyesyesyesyesyesyesyesnoyesyesyesyesyesnonoyesnononoyesyesyesnoyesyesyesyesnoyesyesnoyesyesnoyesyesyesnoyesyesyesyesyesyesyesyesyesyesyesyesyesyesyesnoyesyesyesyesnoyesyesyesnoyesyesnoyesyesnoyesyesyesyesnoyesyesyesyesyesyesnoyesyesnononoyesyesyesyesyesnoyesyesyesnononoyes'
assert ''.join(lookup_column_values('sex', port_header, port_data_rows)) == 'FFFFFMMFMMFFMMMFFFMMMMMMFFMMMMMMMMMFMMFFFMMMFFFMMFFFMFFFFMMFFFFFFFMFFFMMFMFMMFMFMMFMFFFFMMFFFFMFMFFFMMMFMFFMMFMFFMMMMMMMFMFMFMFFMMFFFFFFMFMFMFMFFFMFFMFMMFFMFFFFFMMMMFMFMMFMMMMMMMMMMFFFMMMFFMFMMMMMFFFMMMFMFFMMMFMMFFFFFFFFFFFFFMFMFFFMFFFFFMFFFMMFFMMMMMMFFFFFMMFFFFFMFFFMMMMFMFFFMFMMMMMMMMMMFFFMFFFMMMFFFMMFFMMMFMFFFFMFFFFMFFFMMMMFFFFMFFMMMFFMMFFFMFMFFFMMMFFFFFFFFMFFFFFFMFFFFFFFFMMMFFFMMFFMFFFFFFFMMFFFMMFFFFFFMMFMFMFFMFMMFFFFFFFFFMMFFFFFFFFFMMMMMMMMFFFMFFFFFMFFFFFFMFMFMMFFFMMFFMFFFFFFFFFFFMFFFFFFFMFFMMMFFFFFMFFFFFFFFFFMFFFMFMFMFFMMMFMMMMFFFFFMMMFMFFMMMMFFMFMMMMFMMMMMMMMMFFMFMMMMMFFFFFFFFFFFFFFMFFMMMFFFFFFFFFFFMFFFMFFFFFMMFFFMFFFFFFFFMMMMMFFFFFFMM'
assert ''.join(lookup_column_values('absences', port_header, port_data_rows)) == '42600602002000061022600022680240200440428168001446242004002080200000200062020400101402422464662602246042120216010448244121046461420624020084010862124122420210840626240424220026616006260014640084220100000400000048440032860010660160084163002402161424204150106216104001209442020000020040181614266410421416844807422101010742128084420022042444202822220860841216102286615628001022182012100000020000200000000000222228289042140221222400000020420020680680400618044001041660119020218510513101054110640050000404020040748086320403004021002690110420420211040201601660412250020001282000800001902800806042201051802001181128501228002000243542000600202049648912888404200050300320003008544002005200540202006400000038404411104040000444664'
assert ''.join(lookup_column_values('G2', port_header, port_data_rows)) == '1111131413121213161214121312141713148121312131011111211121111151412121114131213111115101111121712121314912121314151315161013131215129101511911131111111391111911121211111510911131314121311121313916131016101014101514111017914151411131413121311911101412119101212139111114913111111910149118131113117121011121191191013913881010108869887101712169179131411129911101218131513914139121111101215131115101412151313911151312121213111116910131210121391681811101512131313912161013108788141011139791414151214151114119101610128912811813121012151010121211131610111215101112128161111129121613171416129121215810101113171312131218131517161819151513141718151371618101510101212161512151210111215141212910810131118141414181012141211109151411913131314141216121216141312151611131211141715171015101817141117101311121010178111110614910711111609139109111016131511108810141110910121411118913121212111481113810799101010710109991181099131411131671281413119119171797668161457810869141499111010715121114131110119119119151110181314911111410913129913091070118895911111099109886117861010813121318171801113138120100181110110141110121413171512141310168105101181711814169187158717991115121011'
assert ''.join(lookup_column_values('age', port_header, port_data_rows)) == '18171515161616171515151515151516161617161515161615161515161615151515161515161515161515151615161615151615151515161515151616161616151615161515161515161615151617161515151515151615161616151616151516161616161615151515151615161516161515161516171515151616161515191616151817151716161615151716181816161615151515161615161715151515151615181615191715171816161617171516171716161616161617161617161716171616171716171616171716171716161717161716161617171617161616171917161817171718171717171618161817171817171717161616171616181818181717171617171817171717171616171717171816171722181618161816171718171918171818191817172018181817181718171818181918181718171718181817191818171718181718171817181818171718171918181818171717171920191818171718181718191818171717171819191817171717181817171717171819181718181818171718171818171818181817171719172118181717182120191718181918201816161515161516161717151615151617151617151516151517161516161616161516171616151615151618151615161516161615151615161719171516161716181918181916171616161716161716171619172017171617171816161816161816161616181617171617161717161616151515161617171617171717161717161719171716161717181718161716161818191816191617201818191718191819181717171817171818171818181718181817181818181917171818191817181717181818181717181817181818171718181918181718181719181817181918181718'
assert ''.join(lookup_column_values('famsize', port_header, port_data_rows)) == 'GT3GT3LE3GT3GT3LE3LE3GT3LE3GT3GT3GT3LE3GT3GT3GT3GT3GT3GT3LE3GT3GT3LE3LE3GT3GT3GT3GT3LE3GT3GT3GT3GT3LE3GT3GT3LE3GT3GT3GT3LE3LE3GT3GT3LE3LE3LE3GT3GT3GT3LE3LE3LE3GT3LE3GT3GT3GT3LE3GT3GT3GT3LE3GT3LE3LE3GT3GT3LE3LE3GT3GT3GT3GT3GT3GT3GT3GT3GT3GT3GT3GT3LE3LE3GT3GT3LE3GT3GT3LE3GT3GT3LE3GT3LE3GT3GT3GT3GT3GT3GT3GT3GT3GT3GT3GT3GT3GT3GT3LE3LE3GT3GT3LE3GT3GT3GT3GT3GT3GT3GT3GT3LE3GT3GT3GT3LE3GT3GT3GT3GT3GT3LE3GT3LE3GT3GT3GT3GT3GT3GT3GT3LE3GT3GT3GT3LE3GT3GT3GT3GT3GT3LE3GT3LE3GT3GT3LE3GT3GT3GT3GT3LE3LE3LE3GT3GT3GT3GT3LE3GT3GT3GT3LE3LE3GT3LE3GT3LE3GT3GT3GT3GT3GT3GT3GT3LE3GT3LE3LE3GT3GT3GT3GT3GT3GT3GT3LE3GT3GT3GT3LE3GT3GT3GT3GT3GT3GT3GT3LE3GT3LE3GT3LE3GT3GT3GT3GT3GT3GT3GT3GT3GT3GT3GT3LE3GT3GT3LE3LE3GT3LE3GT3GT3GT3GT3GT3GT3GT3GT3GT3LE3LE3GT3LE3GT3GT3GT3LE3GT3LE3GT3GT3LE3LE3GT3GT3GT3GT3LE3GT3GT3LE3GT3LE3LE3LE3LE3LE3LE3GT3GT3GT3GT3GT3GT3GT3GT3GT3GT3LE3GT3GT3GT3GT3GT3GT3LE3LE3LE3GT3LE3GT3LE3GT3LE3GT3GT3LE3GT3GT3GT3GT3GT3GT3GT3LE3GT3GT3GT3GT3GT3GT3LE3GT3GT3GT3LE3GT3GT3GT3GT3GT3LE3GT3LE3LE3LE3GT3GT3GT3GT3GT3GT3GT3LE3GT3GT3LE3LE3GT3GT3LE3GT3LE3LE3LE3GT3GT3GT3LE3GT3GT3GT3GT3GT3GT3GT3LE3GT3GT3GT3GT3GT3GT3GT3LE3LE3LE3GT3LE3GT3GT3LE3GT3LE3GT3GT3GT3GT3GT3GT3GT3LE3GT3GT3GT3GT3GT3LE3GT3GT3GT3LE3GT3GT3LE3GT3GT3GT3GT3GT3GT3GT3GT3LE3GT3LE3LE3LE3GT3LE3LE3LE3GT3GT3LE3GT3GT3GT3LE3GT3GT3GT3GT3GT3GT3GT3LE3LE3GT3GT3GT3GT3LE3LE3LE3GT3GT3GT3GT3GT3LE3LE3GT3GT3LE3GT3GT3GT3GT3GT3GT3GT3GT3GT3GT3LE3GT3GT3GT3GT3GT3LE3GT3GT3GT3GT3GT3GT3GT3GT3LE3LE3LE3GT3LE3GT3GT3GT3GT3GT3LE3GT3LE3GT3GT3LE3GT3GT3GT3GT3GT3GT3LE3GT3LE3GT3GT3GT3GT3LE3LE3GT3LE3GT3GT3LE3GT3LE3GT3GT3GT3GT3GT3LE3GT3LE3GT3GT3LE3GT3GT3LE3LE3LE3GT3LE3GT3LE3GT3GT3GT3LE3LE3GT3LE3LE3LE3LE3GT3GT3GT3GT3LE3GT3GT3LE3GT3GT3LE3GT3GT3LE3GT3GT3LE3GT3GT3GT3GT3LE3LE3GT3GT3LE3GT3GT3GT3GT3GT3GT3GT3GT3GT3GT3GT3GT3GT3LE3GT3GT3LE3GT3GT3LE3GT3LE3GT3LE3GT3GT3GT3LE3GT3LE3GT3GT3LE3GT3GT3GT3LE3LE3GT3GT3GT3LE3LE3GT3GT3GT3GT3GT3GT3LE3LE3LE3LE3GT3LE3GT3GT3LE3GT3GT3GT3LE3GT3GT3LE3GT3GT3LE3GT3GT3GT3GT3GT3GT3GT3GT3GT3GT3GT3LE3GT3LE3LE3'

print("\n(Passed!)")
```

    First five values of 'age' column in the math data:
    ['18', '17', '15', '15', '16']
    
    Spot-checking some additional cases...
    
    (Passed!)
    

**Exercise 2** (1 points). Suppose we wish to get all _unique_ values in a given column. Complete the function, `get_unique_values(col, header, data_rows)`, so that it returns these values, as a **list**.

> The ordering of unique values in your output does not matter.


```python
def get_unique_values(col, header, data_rows):
    assert col in header
    col_id = header.index(col)
    return list(set(lookup_column_values(col, header, data_rows)))

```


```python
# Test cell: `exercise_2_test`

print("Checking ages...")
ages = get_unique_values('age', math_header, math_data_rows)
assert type(ages) is list, "`get_unique_values()` should return a list!"
assert len(ages) == 8
assert all([str(a) in ages for a in range(15, 23)])

print("\nSpot checking some additional cases...")

if False: # Generate test cases
    from random import sample
    for col in sample(math_header, 5):
        values = get_unique_values(col, math_header, math_data_rows)
        print("""
values = get_unique_values('{}', math_header, math_data_rows)
assert len(values) == {}
for v in {}:
    assert v in values, "'{{}}' should be in the output, but isn't.".format(v)
""".format(col, len(values), values))
        
values = get_unique_values('Fedu', math_header, math_data_rows)
assert len(values) == 5
for v in ['4', '1', '3', '0', '2']:
    assert v in values, "'{}' should be in the output, but isn't.".format(v)

values = get_unique_values('traveltime', math_header, math_data_rows)
assert len(values) == 4
for v in ['4', '1', '3', '2']:
    assert v in values, "'{}' should be in the output, but isn't.".format(v)

values = get_unique_values('paid', math_header, math_data_rows)
assert len(values) == 2
for v in ['no', 'yes']:
    assert v in values, "'{}' should be in the output, but isn't.".format(v)

values = get_unique_values('school', math_header, math_data_rows)
assert len(values) == 2
for v in ['MS', 'GP']:
    assert v in values, "'{}' should be in the output, but isn't.".format(v)

values = get_unique_values('sex', math_header, math_data_rows)
assert len(values) == 2
for v in ['F', 'M']:
    assert v in values, "'{}' should be in the output, but isn't.".format(v)

print("\n(Passed!)")
```

    Checking ages...
    
    Spot checking some additional cases...
    
    (Passed!)
    

**A simple analysis task.** The column `'Dalc'` contains the student's self-reported drinking frequency during the weekday. The values are 1 (very low amount of drinking) to 5 (very high amount of drinking); if your function above works correctly then you should see that in the output of the following cell:


```python
print("Unique values of 'Dalc':", get_unique_values('Dalc', math_header, math_data_rows))
```

    Unique values of 'Dalc': ['1', '4', '3', '2', '5']
    

Similarly, `Walc` is the self-reported drinking frequency on the same scale, but for the _weekend_ (instead of _weekday_).

Now, suppose we wish to know whether there is a relationship between these drinking frequencies and the final math grade, which appears in column `'G3'` (on a scale of 0-20, where 0 is a "low" grade and 20 is a "high" grade).

**Exercise 3** (4 points). Create a dictionary named **`dw_avg_grade`** that will help with this analysis. For this exercise, we only care about the math grades (`math_data_rows`); you can ignore the Portuguese grades (`port_data_rows`).

For your dictionary, `dw_avg_grade`, the keys and values should be defined as follows.

1. Each key should be a tuple, `(a, b)`, where `a` is the `'Dalc'` rating and `b` is the `'Walc'` rating. You should convert these ratings from strings to integers. You only need to consider keys that actually occur in the data.
2. Each corresponding value should be the average test score, rounded to two decimal places.


> **Hint.** To get you started, we've used your `lookup_column_values()` function to extract the relevant columns for analysis. From there, consider breaking this problem up into several parts:
> 1. Counting the number of occurrences of (`Dalc`, `Walc`) pairs.
> 2. Summing the grades for each pair.
> 3. Dividing the latter by the former to get the mean. Use [`round()`](https://docs.python.org/3/library/functions.html#round) to do the rounding.


```python
from collections import defaultdict # Optional, but might help

# Relevant data to analyze:
Dalc_values = [int(i) for i in lookup_column_values('Dalc', math_header, math_data_rows)]
Walc_values = [int(i) for i in lookup_column_values('Walc', math_header, math_data_rows)]
G3_values = [int(i) for i in lookup_column_values('G3', math_header, math_data_rows)]

analysis_list = list(map(list, zip(zip(Dalc_values, Walc_values), G3_values)))

pair_counts = defaultdict(int)
pair_values_sum = defaultdict(int)    

for L in analysis_list:
    key = L[0]
    value = L[1]
    pair_counts[key] += 1
    pair_values_sum[key] += value

dw_avg_grade = defaultdict(float)

for key in pair_values_sum.keys():
    dw_avg_grade[key] = round(pair_values_sum[key] / pair_counts[key], 1)
```


```python
# Test cell: `exercise_3_test`

assert type(dw_avg_grade) is dict or type(dw_avg_grade) is defaultdict, "'dw_avg_grade' should be a dictionary (or default dictionary)."

print("Your computed results:")
descending = sorted(dw_avg_grade.items(), key=lambda x: -x[1])
for (dalc, walc), g3 in descending:
    print("(Dalc={}, Walc={}): {}".format(dalc, walc, g3))
    
print("\nSpot checking some of these values...")
test_cases = [((3, 2), 12.0), ((1, 1), 10.8), ((2, 1), 0.0), ((3, 3), 9.75), ((3, 5), 10.0)]
for (d, w), g3 in test_cases:
    your_g3 = dw_avg_grade[(d, w)]
    msg = "({}, {}) == {}, but it should be {}".format(d, w, your_g3, g3)
    assert abs(your_g3 - g3) < 0.1, msg
    
print("\n(Passed!)")
```

    Your computed results:
    (Dalc=3, Walc=2): 12.0
    (Dalc=4, Walc=4): 11.3
    (Dalc=3, Walc=4): 11.2
    (Dalc=1, Walc=3): 11.1
    (Dalc=4, Walc=2): 11.0
    (Dalc=1, Walc=1): 10.8
    (Dalc=1, Walc=5): 10.8
    (Dalc=2, Walc=3): 10.7
    (Dalc=5, Walc=5): 10.7
    (Dalc=1, Walc=2): 10.4
    (Dalc=1, Walc=4): 10.3
    (Dalc=3, Walc=5): 10.0
    (Dalc=3, Walc=3): 9.8
    (Dalc=4, Walc=5): 9.8
    (Dalc=2, Walc=5): 9.2
    (Dalc=2, Walc=2): 8.7
    (Dalc=2, Walc=4): 8.3
    (Dalc=4, Walc=3): 5.0
    (Dalc=2, Walc=1): 0.0
    
    Spot checking some of these values...
    
    (Passed!)
    

**Fin!** If you've reached this point and all tests above pass, you are ready to submit your solution to this problem. Don't forget to save you work prior to submitting.
