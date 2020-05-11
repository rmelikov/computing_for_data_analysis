# Problem 5

This problem will expose you to a different but common data format (JSON), the methods that can be used to load data from JSON files, and also tests your data processing/regular expression skills. These can be valuable when you have to collect or scrape your own data in the future.

There are five exercises (numbered 0-4) worth a total of ten (10) points.

## The Data

We are going to play with a mock dataset in the *JavaScript Object Notation (JSON)* format. (You can check out the wiki intro to this format [here](https://en.wikipedia.org/wiki/JSON) ). 

JSON has it's origins on the web, where it was used as a format to send data from a server to a client, and then handled at the client side using JavaScript (hence the name). It is a common format in data analysis, and is the main format for a great variety of data sources, such as Twitter, Yelp!, Facebook, and many others.

The JSON format is a text format. From your knowledge of Python lists and dictionaries, it should be easy to understand. Let's take a quick look at an example. It's a simple database of individuals with their names, email addresses, gender, location, and lists of friends.

```json
[{
    "city": "Sitovo",
    "name": {
        "last_name": "Ricciardo",
        "first_name": "Nerta"
    },
    "id": 1,
    "email": {
        "personal": "nricciardo0@hostgator.com",
        "working": "nricciardo0@java.com"
    },
    "friends": [
        {
            "last_name": "De'Vere - Hunt",
            "first_name": "Miran"
        },
        {
            "last_name": "Fryers",
            "first_name": "Dorisa"
        },
        {
            "last_name": "Brusin",
            "first_name": "Carina"
        }
    ],
    "gender": "Female"
}, ...]
```

JSON uses a Python-like syntax to describe values. For instance, the above includes simple values, like strings and integers. It also has collections: square brackets (`[...]`) denote lists of comma-separated elements; while curly brackets (`{...}`) mark dictionaries, which contain comma-separated key-value (or "attribute-value") pairs. And just like Python, these collections may be nested: observe that in some of the key-value pairs, the values are lists or dictionaries.

Indeed, after reading this file into a Python object, you will access data elements using the same 0-based index and dictionary key-value notation. For instance, if that object is stored in a Python variable named `data`, then `data[0]["name"]["last_name"] == "Ricciardo"`.

The data you will need is pre-loaded into the Vocareum environment. The function, `fn(f)`, below, will convert filenames into fully qualified path names that point to the right places on Vocareum where the data reside. If you are running in your local environment, you may want to modify this function.

> A file `XXX` referenced in this notebook may be located at https://cse6040.gatech.edu/datasets/json/XXX.


```python
def fn(fn_base, dirname='./'):
    return '{}{}'.format(dirname, fn_base)

# Demo:
fn('MOCK_DATA.json')
```




    './MOCK_DATA.json'



### Loading JSON in Python

There are several ways in Python to read in a JSON (`.json`) file.

Like **`csv`**, Python has a built-in package called **`json`**, to operate on JSON files. (As always, check out the [documentation](https://docs.python.org/3/library/json.html).)


```python
import pandas as pd
import json

json_file = open(fn('MOCK_DATA.json'), encoding="utf8")
json_str = json_file.read()
json_data = json.loads(json_str)

# Demo:
print(json_data[0]["name"]["last_name"]) # Should be 'Ricciardo'
```

    Ricciardo
    


```python
#json.loads(
#    open('MOCK_DATA.json', encoding = 'utf').read()
#)
```

**Exercise 0** (2 points). Complete the following function, **`find_emails(data)`**, so that it returns a  to save all the `working` emails in the input file into a list named `emails` and sort it alphabetically in **descending** order.  


```python
def find_emails(data):
    
    # making sure `data` is a list
    assert type(data) == list
    
    # making sure that all items of the list are dictionaries
    assert all([type(dic) == dict for dic in data])
    
    return sorted(
        [dic['email']['working'] for dic in data],
        reverse = True
    )
```


```python
# Test cell: `find_emails`

emails = find_emails(json_data)
assert len(emails) == 1000
assert type(emails) == list
assert emails[0] == 'zwardropege@home.pl'
assert emails[726] == 'dplanfn@newyorker.com'
assert emails[349] == 'mgerbel3j@blogger.com'
assert emails[85] == 'tbenson70@salon.com'
assert emails[899] == 'bdegowe77@unblog.fr'
assert emails[181] == 'rdurbyngz@aboutads.info'
assert emails[703] == 'ebaudq4@sogou.com'
assert emails[156] == 'rsimicqg@gov.uk'
assert emails[483] == 'jsemplenj@google.ru'
assert emails[249] == 'oblitzer30@dropbox.com'
assert emails[134] == 'sbrandomo9@smh.com.au'

print("\n(Passed!)")
```

    
    (Passed!)
    

**Exercise 1** (2 points). Many of the people in tihs dataset have friends. But sadly, some of them don't. :(  

Complete the function, **`are_u_lonely(lst)`**, to find the people without friends. The input to the function will be the list created from the JSON file's contents. The function should return a list of names constructed by concatenating the first name and last name together separated by a single space.

(For example, Joanne Goodisson is one of these people. She should be an element in the list shown as `"Joanne Goodisson"`.)


```python
def are_u_lonely(lst):
    return [
        dic['name']['first_name'] + ' ' + dic['name']['last_name']
        for dic in lst 
        if not dic['friends']
    ]
```


```python
# Test cell: `are_u_lonely_test`

lonely_guys = are_u_lonely(json_data)
assert len(lonely_guys) == 171, "There are {} lonely guys in your result, shoud be 171".format(len(lonely_guys))
assert lonely_guys[2] == 'Joanne Goodisson', 'Joanne should be one of them, but she is not in your result.'
assert lonely_guys[109] == 'Violetta Swinden', 'Violetta Swinden is missing from your list.'
assert lonely_guys[143] == 'Seumas Turban', 'Seumas Turban is missing from your list.'
assert lonely_guys[78] == 'Giraldo Attard', 'Giraldo Attard is missing from your list.'
assert lonely_guys[127] == 'Gale Ryley', 'Gale Ryley is missing from your list.'
assert lonely_guys[99] == 'Pieter Dillestone', 'Pieter Dillestone is missing from your list.'
assert lonely_guys[139] == 'Latashia Greenhaugh', 'Latashia Greenhaugh is missing from your list.'
assert lonely_guys[46] == 'Melodee Malster', 'Melodee Malster is missing from your list.'
assert lonely_guys[7] == 'Georgine Limprecht', 'Georgine Limprecht is missing from your list.'
assert lonely_guys[23] == 'Ilse Tackley', 'Ilse Tackley is missing from your list.'
assert lonely_guys[29] == 'Camile Theobalds', 'Camile Theobalds is missing from your list.'

print("\n(Passed!)")
```

    
    (Passed!)
    

**Exercise 2** (2 points). Write a function **`edu_users()`**, that finds all users who are using a `'.edu'` domain email address as their personal email, and returns their user ID's as a list.


```python
def edu_users(lst):
    return [
        dic['id']
        for dic in lst
        if dic['email']['personal'].endswith('.edu')
    ]
```


```python
# Test cell: `edu_users_test`

test_users = edu_users(json_data)
assert len(test_users) == 53, "You found {} people using edu emails, but there should be 53.".format(len(test_users))
assert json_data[test_users[-1]-1]['city'] == 'Alajuela'

print("\n(Passed!)")
```

    
    (Passed!)
    

**Exercise 3** (1 point). Write a function that, given the filename of a JSON file, returns the data as a Pandas dataframe.

Pandas has a convenient function to do just that. Check the documentation for [pd.read_json()](https://pandas.pydata.org/pandas-docs/stable/generated/pandas.read_json.html) and use it to complete the function below.


```python
def pd_load_json(file_name):
    return pd.read_json(file_name)
    
pd_load_json(fn("MOCK_DATA.json")).head()
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
      <th>id</th>
      <th>name</th>
      <th>email</th>
      <th>gender</th>
      <th>city</th>
      <th>friends</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>1</td>
      <td>{'first_name': 'Nerta', 'last_name': 'Ricciardo'}</td>
      <td>{'personal': 'nricciardo0@hostgator.com', 'wor...</td>
      <td>Female</td>
      <td>Sitovo</td>
      <td>[{'first_name': 'Miran', 'last_name': 'De'Vere...</td>
    </tr>
    <tr>
      <th>1</th>
      <td>2</td>
      <td>{'first_name': 'Minerva', 'last_name': 'Little...</td>
      <td>{'personal': 'mlittlewood1@icio.us', 'working'...</td>
      <td>Female</td>
      <td>Jiaoyuan</td>
      <td>[{'first_name': 'Dian', 'last_name': 'Hounsham...</td>
    </tr>
    <tr>
      <th>2</th>
      <td>3</td>
      <td>{'first_name': 'Rosa', 'last_name': 'Casswell'}</td>
      <td>{'personal': 'rcasswell2@soup.io', 'working': ...</td>
      <td>Female</td>
      <td>Zhenchuan</td>
      <td>[{'first_name': 'Meghann', 'last_name': 'Vanna...</td>
    </tr>
    <tr>
      <th>3</th>
      <td>4</td>
      <td>{'first_name': 'Loren', 'last_name': 'Bugbee'}</td>
      <td>{'personal': 'lbugbee3@dmoz.org', 'working': '...</td>
      <td>Female</td>
      <td>Rokytne</td>
      <td>[]</td>
    </tr>
    <tr>
      <th>4</th>
      <td>5</td>
      <td>{'first_name': 'Neda', 'last_name': 'De Norman...</td>
      <td>{'personal': 'ndenormanville4@illinois.edu', '...</td>
      <td>Female</td>
      <td>Antipolo</td>
      <td>[{'first_name': 'Andrus', 'last_name': 'Szymon...</td>
    </tr>
  </tbody>
</table>
</div>




```python
# Test cell: `pd_load_json_test`

test_df = pd_load_json(fn('MOCK_DATA.json'))
assert len(test_df) == 1000
assert set(test_df.columns.tolist()) == {'city', 'email', 'friends', 'gender', 'id', 'name'}

print("\n(Passed!)")
```

    
    (Passed!)
    

**Exercise 4** (3 points). You should observe that the personal and working email addresses appear in same column? Complete the function, **`split_emails()`** below, so that it separates them into two new columns named `"personal"` and `"working"`. It should return a new dataframe that has all the same columns, but with the `"email"` column removed and replaced by these two new columns. (See the test cell if this is unclear.)

> Hint: There is a nice way of using `.apply` and `pd.Series` to accomplish this task: [Stack Overflow](https://stackoverflow.com/questions/38231591/splitting-dictionary-list-inside-a-pandas-column-into-separate-columns) is your friend!


```python
def split_emails(file_name):
    from pandas.io.json import json_normalize
    df = pd_load_json(file_name)
    return (
        df
            #.assign(personal = lambda df: df['email'].apply(lambda cell: dict(cell)['personal']))
            #.assign(working = lambda df: df['email'].apply(lambda cell: dict(cell)['working']))
            #.assign(personal = lambda df: json_normalize(df['email'])['personal'])
            #.assign(working = lambda df: json_normalize(df['email'])['working'])
            .join(json_normalize(df['email']))
            .drop('email', axis = 1)
    )
    
split_emails(fn("MOCK_DATA.json")).head()
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
      <th>id</th>
      <th>name</th>
      <th>gender</th>
      <th>city</th>
      <th>friends</th>
      <th>personal</th>
      <th>working</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>1</td>
      <td>{'first_name': 'Nerta', 'last_name': 'Ricciardo'}</td>
      <td>Female</td>
      <td>Sitovo</td>
      <td>[{'first_name': 'Miran', 'last_name': 'De'Vere...</td>
      <td>nricciardo0@hostgator.com</td>
      <td>nricciardo0@java.com</td>
    </tr>
    <tr>
      <th>1</th>
      <td>2</td>
      <td>{'first_name': 'Minerva', 'last_name': 'Little...</td>
      <td>Female</td>
      <td>Jiaoyuan</td>
      <td>[{'first_name': 'Dian', 'last_name': 'Hounsham...</td>
      <td>mlittlewood1@icio.us</td>
      <td>mlittlewood1@vimeo.com</td>
    </tr>
    <tr>
      <th>2</th>
      <td>3</td>
      <td>{'first_name': 'Rosa', 'last_name': 'Casswell'}</td>
      <td>Female</td>
      <td>Zhenchuan</td>
      <td>[{'first_name': 'Meghann', 'last_name': 'Vanna...</td>
      <td>rcasswell2@soup.io</td>
      <td>rcasswell2@apple.com</td>
    </tr>
    <tr>
      <th>3</th>
      <td>4</td>
      <td>{'first_name': 'Loren', 'last_name': 'Bugbee'}</td>
      <td>Female</td>
      <td>Rokytne</td>
      <td>[]</td>
      <td>lbugbee3@dmoz.org</td>
      <td>lbugbee3@odnoklassniki.ru</td>
    </tr>
    <tr>
      <th>4</th>
      <td>5</td>
      <td>{'first_name': 'Neda', 'last_name': 'De Norman...</td>
      <td>Female</td>
      <td>Antipolo</td>
      <td>[{'first_name': 'Andrus', 'last_name': 'Szymon...</td>
      <td>ndenormanville4@illinois.edu</td>
      <td>ndenormanville4@tamu.edu</td>
    </tr>
  </tbody>
</table>
</div>




```python
# Test cell: `Exercise 5`

test_df = split_emails(fn('MOCK_DATA.json'))
assert len(test_df) == 1000
assert set(test_df.columns.tolist()) == {'city', 'friends', 'gender', 'id', 'name', 'personal', 'working'}
assert test_df.personal[0] == 'nricciardo0@hostgator.com'
assert test_df.personal[999] == 'bbretonrr@pen.io'

print("\n(Passed!)")
```

    
    (Passed!)
    

**Fin!** You have reached the end of this problem. Don't forget to submit it before moving on.
