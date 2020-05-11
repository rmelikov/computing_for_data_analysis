# Problem 14: Scraping data from "FiveThirtyEight"

There are a ton of fun interactive visualizations at the website, [FiveThirtyEight](http://fivethirtyeight.com). For example, consider the one that tracks the US President's approval ratings: https://projects.fivethirtyeight.com/trump-approval-ratings/

Here is a screenshot of the interactive graph it contains:

![538 DJT Approval Meter](https://cse6040.gatech.edu/datasets/538-djt-pop/538-demo.png)

In it, you can select each day ("movable cursor") and get information about the approval ratings for that day.

As it turns out, this visualization is implemented in JavaScript and all of the individual data items are embedded within the web page itself. For example, here is a 132-page PDF file, which is the source code for the web page taken on September 6, 2018: [PDF file](https://cse6040.gatech.edu/datasets/538-djt-pop/2018-09-06.pdf). The raw data being rendered in the visualization starts on page 50.

Of course, that means you can use your Python-fu to try to extract this data for your own purposes! Indeed, that is your task for this problem.

> Although the data in this problem comes from an HTML file with embedded JavaScript, you do **not** need to know anything about HTML or JavaScript to solve this problem. It is purely an exercise of rudimentary Python and computational problem solving.

# Reading the raw HTML file

Let's read the raw contents of the FiveThirtyEight approval ratings page (i.e., the same contents as the PDF) into a variable named `raw_html`.

> Like the groceries problem in Notebook 2, this cell contains a bunch of code for getting the data file you need, which you can ignore.


```python
def download(url, local_file, overwrite=False):
    import os, requests
    if not os.path.exists(local_file) or overwrite:
        print("Downloading: {} ...".format(url))
        r = requests.get(url)
        with open(local_file, 'wb') as f:
            f.write(r.content)
        return True
    return False # File existed already

def get_checksum(local_file):
    import io, hashlib
    with io.open(local_file, 'rb') as f:
        body = f.read()
        body_checksum = hashlib.md5(body).hexdigest()
        return body_checksum

def download_or_load_locally(file, local_dir="", url_base=None, checksum=None):
    if url_base is None: url_base = "https://cse6040.gatech.edu/datasets/"
    local_file = "{}{}".format(local_dir, file)
    remote_url = "{}{}".format(url_base, file)
    download(remote_url, local_file)
    if checksum is not None:
        body_checksum = get_checksum(local_file)
        assert body_checksum == checksum, \
            "Downloaded file '{}' has incorrect checksum: '{}' instead of '{}'".format(local_file,
                                                                                       body_checksum,
                                                                                       checksum)        
    print("'{}' is ready!".format(file))
    
def on_vocareum():
    import os
    return os.path.exists('.voc')

if on_vocareum():
    URL_BASE = None
    DATA_PATH = "./resource/asnlib/publicdata/538-djt-pop/"
else:
    URL_BASE = "https://cse6040.gatech.edu/datasets/538-djt-pop/"
    DATA_PATH = ""
datasets = {'2018-09-06.html': '291a7c1cbf15575a48b0be8d77b7a1d6'}

for filename, checksum in datasets.items():
    download_or_load_locally(filename, url_base=URL_BASE, local_dir=DATA_PATH, checksum=checksum)

with open('{}{}'.format(DATA_PATH, '2018-09-06.html')) as fp:
    raw_html = fp.read()
print("\n(All data appears to be ready.)")
```

    '2018-09-06.html' is ready!
    
    (All data appears to be ready.)
    

**File snippets.** Run the following code cell. It takes the `raw_html` string and prints the substring just around the start of the raw data you'll need, i.e., starting at page 50 of the PDF:


```python
sample_offset, sample_len = 69950, 1500
print(raw_html[sample_offset:sample_offset+sample_len])
```

    Page = {};
    interactivePage.navSlug = 'approval';
    var pathPrefix="/trump-approval-ratings/";
    var subgroup="All polls";
    var showMoreCutoff=5;
    var approval=[{"date":"2017-01-23","future":false,"subgroup":"All polls","approve_estimate":"45.46693","approve_hi":"50.88971","approve_lo":"40.04416","disapprove_estimate":"41.26452","disapprove_hi":"46.68729","disapprove_lo":"35.84175"},{"date":"2017-01-24","future":false,"subgroup":"All polls","approve_estimate":"45.44264","approve_hi":"50.82922","approve_lo":"40.05606","disapprove_estimate":"41.87849","disapprove_hi":"47.26508","disapprove_lo":"36.49191"},{"date":"2017-01-25","future":false,"subgroup":"All polls","approve_estimate":"47.76497","approve_hi":"52.66397","approve_lo":"42.86596","disapprove_estimate":"42.52911","disapprove_hi":"47.42811","disapprove_lo":"37.63011"},{"date":"2017-01-26","future":false,"subgroup":"All polls","approve_estimate":"44.37598","approve_hi":"48.93261","approve_lo":"39.81936","disapprove_estimate":"41.06081","disapprove_hi":"45.61743","disapprove_lo":"36.50418"},{"date":"2017-01-27","future":false,"subgroup":"All polls","approve_estimate":"44.13586","approve_hi":"48.70494","approve_lo":"39.56679","disapprove_estimate":"41.67268","disapprove_hi":"46.24175","disapprove_lo":"37.1036"},{"date":"2017-01-28","future":false,"subgroup":"All polls","approve_estimate":"43.87527","approve_hi":"48.46821","approve_lo":"39.28233","disapprove_estimate":"41.91362","disapprove_hi":"46.50656","disapprove_lo":"37.32067
    

Run the following code cell to see the end of the raw data region.


```python
sample_end = 257500
print(raw_html[sample_end:sample_end+sample_len])
```

    s","approve_estimate":"41.46994","approve_hi":"53.69857","approve_lo":"29.24131","disapprove_estimate":"51.94407","disapprove_hi":"63.94288","disapprove_lo":"39.94526"},{"date":"2019-05-10","future":true,"subgroup":"All polls","approve_estimate":"41.47093","approve_hi":"53.72246","approve_lo":"29.2194","disapprove_estimate":"51.94225","disapprove_hi":"63.96438","disapprove_lo":"39.92012"},{"date":"2019-05-11","future":true,"subgroup":"All polls","approve_estimate":"41.4719","approve_hi":"53.74633","approve_lo":"29.19748","disapprove_estimate":"51.94044","disapprove_hi":"63.98589","disapprove_lo":"39.895"},{"date":"2019-05-12","future":true,"subgroup":"All polls","approve_estimate":"41.47285","approve_hi":"53.77016","approve_lo":"29.17555","disapprove_estimate":"51.93866","disapprove_hi":"64.0074","disapprove_lo":"39.86993"},{"date":"2019-05-13","future":true,"subgroup":"All polls","approve_estimate":"41.47378","approve_hi":"53.79396","approve_lo":"29.15361","disapprove_estimate":"51.9369","disapprove_hi":"64.02892","disapprove_lo":"39.84487"},{"date":"2019-05-14","future":true,"subgroup":"All polls","approve_estimate":"41.47469","approve_hi":"53.81773","approve_lo":"29.13165","disapprove_estimate":"51.93515","disapprove_hi":"64.05045","disapprove_lo":"39.81984"}];
      </script>
      <div class="container">
       <div id="footer">
        <div class="notes">
         <p>
          When the dates of tracking polls from the same pollster overlap, only the most recent version is shown.
         </p>
       
    

Please make the following observations about the file snippets shown above:

- The raw data of approval ratings begins with the text, `'var approval=['` and ends with a closing square bracket, `']'`. No other square brackets appear between these two.
- Each "data point" or "data record" is encoded in JavaScript Object Notation (JSON), which is essentially the same as a Python dictionary. That is, it is enclosed in curly brackets, `{...}` and contains a number of key-value pairs. These include the date (`"date":"yyyy-mm-dd"`), approval and disapproval rating estimates (`"approve_estimate":"45.46693"` and `"disapprove_estimate":"41.26452"`), as well as upper and lower error bounds (`"..._hi"` and `"..._lo"`). The estimates correspond to the green (approval) and orange (disapproval) lines, and the error bounds form the shaded regions around those lines.
- Each data record includes a key named `"future"`. That's because FiveThirtyEight has projected the ratings into the future, so some records correspond to observed values (`"future":false`) while others correspond to extrapolated values (`"future":true`).

In addition, for the exercises below, you may assume the data records are encoded in the same way, e.g., the fields appear in the same order and there are no variations in punctuation or whitespace from what you see in the above snippets.

## Your task: Extracting the approval ratings

**Exercise 0** (1 point). Recall that the data begins with `'var approval=[...'` and ends with a closing square bracket, `']'`. Complete the function, `extract_approval_raw(html)`, below. The input variable, `html`, is a string corresponding to the raw HTML file. Your function should return the substring beginning immediately **after** the opening square bracket and up to, but **excluding**, the last square bracket. It should return exactly that substring from the file, and should not otherwise modify it.

> While you don't have to use regular expressions for this problem, if you wish to, observe that the cell below imports the `re` module.


```python
import re

def extract_approval_raw(html):
    assert isinstance(html, str), "`html` is not a string."
    match = re.search(r'var\s+approval\s*=\s*\[([^\]]*)\];', html)
    if match:
        return match.groups(0)[0]
    return ''
    
raw_data = extract_approval_raw(raw_html)
print("type(raw_data) == {}   (should be a string!)\n".format(type(raw_data)))
print("=== First and last 300 characters ===\n{}\n   ...   \n{}".format(raw_data[:300], raw_data[-300:]))
```

    type(raw_data) == <class 'str'>   (should be a string!)
    
    === First and last 300 characters ===
    {"date":"2017-01-23","future":false,"subgroup":"All polls","approve_estimate":"45.46693","approve_hi":"50.88971","approve_lo":"40.04416","disapprove_estimate":"41.26452","disapprove_hi":"46.68729","disapprove_lo":"35.84175"},{"date":"2017-01-24","future":false,"subgroup":"All polls","approve_estimat
       ...   
    e_estimate":"51.9369","disapprove_hi":"64.02892","disapprove_lo":"39.84487"},{"date":"2019-05-14","future":true,"subgroup":"All polls","approve_estimate":"41.47469","approve_hi":"53.81773","approve_lo":"29.13165","disapprove_estimate":"51.93515","disapprove_hi":"64.05045","disapprove_lo":"39.81984"}
    


```python
# Test cell: `test__extract_approval_raw` (1 point)

raw_data = extract_approval_raw(raw_html)
assert isinstance(raw_data, str), "Your function did not return a string!"
assert len(raw_data) == 188678, "Did your function return all of the substring? It should have {} characters, but your function has {} instead, making it a bit too {}.".format(188678, len(raw_data), "short" if len(raw_data) < 188678 else "long")

print("\n(Passed!)")
```

    
    (Passed!)
    

**Exercise 1** (3 points). Complete the function, `extract_approval_estimates(data_substr)`, below. It takes as input a string, `data_substr`, which would be a data substring like the one returned by `extract_approval_raw()`. It should return a **dictionary** where

- each key is a date, stored as a string in the form `'2017-01-23'` (**without the quotes**);
- each corresponding value is the approval rating, **stored as a `float`**.

For example, executing

```python
    approvals = extract_approval_estimates(data_substr)
    print(type(approvals['2017-01-23']), approvals['2017-01-23'])
```

should display `<class 'float'> 45.46693`. (Refer to the first entry of ratings from the PDF, which begins on page 50.)

Also, your function should keep **only** records for which `"future":false`, that is, **it should not include the extrapolated values**.

You may make the following assumptions:

1. Dates are *not* duplicated.
2. All encountered dates have valid approval estimates.
3. All data records are encoded using the same pattern as shown in the file snippets above.


```python
def extract_approval_estimates(data_substr):
    assert isinstance(data_substr, str), "`data_substr` is not a string."
    assert data_substr[0] == '{' and data_substr[-1] == '}', "Input may be malformed."
    estimates = {}
    pattern = r"\"date\":\"(\d{4}-\d{2}-\d{2})\",\"future\":(false|true),.*\"approve_estimate\":\"([^\"]*)\""
    for record in data_substr.split('}'):
        matches = re.search(pattern, record)
        if matches:
            date, future, rating = matches.groups()
            if future == "true": continue
            assert date not in estimates, "Duplicate date detected!"
            estimates[date] = float(rating)
    return estimates

approvals = extract_approval_estimates(raw_data)
print("Found {} data records.".format(len(approvals)))
```

    Found 592 data records.
    


```python
# Test cell 0: test__extract_approval_estimates__0 (2 points)

num_approvals_expected, min_date_expected, max_date_expected = 592, '2017-01-23', '2018-09-06'

approvals = extract_approval_estimates(raw_data)
assert isinstance(approvals, dict), "Your function returned an object of type {} instead of a dictionary (type `dict`).".format(type(approvals))
assert len(approvals) == num_approvals_expected, \
       "Your function should have found {} records but has {} instead, which is too {}.".format(num_approvals_expected,
                                                                                                len(approvals),
                                                                                                "few" if len(approvals) < num_approvals_expected else "many")
for date_expected, date_occurred in zip([min_date_expected, max_date_expected],
                                        [min(approvals.keys()), max(approvals.keys())]):
    assert date_occurred == date_expected, "A record for {} is missing!".format(date_expected)

print("\n(Passed!)")
```

    
    (Passed!)
    


```python
# Test cell 1: test__extract_approval_estimates__1__HIDDEN (1 points)

def sample_ratings(ratings, k):
    from random import sample
    ratings_sample = []
    for date in sample(ratings.keys(), k):
        ratings_sample.append((date, ratings[date]))
    return ratings_sample

if False:
    print(sample_ratings(approvals, 20))

import math
sample_approvals = [('2017-06-04', 38.89021), ('2017-10-30', 37.26442), ('2017-03-07', 44.3455), ('2017-02-19', 44.10547), ('2017-05-18', 39.45036), ('2017-09-04', 37.50347), ('2017-03-26', 42.08798), ('2018-06-29', 41.75614), ('2017-10-24', 37.1561), ('2017-11-23', 38.42519), ('2017-11-06', 37.69767), ('2017-06-17', 38.7223), ('2018-02-17', 41.3879), ('2017-05-16', 39.90671), ('2017-03-18', 42.70947), ('2018-07-17', 42.0509), ('2017-10-31', 37.24255), ('2017-02-28', 42.93886), ('2018-08-17', 41.97367), ('2018-08-28', 41.37738)]
for date, value in sample_approvals:
    assert date in approvals, "Approvals is missing a record for the date '{}'!".format(date)
    assert math.isclose(approvals[date], value, abs_tol=1e-5), \
           "Approval rating for {} should be {}, not {}.".format(date, value, approvals[date])
    
print("\n(Passed!)")
```

    
    (Passed!)
    

**Exercise 2** (2 points). Complete the function, `extract_disapproval_estimates(data_substr)`, below. It should behave just like `extract_approval_estimates()` except that it should extract disapproval estimates rather than approval estimates.

For instance, executing

```python
disapprovals = extract_disapproval_estimates(raw_data)
print(type(disapprovals['2017-01-23']), disapprovals['2017-01-23'])
```

should display, `<class 'float'> 41.26452`.


```python
def extract_disapproval_estimates(data_substr):
    assert isinstance(data_substr, str), "`data_substr` is not a string."
    assert data_substr[0] == '{' and data_substr[-1] == '}', "Input may be malformed."
    estimates = {}
    pattern = r"\"date\":\"(\d{4}-\d{2}-\d{2})\",\"future\":(false|true),.*\"disapprove_estimate\":\"([^\"]*)\""
    for record in data_substr.split('}'):
        matches = re.search(pattern, record)
        if matches:
            date, future, rating = matches.groups()
            if future == "true": continue
            assert date not in estimates, "Duplicate date detected!"
            estimates[date] = float(rating)
    return estimates

disapprovals = extract_disapproval_estimates(raw_data)
print("Found {} data records.".format(len(disapprovals)))
```

    Found 592 data records.
    


```python
# Test cell 0: test__extract_disapproval_estimates__0 (1 point)

disapprovals = extract_disapproval_estimates(raw_data)
assert isinstance(disapprovals, dict), "Your function returned an object of type {} instead of a dictionary (type `dict`).".format(type(disapprovals))

for date in approvals.keys():
    assert date in disapprovals, "The date '{}' is missing.".format(date)

print("\n(Passed!)")
```

    
    (Passed!)
    


```python
# Test cell 1: test__extract_disapproval_estimates__1__HIDDEN (1 point)

if False:
    print(sample_ratings(disapprovals, 20))

import math
sample_disapprovals = [('2018-05-27', 52.33014), ('2017-07-18', 55.6564), ('2017-04-22', 52.33886), ('2017-05-14', 53.41823), ('2018-07-18', 52.83702), ('2018-03-31', 53.42228), ('2018-05-21', 52.68593), ('2017-05-18', 54.28648), ('2017-07-04', 54.46843), ('2018-08-31', 54.46505), ('2018-07-20', 52.74246), ('2018-02-13', 53.64082), ('2017-07-07', 55.12906), ('2017-12-22', 57.08867), ('2018-06-20', 51.4082), ('2017-12-20', 56.98474), ('2017-09-27', 54.94124), ('2018-06-13', 52.53515), ('2018-02-20', 53.43631), ('2017-06-25', 55.20745)]
for date, value in sample_disapprovals:
    assert date in disapprovals, "Disapprovals is missing a record for the date '{}'!".format(date)
    assert math.isclose(disapprovals[date], value, abs_tol=1e-5), \
           "Disapproval rating for {} should be {}, not {}.".format(date, value, disapprovals[date])
    
print("\n(Passed!)")
```

    
    (Passed!)
    

**Exercise 3** (1 point). Complete the function, `filter_dates_by_month(month_str, dates)`, below, so that it filters a list of dates by a particular target month and year. Its inputs are as follows.

- The `month_str` argument is a string of the form `yyyy-mm`. This string encodes a target month and year.
- The `dates` argument is a Python set of dates of the form `yyyy-mm-dd`. Think of these as the keys in the approval and disapproval ratings dictionaries, for instance.

The function should return a **Python set** containing only those elements of `dates` that match the given `month_str`. For example, the call

```python
  filter_dates_by_month('2018-07', {'2018-05-02', '2018-07-05', '2018-07-01', '2017-07-02', '2016-07-31'})
```

would return

```python
  {'2018-07-05', '2018-07-01'}
```


```python
def filter_dates_by_month(month_str, dates):
    assert isinstance(month_str, str), "`month_str` is of type {}, rather than `str`.".format(type(month_str))
    assert re.match(r'\d{4}-\d{2}$', month_str), "`month_str == '{}'` is invalid.".format(month_str)
    return {date for date in dates if re.match(month_str + r'-\d{2}', date)}
    # Alternative: `return {date for date in dates if date[:7] == month_str}`

```


```python
# Test cell: `test__filter_dates_by_month` (1 point)

def test__filter_dates_by_month():
    from random import sample
    padded_digits = lambda a, b: sample(range(a, b), 1)[0]
    rand_year = lambda: padded_digits(0, 10000)
    rand_month = lambda: padded_digits(1, 13)
    rand_day = lambda m: padded_digits(1, 28 if m == 2 else 31 if m in [1, 3, 5, 7, 8, 10, 12] else 30)
    coin_flip = lambda: sample([False, True], 1)[0]
    
    yyyy_0, mm_0 = rand_year(), rand_month()
    month_str = '{:04d}-{:02d}'.format(yyyy_0, mm_0)
    all_dates = set()
    answer = set()
    for _ in range(10):
        match = coin_flip()
        if match:
            yyyy, mm = yyyy_0, mm_0
        else:
            yyyy, mm = rand_year(), rand_month()
        date = '{:04d}-{:02d}-{:02d}'.format(yyyy, mm, rand_day(mm))
        all_dates |= {date}
        if match:
            answer |= {date}
    filtered_dates = filter_dates_by_month(month_str, all_dates)
    print("All dates: {}".format(all_dates))
    print("Target month: {}".format(month_str))
    print("Your matches: {}".format(filtered_dates))
    if filtered_dates != answer:
        print("*** Your solution is incorrect! ***")
        excesses = filtered_dates - answer
        omissions = answer - filtered_dates
        if excesses:
            print("It erroneously includes {}.".format(excesses))
        if omissions:
            print("It erroneously omits {}.".format(omissions))
        assert False
    else:
        print("==> Looks good!\n")

for k in range(5):
    print("=== Test case {} ===".format(k))
    test__filter_dates_by_month()
```

    === Test case 0 ===
    All dates: {'4138-06-13', '9135-09-08', '5077-02-27', '5669-08-23', '4138-06-10', '5702-09-08', '4138-06-18', '4867-04-23', '5342-05-29', '4138-06-08'}
    Target month: 4138-06
    Your matches: {'4138-06-10', '4138-06-08', '4138-06-13', '4138-06-18'}
    ==> Looks good!
    
    === Test case 1 ===
    All dates: {'8975-02-19', '6263-06-28', '9942-09-09', '8268-07-14', '9211-05-21', '4064-08-25', '9942-09-18', '8266-11-06', '1882-12-30', '9942-09-22'}
    Target month: 9942-09
    Your matches: {'9942-09-22', '9942-09-18', '9942-09-09'}
    ==> Looks good!
    
    === Test case 2 ===
    All dates: {'3475-11-05', '7378-09-17', '4687-11-27', '8803-03-16', '8598-09-01', '8803-03-18', '8803-03-12', '8803-03-01', '8803-03-02', '8803-03-21'}
    Target month: 8803-03
    Your matches: {'8803-03-16', '8803-03-18', '8803-03-01', '8803-03-12', '8803-03-02', '8803-03-21'}
    ==> Looks good!
    
    === Test case 3 ===
    All dates: {'9273-11-12', '8249-01-22', '8165-10-20', '9273-11-19', '5669-08-23', '1036-03-26', '9273-11-03', '9273-11-11', '9273-11-05', '0098-04-27'}
    Target month: 9273-11
    Your matches: {'9273-11-12', '9273-11-19', '9273-11-03', '9273-11-11', '9273-11-05'}
    ==> Looks good!
    
    === Test case 4 ===
    All dates: {'0484-06-26', '6863-04-14', '5170-05-30', '6863-04-24', '1666-10-30', '0497-05-02', '3303-08-18', '6863-04-09', '6418-07-22', '6863-04-20'}
    Target month: 6863-04
    Your matches: {'6863-04-20', '6863-04-09', '6863-04-24', '6863-04-14'}
    ==> Looks good!
    
    

**Exercise 4** (3 points). Let the _discrepancy rating_ of a given day be that day's approval rating minus its disapproval rating. Complete the function, `avg_discrepancy_in_month(month_str, approvals, disapprovals)`, below, so that it returns the **average** daily discrepancy rating for a given month.

- The `month_str` argument is a string of the form `yyyy-mm`. This string encodes a target month and year.
- The `approvals` argument is a dictionary, whose keys are dates and whose values are approval ratings.
- The `disapprovals` argument is a dictionary, whose keys are dates and whose values are disapproval ratings.

You may assume that if a date is in `approvals` then it is also in `disapprovals`, and vice-versa. You can ignore missing dates; that is, compute the averages based on whatever dates exist in the input dictionaries.


```python
def avg_discrepancy_in_month(month_str, approvals, disapprovals):
    assert isinstance(month_str, str), "`month_str` is of type {}, rather than `str`.".format(type(month_str))
    assert re.match(r'\d{4}-\d{2}$', month_str), "`month_str == '{}'` is invalid.".format(month_str)
    assert all([date in disapprovals for date in approvals]), "`disapprovals` is missing dates that appear in `approvals`."
    assert all([date in approvals for date in disapprovals]), "`approvals` is missing dates that appear in `disapprovals`."
    days = filter_dates_by_month(month_str, set(approvals.keys()))
    discrepancies = {day: approvals[day] - disapprovals[day] for day in days}
    return avg_rating(discrepancies)
    
def avg_rating(ratings, days=None):
    if days is None:
        days = ratings.keys()
    assert all([day in ratings for day in days]), "`ratings` is missing days!"
    ratings_subset = [ratings[day] for day in days]
    if len(ratings_subset) > 0:
        return sum(ratings_subset) / len(ratings_subset)
    return 0.0

for month in ['2017-01', '2018-09']:
    disc = avg_discrepancy_in_month(month, approvals, disapprovals)
    print("Average daily discrepancy for {}: {}".format(month, disc))
```

    Average daily discrepancy for 2017-01: 2.799275555555555
    Average daily discrepancy for 2018-09: -14.008345
    


```python
# Test cell: test__avg_discrepancy_in_month__0 (1 point)

import math
for month, disc_true in [('2017-01', 2.799275555555555), ('2018-09', -14.008344999999997)]:
    disc = avg_discrepancy_in_month(month, approvals, disapprovals)
    assert math.isclose(disc, disc_true, abs_tol=1e-2), \
           "Your average daily discrepancy is {} instead of {}.".format(disc, disc_true)
    
print("\n(Passed!)")
```

    
    (Passed!)
    


```python
# Test cell: test__avg_discrepancy_in_month__1__HIDDEN (2 points)

all_complete_months = ['2017-02', '2017-03', '2017-04', '2017-05', '2017-06', '2017-07', '2017-08', '2017-09',
                       '2017-10', '2017-11', '2017-12', '2018-01', '2018-02', '2018-03', '2018-04', '2018-05',
                       '2017-06', '2018-07', '2018-08']

if False: # Generate solutions
    solutions = [(month, avg_discrepancy_in_month(month, approvals, disapprovals)) for month in all_complete_months]
else: # Assume solutions
    solutions = [('2017-02', -5.236897857142858), ('2017-03', -7.590843548387097), ('2017-04', -11.404569), ('2017-05', -13.227069354838711), ('2017-06', -16.525405666666668), ('2017-07', -16.39675064516129), ('2017-08', -19.508681612903224), ('2017-09', -16.797917), ('2017-10', -18.355616129032256), ('2017-11', -17.916010999999994), ('2017-12', -19.313219354838708), ('2018-01', -16.359479999999998), ('2018-02', -13.414909642857141), ('2018-03', -13.186816129032259), ('2018-04', -13.382937666666669), ('2018-05', -10.298456774193546), ('2017-06', -16.525405666666668), ('2018-07', -11.030132580645164), ('2018-08', -11.177329354838706)]
print(solutions)

import math
for month, disc_true in solutions:
    disc = avg_discrepancy_in_month(month, approvals, disapprovals)
    assert math.isclose(disc, disc_true, abs_tol=1e-2), \
           "Your average daily discrepancy is {} instead of {}.".format(disc, disc_true)

print("\n(Passed!)")
```

    [('2017-02', -5.236897857142858), ('2017-03', -7.590843548387097), ('2017-04', -11.404569), ('2017-05', -13.227069354838711), ('2017-06', -16.525405666666668), ('2017-07', -16.39675064516129), ('2017-08', -19.508681612903224), ('2017-09', -16.797917), ('2017-10', -18.355616129032256), ('2017-11', -17.916010999999994), ('2017-12', -19.313219354838708), ('2018-01', -16.359479999999998), ('2018-02', -13.414909642857141), ('2018-03', -13.186816129032259), ('2018-04', -13.382937666666669), ('2018-05', -10.298456774193546), ('2017-06', -16.525405666666668), ('2018-07', -11.030132580645164), ('2018-08', -11.177329354838706)]
    
    (Passed!)
    

**Fin!** Remember to test your solutions by running them as the autograder will: restart the kernel and run all cells from "top-to-bottom." Also remember to submit to the autograder; otherwise, you will **not** get credit for your hard work!
