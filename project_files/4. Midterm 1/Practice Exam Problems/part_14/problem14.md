# Problem 14: Kaggle ML/DS Survey

_Version 2.5_


**Pro-tips.** If your program behavior seem strange, try resetting the kernel and rerunning everything. If you mess up this notebook or just want to start from scratch, save copies of all your partial responses and use `Actions` $\rightarrow$ `Reset Assignment` to get a fresh, original copy of this notebook. (_Resetting will wipe out any answers you've written so far, so be sure to stash those somewhere safe if you intend to keep or reuse them!_)

In this problem, you'll be working with a subset of the Kaggle Machine Learning & Data Science Survey 2018 dataset. You are provided with the responses collected to the survey questions in the form of a csv. The survey dataset in the form of responses to Multiple Choice Questions (MCQs).

You are expected to clean this dataset, which is quite messy, and **recreate the questionnaire** from this dataset, i.e., the questions and the multiple choices against each question are to be extracted from the csv file. Read on for more details.

> (The whole dataset, if you're interested as data science/analytics students and practitioners, can be found here : https://www.kaggle.com/kaggle/kaggle-survey-2018)

Let's first look at the data in the CSV. Run the below cell to load the survey response dataset.


```python
import pandas as pd
data = pd.read_csv("problem1.csv", dtype="unicode")
data.head(5)
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
      <th>Q1</th>
      <th>Q2</th>
      <th>Q3</th>
      <th>Q4</th>
      <th>Q5</th>
      <th>Q6</th>
      <th>Q7</th>
      <th>Q9</th>
      <th>Q10</th>
      <th>Q12_MULTIPLE_CHOICE</th>
      <th>...</th>
      <th>Q31_Part_7</th>
      <th>Q31_Part_8</th>
      <th>Q31_Part_9</th>
      <th>Q31_Part_10</th>
      <th>Q31_Part_11</th>
      <th>Q31_Part_12</th>
      <th>Q31_OTHER_TEXT</th>
      <th>Q32</th>
      <th>Q32_OTHER</th>
      <th>Q48</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>What is your gender? - Selected Choice</td>
      <td>What is your age (# years)?</td>
      <td>In which country do you currently reside?</td>
      <td>What is the highest level of formal education ...</td>
      <td>Which best describes your undergraduate major?...</td>
      <td>Select the title most similar to your current ...</td>
      <td>In what industry is your current employer/cont...</td>
      <td>What is your current yearly compensation (appr...</td>
      <td>Does your current employer incorporate machine...</td>
      <td>What is the primary tool that you use at work ...</td>
      <td>...</td>
      <td>Which types of data do you currently interact ...</td>
      <td>Which types of data do you currently interact ...</td>
      <td>Which types of data do you currently interact ...</td>
      <td>Which types of data do you currently interact ...</td>
      <td>Which types of data do you currently interact ...</td>
      <td>Which types of data do you currently interact ...</td>
      <td>Which types of data do you currently interact ...</td>
      <td>What is the type of data that you currently in...</td>
      <td>What is the type of data that you currently in...</td>
      <td>Do you consider ML models to be "black boxes" ...</td>
    </tr>
    <tr>
      <th>1</th>
      <td>Female</td>
      <td>45-49</td>
      <td>United States of America</td>
      <td>Doctoral degree</td>
      <td>Other</td>
      <td>Consultant</td>
      <td>Other</td>
      <td>NaN</td>
      <td>I do not know</td>
      <td>Cloud-based data software &amp; APIs (AWS, GCP, Az...</td>
      <td>...</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>-1</td>
      <td>NaN</td>
      <td>-1</td>
      <td>NaN</td>
    </tr>
    <tr>
      <th>2</th>
      <td>Male</td>
      <td>30-34</td>
      <td>Indonesia</td>
      <td>Bachelor’s degree</td>
      <td>Engineering (non-computer focused)</td>
      <td>Other</td>
      <td>Manufacturing/Fabrication</td>
      <td>10-20,000</td>
      <td>No (we do not use ML methods)</td>
      <td>Basic statistical software (Microsoft Excel, G...</td>
      <td>...</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>-1</td>
      <td>NaN</td>
      <td>-1</td>
      <td>NaN</td>
    </tr>
    <tr>
      <th>3</th>
      <td>Female</td>
      <td>30-34</td>
      <td>United States of America</td>
      <td>Master’s degree</td>
      <td>Computer science (software engineering, etc.)</td>
      <td>Data Scientist</td>
      <td>I am a student</td>
      <td>0-10,000</td>
      <td>I do not know</td>
      <td>Local or hosted development environments (RStu...</td>
      <td>...</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>Text Data</td>
      <td>Time Series Data</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>-1</td>
      <td>Time Series Data</td>
      <td>-1</td>
      <td>I am confident that I can explain the outputs ...</td>
    </tr>
    <tr>
      <th>4</th>
      <td>Male</td>
      <td>35-39</td>
      <td>United States of America</td>
      <td>Master’s degree</td>
      <td>Social sciences (anthropology, psychology, soc...</td>
      <td>Not employed</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>Local or hosted development environments (RStu...</td>
      <td>...</td>
      <td>NaN</td>
      <td>Tabular Data</td>
      <td>Text Data</td>
      <td>Time Series Data</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>-1</td>
      <td>Numerical Data</td>
      <td>-1</td>
      <td>Yes, most ML models are "black boxes"</td>
    </tr>
  </tbody>
</table>
<p>5 rows × 144 columns</p>
</div>



There are a total of 144 columns in the dataset. Observe the following features of its structure:

- The question number appears as column headings, e.g., `Q1`, `Q2`, and so on.
- The first row is the text of question itself, as it appeared in the survey.
- Each one of the remaining rows is someone's response to the survey.
- For many questions in the dataset, the responses are stored across multiple columns, where each column represents a choice offered in the MCQ. For example, `Q13` has 15 different choices and is stored across 15 columns, each with the column name `Q13_Part_1`, `Q13_Part_2`,.. upto `Q13_Part_15`. 


```python
#count NAs in a dataset
data.isna().sum()
```




    Q1                    0
    Q2                    0
    Q3                    0
    Q4                    9
    Q5                   89
                      ...  
    Q31_Part_12       13764
    Q31_OTHER_TEXT        0
    Q32                2951
    Q32_OTHER             0
    Q48                1533
    Length: 144, dtype: int64



**Exercise 0** (ungraded): Run the below code to store the first row containing the questions along with the column names into another dataframe `data_copy` so you can work with it later. We also drop this row of questions from `data` so that the dataframe only contains responses.


```python
print ("Shape of initial dataframe ==>",data.shape)
data_copy = data[0:1].copy()

data = data[1:]
print (data.shape)
```

    Shape of initial dataframe ==> (14054, 144)
    (14053, 144)
    


```python
data_copy
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
      <th>Q1</th>
      <th>Q2</th>
      <th>Q3</th>
      <th>Q4</th>
      <th>Q5</th>
      <th>Q6</th>
      <th>Q7</th>
      <th>Q9</th>
      <th>Q10</th>
      <th>Q12_MULTIPLE_CHOICE</th>
      <th>...</th>
      <th>Q31_Part_7</th>
      <th>Q31_Part_8</th>
      <th>Q31_Part_9</th>
      <th>Q31_Part_10</th>
      <th>Q31_Part_11</th>
      <th>Q31_Part_12</th>
      <th>Q31_OTHER_TEXT</th>
      <th>Q32</th>
      <th>Q32_OTHER</th>
      <th>Q48</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>What is your gender? - Selected Choice</td>
      <td>What is your age (# years)?</td>
      <td>In which country do you currently reside?</td>
      <td>What is the highest level of formal education ...</td>
      <td>Which best describes your undergraduate major?...</td>
      <td>Select the title most similar to your current ...</td>
      <td>In what industry is your current employer/cont...</td>
      <td>What is your current yearly compensation (appr...</td>
      <td>Does your current employer incorporate machine...</td>
      <td>What is the primary tool that you use at work ...</td>
      <td>...</td>
      <td>Which types of data do you currently interact ...</td>
      <td>Which types of data do you currently interact ...</td>
      <td>Which types of data do you currently interact ...</td>
      <td>Which types of data do you currently interact ...</td>
      <td>Which types of data do you currently interact ...</td>
      <td>Which types of data do you currently interact ...</td>
      <td>Which types of data do you currently interact ...</td>
      <td>What is the type of data that you currently in...</td>
      <td>What is the type of data that you currently in...</td>
      <td>Do you consider ML models to be "black boxes" ...</td>
    </tr>
  </tbody>
</table>
<p>1 rows × 144 columns</p>
</div>



**Exercise 1** (1 point). Several of the survey questions offered the choice, `Other`. These are encoded in the **column names**. For instance, Question 31 has 12 parts and an "other" option (`Q31_OTHER_TEXT`), as does Question 32 (`Q32_OTHER`). Remove these columns from the `data` dataframe. (That is, overwrite `data` with a version of itself omitting columns that contain the substring `OTHER`.)

> **Note.** There are also non-split questions having an option named `Other`. For instance, see `Q5` in row 1 or `Q6` in row 2. Do **not** remove these columns.


```python
# get the list of columns
# list(data)
```


```python
# exclude columns with a particular string or just drop columns
#data = data.loc[:,~data.columns.str.contains('other', case = False)]
#data.drop(data.filter(regex = 'OTHER').columns, axis = 1)
#data.drop(data.filter(like = 'OTHER').columns, axis = 1)
#data.drop(list(data.filter(regex='OTHER')), axis = 1)
#data[[c for c in data.columns if 'OTHER' not in c ]]
data = data.filter(regex='^((?!OTHER).)*$')
```


```python
# Test cell : `exercise1` (exposed)

assert data.shape[0]==14053,"Incorrect number of rows in the dataframe!"

if data.shape[1]!=138:
    how_many = "few" if data.shape[1]<138 else "many"
    
    
    
    assert data.shape[1]==138,f"Too {how_many} columns in the dataframe!"
print("Exposed tests passed! \n Note that you will need to hit 'submit' for the autograder to run the hidden tests and award you points. The exposed tests will help you debug, but they do not guarantee that your solution is accurate.")
```

    Exposed tests passed! 
     Note that you will need to hit 'submit' for the autograder to run the hidden tests and award you points. The exposed tests will help you debug, but they do not guarantee that your solution is accurate.
    


```python
# Test cell : `exercise1` (hidden)

print("Checking the columns in your dataframe..")

###
### AUTOGRADER TEST - DO NOT REMOVE
###

```

    Checking the columns in your dataframe..
    

**Exercise 2** (3 points) . Create a dictionary `mapping` to store the information of split questions such that:

Key : question number where the responses to a single question are stored across a number of columns.<br>Value : list of all columns corresponding to that question.

Sample output:

```
mapping =
{
 'Q13': ['Q13_Part_1', 'Q13_Part_2',...],

'Q15': ['Q15_Part_1', 'Q15_Part_2', 'Q15_Part_3', 'Q15_Part_4', 'Q15_Part_5', 'Q15_Part_6', 'Q15_Part_7'], 
 
 
 ..
 }
```
Hint : You may notice patterns in the columns names of questions split across columns. Using df.filter() is one way to capture or match such patterns. https://pandas.pydata.org/pandas-docs/stable/reference/api/pandas.DataFrame.filter.html


```python
values = data.filter(regex='_Part_')
keys = set(values.columns.str.replace('_Part.*', ''))
mapping2 = {}
for q in keys:
    mapping2[q] = list(values.filter(regex=q, axis=1))
```


```python
from itertools import groupby as gb
questions = list(data.filter(regex='_Part_'))
mapping = {k : list(v) for k, v in gb(questions, lambda element: element.split('_')[0])}
#vals = [list(i) for j, i in groupby(questions, lambda a: a.partition('_')[0])] 
#keys = list(set([x.split('_')[0] for x in list(data.filter(regex='_Part_'))]))
#mapping2 = {k : v for v in vals for k in keys if all(k in s for s in v)}
# check if dictionaries are equal
#mapping == mapping2
```


```python
# Test cell : `exercise2` (hidden)

###
### AUTOGRADER TEST - DO NOT REMOVE
###

```

**Exercise 3** (4 points). Here is the final cleaning step! Complete the function, `reduce_mapping()`, below, to return a mapping `q_mapping` of each question to its possible choices across **all** questions (and not just the ones you extracted in Exercise 2.)

We have provided a list of values to be removed from the values you collect from the dataframe in `na_list`. The items in this list should not appear among the choices.

You will find the questions dataframe `data_copy` from `Exercise 0` useful for this task. Make sure you "trim the questions" to conform to the format shown in the example fragment below:
```python
q_mapping = 
{
 'What is your gender?': ['Female',
                          'Male',
                          'Prefer not to say',
                          'Prefer to self-describe'],
 'What machine learning frameworks have you used in the past 5 years? (Select all that apply)': ['Scikit-Learn',
                                                                                                 'TensorFlow',
                                                                                                 'Keras',
                                                                                                 'PyTorch',
                                                                                                 'Spark MLlib',
                                                                                                 'H20',
                                                                                                 'Fastai',
                                                                                                 'Mxnet',
                                                                                                 'Caret',
                                                                                                 'Xgboost',
                                                                                                 'mlr',
                                                                                                 'Prophet',
                                                                                                 'randomForest',
                                                                                                 'lightgbm',
                                                                                                 'catboost',
                                                                                                 'CNTK',
                                                                                                 'Caffe'],
                                                                                                
 'Select the title most similar to your current role (or most recent title if retired):' : [....,
                                                                                            ....],
 ... etc....
 }
 ```
For example, the text for question `Q1` in the original dataframe is "`What is your gender? - Selected Choice`". In the final output, observe that the corresponding key for `q_mapping` has the substring " ` - Selected Choice`" removed. As another example, the text of the first option of question 19 (`Q19_Part1`) is, "`What machine learning frameworks have you used in the past 5 years? (Select all that apply) - Selected Choice - Scikit-Learn`". In the final output, " ` - Selected Choice - `" is removed from the key and "`Scikit-Learn`" appears as a possible value.

> **Hint.** One way to solve this problem is to figure out a way to trim the questions to the required format, do it for the columns (values) you mapped in Exercise 2, and then for the remaining columns in the dataframe.

Note : You do not need to spend effort on **re-arranging or introducing** punctuations to questions. Any question mark/colon appearing in a question can (and should) be included in your key as it originally appeared.


```python
def reduce_data(mapping):
    na_list = ['Other','None']
    import numpy as np
    q_mapping = {
        ''.join(v1) : [v for v in v2 if v is not np.nan and v not in na_list]
        for k1, v1 in data_copy.to_dict('list').items()
        for k2, v2 in data.to_dict('list').items()
        if k1 == k2
    }
    return q_mapping

q_mapping = reduce_data(mapping)
```


```python
# Test cell : `exercise3` (exposed)
import itertools
assert isinstance(q_mapping,dict),"`q_mapping` is not a dict"
assert all(non_value not in list(itertools.chain(*q_mapping.values())) for non_value in ['Other','None','nan']) ,"Invalid choices found in q_mapping!"
print("Exposed tests passed! \n Note that you will need to hit 'submit' for the autograder to run the hidden tests and award you points. The exposed tests will help you debug, but they do not guarantee that your solution is accurate.")
```

    Exposed tests passed! 
     Note that you will need to hit 'submit' for the autograder to run the hidden tests and award you points. The exposed tests will help you debug, but they do not guarantee that your solution is accurate.
    


```python
# Test cell : `exercise3` (hidden)

###
### AUTOGRADER TEST - DO NOT REMOVE
###

```

Take a look at how the data you extracted looks! Run the below code.



```python
import collections
final_data = collections.OrderedDict(sorted(q_mapping.items()))
display(final_data)
```

Looks good! Run the below code to write it to a csv so you can look at the Questionnaire you extracted!


```python
list_of_qs = []
responses=[]
for question,answers in q_mapping.items():
    list_of_qs.append(question)
    list_of_qs.extend(['']*(len(answers)-1))
    responses.extend(answers)
import pandas as pd
a = pd.DataFrame({'Questions':list_of_qs, 'Choices':responses})
a.to_csv("Questionnaire.csv",encoding='utf-8-sig',index=None)
```


```python
#Don't forget to run this cell. You should recognize this code!

def canonicalize_tibble(X):
    var_names = sorted(X.columns)
    Y = X[var_names].copy()
    Y.sort_values(by=var_names, inplace=True)
    Y.reset_index(drop=True, inplace=True)
    return Y

def tibbles_are_equivalent (A, B):
    A_canonical = canonicalize_tibble(A)
    B_canonical = canonicalize_tibble(B)
    cmp = A_canonical.eq(B_canonical)
    return cmp.all().all()
```

**Exercise 4** (2 points) . Let's do some ranking of the programming languages! 

In the code cell below, we supply you with a dataframe named `ranking_data`, which is a copy of questions 6 (job title) and 17 (programming language). Using `ranking_data`, create a new dataframe, `ranking_data_summary`, that counts how many times each programming language was reported by people whose job title is `'Data Scientist'`. This new dataframe should be sorted in descending order by count.

In addition:

* Response Q6 contains the selected career fields. Filter so the results only factor in entries corresponding to *Data Scientist*
* Response Q17 contains the programming languages. You must get total counts for each language and sort in descending order. Call the column with the counts, 'counts'
* Reset the index
* Change the column name Q17 to 'Language'


```python
#copies the relevant columns into a dataframe for you. Don't modify
ranking_data = data[['Q6','Q17']].copy()

ranking_data_summary = (
    ranking_data
        .query('Q6 == "Data Scientist"')
        .groupby('Q17', as_index = False)
        .count()
        .rename(columns = {'Q6' : 'counts', 'Q17' : 'Language'})
        .sort_values(by = 'counts', ascending = False)
        .reset_index(drop = True)
)
ranking_data_summary #.head(5)

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
      <th>Language</th>
      <th>counts</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>Python</td>
      <td>1619</td>
    </tr>
    <tr>
      <th>1</th>
      <td>R</td>
      <td>465</td>
    </tr>
    <tr>
      <th>2</th>
      <td>SQL</td>
      <td>141</td>
    </tr>
    <tr>
      <th>3</th>
      <td>SAS/STATA</td>
      <td>31</td>
    </tr>
    <tr>
      <th>4</th>
      <td>Java</td>
      <td>25</td>
    </tr>
    <tr>
      <th>5</th>
      <td>MATLAB</td>
      <td>19</td>
    </tr>
    <tr>
      <th>6</th>
      <td>C/C++</td>
      <td>17</td>
    </tr>
    <tr>
      <th>7</th>
      <td>Scala</td>
      <td>16</td>
    </tr>
    <tr>
      <th>8</th>
      <td>C#/.NET</td>
      <td>11</td>
    </tr>
    <tr>
      <th>9</th>
      <td>Javascript/Typescript</td>
      <td>10</td>
    </tr>
    <tr>
      <th>10</th>
      <td>Other</td>
      <td>9</td>
    </tr>
    <tr>
      <th>11</th>
      <td>PHP</td>
      <td>6</td>
    </tr>
    <tr>
      <th>12</th>
      <td>Visual Basic/VBA</td>
      <td>6</td>
    </tr>
    <tr>
      <th>13</th>
      <td>Go</td>
      <td>4</td>
    </tr>
    <tr>
      <th>14</th>
      <td>Bash</td>
      <td>3</td>
    </tr>
    <tr>
      <th>15</th>
      <td>Ruby</td>
      <td>1</td>
    </tr>
  </tbody>
</table>
</div>




```python
###
### AUTOGRADER TEST - DO NOT REMOVE
###

```


```python
#visible tests

assert type(ranking_data_summary) == pd.DataFrame,\
    "Your output isn't a dataframe. Try checking the type of your output step by step."

assert ranking_data_summary.shape == (16,2), \
    "Your output shape seems wrong. Did you accidently drop columns or categories?"

assert ranking_data_summary.iloc[0,1] == 1619, \
    "Your top result has the wrong count. Did you setup your count correctly? What about your sort?"
    
    
print("Exposed tests passed! \n Note that you will need to hit 'submit' for the autograder to run the hidden tests and award you points. The exposed tests will help you debug, but they do not guarantee that your solution is accurate.")

###
### AUTOGRADER TEST - DO NOT REMOVE
###

print("\n(Passed!)")
```

    Exposed tests passed! 
     Note that you will need to hit 'submit' for the autograder to run the hidden tests and award you points. The exposed tests will help you debug, but they do not guarantee that your solution is accurate.
    
    (Passed!)
    

**Fin!** You’ve reached the end of this part. Don’t forget to restart and run all cells again to make sure it’s all working when run in sequence; and make sure your work passes the submission process. Good luck!
