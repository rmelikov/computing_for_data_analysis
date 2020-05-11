**Important note**! Before you turn in this lab notebook, make sure everything runs as expected:

- First, restart the kernel -- in the menubar, select Kernel → Restart.
- Then run all cells -- in the menubar, select Cell → Run All.

Make sure you fill in any place that says YOUR CODE HERE or "YOUR ANSWER HERE."

# Data Wrangling for Machine Learning

In this exercise we will take you through the basics of data cleaning that often is the majority of your work before fitting a training a machine learning model.

Data often has a lot of missing values, incorrect data types, rows that need to be removed etc., and this problemgives you a flavor of what is often required before any sort of descriptive, predictive, or prescriptive analysis.

For this exercise, we will be using a dataset with credit approval scores.


```python
import pandas as pd
import numpy as np
import re

pd.options.mode.chained_assignment = None
```


```python
credit = pd.read_csv('creditapproval.csv')
```

Let us take a look at the data.


```python
print('The dataset has {} rows.'.format(len(credit))) 
credit.head()
```

    The dataset has 624 rows.
    




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
      <th>Predictor A</th>
      <th>Predictor B</th>
      <th>Predictor C</th>
      <th>Predictor D</th>
      <th>Predictor E</th>
      <th>Predictor F</th>
      <th>Predictor G</th>
      <th>Response</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>30.83</td>
      <td>G</td>
      <td>1.25</td>
      <td>1</td>
      <td>t</td>
      <td>202.0</td>
      <td>Mrketing 1</td>
      <td>1.0</td>
    </tr>
    <tr>
      <th>1</th>
      <td>58.67</td>
      <td>J</td>
      <td>3.04</td>
      <td>6</td>
      <td>t</td>
      <td>43.0</td>
      <td>Mkt6</td>
      <td>1.0</td>
    </tr>
    <tr>
      <th>2</th>
      <td>24.50</td>
      <td>B</td>
      <td>1.50</td>
      <td>0</td>
      <td>f</td>
      <td>280.0</td>
      <td>M0</td>
      <td>1.0</td>
    </tr>
    <tr>
      <th>3</th>
      <td>27.83</td>
      <td>B</td>
      <td>3.75</td>
      <td>5</td>
      <td>f</td>
      <td>100.0</td>
      <td>Marketing 5</td>
      <td>1.0</td>
    </tr>
    <tr>
      <th>4</th>
      <td>20.17</td>
      <td>B</td>
      <td>1.71</td>
      <td>0</td>
      <td>t</td>
      <td>120.0</td>
      <td>Spend 0</td>
      <td>1.0</td>
    </tr>
  </tbody>
</table>
</div>



The data hence has 7 predictors and 1 response variable.

In machine learning or predictive modelling in general, you use predictors (in this case 7 of them) to predict its corresponding response.

But before we can move onto predictive modelling, we need to clean the data. Data cleaning often is the most important part of machine learning and we exactly going to do that bit.

For instance, let's check for columns having missing values.


```python
credit.isnull().any()
```




    Predictor A     True
    Predictor B     True
    Predictor C    False
    Predictor D    False
    Predictor E    False
    Predictor F     True
    Predictor G    False
    Response        True
    dtype: bool



So the columns **Predictor A, Predictor B, Predictor F, and Response** all have missing values. You'll treat these cases through the exercises below.

Data without a **Response variable** can neither be used in training the model or in testing it. We would hence like to remove rows that have **both Predictor F** and **Response as NaN values**


**Exercise 0** (1 point): Create a new dataframe named **`creditwithresponse`** that is a copy of **`credit`** but with any rows missing **either** `Predictor F` **or** `Response` removed.


```python
# my solution
creditwithresponse = credit.dropna(subset = ['Predictor F', 'Response'])

# school solution
#has_F_and_Response = credit['Predictor F'].notnull() & credit['Response'].notnull()
#creditwithresponse = credit[has_F_and_Response]
```


```python
##Test cell: Exercise 0
assert len(creditwithresponse) == 612, "The length of your newly created dataframe does not match the solution"
assert len(creditwithresponse[creditwithresponse['Predictor F'].isnull()]) == 0, "Some NaN values still exist in your new dataframe."
assert len(creditwithresponse[creditwithresponse['Response'].isnull()]) == 0, "Almost there! Though some NaN values still exist in your new dataframe."

print("\n(Passed!)")
```

    
    (Passed!)
    

What about the other predictors?

One technique is to replace missing values with sensible substitutes. For instance, we might replace a missing value with the **mean** of the remaining values in the case of a numerical variable, or the **mode** in the case of a categorical (discrete) variable.

So, for instance, suppose a numerical predictor has the values `[1.0, 6.5, 3.5, NaN, 5.0]`. Then, you might replace the `NaN` with the mean of the known values, `[1.0, 6.5, 3.5, 5.0]`, which is 4.0.

**Exercise 1 (3 points)**: Create a function called **`imputevalue()`** that takes, as its inputs, a dataframe, the name of a column in that dataframe, and the replacement method. The replacement method will be a string, either `"mean"` or `"mode"`.

With these three inputs, your function should do the following:

1. Create a copy of the dataframe (i.e., the original should remain intact).
2. Compute the **mean** or **mode** of the column **without** the NaN values.
3. Replace the NaN's in that column with the computed mean/mode.
4. Return this new dataframe (i.e., not just the column containing the newly imputed values).


```python
def imputevalue(df, col, func):
    assert func in ['mean', 'mode'], "You might have edited the assertion in this code cell, please reload this cell"
    
    def my_solution():
        if func == 'mean':
            return df.fillna(df.mean()[col])
        else:
            return df.fillna(df.mode()[col][0])
    
    def school_solution():
        dfcopy = df.copy()
        dfcopy2 = dfcopy[dfcopy[col].notnull()]
        if func=='mean':
            mean = np.mean(dfcopy2[col])
            dfcopy.loc[dfcopy[col].isnull(), col] = mean
        else:
            mode = dfcopy2.loc[:, col].mode()
            dfcopy.loc[dfcopy[col].isnull(), col] = mode[0]
        return dfcopy

    return my_solution()
```


```python
##Test cell: Exercise 1
pd.options.mode.chained_assignment = None

df2 = imputevalue(creditwithresponse, 'Predictor A', 'mean')
assert not(df2.equals(creditwithresponse)), 'You have not created a copy of the dataframe'
assert (round(np.mean(df2['Predictor A']), 2) >= 31.8) & (round(np.mean(df2['Predictor A']), 2)<=31.9), "The imputed value is incorrect. Please check your code"

df2 = imputevalue(creditwithresponse, 'Predictor B', 'mode')
assert df2.loc[:,'Predictor B'].mode()[0] == 'B', "The imputed value is incorrect. Please check your code"

credit_imputed_temp = imputevalue(creditwithresponse, 'Predictor A', 'mean')
credit_imputed = imputevalue(credit_imputed_temp, 'Predictor B', 'mode')

assert credit_imputed['Predictor A'].notnull().all()==True, 'There are still some missing values in Predictor A'
assert credit_imputed['Predictor B'].notnull().all()==True, 'There are still some missing values in Predictor B'

print("\n(Passed!)")
```

    
    (Passed!)
    

Using the preceding techniques (removing missing rows or imputing values), we've covered all variables except `Predictor G`. Let's treat that one next. First, let's inspect it:


```python
credit_imputed['Predictor G'].head()
```




    0     Mrketing 1
    1           Mkt6
    2             M0
    3    Marketing 5
    4        Spend 0
    Name: Predictor G, dtype: object



This column actually contains marketing expenditures in thousands of dollars. For example, `'Marketing 1'` means that a total of $1000 was spent on this marketing campaign.

As you can see, these data were not entered in a consistent way, except that a numerical value does appear. In this exercise you are required to extract the numbers from the column's values, e.g., extract **`1`** from `'Marketing 1'`.

Please note that the following facts about the values in the column 'Predictor G'.
1. Each value begins with a string of alphabetic characters. This string may vary from row to row.
2. A space may or may not follow that initial string of alphabetic characters.
3. The string ends with a sequence of digits.

Refer to the sample values from the call to `.head()` above.

**Exercise 2 (3 points)**: Create a function **`strip_text()`** that takes a **`(dataframe, column)`** as inputs and returns a **dataframe** according to the desciption below.

With these two inputs, your function should:

1. Create a copy of the dataframe, i.e., the original should remain intact.
2. For the given column, remove all the text in the column so that it contains only numbers (integers).
3. Return this new dataframe, i.e., not just the column containing the newly imputed values.


```python
def strip_text(df, col):
    df_copy = df.copy()
    
    def my_solution():
        df_copy[col].replace(regex = True, inplace = True, to_replace = r'[^0-9]', value = r'')
        df_copy[col] = df_copy[col].astype('int64')
        
    
    def school_solution():
        df_copy[col] = df_copy[col].apply(lambda x : int(''.join(re.findall("[0-9]", x))))
    
    #my_solution()
    school_solution()
    
    return df_copy
```


```python
##Test cell: Exercise 2

instr = pd.DataFrame(['Rich0','Rachel    2', 'Sam123', 'Ben 012', 'Evan 999', 'Chinmay12', '   Raghav12'])
instr2 = instr.rename(columns={0:'col1'})
assert strip_text(instr2,'col1').equals(pd.DataFrame([0,2,123,12,999,12,12]).rename(columns={0:'col1'})),"Please check your output by running your function on the 'instr' dataframe"

credit_cleaned = strip_text(credit_imputed,'Predictor G')
assert not(credit_cleaned.equals(credit_imputed)), 'You have not created a copy of the dataframe'
assert credit_cleaned['Predictor G'].dtype  == 'int64', "Output data type does not match"
assert len(credit_cleaned) == 612, "Your dataframe output is not of the appropriate length"
assert (round(np.mean(credit_cleaned['Predictor G']),2) >= 2.62) & (round(np.mean(credit_cleaned['Predictor G']),2)<=2.64), "The imputed data does not match. You could try replicating these tests on the 'instr' dataframe above."
assert (round(np.sum(credit_cleaned['Predictor G']),2) >= 1611.0) & (round(np.sum(credit_cleaned['Predictor G']),2) <= 1613.0) , "The imputed data does not match. You could try replicating these tests on the 'instr' dataframe above."

print("\n(Passed!)")
```

    
    (Passed!)
    

Now that you have cleaned your dataset, let's do one final check to see if we still have any missing values.


```python
credit_cleaned.isnull().any()
```




    Predictor A    False
    Predictor B    False
    Predictor C    False
    Predictor D    False
    Predictor E    False
    Predictor F    False
    Predictor G    False
    Response       False
    dtype: bool



You should see all `False` values, meaning there is no missing data in any of the columns. If so, great!

**Creating Interaction Terms in the Data**

Sometimes, for analysis purposes, it is better to create _interaction predictors_, which are new predictors that modify or combine existing predictors. For example, in a marketing scenario, spending on TV marketing might have a quadratic relationship with the sales of the product. We would hence want to include ** $(\mathrm{TV\ marketing})^2$ **  as a predictor to better capture the relationship.

In this final exercise we will create a new predictor that is a combination of the predictors in the dataset **`credit_cleaned`**.

**Exercise 3 (3 points):** Create a function **`familiarity()`** that takes as its inputs a dataframe (`df`), the names of three input columns (`column1`, `column2`, `column3`), and the name of a new output column (`columnnew`). It should compute for the values of this new column what appears in the formula below.

**$$\mathtt{columnnew} = \frac{\mathtt{column1}}{e^{\mathtt{column2}}} - \sqrt{\mathtt{column3}},$$**

where **$$\sqrt{\mathtt{column3}} = (\mathtt{column3})^{0.5}.$$**

The return value for the function will be a dataframe with the new column, **`columnnew`**, **in addition** to all the original columns in the dataframe.

> **Note.** If a value in column 3 is negative, so that the square-root is undefined, set the corresponding value in `columnnew` to zero (0).


```python
def familiarity(df, column1, column2, column3, columnnew):
    
    def my_solution():
    
        df[columnnew] = df.apply(
            lambda row : 0 if row[column3] < 0 else row[column1] / np.exp(row[column2]) - row[column3] ** 0.5, 
            axis = 1
        )
        
        return df
    
    def school_solution():
        dfcopy = df.copy()
        dfcopypos = dfcopy[dfcopy[column3]>=0]
        dfcopyneg = dfcopy[dfcopy[column3]<0]
        dfcopypos[columnnew] = dfcopypos[column1]/np.exp(dfcopypos[column2])-np.sqrt(dfcopypos[column3])
        dfcopyneg[columnnew] = 0
        return dfcopypos.append(dfcopyneg)
    
    return my_solution()
```


```python
##Test cell: Exercise 3

d={'col1':[1,2,3,4,5], 'col2':[2,3,4,0,4], 'col3':[-9,2,-8,0,0]}
df = pd.DataFrame(d)
dffamiliarity = familiarity(df, 'col1', 'col2', 'col3', 'colnew')
assert dffamiliarity.loc[dffamiliarity['col3']<0,'colnew'].all()==0, "The non negative case for col3 is failing. Please check your code"


credit_final = familiarity(credit_cleaned, 'Predictor A', 'Predictor C', 'Predictor G', 'Predictor H')
assert 'Predictor H' in credit_final, "Column 'Predictor H' does not exist"
assert len(credit_final) == 612, "The length of the dataframe does not match the required length"
assert (round(np.sum(credit_final['Predictor H']),2) >= 7262.4) & (round(np.sum(credit_final['Predictor H']),2) <= 7262.5), "The sum of values in Predictor H do not match the required vlue"

print("\n(Passed!)")
```

    
    (Passed!)
    

At this point, you have completed all the exercises and can go ahead and submit the notebook!

However, we have however added a small piece of code below to give you an idea of how simple it is to create a predicitive model in Python. It is **not graded** and hence you can submit this notebook, complete other notebooks and come back and have a look at it!


```python
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
from sklearn.ensemble import RandomForestClassifier

#Split the dataset into predictors and response
datax = credit_final[['Predictor A', 'Predictor C', 'Predictor D', 'Predictor F', 'Predictor G', 'Predictor H']]
datay = credit_final[['Response']]

#Traintest split - Test sets are created to test the accuracy of your model on a piece of data that is not used to train the model
X_train, X_test, y_train, y_test = train_test_split(datax, np.ravel(datay), test_size=0.20, random_state=42)

forest = RandomForestClassifier(n_estimators=500) #Number of decision trees in the forest = 500
forest.fit(X_train,y_train) #Train the classifier using the train data
forest_pred = forest.predict(X_test) #Predict the classes for the test data
print("The testing accuracy of the random forest classifier is: ",accuracy_score(y_test, forest_pred)) #Print the accuracy of the model
```

    The testing accuracy of the random forest classifier is:  0.7317073170731707
    

The accuracy of the model above is about 70% or so. Due to the nature of the dataset being artificial, we don't expect a higher accuracy. Instead, our purpose here is to give you an idea as to how easy it is to do predictive modelling (machine learning) in Python.


**Fin!** That's the end of this problem. Don't forget to restart and run this notebook from the beginning to verify that it works top-to-bottom before submitting. You can move on to the next problem
