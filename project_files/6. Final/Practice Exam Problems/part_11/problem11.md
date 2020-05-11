# Problem 11: Text Preprocessing for Machine Learning

For this problem, we are going to do some preprocessing of text data to get it ready for use in a so-called _Word2vec_ model.

In a Word2vec model, the input is usually a large corpus of text while the output is a collection of points in some vector space. Each word of the corpus is assigned to a point, with the goal of having words with a similar "meaning" appearing close to one another in the space. If you feel you need further information about this topic, see: https://en.wikipedia.org/wiki/Word2vec

Our data consists of reviews on Amazon.com for sales made on musical instruments. The raw review data can be found here: http://jmcauley.ucsd.edu/data/amazon/

Let's first load the data from a .json file.  


```python
import pandas as pd
df = pd.read_json("reviews_Musical_Instruments_5.json", lines=True)
df.head()
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
      <th>reviewerID</th>
      <th>asin</th>
      <th>reviewerName</th>
      <th>helpful</th>
      <th>reviewText</th>
      <th>overall</th>
      <th>summary</th>
      <th>unixReviewTime</th>
      <th>reviewTime</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>A2IBPI20UZIR0U</td>
      <td>1384719342</td>
      <td>cassandra tu "Yeah, well, that's just like, u...</td>
      <td>[0, 0]</td>
      <td>Not much to write about here, but it does exac...</td>
      <td>5</td>
      <td>good</td>
      <td>1393545600</td>
      <td>02 28, 2014</td>
    </tr>
    <tr>
      <th>1</th>
      <td>A14VAT5EAX3D9S</td>
      <td>1384719342</td>
      <td>Jake</td>
      <td>[13, 14]</td>
      <td>The product does exactly as it should and is q...</td>
      <td>5</td>
      <td>Jake</td>
      <td>1363392000</td>
      <td>03 16, 2013</td>
    </tr>
    <tr>
      <th>2</th>
      <td>A195EZSQDW3E21</td>
      <td>1384719342</td>
      <td>Rick Bennette "Rick Bennette"</td>
      <td>[1, 1]</td>
      <td>The primary job of this device is to block the...</td>
      <td>5</td>
      <td>It Does The Job Well</td>
      <td>1377648000</td>
      <td>08 28, 2013</td>
    </tr>
    <tr>
      <th>3</th>
      <td>A2C00NNG1ZQQG2</td>
      <td>1384719342</td>
      <td>RustyBill "Sunday Rocker"</td>
      <td>[0, 0]</td>
      <td>Nice windscreen protects my MXL mic and preven...</td>
      <td>5</td>
      <td>GOOD WINDSCREEN FOR THE MONEY</td>
      <td>1392336000</td>
      <td>02 14, 2014</td>
    </tr>
    <tr>
      <th>4</th>
      <td>A94QU4C90B1AX</td>
      <td>1384719342</td>
      <td>SEAN MASLANKA</td>
      <td>[0, 0]</td>
      <td>This pop filter is great. It looks and perform...</td>
      <td>5</td>
      <td>No more pops when I record my vocals.</td>
      <td>1392940800</td>
      <td>02 21, 2014</td>
    </tr>
  </tbody>
</table>
</div>



From the data, we are interested in the `reviewText` column.  Additionally, for our analysis, we will only consider the first 500 rows of the data.


```python
text = df['reviewText'].values
text = text[:500]
text[:3]
```




    array(["Not much to write about here, but it does exactly what it's supposed to. filters out the pop sounds. now my recordings are much more crisp. it is one of the lowest prices pop filters on amazon so might as well buy it, they honestly work the same despite their pricing,",
           "The product does exactly as it should and is quite affordable.I did not realized it was double screened until it arrived, so it was even better than I had expected.As an added bonus, one of the screens carries a small hint of the smell of an old grape candy I used to buy, so for reminiscent's sake, I cannot stop putting the pop filter next to my nose and smelling it after recording. :DIf you needed a pop filter, this will work just as well as the expensive ones, and it may even come with a pleasing aroma like mine did!Buy this product! :]",
           'The primary job of this device is to block the breath that would otherwise produce a popping sound, while allowing your voice to pass through with no noticeable reduction of volume or high frequencies. The double cloth filter blocks the pops and lets the voice through with no coloration. The metal clamp mount attaches to the mike stand secure enough to keep it attached. The goose neck needs a little coaxing to stay where you put it.'],
          dtype=object)



Though these are solid reviews for the instruments, they are not too great as inputs for doing machine learning. So let's do a bit of cleaning!

**Exercise 0** (3 points). This exercise has two parts.

**Part A** (2 pts) Complete the function **clean_review** that, given a review, returns a cleaned version according to the following processing steps.
   1. First, retain only the following characters and replace all other characters with a space: 
        - alphanumerical
        - comma
        - exclamation mark
        - question mark
        - quotation mark (')
        - open or closed parenthesis
   2. Next, add a leading and trailing space for the following characters (For example, "," => "\s,\s" ):
        - comma
        - exclamation mark
        - question mark
        - open or closed parenthesis
   3. Next, replace any two spaces with a single space. For example, "\s\s" => "\s"
   4. Finally, return the review with all characters converted to lowercase and remove all leading and trailing spaces      


```python
import re
def clean_review(review):
    ### BEGIN SOLUTION
    review = re.sub(r"[^A-Za-z0-9(),!?\']", " ", review)
    review = re.sub(r"\?", " \? ", review)
    review = re.sub(r",", " , ", review)
    review = re.sub(r"!", " ! ", review)
    review = re.sub(r"\(", " \( ", review)
    review = re.sub(r"\)", " \) ", review)
    review = re.sub(r"\s{2,}", " ", review)
    return review.strip().lower()
    ### END SOLUTION

```

**Part B** (1 pt) Apply the changes from Part A to each review in the list **`text`** and return a new list **`text_new`** with the same reviews cleaned. The ordering of the reviews should be preserved.


```python
### BEGIN SOLUTION
text_new = [clean_review(rev) for rev in text]
### END SOLUTION
```


```python
## Test cell
testText = df['reviewText'].values[501]
cleanedText = clean_review(testText)
assert cleanedText.islower()
assert cleanedText.startswith(' ') == False
assert cleanedText.endswith(' ') == False
assert '  ' not in cleanedText
if cleanedText.find(',') != -1:
    assert ' , ' in cleanedText
if cleanedText.find('?') != -1:
    assert ' ? ' in cleanedText
if cleanedText.find('!') != -1:
    assert ' ! ' in cleanedText
if cleanedText.find('(') != -1:
    assert ' ( ' in cleanedText
if cleanedText.find(')') != -1:
    assert ' ) ' in cleanedText
assert len(text_new) == 500


print("\n(Passed!)")
```

    
    (Passed!)
    

**Exercise 1** (3 points): Next, create a unique index for each element in **`text_new`**. Replace each element in a review with its index value. Index values should start from 1; you should scan the reviews in the order that they appear in the list, and scan the words in each review from left-to-right. Return a list of lists called **`text_data`** where each list in **`text_data`** represents a review in **`text_new`** and contains the indices for each element in the review.

For example, consider the two reviews:

    ['i love georgia tech soooo much', 'i learned soooo much in 6040 ,']
    
These two reviews would become:

    [[1, 2, 3, 4, 5, 6], [1, 7, 5, 6, 8, 9, 10]]


```python
### BEGIN SOLUTION
idx = 0
word_idx = {}
for rev in text_new:
    for word in rev.split(" "):
        if word not in word_idx:
            word_idx[word] = idx+1
            idx = idx+1

def rev_to_idx(sent):
    return [word_idx[w] for w in sent.split(" ")]

text_data = [rev_to_idx(sent) for sent in text_new]
### END SOLUTION
```


```python
## Test cell
assert isinstance(text_data, (list,))
assert isinstance(text_data[7], (list,))
assert len(text_data) == 500
assert text_data[2] == [17, 104, 105, 28, 91, 106, 26, 3, 107, 17, 108, 109, 110, 111, 112, 68, 113, 114, 7, 115, 116, 117, 118, 3, 119, 120, 98, 121, 122, 123, 28, 124, 125, 126, 127, 17, 54, 128, 82, 129, 17, 130, 47, 131, 17, 118, 120, 98, 121, 132, 17, 133, 134, 135, 136, 3, 17, 137, 138, 139, 140, 3, 141, 9, 142, 17, 143, 144, 145, 68, 146, 147, 3, 148, 149, 89, 150, 9]
assert text_data[300] == [183, 665, 913, 47, 183, 665, 2940, 17, 157, 2941, 50, 110, 366, 26, 17, 595, 2942, 109, 1949, 2943, 17, 144, 17, 2942, 145, 3, 503, 636, 35, 17, 2942, 2944, 9, 2945, 1543, 17, 242, 175, 9, 61, 24, 2946, 2947, 227, 242, 3, 17, 2948, 149, 9, 2949, 9, 110, 1812]


print("\n(Passed!)")
```

    
    (Passed!)
    

Great job! We are almost there. 

The inputs into a Word2vec algorithm are `pivots` and `targets`. You should build these as follows, referring also to the example below.

First, suppose you are given a _window size_, $w$, which is a positive integer. For each review, a word with $w$ words to its left and $w$ words to its right is called a _pivot_. Then, given a pivot, any word to its left or right is a _target_.

**Example.** Consider the following two reviews after undergoing the steps in the exercises above:

    [[1, 2, 3, 4, 5, 6, 8], [9, 10, 11, 12, 13, 14]]

If the window size $w=2$, then:
1. The pivots are [3, 4, 5], [11, 12]
    Note: 1, 2, 6, and 8 are not pivots for the first review because they do not have 2 words to the left and right of them. 
2. and the targets are:
    3. [1, 2, 4, 5] for pivot 3
    4. [2, 3, 5, 6] for pivot 4
    5. etc.

Thus, continuing our example, the inputs to Word2vec would be the following two lists:
    
    targets: [1, 2, 4, 5, 2, 3, 5, 6, 3, 4, 6, 8, 9, 10, 12, 13, 10, 11, 13, 14]

    pivots: [3, 3, 3, 3, 4, 4, 4, 4, 5, 5, 5, 5, 11, 11, 11, 11, 12, 12, 12, 12]
    
Note that `pivots` and `targets` have the same length and are paired, that is, `targets[i]` is the target associated with pivot `pivots[i]`. Conceptually, these are the inputs in the Word2vec model.

For Exercise 2 below, we will take it just a little bit further:

**Exercise 2** (4 points): Given a window size and an input like **`text_data`**, complete the function **create_pivots_targets** that returns a list of tuples where each tuple represents a pivot-target pair, as defined above. That is, for the example above:

```python
    pivots_targets = [(3, 1), (3, 2), (3, 4), (3, 5), (4, 2), (4, 3), ... ,(12, 13), (12, 14)]
```

The returned values should use the same ordering convention as the input: reviews should be processed in the order in which they appear in the input data, and targets should appear in the order in which they appear their reviews.


```python
import numpy as np
def create_pivots_targets(window, data):
    ### BEGIN SOLUTION
    pivot_words = []
    target_words = []
    data = np.array(data)
    for i in range(data.shape[0]): # get the index of the current review
        pivot_idx = data[i][window:-window] # get the pivots for that review

        for j in range(len(pivot_idx)): # get the index of the current pivot in the list of pivots

            #get pivot value at the current pivot index
            pivot = pivot_idx[j]
            
            # get targets for current pivot
            targets = np.array([])
            neg_targets = data[i][j:j+window] # left of the pivot
            pos_targets = data[i][j+window+1: j+(2*window)+1] # right of the pivot

            targets = np.append(neg_targets, pos_targets).flatten().tolist()
            #targets = np.append(targets, [neg_targets, pos_targets]).flatten().tolist()

            for z in range(window*2):
                pivot_words.append(pivot)
                target_words.append(targets[z])
    
    pivots_targets = list(zip(pivot_words, target_words)) 
    ### END SOLUTION
    return pivots_targets
```


```python
## Test cell
output = create_pivots_targets(2, text_data)
output2 = create_pivots_targets(4, text_data)
assert len(output) == 184968
assert len(output2) == 353944
assert isinstance(output, (list,))
assert len(output2[7]) == 2
assert isinstance(output[300], tuple)
assert output[25] == (9,8)
assert output2[70] == (13,15)

print("\n(Passed!)")
```

    
    (Passed!)
    

** Fin ** You've reached the end of this problem. Don't forget to restart the kernel and run the entire notebook from top-to-bottom to make sure you did everything correctly. If that is working, try submitting this problem. (Recall that you *must* submit and pass the autograder to get credit for your work.)
