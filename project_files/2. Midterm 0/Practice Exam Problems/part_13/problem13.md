# Problem 13: Regular Expressions and Sentiment Analysis

In this problem, you will implement regular expressions and perform sentiment analysis on movie review data.

This problem has 3 exercises, numbered 0 through 2 and worth a total of 10 points. Note that Exercises 1 and 2 do not depend on Exercise 0, in case you get stuck on that first one.

## Overview

Suppose you are given a collection of textual movie reviews, such as those collected by the [Internet Movie Database (IMDB)](http://www.imdb.com) or [Rotten Tomatoes](https://www.rottentomatoes.com/). In the next set of exercises, you will implement a simple _sentiment analysis_ of these reviews. Such an analysis inspects the words in the review and tries to determine whether the review is "positive" or "negative."

> What you will implement is a simplified version of an idea originally proposed in the research article, "[Mining and summarizing customer reviews](https://www.cs.uic.edu/~liub/publications/kdd04-revSummary.pdf)," by Minqing Hu and Bing Liu, which appears in the _Proceedings of the ACM SIGKDD International Conference on Knowledge Discovery and Data Mining (KDD)_, Aug 22-25, 2004.

## The dataset

The dataset for this problem consists of 50 movie reviews. **Each review is stored in its own text file.** Your task is to process the files, analyze each review based on the rules below, and return top 5 positive reviews.

Along with the reviews, there are two more files:
1. `positive.txt`: A collection of 2005 positive words 
2. `negative.txt`: A collection of 4783 negative words

The analysis you will implement is based on the following simple idea. Given a review, count the number of its words that are positive and negative. The overall sentiment score will simply be the difference between the positive word count and the negative word count.

> This scheme makes many simplifying assumptions, like words having equal weight, repeated words being counted multiple times, and no normalization for review length, but for now just take the scheme as-is.

Before starting, please run the following code to set up positive words list `positive_words` and negative words list `negative_words`. This code cell will also print the filenames of 10 of the 50 reviews (i.e., `review_list[0:10]`).

> The code below will only work when running on Vocareum. If you opt to work on the local problem on your local machine, the following is [a link to the dataset](https://cse6040.gatech.edu/datasets/movie-reviews.zip), but you will need to figure out how to adapt the code below to work in your local envirionment. (We will **not** provide any technical support.)


```python
import os

DATA_PATH = "./data/"

# Get a list of positive words: positive_words
with open(DATA_PATH + "positive.txt") as fp:
    positive_words = set(fp.read().splitlines())

# Get a list of negative words: negative_words
with open(DATA_PATH + "negative.txt") as fp:
    negative_words = set(fp.read().splitlines())

assert len(positive_words) == 2005, "The file containing positive words may have been corrupted!"
assert len(negative_words) == 4783, "The file containing negative words may have been corrupted!"

# A list of names for the 50 review files
PATH = DATA_PATH + "reviews"
review_list = os.listdir(PATH)

# See the name of 10 review files
print("Here is a sample of 10 of the review filenames:")
review_list[0:10]
```

    Here is a sample of 10 of the review filenames:
    




    ['review_11542_7.txt',
     'review_11560_9.txt',
     'review_11870_1.txt',
     'review_11891_1.txt',
     'review_11916_3.txt',
     'review_11949_3.txt',
     'review_11950_2.txt',
     'review_11951_1.txt',
     'review_11954_1.txt',
     'review_11956_3.txt']



## Review preprocessing

To help clean up the review data, your friend has written a function, `clean_text(s)`, that converts an input string `s` into a list of words. Here is your friend's code (run this cell):


```python
def next_char(input_string, offset):
    assert type(input_string) is str and type(offset) is int and offset >= 0
    return input_string[offset+1] if offset+1 < len(input_string) else ''

def append_char(input_string, new_char):
    assert (input_string is None or type(input_string) is str) and type(new_char) is str
    return ('' if input_string is None else input_string) + new_char

def clean_text(s):
    words = []
    new_word = None
    hyphen_or_apostrophe_count = 0
    for k, c in enumerate(s):
        # Determine an action to take based on `c`, which is `s[k]`
        action = None
        if c.isalnum():
            action = 'append'
        elif c in ['-', "'"]:
            if new_word is not None and hyphen_or_apostrophe_count == 0 and next_char(s, k).isalnum():
                action = 'append'
                hyphen_or_apostrophe_count = 1
        action = action or 'skip_and_start_new_word'

        # Take the action
        assert action in ['append', 'skip_and_start_new_word']
        if action == 'append':
            new_word = append_char(new_word, c)
        else: # action == 'skip_and_start_new_word'
            if new_word is not None:
                words.append(new_word.lower())
            new_word = None
            hyphen_or_apostrophe_count = 0
    return words
```

Here is what it produces on a sample input (run this cell):


```python
sample_input = """
This film is based on Isabel Allende's not-so-much-better novel. I hate Meryl
Streep and Antonio Banderas (in non-Spanish films), and the other actors,
including Winona, my favourite actress and Jeremy Irons try hard to get over
such a terrible script.
"""

sample_output = clean_text(sample_input)
print("=== Output of your friend's code: ===\n\n{}\n".format(sample_output))
```

    === Output of your friend's code: ===
    
    ['this', 'film', 'is', 'based', 'on', 'isabel', "allende's", 'not-so', 'much-better', 'novel', 'i', 'hate', 'meryl', 'streep', 'and', 'antonio', 'banderas', 'in', 'non-spanish', 'films', 'and', 'the', 'other', 'actors', 'including', 'winona', 'my', 'favourite', 'actress', 'and', 'jeremy', 'irons', 'try', 'hard', 'to', 'get', 'over', 'such', 'a', 'terrible', 'script']
    
    

Observe the following properties about your friend's code.

1. It scans the input string `s` from left-to-right, one character at a time, building words as it goes.
2. It ensures that a word can only begin and end with a single letter or number.
3. It allows at most one hyphen (`"-"`) **or** one apostrophe (`"'"`) in a word, but no more than one of these per word.
4. It starts a new word whenever items (2) and (3) above would be violated or whenever it encounters any character that is not alphanumeric, a hyphen, or an apostrophe.

If you inspect the output closely, you'll see that the input string `"not-so-much-better"` becomes `["not-so", "much-better"]`. You might trace through the code to see why that happens.

**Exercise 0** (5 points). Come up with a regular expression pattern that can compute the same result as your friend's function when called in the following way:

```python
    # YOU define your pattern:
    your_regex_pattern = r'...'
    
    # And calling `re.findall()` as follows will match your friend's code:
    assert re.findall(your_regex_pattern, sample_input.lower(), re.VERBOSE) == clean_text(sample_input)
```

This problem is tricky! To get it right, you need to be able to read and reason about what the Python code does and understand how regular expressions work to come up with a valid solution.


```python
import re

your_regex_pattern = r"\w+[\-']?\w+|\w+"

# The following will demo your solution:
sample_output_regex = re.findall(your_regex_pattern, sample_input.lower(), re.VERBOSE)
print(">>> Your chosen regex pattern: {}\n".format(your_regex_pattern))
print(">>> Recall what your friend's code produced:\n{}\n".format(sample_output))
print(">>> Here is what your regex produces on the sample sentence from above:\n{}".format(sample_output_regex))
```

    >>> Your chosen regex pattern: \w+[\-']?\w+|\w+
    
    >>> Recall what your friend's code produced:
    ['this', 'film', 'is', 'based', 'on', 'isabel', "allende's", 'not-so', 'much-better', 'novel', 'i', 'hate', 'meryl', 'streep', 'and', 'antonio', 'banderas', 'in', 'non-spanish', 'films', 'and', 'the', 'other', 'actors', 'including', 'winona', 'my', 'favourite', 'actress', 'and', 'jeremy', 'irons', 'try', 'hard', 'to', 'get', 'over', 'such', 'a', 'terrible', 'script']
    
    >>> Here is what your regex produces on the sample sentence from above:
    ['this', 'film', 'is', 'based', 'on', 'isabel', "allende's", 'not-so', 'much-better', 'novel', 'i', 'hate', 'meryl', 'streep', 'and', 'antonio', 'banderas', 'in', 'non-spanish', 'films', 'and', 'the', 'other', 'actors', 'including', 'winona', 'my', 'favourite', 'actress', 'and', 'jeremy', 'irons', 'try', 'hard', 'to', 'get', 'over', 'such', 'a', 'terrible', 'script']
    


```python
# Test cell: `test_clean_text` (5 points)

def clean_text_regex(s):
    assert type(s) is str
    return re.findall(your_regex_pattern, s.lower(), re.VERBOSE)

for review_id in ['8_7', '144_2', '3905_1']:
    review_file = DATA_PATH + "reviews/review_{}.txt".format(review_id)
    with open(review_file) as fp:
        review_text = fp.read()
    cleaned_text = clean_text(review_text)
    cleaned_text_regex = clean_text_regex(review_text)
    assert cleaned_text == cleaned_text_regex, \
           "[{}] Your friend's results do not match yours: {}".format(review_file,
                                                                    list(zip(cleaned_text, cleaned_text_regex)))

print("\n(Passed!)")
```

    
    (Passed!)
    

**Exercise 1** (2 points). Complete the function, `score_text(clean_text)`, below. The input is a cleaned text, that is, a list of words converted from a string using the `clean_text()` function. It should return the score of the text, given the positive and negative word lists (the global variables, `positive_words` and `negative_words`, defined previously).

Here is how to calculate a sentiment score:

- Let `positive_score` be the number of words in the cleaned text that are also in `positive_words` list.
- Let `negative_score` be the number of words in the cleaned text that are also in `negative_words` list.
- The sentiment score is `positive_score - negative_score`.

> The test cells below will use your friend's implementation of `clean_text()`, **not** your regular expression. So if you don't have a working solution for Exercise 0, you can still attempt Exercises 1 and 2.


```python
def score_text(cleaned_text):
    assert type(cleaned_text) is list
    positive_score = sum([c in positive_words for c in cleaned_text])
    negative_score = sum([c in negative_words for c in cleaned_text])
    return positive_score - negative_score

```


```python
# Test cell: `test_score_text` (2 points)

for review_id, score_soln in [('8_7', 0), ('144_2', -8), ('3905_1', -8)]:
    review_file = DATA_PATH + "reviews/review_{}.txt".format(review_id)
    with open(review_file) as fp:
        review_text = fp.read()
    cleaned_text = clean_text(review_text)
    score = score_text(cleaned_text)
    assert score == score_soln, "[{}] Your score (={}) does not match the true solution (={}).".format(review_file,
                                                                                                       score,
                                                                                                       score_soln)
print('\n(Passed!)')
```

    
    (Passed!)
    

**Exercise 2** (3 points). Complete a function `top_score_files(path)`, below. To help you out, we are giving you all of the lines of code you will need. Therefore, solving the problem is then just a matter of putting them in the right order **and**, since it's Python, with the right level of indentation for each statement.

Regarding the function's input (`path`) and return value, here is what you need to know.

* The `path` variable is the name of a file directory containing all 50 review files. Therefore, the function will need to iterate over all the review files in that directory. As a hint, observe that the code cell imports the `os` module and that one of the lines of code calls `os.listdir(path)`.
* Your function should return a **list of dictionaries**. Each entry of the list corresponds to a review, and the list should be sorted in **descending** order of score (i.e., from most positive at entry 0 to most negative in the last entry).
* Furthermore, each list entry must be a **dictionary**. Each dictionary should have the form, `{'filename': ..., 'score': ...}`. In other words, there are two keys (`'filename'` and `'score'`) whose values are the name of the review file and score as would be computed by `score_text()`.

With that as background, here is the code you need to unscramble.

```python
final_scores.append(dic)
return newlist[0:5]
final_scores = []
file_score = score_text(clean_text(file))
dic = {'filename': f, 'score': file_score}
file = fp.read()
with open(path + "/" + f) as fp:
newlist = sorted(final_scores, key=lambda k: k['score'], reverse=True)
for f in os.listdir(path):
```

> You can, of course, also write your own implementation from scratch. But if you use the lines we've provided above, you are guaranteed that a solution exists using exactly those lines.


```python
import os

def top_score_files(path):
    final_scores = []
    for f in os.listdir(path):
        with open(path + "/" + f) as fp:
            file = fp.read()
            file_score = score_text(clean_text(file))
            dic = {'filename': f, 'score': file_score}
        final_scores.append(dic)
    newlist = sorted(final_scores, key=lambda k: k['score'], reverse=True) 
    return newlist[0:5]

```


```python
# Test cell: `test_top_score_files` (3 points)

path = DATA_PATH + "reviews"
test = top_score_files(path)
assert type(test) is list
assert len(test) == 5
assert test[0] == {'filename': 'review_6681_10.txt', 'score': 17}
assert test[1] == {'filename': 'review_6686_9.txt', 'score': 11}
assert test[2] == {'filename': 'review_5407_7.txt', 'score': 10}
assert test[3] == {'filename': 'review_8561_3.txt', 'score': 8} or test[3] == {'filename': 'review_334_10.txt', 'score': 8}
assert test[4] == {'filename': 'review_8561_3.txt', 'score': 8} or test[4] == {'filename': 'review_334_10.txt', 'score': 8}

print('\n(Passed!)')
```

    
    (Passed!)
    

**Fin!** Remember to test your solutions by running them as the autograder will: restart the kernel and run all cells from "top-to-bottom." Also remember to submit to the autograder; otherwise, you will **not** get credit for your hard work!
