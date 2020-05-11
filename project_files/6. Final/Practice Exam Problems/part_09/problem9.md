# Problem 9: Board Game Similarity 

In this problem you will analyze a dataset that contains information about board games. The games will be represented by their category tags and you will attempt to find similar board games using a _cosine similarity_ measure: https://en.wikipedia.org/wiki/Cosine_similarity

This dataset derives from one available on Kaggle: https://www.kaggle.com/mrpantherson/board-game-data/data

The original source for this data is https://boardgamegeek.com/

Let's start by inspecting the dataset.


```python
import pandas as pd
import numpy as np
import random
from IPython.display import display
import ast

# Import the dataset
data = pd.read_csv("bgg_db_2018_01.csv",encoding = 'latin-1')

# Display the data
display(data.head())
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
      <th>rank</th>
      <th>bgg_url</th>
      <th>game_id</th>
      <th>names</th>
      <th>min_players</th>
      <th>max_players</th>
      <th>avg_time</th>
      <th>min_time</th>
      <th>max_time</th>
      <th>year</th>
      <th>avg_rating</th>
      <th>geek_rating</th>
      <th>num_votes</th>
      <th>image_url</th>
      <th>age</th>
      <th>mechanic</th>
      <th>owned</th>
      <th>category</th>
      <th>designer</th>
      <th>weight</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>1</td>
      <td>https://boardgamegeek.com/boardgame/174430/glo...</td>
      <td>174430</td>
      <td>Gloomhaven</td>
      <td>1</td>
      <td>4</td>
      <td>150</td>
      <td>90</td>
      <td>150</td>
      <td>2017</td>
      <td>9.01310</td>
      <td>8.52234</td>
      <td>9841</td>
      <td>https://cf.geekdo-images.com/images/pic2437871...</td>
      <td>12</td>
      <td>Action / Movement Programming, Co-operative Pl...</td>
      <td>18217</td>
      <td>Adventure, Exploration, Fantasy, Fighting, Min...</td>
      <td>Isaac Childres</td>
      <td>3.7720</td>
    </tr>
    <tr>
      <th>1</th>
      <td>2</td>
      <td>https://boardgamegeek.com/boardgame/161936/pan...</td>
      <td>161936</td>
      <td>Pandemic Legacy: Season 1</td>
      <td>2</td>
      <td>4</td>
      <td>60</td>
      <td>60</td>
      <td>60</td>
      <td>2015</td>
      <td>8.66575</td>
      <td>8.49837</td>
      <td>23489</td>
      <td>https://cf.geekdo-images.com/images/pic2452831...</td>
      <td>13</td>
      <td>Action Point Allowance System, Co-operative Pl...</td>
      <td>38105</td>
      <td>Environmental, Medical</td>
      <td>Rob Daviau, Matt Leacock</td>
      <td>2.8056</td>
    </tr>
    <tr>
      <th>2</th>
      <td>3</td>
      <td>https://boardgamegeek.com/boardgame/182028/thr...</td>
      <td>182028</td>
      <td>Through the Ages: A New Story of Civilization</td>
      <td>2</td>
      <td>4</td>
      <td>240</td>
      <td>180</td>
      <td>240</td>
      <td>2015</td>
      <td>8.65702</td>
      <td>8.32401</td>
      <td>10679</td>
      <td>https://cf.geekdo-images.com/images/pic2663291...</td>
      <td>14</td>
      <td>Action Point Allowance System, Auction/Bidding...</td>
      <td>14147</td>
      <td>Card Game, Civilization, Economic</td>
      <td>Vlaada Chvátil</td>
      <td>4.3538</td>
    </tr>
    <tr>
      <th>3</th>
      <td>4</td>
      <td>https://boardgamegeek.com/boardgame/12333/twil...</td>
      <td>12333</td>
      <td>Twilight Struggle</td>
      <td>2</td>
      <td>2</td>
      <td>180</td>
      <td>120</td>
      <td>180</td>
      <td>2005</td>
      <td>8.35188</td>
      <td>8.21012</td>
      <td>29923</td>
      <td>https://cf.geekdo-images.com/images/pic361592.jpg</td>
      <td>13</td>
      <td>Area Control / Area Influence, Campaign / Batt...</td>
      <td>41094</td>
      <td>Modern Warfare, Political, Wargame</td>
      <td>Ananda Gupta, Jason Matthews</td>
      <td>3.5446</td>
    </tr>
    <tr>
      <th>4</th>
      <td>5</td>
      <td>https://boardgamegeek.com/boardgame/167791/ter...</td>
      <td>167791</td>
      <td>Terraforming Mars</td>
      <td>1</td>
      <td>5</td>
      <td>120</td>
      <td>120</td>
      <td>120</td>
      <td>2016</td>
      <td>8.38331</td>
      <td>8.17328</td>
      <td>20468</td>
      <td>https://cf.geekdo-images.com/images/pic3536616...</td>
      <td>12</td>
      <td>Card Drafting, Hand Management, Tile Placement...</td>
      <td>26145</td>
      <td>Economic, Environmental, Industry / Manufactur...</td>
      <td>Jacob Fryxelius</td>
      <td>3.2465</td>
    </tr>
  </tbody>
</table>
</div>


Let's concentrate on the 'category' column of the data


```python
print(data.loc[0, 'category'])
print(data.loc[1, 'category'])
```

    Adventure, Exploration, Fantasy, Fighting, Miniatures
    Environmental, Medical
    

Observe that each game may be assigned one or more categories. Let us first get a list of all categories from this dataset.

**Exercise 0** (2 points). Write some code to create a **list** named `categories` that lists **unique** categories sorted alphabetically from the `category` column of the data.


```python
# method 1
#category = data['category'].unique().tolist()
#categories = []
#for i in category:
#    categories.extend(i.split(', '))
#categories = sorted(list(set(categories)))


# method 2
#category = data['category'].unique().tolist()
#categories = sorted(
#    list(
#        set(
#            [item for sublist in [i.split(', ') for i in category] for item in sublist]
#        )
#    )
#)


# method 3
categories = []
data['category'].apply(lambda x: categories.extend(x.split(', ')))
categories = sorted(list(set(categories)))

print("You found {} unique categories, the first few of which are:\n{}.".format(len(categories), categories[:5]))
```

    You found 84 unique categories, the first few of which are:
    ['Abstract Strategy', 'Action / Dexterity', 'Adventure', 'Age of Reason', 'American Civil War'].
    


```python
# Test Cell: Exercise 0

assert len(categories) == 84
assert categories == sorted(categories)
assert isinstance(categories,list)

assert categories[0] == 'Abstract Strategy'
assert categories[1] == 'Action / Dexterity'
assert categories[15] == "Children's Game"
assert categories[20] == 'Comic Book / Strip'
assert categories[27] == 'Expansion for Base-game'
assert categories[36] == 'Korean War'
assert categories[46] == 'Movies / TV / Radio theme'
assert categories[59] == 'Post-Napoleonic'
assert categories[61] == 'Print & Play'
assert categories[-1] == 'none'

print('\n Passed!')
```

    
     Passed!
    

**Exercise 1** (3 points). Write a function 
``` python
def get_normalized_category_vector(game_categories,categories):
    ...
```

that takes two inputs

1. `game_categories`, a string of comma separated categories (think of this input as an entry from the `category` column of the data); and
2. `categories`, a list of alphabetically sorted categories created in Exercise 0;

and returns a _normalized category vector_, defined below, as a 1-D numpy array.

A _category vector_ is defined as a vector of 1's and 0's where an entry is 1 if the board game has the corresponding category as one of its categories, or 0 otherwise.

For example, suppose:

```python
# Game categories
game_categories = 'Environmental, Medical'
# List of all alphabetically sorted categories categories
categories = ['Adventure', 'Environmental', 'Exploration', 'Fantasy', 'Fighting', 'Medical', 'Miniatures']
```

Then,

```python
assert game_category_vector == np.array([0, 1, 0, 0, 0, 1, 0])
```

The _normalized category vector_ is a category vector that is normalized to have a 2-norm length of one. For the preceding example:

```python
assert get_normalized_category_vector(game_categories, categories) == np.array([0, 1/sqrt(2), 0, 0, 0, 1/sqrt(2), 0])
```


```python
def get_normalized_category_vector(game_categories, categories):
    assert type(game_categories) is str
    
    result = [1 if i in game_categories.split(', ') else 0 for i in categories]
    return result/np.linalg.norm(result)
    
    #result = np.array([1 if i in game_categories.split(', ') else 0 for i in categories])
    #from sklearn.preprocessing import normalize
    #return normalize(result[:, np.newaxis], axis = 0).ravel()


    #result = np.zeros(len(categories), dtype=int)
    #L = game_categories.split(', ')
    #for v in L:
    #    result[categories.index(v)] = 1
    #return result/np.linalg.norm(result)
```


```python
# Test Cell: Exercise 1

test0 = 'Environmental, Medical'
toy_categories = ['Adventure', 'Environmental', 'Exploration', 'Fantasy', 'Fighting', 'Medical', 'Miniatures']
your_result = get_normalized_category_vector(test0,toy_categories)
true_result = np.array([0.        , 0.70710678, 0.        , 0.        , 0.        ,
       0.70710678, 0.        ])

assert len(your_result)==len(true_result), "The length does not match!"
assert np.all(np.isclose(your_result,true_result)), "The result is not correct!"

# check a random sample

df = data.sample(n=100)
for category in df['category']:
    l = len(category.split(', '))
    v = get_normalized_category_vector(category,categories)
    a = v[np.nonzero(v)]
    b = np.ones(l)/np.sqrt(l)
    assert len(v)==len(categories), "The length of your vector is not correct!"
    assert np.all(np.isclose(a,b)), "The result is not correct!"

print("\n Passed!")
```

    
     Passed!
    

**Exercise 2** (1 points). Write a function 

``` python
def get_similarity_score(v1, v2):
    ...
```

that takes two normalized category vectors (as 1-D numpy arrays) as inputs and returns a _cosine similarity score_ as an output.

The cosine similarity of two normalized vectors is their dot product. As an example:

``` python
v1 = np.array([0, 1/sqrt(2), 0, 0, 0, 1/sqrt(2), 0])
v2 = np.array([1/sqrt(3), 1/sqrt(3), 0, 0, 0, 0, 1/sqrt(3)])

assert get_similarity_score(v1, v2) == 1/sqrt(6)
```

If you feel you need more details, see: https://en.wikipedia.org/wiki/Cosine_similarity.


```python
def get_similarity_score(v1, v2):
    assert len(v1)==len(v2)
    
    return v1.dot(v2)
```


```python
# Test Cell: Exercise 2

v1 = np.array([0, 1/np.sqrt(2), 0, 0, 0, 1/np.sqrt(2), 0])
v2 = np.array([1/np.sqrt(3), 1/np.sqrt(3), 0, 0, 0, 0, 1/np.sqrt(3)])
your_score = get_similarity_score(v1, v2)
true_score = 1/np.sqrt(6)
assert np.isclose(your_score,true_score), \
    "The result is not correct! \n your score = {} \n true score = {}".format(your_score,true_score)


# Test a random sample from the dataset

for i in range(100):
    c1 = data['category'].sample(1).iloc[0]
    v1 = get_normalized_category_vector(c1,categories)
    c2 = data['category'].sample(1).iloc[0]
    v2 = get_normalized_category_vector(c2,categories)
    your_score = get_similarity_score(v1, v2)
    com_ind = np.intersect1d(np.nonzero(v1), np.nonzero(v2))
    true_score=0
    for i in com_ind:
        true_score += v1[i]*v2[i]

    assert np.isclose(your_score, true_score), \
        "The result is not correct! \n cat1 = {} \n cat2 = {} \n your score = {} \n true score = {}".format(c1,c2,your_score,true_score)

print("\n Passed!")
```

    
     Passed!
    

**Instructions to prepare for Exercise 3.** The final objective of this problem is to create a "game graph." In such a graph, each game is a node; an edge connecting the nodes exists if the similarity score between the two corresponding board games is greater than a predefined threshold. The example below shows how a game graph is generated.

For instance, consider the categories of the following 3 games. The code prints categories of these 3 games:

```python
for i in range(6,9): print(i, data.loc[i,'category'])
```
- Civilization, Economic, Fantasy, Territory Building
- Civilization, Economic, Fighting, Miniatures, Science Fiction, Territory Building
- Ancient, Card Game, City Building, Civilization

**Step 1)** Below is a **unique list of alphabetically sorted categories for this example**.

```python
['American West', 'Ancient', 'Card Game', 'City Building', 'Civilization', 'Economic', 'Fantasy', 'Fighting', 'Miniatures', 'Science Fiction', 'Territory Building']
```

> In the case of complete dataset, you should have found 84 categories in Exercise 0.

**Step 2)** Below are **normalized category vectors for these games**

```python
# Game 6
[ 0.   0.   0.   0.   0.5  0.5  0.5  0.   0.   0.   0.5]

# Game 7
[ 0.   0.   0.   0.   0.40824829  0.40824829  0.    0.40824829  0.40824829  0.40824829  0.40824829]

# Game 8
[ 0.   0.5  0.5  0.5  0.5  0.   0.   0.   0.   0.   0. ]
```

> In the case of complete dataset, these vectors will have the size of $1 \times 84$.

**Step 3)** Getting the similarity score between these metrices we can produce a $3 \times 3$ matrix, where each row and column represent a game. The **similarity matrix** for this game can be written as,

```python
[[ 1.          0.61237244  0.25      ]
 [ 0.61237244  1.          0.20412415]
 [ 0.25        0.20412415  1.        ]]
```

Note that similarity of the game with itself will always be 1. Therefore the diagonal entries of this matrix will always be 1.

> In the case of complete dataset, this matrix will be of the size 4999x4999 as there are a total of 4999 games.

**Step 4)** With the `THRESHOLD` = 0.6, the final **game graph** becomes

```python
[[1 1 0]
 [1 1 0]
 [0 0 1]]
```

This shows that games with index 6 and 7 in the original dataset are similar to each other.

**Exercise 3** (4 points). Write some code to create a sparse CSR matrix named **`game_graph`** that represents a game graph as described previously.

A few points to note:
1. The input dataset, named **`data`**, has 4999 games.
2. Take the index of a game in the input dataframe to be the game's index. The index 0 of the input dataframe should also corresponds row 0 and column 0 of the output sparse matrix.
3. You will need to calculate normalized category vector for each of the games.
4. You will then need to find similarity between each pair of the games.
5. The final output **game_graph** should be a 4999x4999 CSR sparse matrix.

A few more points to note:
1. 4999x4999 is a fairly large matrix.
2. 4999 normalized game category vectors, each of size (1x84) also forms a large matrix.
3. Be cautious when using for loops with normal numpy arrays as they will take a considerable amount of time to run.
4. Storing these large matrices into a sparse matrix format would improve the performance significantly.
5. For efficiency's sake, sparse matrix operations like `vstack()`, `transpose()`, and `dot()`, may prove to be convenient.

## Here is my solution

It works. However, it is not as slick and as fast as the solution provided by the school.

The idea that I used was to permute all normalized vectors ($n^r$) and then compute a similarity score for each permuation. Needless to say that it takes longer.

The idea that the instructor used was that if you need to compute a dot product between every vector in an matrix or an array of vectors, then you can do this by computing $A \cdot A^T$ (among few other ideas that instructor used).



```Python
import scipy.sparse as sp
from itertools import product
THRESHOLD = 0.6
num_games = len(data)


category = data['category'].tolist()

normalized_category_vectors = {}
for i, game_category in enumerate(category):
    normalized_category_vectors[i] = get_normalized_category_vector(game_category, categories)

I, J = [], []
V = []

for ai, aj in product(np.array(list(normalized_category_vectors.items())), repeat = 2):
    V.append(get_similarity_score(ai[1], aj[1]))
    I.append(ai[0])
    J.append(aj[0])

game_graph = sp.csr_matrix((V, (I, J)), shape=(num_games, num_games))

game_graph = (game_graph > THRESHOLD).astype(int)
```


```python
# school solution
import scipy.sparse as sp
THRESHOLD = 0.6
num_games = len(data)

game_cat_vectors = []

# Get normalized game category vectors
for i in range(num_games):
    game_cat_vectors.append(sp.csr_matrix(get_normalized_category_vector(data.loc[i, 'category'], categories)))

# Create a 4999 cross 84 matrix
game_cat_vectors = sp.vstack(game_cat_vectors)

# Take a dot product to get similarity scores
game_graph = game_cat_vectors.dot(game_cat_vectors.transpose())

# Convert to binary based on the threshold
game_graph = (game_graph>THRESHOLD).astype(int)
```


```python
# Test Cell: Exercise 3

assert type(game_graph) is sp.csr.csr_matrix, "We require a sparse matrix in the CSR format!"
assert game_graph.shape[0]==4999, "The number of rows is not correct!"
assert game_graph.shape[1]==4999, "The number of columns is not correct!"
assert sp.csr_matrix.count_nonzero(game_graph)==606977, "The number of nonzero entries is not correct!"

# Check the result randomly 

ind1 = np.random.randint(0, len(data),100)
ind2 = np.random.randint(0, len(data),100)
for i in ind1:
    for j in ind2:
        v1 = get_normalized_category_vector(data.loc[i,]['category'],categories)
        v2 = get_normalized_category_vector(data.loc[j,]['category'],categories)
        sim_score = v1.dot(v2)
        con = 1 if sim_score > THRESHOLD else 0
        assert game_graph[i,j]==game_graph[j,i]==con,"Your result is wrong at position {} or {}".format((i,j), (j,i))
        if not game_graph[i,j]==game_graph[j,i]==con:
            print("Your result at position {} is {}".format((i,j),game_graph[i,j])) 
            print("Your result at position {} is {}".format((j,i),game_graph[j,i]))
            print("Correct value at both positions should be {}".format(con))


print("\n Passed!")

```

    
     Passed!
    

**Fin!** That's the end of this problem. Don't forget to restart and run this notebook from the beginning to verify that it works top-to-bottom before submitting. You can move on to the next problem
