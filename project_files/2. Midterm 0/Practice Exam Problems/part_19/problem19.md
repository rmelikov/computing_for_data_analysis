# Problem 3: Gaussian Naïve Bayes Classification for Predicting Protein Localization Sites

_Version 1.2_

This notebook concerns a machine learning method called the _(Gaussian) naïve Bayes classifier_. It finds uses in text categorization, spam filters, and biomedicine.

In this problem, you'll apply naïve Bayes to the problem of predicting protein localization sites in [E.Coli bacteria](https://en.wikipedia.org/wiki/Escherichia_coli). In essence, the problem is to estimate where particular proteins reside in a cell. If, later on, you want to learn more, see the references at the end of this notebook.

This notebook asks you to implement the method using only constructs from basic Python, without auxiliary libraries like `numpy` or `sklearn`.

## Setup and data

You will use a modified version of [a publicly available dataset](https://archive.ics.uci.edu/ml/datasets/ecoli). Run the code cell below to load this data, whose output we'll explain afterwards.


```python
import math
from problem_utils import load_data
from pprint import pprint # For pretty-printing Python data structures

x_train, y_train, x_test, y_test = load_data('ecoli-mod.mat')

print ("\nTraining data for first 5 samples:")
pprint(x_train[:5], width=100)
print ("\nTraining results for first 5 samples:")
print(y_train[:5])
print(f"\nThe possible class labels are: {set(y_train + y_test)}")
```

    Loading dataset: ecoli-mod.mat...
    
    Some information about the given data:
    218 total training samples having 5 features each
    109 total testing samples having 5 features each
    
    Training data for first 5 samples:
    [{'feature_1': 0.44, 'feature_2': 0.28, 'feature_3': 0.43, 'feature_4': 0.27, 'feature_5': 0.37},
     {'feature_1': 0.31, 'feature_2': 0.36, 'feature_3': 0.58, 'feature_4': 0.94, 'feature_5': 0.94},
     {'feature_1': 0.58, 'feature_2': 0.55, 'feature_3': 0.57, 'feature_4': 0.7, 'feature_5': 0.74},
     {'feature_1': 0.38, 'feature_2': 0.44, 'feature_3': 0.43, 'feature_4': 0.2, 'feature_5': 0.31},
     {'feature_1': 0.29, 'feature_2': 0.28, 'feature_3': 0.5, 'feature_4': 0.42, 'feature_5': 0.5}]
    
    Training results for first 5 samples:
    ['class_1', 'class_2', 'class_2', 'class_1', 'class_1']
    
    The possible class labels are: {'class_4', 'class_3', 'class_2', 'class_1', 'class_5'}
    

**About these data.** The data is split into _training data_, which you'll use to build a predictive model, and _testing data_, which you'll use to test the accuracy of the model. There are four variables of interest: `x_train` and `y_train`, which hold the training data, and `x_test` and `y_test`, which hold the testing data. More specifically:

- `x_train` is a _list of dictionaries_. Each element `x_train[i]` is the `i`-th data point of the training set. The point is represented by a _feature vector_, which is a linear algebraic vector having five components. Each vector is stored as a dictionary, with its components named by the keys `'feature_1'` through `'feature_5'`.

- `y_train` is a _list of strings_. Each element `y_train[i]` is a _class label_ for the `i`-th data point. Observe from the output above that there are five possible class labels, `'class_1'` through `'class_5'`.

- `x_test` and `y_test` are similar to the above, except that they hold values for the test data. Our goal is to build a model of the training data that can closely predict the true class labels, `y_test`, given only the feature vectors in `x_test`.

## Background on Bayes' Theorem

Recall _Bayes' theorem_ (or Bayes' law or Bayes' rule) from Topic 3. It is a statement about the quantitative relationships among the [conditional probablilities](https://stats.stackexchange.com/questions/239014/bayes-theorem-intuition) (recall Notebook 2) of several events. One common use of Bayes' theorem is to "reverse" a conditional relationship, such as estimating the probability that [rich people are happy](https://www.quora.com/What-is-an-intuitive-explanation-of-Bayes-Rule) given knowledge of the probability that a happy person is rich. Another use is to [update one's belief](https://arbital.com/p/bayes_rule/?l=1zq) using prior knowledge when new information arrives, like updating the probability that a person has cancer when he or she now tests positive, knowing some background information on the accuracy of the test.

The mathematical statement of Bayes' theorem is

$${\displaystyle P(A\mid B)={\frac {P(B\mid A)P(A)}{P(B)}}},$$

where ${\displaystyle A}$ and ${\displaystyle B}$ are events and ${\displaystyle P(B)\neq 0}$.

- ${\displaystyle P(A\mid B)}$ is a posterior probability of event ${\displaystyle A}$ given event ${\displaystyle B}$, or just **posterior**.
- ${\displaystyle P(B\mid A)}$ is a conditional probability of event ${\displaystyle B}$ given event ${\displaystyle A}$, also called **likelihood**.
- ${\displaystyle P(A)}$ is prior probability of event ${\displaystyle A}$ independently of ${\displaystyle B}$, also called just **prior**.
- ${\displaystyle P(B)}$ is the probabilities of observing ${\displaystyle B}$ independently ${\displaystyle A}$, also called marginal likelihood or model **evidence**.

In words, the formula would be

$$ \text{posterior} \ = \ \frac{\text{prior} \times \text{likelihood}}{\text{evidence}}.$$

You can read more about Bayes' Theorem on its [Wiki page](https://en.wikipedia.org/wiki/Bayes%27_theorem).

## Part 0: The Naïve Bayes Model

We can use Bayes' theorem to make predictions not only for toy problems with two events, but also for much more complex problems, such as multinomial classification for data with multiple features. However, finding the likelihood of a multidimensional feature vector given the assigned class can be very hard and even [intractable](https://en.wikipedia.org/wiki/Naive_Bayes_classifier#Probabilistic_model). However, we can greatly simplify the computation if we make a _naïve_ assumption that all the features are conditionally independent. In this case, the probability of observing a class label for a given feature vector becomes

$${\displaystyle p(C_{k}\mid \mathbf {x} )={\frac {p(C_{k})\ \prod _{i=1}^{n}p(x_{i}\mid C_{k})}{p(\mathbf {x} )}}\,}$$

Moreover, since evidence $p(\mathbf {x} )$ does not depend on $C$, in practice we can omit it and use the formula

$${\displaystyle {\begin{aligned}p(C_{k}\mid \mathbf {x} )&\varpropto p(C_{k})\prod _{i=1}^{n}p(x_{i}\mid C_{k})\,\end{aligned}}} \tag{1}$$

where ${\displaystyle \varpropto }$ denotes proportionality.

**Learning class priors.** Before moving on, let's write a function that computes class priors, $p(C_k)$ using the vector of class assignments. Recall that `y_train` holds the class labels.

**Exercise 0** (2 points). Write a function `prior(data)` that takes a list of class values as inputs and returns a dictionary containing the priors. In this dictionary, the class names serve as keys and the probability of the occurrence of each class as values.

We can find prior of class $K$ simply dividing number of data samples of class $K$ by total number of data samples. For example, if

```python
data = ['class_1' , 'class_2' , 'class_3' , 'class_4' , 'class_3' , 'class_2' , 'class_1' , 'class_3' ]
```

then your function should produce the output,

```python
prior(data) == {'class_1': 0.25,'class_2': 0.25,'class_3': 0.375, 'class_4': 0.125 }
```


```python
def prior(data):
    assert isinstance(data, list), f"Input `data` has type `{type(data)}`, which does not derive from `list` as expected."
    from collections import Counter
    count = Counter (data)
    return {k:v/sum(count.values()) for k,v in count.items()}

```


```python
# Test cell: `test_prior_script`

#Test Case 1
test_data = ['a1' , 'a2' , 'b1' , 'b2' , 'b1' , 'a2' , 'a1' , 'b1' ]
p1 = prior(test_data)
p1_test = {'a1': 0.25,'a2': 0.25,'b1': 0.375, 'b2': 0.125 }
assert isinstance(p1, dict), f"Input `p1` has type `{type(p1)}`, which does not derive from `dict` as expected."
assert p1 == p1_test, f"The Output for Test Case 1 is incorrect, returned: \n{p1}\n, should be: \n{p1_test}\n"

#Test Case 2
test_data2 = ['a1' , 'a2' , 'b1' , 'b2' , 'b1' , 'a2' , 'a1' , 'b1' , 'a1' , 'a2' , 'b1' , 'b2' , 'b1' , 'a2' , 'a1' , 'b1' , 'a2' , 'a2' , 'b1' , 'b2' , 'b1' , 'c1' , 'a1' , 'b2' , 'a1' , 'b2' , 'b1' , 'c2' , 'b1' , 'a2' , 'a3' , 'b1',  'a3' , 'a2' , 'b4' , 'b2' , 'b1' , 'a2' , 'a1' , 'b1' , 'a3' , 'a2' , 'b1' , 'b2' , 'b1' , 'a2' , 'a3' , 'b2' ]
p2 = prior(test_data2)
p2_test = {'a1': 0.14583333, 'a2': 0.22916667, 'b1': 0.3125, 'b2': 0.16666667, 'c1': 0.02083333, 'c2': 0.02083333, 'a3': 0.08333333, 'b4': 0.02083333}
assert isinstance(p2, dict), f"Input `p2` has type `{type(p2)}`, which does not derive from `dict` as expected."
for k in p2_test.keys():
    assert math.isclose(p2[k], p2_test[k], abs_tol=1e-8), f"The Output for Test Case 2 is incorrect, returned: \n{p2:.8f}\n, should be: \n{p2_test}\n"

print("\n(Passed!)")
```

    
    (Passed!)
    

Assuming your implementation really is correct, run the code below to inspect the priors for our dataset.


```python
prior_val = prior(y_train)
print("\nClass Priors:")
print(prior_val)
```

    
    Class Priors:
    {'class_1': 0.4724770642201835, 'class_2': 0.23394495412844038, 'class_4': 0.10091743119266056, 'class_3': 0.14220183486238533, 'class_5': 0.05045871559633028}
    

## Part 1: The Gaussian Naïve Bayes model

In real problems we often do not know the true underlying data distributions. Instead, we estimate them from the data, often assuming the form (but not the parameters) of the data distribution.

When dealing with continuous data, a typical assumption is that any continuous values associated with each class are distributed according to a [normal (or Gaussian) distribution](https://en.wikipedia.org/wiki/Normal_distribution). Under this assumption, the likelihood is

$${\displaystyle p(x=f_i\mid C_{k})= N(f_i;µ_{i,k},σ_{i,k}^2)}$$

$${\displaystyle p(x=f_i\mid C_{k})={\frac {1}{\sqrt {2\pi \sigma _{i,k}^{2}}}}\, \exp\left({-{\frac {(f_i-\mu _{i,k})^{2}}{2\sigma _{i,k}^{2}}}}\right)} \tag{2}$$

where $µ_{i,k}$ is the mean of feature $f_i$ and $σ_{i,k}^2$ is its variance.

**Exercise 1.a** (1 point). Write two functions, `mean(vector)` and `var(vector)`, to compute the mean and variance of the components of an input vector, respectively. The vector, `vector`, is given as a list of values and you need to return the mean and variance of those values.

Recall that the mean and variance are defined by

$$ \mathbf {mean}(\mathbf {x}) = \mu(\mathbf {x})= \frac{1}{n}\sum_{i=1}^n{x_i} \quad \mbox{and} \quad
\mathbf {var}(\mathbf {x}) = \frac{1}{n} \sum_{i=1}^n (x_i - \mu)^2.
$$

For example, suppose `vector = [1, 2, 3]`. Then,

```python
mean([1, 2, 3]) = 2.0 
var([1, 2, 3]) = 0.66666667
```

In the context of our problem, these functions would be helpful while implementing equation (2). 


```python
def mean(vector):
    return sum(vector) / len(vector)

def var(vector):
    avg = mean(vector)
    return sum([(x - avg)**2 for x in vector]) / len(vector)

```


```python
# Test cell: `test_mean_var`

#Test Case 1
l = [1.0 , 5.272 , 6.2734 , 32.4824 , 8.2876 , 43.3242 ]
m = mean(l)
v = var(l)
assert isinstance(m, float), f"Input `m` has type `{type(m)}`, which does not derive from `float` as expected."
assert isinstance(v, float), f"Input `v` has type `{type(v)}`, which does not derive from `float` as expected."
assert math.isclose(m, 16.1066, abs_tol=1e-8), f"Test Case 1 for Mean Failed as `{m:.8f}` != 16.1066"
assert math.isclose(v, 252.06517989, abs_tol=1e-8), f"Test Case 1 for Variance Failed as `{v:.8f}` != 252.06517989"

#Test Case 2
l1 = [11.0423 , 6.34324 , 7.347234 , 426.244 , 247. , 232.4332476 , 9. , 9. ,9. , -3.432 , 0. , -34.234 ]
m1 = mean(l1)
v1 = var(l1)
assert isinstance(m1, float), f"Input `m` has type `{type(m1)}`, which does not derive from `float` as expected."
assert isinstance(v1, float), f"Input `v` has type `{type(v1)}`, which does not derive from `float` as expected."
assert math.isclose(m1, 76.64533513, abs_tol=1e-8), f"Test Case 2 for Mean Failed as `{m1:.8f}` != 76.64533513"
assert math.isclose(v1, 18988.91413866, abs_tol=1e-8), f"Test Case 2 for Variance Failed as `{v1:.8f}` != 18988.91413866"

print("\n(Passed!)")
```

    
    (Passed!)
    

**Exercise 1.b** (2 points). Using `mean(l)` and `var(l)`, implement a function, `likelihood(x, y)`, to compute the parameters of the Gaussian likelihood of the data, `x` and `y`.

In particular, the input `x` is a list of dictionaries containing feature vectors (e.g., `x_train` from the training data) and `y` is a list of class labels (e.g., `y_train`). The output is a _pair of dictionaries of dictionaries_ (yikes!!), one holding means and the other variances. It's easiest to understand the inputs and outputs by example, so let's start there.

Suppose the inputs `x` and `y` correspond to four data points, where the feature vectors have three components and there are two distinct class labels:

```python
x = [{'feature_1': 0.58, 'feature_2': 0.55, 'feature_3': 0.57},
     {'feature_1': 0.38, 'feature_2': 0.44, 'feature_3': 0.43},
     {'feature_1': 0.29, 'feature_2': 0.28, 'feature_3': 0.5},
     {'feature_1': 0.98, 'feature_2': 0.74, 'feature_3': 0.32}] 
y = ['class_1', 'class_2', 'class_1', 'class_2']
```

Your function should return two outputs,

```python
#Note: The result is spread out in different lines to provide more clarity
dist_mean, dist_var = likelihood(x_train, y_train)
```

where

```python
# Dictionary corresponding to Mean of each feature in a particular class. 
dist_mean == {'class_1': {'feature_1': 0.435, 'feature_2': 0.415, 'feature_3': 0.535},
              'class_2': {'feature_1': 0.680, 'feature_2': 0.59, 'feature_3': 0.375}}

# Dictionary corresponding to Variance of each feature in a particular class.
dist_var == {'class_2': {'feature_1': 0.09, 'feature_2': 0.0225, 'feature_3': 0.003025},
             'class_1': {'feature_1': 0.021025, 'feature_2': 0.018225, 'feature_3': 0.001225}}
```

Consider `dist_mean`. It is a dictionary whose keys are class labels and whose values are _mean_ feature vectors. For instance, consider the vectors for the `'class_1'` data points in `x`. In the `'feature_1'` component, the values that occur are 0.58 and 0.29; therefore, `dist_mean['class_1']['feature_1'] == (0.58 + 0.29) / 2 == 0.435`.

In the function you are to complete, below, we've created two empty dictionaries to hold your results and return them. You need to supply the code that populates them.


```python
def likelihood(x, y):
    dist_mean = {}
    dist_var = {}
    ### BEGIN SOLUTION
    for label in set(y):
        count_list = [x[n] for n in range(len(x)) if y[n] == label]
        dist_mean[label] = {k: mean([el[k] for el in count_list]) for k in count_list[0].keys()}
        dist_var[label] = {k: var([el[k] for el in count_list]) for k in count_list[0].keys()}
    ### END SOLUTION
    return dist_mean, dist_var
```


```python
# Test cell: `test_likelihood`

#Test Case 1
training_data = [{'feature_1': 0.58, 'feature_2': 0.55, 'feature_3': 0.57, 'feature_4': 0.7, 'feature_5': 0.74},{'feature_1': 0.38, 'feature_2': 0.44, 'feature_3': 0.43, 'feature_4': 0.2, 'feature_5': 0.31},{'feature_1': 0.29, 'feature_2': 0.28, 'feature_3': 0.5, 'feature_4': 0.42, 'feature_5': 0.5},{'feature_1': 0.98, 'feature_2': 0.74, 'feature_3': 0.32, 'feature_4': 0.25, 'feature_5': 0.11},{'feature_1': 0.08, 'feature_2': 0.69, 'feature_3': 0.84, 'feature_4': 0.85, 'feature_5': 0.17} ] 

training_result = ['class_1' , 'class_2' , 'class_1' , 'class_2','class_2']

mean1, var1 = likelihood(training_data, training_result)
assert isinstance(mean1, dict), f"Input `mean1` has type `{type(mean1)}`, which does not derive from `dict` as expected."
assert isinstance(var1, dict), f"Input `var1` has type `{type(var1)}`, which does not derive from `dict` as expected."
mean1_test = {'class_1': {'feature_1': 0.435, 'feature_2': 0.415, 'feature_3': 0.535, 'feature_4': 0.56, 'feature_5': 0.62},'class_2': {'feature_1': 0.48, 'feature_2': 0.62333333, 'feature_3': 0.53, 'feature_4': 0.43333333, 'feature_5': 0.19666667}}
var1_test = {'class_1': {'feature_1': 0.021025, 'feature_2': 0.018225, 'feature_3': 0.001225, 'feature_4': 0.0196, 'feature_5': 0.0144},'class_2': {'feature_1': 0.14, 'feature_2': 0.01722222, 'feature_3': 0.05006667, 'feature_4': 0.08722222, 'feature_5': 0.00702222}}

for cl in mean1_test.keys():
    for f in mean1_test[cl].keys():
        assert math.isclose(mean1[cl][f], mean1_test[cl][f], abs_tol=1e-8), f"The Output for Test Case 1 is incorrect, returned: \n{mean1:.8f}\n, should be: \n{mean1_test}\n"
for cl in var1_test.keys():
    for f in var1_test[cl].keys():
        assert math.isclose(var1[cl][f], var1_test[cl][f], abs_tol=1e-8), f"The Output for Test Case 1 is incorrect, returned: \n{var1:.8f}\n, should be: \n{var1_test}\n"

#Test Case 2
training_data2 = [{'feature_1': 0.9238, 'feature_2': 0.34, 'feature_3': 0.57, 'feature_4': 0.7, 'feature_5': 0.747},{'feature_1': 0.3842, 'feature_2': 0.4234, 'feature_3': 0.2343, 'feature_4': 0.2, 'feature_5': 0.331},{'feature_1': 0.2129, 'feature_2': 0.0228, 'feature_3': 0.425, 'feature_4': 0.835, 'feature_5': 0.587}] 

training_result2 = ['class_1' , 'class_2' , 'class_1']

mean2, var2 = likelihood(training_data2, training_result2)
assert isinstance(mean2, dict), f"Input `mean2` has type `{type(mean2)}`, which does not derive from `dict` as expected."
assert isinstance(var2, dict), f"Input `var2` has type `{type(var2)}`, which does not derive from `dict` as expected."
mean2_test = {'class_1': {'feature_1': 0.56835, 'feature_2': 0.1814, 'feature_3': 0.4975, 'feature_4': 0.7675, 'feature_5': 0.667},'class_2': {'feature_1': 0.3842, 'feature_2': 0.4234, 'feature_3': 0.2343, 'feature_4': 0.2, 'feature_5': 0.331} }
var2_test = {'class_1': {'feature_1': 0.1263447, 'feature_2': 0.02515396, 'feature_3': 0.00525625, 'feature_4': 0.00455625, 'feature_5': 0.0064},'class_2': {'feature_1': 0.0, 'feature_2': 0.0, 'feature_3': 0.0, 'feature_4': 0.0, 'feature_5': 0.0}}

for cl in mean2_test.keys():
    for f in mean2_test[cl].keys():
        assert math.isclose(mean2[cl][f], mean2_test[cl][f], abs_tol=1e-8), f"The Output for Test Case 2 is incorrect, returned: \n{mean2:.8f}\n, should be: \n{mean2_test}\n"
for cl in var2_test.keys():
    for f in var2_test[cl].keys():
        assert math.isclose(var2[cl][f], var2_test[cl][f], abs_tol=1e-8), f"The Output for Test Case 2 is incorrect, returned: \n{var2:.8f}\n, should be: \n{var2_test}\n"

print("\n(Passed!)")
```

    
    (Passed!)
    

**Inspecting the results.** Now that you have successfully defined the function to calculate the likelihood, let us see the output of the function. We have also provided a helper function for pretty printing the mean and variance dictionaries.


```python
# Helper function for pretty printing the Mean and Variance dictionaries.
def pretty_print_mean_var(m,v):
    
    # Convert the contents of a dictionary to be used for displaying in a user friendly manner.
    def formatted_dict(d):
        import json
        return json.dumps(d,sort_keys=True,indent=4)
        
    print("\nPretty Printing Output:")
    print("Mean:\n",formatted_dict(m))
    print("Variance:\n",formatted_dict(v))
```


```python
dist_mean, dist_var = likelihood(x_train,y_train)

print("\nOriginal Output:\n")
print("Mean:\n",dist_mean)
print("Variance:\n",dist_var)

pretty_print_mean_var(dist_mean,dist_var)
```

    
    Original Output:
    
    Mean:
     {'class_4': {'feature_1': 0.7390909090909091, 'feature_2': 0.47045454545454546, 'feature_3': 0.5813636363636364, 'feature_4': 0.7490909090909091, 'feature_5': 0.7690909090909092}, 'class_3': {'feature_1': 0.651290322580645, 'feature_2': 0.7158064516129031, 'feature_3': 0.4303225806451614, 'feature_4': 0.47000000000000014, 'feature_5': 0.3870967741935484}, 'class_2': {'feature_1': 0.47176470588235286, 'feature_2': 0.5054901960784315, 'feature_3': 0.526862745098039, 'feature_4': 0.7554901960784315, 'feature_5': 0.7101960784313728}, 'class_1': {'feature_1': 0.3594174757281553, 'feature_2': 0.4072815533980582, 'feature_3': 0.45339805825242735, 'feature_4': 0.3079611650485437, 'feature_5': 0.3930097087378642}, 'class_5': {'feature_1': 0.6890909090909091, 'feature_2': 0.69, 'feature_3': 0.7672727272727272, 'feature_4': 0.48181818181818187, 'feature_5': 0.3154545454545455}}
    Variance:
     {'class_4': {'feature_1': 0.010844628099173555, 'feature_2': 0.009504338842975206, 'feature_3': 0.003939049586776858, 'feature_4': 0.005635537190082647, 'feature_5': 0.004780991735537189}, 'class_3': {'feature_1': 0.010817689906347553, 'feature_2': 0.021353381893860563, 'feature_3': 0.006370863683662852, 'feature_4': 0.012438709677419358, 'feature_5': 0.01572382934443288}, 'class_2': {'feature_1': 0.04263414071510957, 'feature_2': 0.007828681276432143, 'feature_3': 0.016327412533640905, 'feature_4': 0.012295347943098807, 'feature_5': 0.036327412533640906}, 'class_1': {'feature_1': 0.015135582995569788, 'feature_2': 0.007635328494674332, 'feature_3': 0.009022433782637385, 'feature_4': 0.009072542181166932, 'feature_5': 0.008968611556225848}, 'class_5': {'feature_1': 0.004190082644628099, 'feature_2': 0.010818181818181817, 'feature_3': 0.004256198347107437, 'feature_4': 0.006651239669421486, 'feature_5': 0.0107702479338843}}
    
    Pretty Printing Output:
    Mean:
     {
        "class_1": {
            "feature_1": 0.3594174757281553,
            "feature_2": 0.4072815533980582,
            "feature_3": 0.45339805825242735,
            "feature_4": 0.3079611650485437,
            "feature_5": 0.3930097087378642
        },
        "class_2": {
            "feature_1": 0.47176470588235286,
            "feature_2": 0.5054901960784315,
            "feature_3": 0.526862745098039,
            "feature_4": 0.7554901960784315,
            "feature_5": 0.7101960784313728
        },
        "class_3": {
            "feature_1": 0.651290322580645,
            "feature_2": 0.7158064516129031,
            "feature_3": 0.4303225806451614,
            "feature_4": 0.47000000000000014,
            "feature_5": 0.3870967741935484
        },
        "class_4": {
            "feature_1": 0.7390909090909091,
            "feature_2": 0.47045454545454546,
            "feature_3": 0.5813636363636364,
            "feature_4": 0.7490909090909091,
            "feature_5": 0.7690909090909092
        },
        "class_5": {
            "feature_1": 0.6890909090909091,
            "feature_2": 0.69,
            "feature_3": 0.7672727272727272,
            "feature_4": 0.48181818181818187,
            "feature_5": 0.3154545454545455
        }
    }
    Variance:
     {
        "class_1": {
            "feature_1": 0.015135582995569788,
            "feature_2": 0.007635328494674332,
            "feature_3": 0.009022433782637385,
            "feature_4": 0.009072542181166932,
            "feature_5": 0.008968611556225848
        },
        "class_2": {
            "feature_1": 0.04263414071510957,
            "feature_2": 0.007828681276432143,
            "feature_3": 0.016327412533640905,
            "feature_4": 0.012295347943098807,
            "feature_5": 0.036327412533640906
        },
        "class_3": {
            "feature_1": 0.010817689906347553,
            "feature_2": 0.021353381893860563,
            "feature_3": 0.006370863683662852,
            "feature_4": 0.012438709677419358,
            "feature_5": 0.01572382934443288
        },
        "class_4": {
            "feature_1": 0.010844628099173555,
            "feature_2": 0.009504338842975206,
            "feature_3": 0.003939049586776858,
            "feature_4": 0.005635537190082647,
            "feature_5": 0.004780991735537189
        },
        "class_5": {
            "feature_1": 0.004190082644628099,
            "feature_2": 0.010818181818181817,
            "feature_3": 0.004256198347107437,
            "feature_4": 0.006651239669421486,
            "feature_5": 0.0107702479338843
        }
    }
    

Great Work! Now that we have computed the conditional mean and variance, we can finally move onto the core part of the algorithm.

## Part 2: Gaussian Naïve Bayes Classifier

The Naïve Bayes classifier combines the Naïve Bayes model with a _decision rule_, meaning a scheme that decides what label to assign to a given feature vector.

One common rule is to pick the hypothesis that is most probable. This approach is known as the _maximum a posteriori_ or MAP decision rule. The corresponding _Bayes classifier_ assigns a class label ${\displaystyle {\hat {y}}=C_{k}}$ for some $k$ as follows:

$${\displaystyle {\hat {y}}={\underset {k\in \{1,\dots ,K\}}{\operatorname {argmax} }}\ p(C_{k})\displaystyle \prod _{i=1}^{n}p(f_{i}\mid C_{k}).} \tag{3}$$

Unfortunately, an "obvious" implementation of this rule can have numerical instabilities because it requires multiplying exponentials with very different numerical ranges. Moreover, we can end up with very small numbers, which reduce accuracy and computation performance. To avoid these issues, can instead take argmax of _logarithm_ of the posterior, which is more stable and produces the same result. (Recall Problem 9 of the Practice Problems Midterm 1!) Applying $\log$ to equation (3) yields

$${\displaystyle {\hat {y}} = {\underset {k\in \{1,\dots ,K\}}{\operatorname {argmax} }}\log \left( p(C_{k})\displaystyle \prod _{i=1}^{n}p(f_{i}\mid C_{k}) \right) = {\underset {k\in \{1,\dots ,K\}}{\operatorname {argmax} }}\ \log p(C_{k}) +\displaystyle \sum _{i=1}^{n}\log p(f_{i}\mid C_{k})} \tag{4}.$$

You already wrote code to compute $p(C_k)$, so now let's work on the second term of (4), the log-likelihood.

**Computing log-likelihood of a single class.** The rightmost sum of equation (4) above is **log-likelihood.** In particular, it is the logarithm of likelihood of class $k$ given the feature vector $\mathbf{f}$.

To discern its mathematical form, suppose we substitute equation (2) into the log-likelihood term. Then,

$$ L(\mathbf {f} \mid C_{k}) =  \sum _{i=1}^{n}\log p(f_{i}\mid C_{k}) = \sum _{i=1}^{n} \log \left( {\frac {1}{\sqrt {2\pi \sigma _{i,k}^{2}}}}\,\exp\left(-{\frac {(f_i-\mu _{i,k})^{2}}{2\sigma _{i,k}^{2}}}\right) \right) = 
\sum _{i=1}^{n} \left( -0.5 \log (2 \pi \sigma _{i,k}^{2}) - 0.5 \frac {(f_i-\mu _{i,k})^{2}}{\sigma _{i,k}^{2}} \right) $$


**Exercise 2.a** (1 point). Complete the `log_likelihood(x, m, v)` function, below. The inputs are:

- `x`, a feature vector $\mathbf{f}$ of **one** data point from, say, `x_test`, which you'll recall is a dictionary with features as keys and scores as values;
- `m`, a dictionary of means $\mu _{i,k}$ for a single class $k$, with features as keys and their mean scores as values.
- `v` is a dictionary of variances $\sigma _{i,k}^{2}$ for a single class $k$, with features as keys and the variance of their scores as values.


```python
def log_likelihood(x, m, v):
    return -0.5 *  sum([math.log(2 * math.pi * v[f]) + ((x[f] - m[f])**2) / v[f] for f in x.keys()])

```


```python
# Test cell: `test_logprob`

#Test Case 1
training_data = {'feature_1': 0.58, 'feature_2': 0.55, 'feature_3': 0.57, 'feature_4': 0.7, 'feature_5': 0.74}
mean1_test = {'feature_1': 0.435, 'feature_2': 0.415, 'feature_3': 0.535, 'feature_4': 0.56, 'feature_5': 0.62}
var1_test = {'feature_1': 0.021025, 'feature_2': 0.018225, 'feature_3': 0.001225, 'feature_4': 0.0196, 'feature_5': 0.0144}

res1 = log_likelihood(training_data,mean1_test,var1_test)
assert math.isclose(res1, 4.277592981147556, abs_tol=1e-8), f"The Output for Test Case 1 is incorrect, returned: \n{res1:.8f}\n, should be: \n{4.277592981147556}\n"

#Test Case 2
training_data2 = {'feature_1': 0.9238, 'feature_2': 0.34, 'feature_3': 0.57, 'feature_4': 0.7, 'feature_5': 0.747}
mean2_test = {'feature_1': 0.56835, 'feature_2': 0.1814, 'feature_3': 0.4975, 'feature_4': 0.7675, 'feature_5': 0.667}
var2_test = {'feature_1': 0.1263447, 'feature_2': 0.02515396, 'feature_3': 0.00525625, 'feature_4': 0.00455625, 'feature_5': 0.0064}

res2 = log_likelihood(training_data2,mean2_test,var2_test)
assert math.isclose(res2, 3.6265730328980883, abs_tol=1e-8), f"The Output for Test Case 2 is incorrect, returned: \n{res2:.8f}\n, should be: \n{3.6265730328980883}\n"

print("\n(Passed!)")
```

    
    (Passed!)
    

**Implementing Gaussian Naive Bayes classifier.** You now have everything you need to implement the Gaussian naïve Bayes classifier from equation (4).

$${\displaystyle {\hat {y}} = {\underset {k\in \{1,\dots ,K\}}{\operatorname {argmax} }}\ \log p(C_{k}) +\displaystyle \sum _{i=1}^{n}\log p(f_{i}\mid C_{k})} \tag{4}$$

In particular, recall that you have written these functions:
- `prior()`, which returns **prior** for each class $C_k$;
- `likelihood`, which returns **mean** and **variance** parameters for likelihood distributed as Gaussian;
- and `log_likelihood`, which returns the logarithm of likelihood for Gaussian with given **mean** and **variance**.

Let's now implement a function, `naive_bayes_classifier`, so that it returns a class prediction for testing features $\mathbf{x}$ given **mean** and **variance** parameters for the likelihood and **prior** vector for classes. 

**Exercise 2.b** (4 points: 2 points "exposed" and 2 points hidden).

Complete `naive_bayes_classifier(x, dist_mean, dist_var, prior)` function, using the `log_likelihood()` function and equation (4). Here the inputs are

- `x`, which is a full list of test samples (e.g., `x == x_test`);
- `dist_mean` and `dist_var`, which are the results of a call to `likelihood()`;
- and `prior()`, which is a dict of class priors as returned by a call to `prior()`.

Your function should return a _list of strings_. Each element in the list would be the _predicted class label_ for the i-th data point in the `x_test` test sample. For additional reference, the output of the function should be similar to the contents of the _list_ of class labels in `y_train`.


```python
def naive_bayes_classifier(x_test, dist_mean, dist_var, prior):
    y_pred = []
    for x in x_test:
        pred = {l : math.log(prior[l]) + log_likelihood(x, dist_mean[l], dist_var[l]) for l in dist_mean.keys()}
        y_pred.append(max(pred.items(), key=lambda x:x[1])[0])
    return y_pred
```


```python
# Test cell: `test_nbclassifier_1`

#Test Case 1
nb_test_data_1 = [{'feature_1': 0.58, 'feature_2': 0.55, 'feature_3': 0.57, 'feature_4': 0.7, 'feature_5': 0.74},
                  {'feature_1': 0.38, 'feature_2': 0.44, 'feature_3': 0.43, 'feature_4': 0.2, 'feature_5': 0.31},
                  {'feature_1': 0.29, 'feature_2': 0.28, 'feature_3': 0.5, 'feature_4': 0.42, 'feature_5': 0.5},
                  {'feature_1': 0.98, 'feature_2': 0.74, 'feature_3': 0.32, 'feature_4': 0.25, 'feature_5': 0.11},
                  {'feature_1': 0.08, 'feature_2': 0.69, 'feature_3': 0.84, 'feature_4': 0.85, 'feature_5': 0.17} ] 

nb_p1 = {'class_1': 0.4, 'class_2': 0.6}
nb_m1 = {'class_1': {'feature_1': 0.43499999999999994, 'feature_2': 0.41500000000000004, 'feature_3': 0.5349999999999999, 'feature_4': 0.5599999999999999, 'feature_5': 0.62}, 'class_2': {'feature_1': 0.48, 'feature_2': 0.6233333333333333, 'feature_3': 0.5299999999999999, 'feature_4': 0.43333333333333335, 'feature_5': 0.19666666666666666}}
nb_v1 = {'class_1': {'feature_1': 0.021024999999999995, 'feature_2': 0.018225, 'feature_3': 0.0012249999999999982, 'feature_4': 0.019599999999999996, 'feature_5': 0.0144}, 'class_2': {'feature_1': 0.13999999999999999, 'feature_2': 0.01722222222222222, 'feature_3': 0.050066666666666655, 'feature_4': 0.08722222222222221, 'feature_5': 0.007022222222222222}}

pred1 = naive_bayes_classifier(nb_test_data_1, nb_m1, nb_v1, nb_p1)
assert pred1, f"The resultant list for Test Case 1 is empty."
pred_test1 = ["class_1","class_2","class_1","class_2","class_2"]

for n, x in enumerate (pred1):
    assert x == pred_test1[n], "The result for Test Case 1 is incorrect" 
    
print("\n(Passed!)")
```

    
    (Passed!)
    


```python
# Test cell: `test_nbclassifier_2`

print("""
This test cell will be replaced with one hidden test case.
You will only know the result after submitting to the autograder.
If the autograder times out, then either your solution is highly
inefficient or contains a bug (e.g., an infinite loop).
""")

###
### AUTOGRADER TEST - DO NOT REMOVE
###

```

    
    This test cell will be replaced with one hidden test case.
    You will only know the result after submitting to the autograder.
    If the autograder times out, then either your solution is highly
    inefficient or contains a bug (e.g., an infinite loop).
    
    

## Summary (no additional exercises beyond this point)

If you've completed the above, you have implemented a Gaussian naïve Bayes classifier! The remaining cells run your classifier and assess its accuracy.


```python
print("Naive Bayes Classifier Output:\n")
nb_result = naive_bayes_classifier(x_test, dist_mean, dist_var, prior_val)
print(nb_result)

print("\nPretty Printing to display only the class number: \n")
pred = [int(s[-1]) for s in nb_result]
print(pred)
```

    Naive Bayes Classifier Output:
    
    ['class_3', 'class_5', 'class_2', 'class_4', 'class_1', 'class_1', 'class_2', 'class_2', 'class_1', 'class_5', 'class_1', 'class_1', 'class_1', 'class_3', 'class_1', 'class_3', 'class_1', 'class_1', 'class_1', 'class_4', 'class_4', 'class_3', 'class_1', 'class_3', 'class_3', 'class_2', 'class_1', 'class_3', 'class_1', 'class_1', 'class_2', 'class_2', 'class_2', 'class_1', 'class_1', 'class_4', 'class_1', 'class_1', 'class_5', 'class_2', 'class_2', 'class_1', 'class_1', 'class_1', 'class_1', 'class_3', 'class_2', 'class_1', 'class_5', 'class_3', 'class_4', 'class_4', 'class_2', 'class_1', 'class_3', 'class_1', 'class_4', 'class_3', 'class_4', 'class_4', 'class_2', 'class_3', 'class_2', 'class_5', 'class_4', 'class_4', 'class_4', 'class_5', 'class_5', 'class_1', 'class_1', 'class_2', 'class_3', 'class_3', 'class_4', 'class_4', 'class_1', 'class_5', 'class_4', 'class_3', 'class_3', 'class_2', 'class_1', 'class_1', 'class_2', 'class_3', 'class_3', 'class_1', 'class_2', 'class_4', 'class_2', 'class_1', 'class_1', 'class_3', 'class_1', 'class_1', 'class_3', 'class_2', 'class_1', 'class_2', 'class_1', 'class_3', 'class_2', 'class_1', 'class_1', 'class_4', 'class_4', 'class_1', 'class_1']
    
    Pretty Printing to display only the class number: 
    
    [3, 5, 2, 4, 1, 1, 2, 2, 1, 5, 1, 1, 1, 3, 1, 3, 1, 1, 1, 4, 4, 3, 1, 3, 3, 2, 1, 3, 1, 1, 2, 2, 2, 1, 1, 4, 1, 1, 5, 2, 2, 1, 1, 1, 1, 3, 2, 1, 5, 3, 4, 4, 2, 1, 3, 1, 4, 3, 4, 4, 2, 3, 2, 5, 4, 4, 4, 5, 5, 1, 1, 2, 3, 3, 4, 4, 1, 5, 4, 3, 3, 2, 1, 1, 2, 3, 3, 1, 2, 4, 2, 1, 1, 3, 1, 1, 3, 2, 1, 2, 1, 3, 2, 1, 1, 4, 4, 1, 1]
    

And finally, below you can see accuracy of our classifier, as well as its per-class statistics.


```python
from problem_utils import assess_accuracy
assess_accuracy('ecoli-mod.mat', pred)
```

    Loading dataset: ecoli-mod.mat...
    Accuracy: 0.8348623853211009
    
    
    Class 1
    Prediction positive: 41
    Condition positive: 40
    True positive: 39
    Precision: 0.9512195121951219
    Recall: 0.975
    
    
    Class 2
    Prediction positive: 21
    Condition positive: 26
    True positive: 17
    Precision: 0.8095238095238095
    Recall: 0.6538461538461539
    
    
    Class 3
    Prediction positive: 21
    Condition positive: 21
    True positive: 19
    Precision: 0.9047619047619048
    Recall: 0.9047619047619048
    
    
    Class 4
    Prediction positive: 18
    Condition positive: 13
    True positive: 9
    Precision: 0.5
    Recall: 0.6923076923076923
    
    
    Class 5
    Prediction positive: 8
    Condition positive: 9
    True positive: 7
    Precision: 0.875
    Recall: 0.7777777777777778
    
    
    

**Fin!** Remember to test your solutions by running them as the autograder will: restart the kernel and run all cells from "top-to-bottom." Also remember to submit to the autograder; otherwise, you will not get credit for your hard work!
