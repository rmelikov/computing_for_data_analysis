# Problem 12: Snowball Poem Generator

This problem will test your ability to work with Python data structures as well as working with text data. This problem is vaguely based on work by [*Paul Thompson*](https://github.com/nossidge). (However, you won't find the answer on his GitHub.)

We will work through the creation of a **Snowball poem generator**. Snowball poems are a type of constraint to poetry belonging to the [Oulipo group](https://en.wikipedia.org/wiki/Oulipo). The Oulipo group is a collection of mathematicians and linguists who create writings based on particular linguistic constraints. Snowball poems are governed by the constraint that each successive word be one letter longer than the previous word, and each word takes up a single line. For example:

o  
we  
all  
have  
heard  
people  
believe  
anything  

The texts we will use are *Life on the Mississippi*, *Adventures of Huckleberry Finn*, and *The Adventures of Tom Sawyer* by Mark Twain, obtained from *Project Gutenberg*. 

For this task, much like our Shakespeare generator on the first midterm, we also include the constraint that each word must follow the preceding word at some point in the source text. 

We have cleaned and prepared the data for you, removing all numbers, punctuation outside of pre-determined "stop characters", ".!?;:", all chapter headers, and filtering out all non-English words. 

Run the following code cell to read in the data and display the first 100 characters. 


```python
life_on_miss = open("LifeOnMississippiCleaned.txt", "r").read().lower()
huck_finn = open("HuckFinnCleaned.txt", "r").read().lower()
tom_saw = open("TomSawyerCleaned.txt", "r").read().lower()
print(life_on_miss[:100])
print(huck_finn[:100])
print(tom_saw[:100])

full_text = life_on_miss + " " + huck_finn + " " + tom_saw
```

    the mississippi is well worth reading about. it is not a commonplace river but on the contrary is in
    you dont know about me without you have read a book by the name of the adventures of tom sawyer; but
    tom! no answer. tom! no answer. whats gone with that boy i wonder? you tom! no answer. the old lady 
    

**Exercise 0 (3 pts):** You will now create the first data structure, `snowball_mapping`, a `defaultdict` mapping a word of length `n`, to a set of all words following it in the text that are of length `n + 1`. In other words, each key should be a word, and the values should be the list of words that follow the key word in the source text *and* are one character longer than the key word.

For example, if our source text was:

```python
'This is the full source material. It is not a full novel written by the very esteemed Mark Twain. Woohoo abcdefg.'
```

`snowball_mapping` will look like:

```python 
snowball_mapping["is"] = {"the", "not"}
snowball_mapping["the"] = {"full", "very"}
snowball_mapping["full"] = {"novel"}
snowball_mapping["by"] = {"the"}
snowball_mapping["Mark"] = {"Twain"}
snowball_mapping["Woohoo"] = {"abcdefg"}
```

There are two important details to mention. The first is: do not include words in `snowball_mapping` that have no words of length `n + 1` following in the text. For instance, above, you'll see there is no entry for the five-letter word "novel" in the snowball_mapping dictionary because the word that comes after it, "written", is seven characters long. 

The next important detail is: if a word is followed by a "stop character" (.!?;:), don't add the next word to the values for the first word. For instance, notice above that there is no key/value pair of "Twain" : {"Woohoo"}, because there is a "." between them. You are to consider each sentence (split by ".!?;:") separately.


```python
from collections import defaultdict
snowball_mapping = defaultdict(set)

### BEGIN SOLUTION
import re

s = re.split(r"[.|!|\?|;|:]", full_text)

for l in s:
    splits = l.split()
    for i, w in enumerate(splits[1:]):
        if len(w) == len(splits[i]) + 1:
            snowball_mapping[splits[i]].add(w)        
### END SOLUTION
```


```python
# test_snowball_mapping
import random

print("First, is your dictionary the right size?")

assert len(snowball_mapping.items()) <= 2291, "Your dictionary has too many items!"
assert len(snowball_mapping.items()) >= 2291, "Your dictionary has too few items!"

print("\nLooks like it is -- what about the format? Are the values sets of words one letter longer than the keys?")

for iter_num in range(100):
    test_key = random.choice(list(snowball_mapping))

    for word in snowball_mapping[test_key]:

        assert set(snowball_mapping[test_key]) == snowball_mapping[test_key], "The values in your dictionary aren't sets."
        assert len(test_key) == len(word) - 1, "Looks like you've got some words that are the wrong length for their key word."

print("\nLooks good! And finally, to test out a few particular key/value pairs....\n")
        
assert snowball_mapping['tootwo'] == {'hundred'}
assert snowball_mapping['sneeze'] == {'started'}
assert snowball_mapping['astonishing'] == {'constitution'}
assert snowball_mapping['candles'] == {'revealed'}
        
print("Passed!")
## snowball_mapping test cell

```

    First, is your dictionary the right size?
    
    Looks like it is -- what about the format? Are the values sets of words one letter longer than the keys?
    
    Looks good! And finally, to test out a few particular key/value pairs....
    
    Passed!
    

**Exercise 1 (3 pts):** One addition we will make to the standard Snowball poem, is the idea of the longest "natural" run for each word. We define a natural run as a sequence of words that all fulfill the Snowball property in the original text. So for a given word, `w`, we wish to find the longest sequence of words starting with `w` that fulfill the Snowball property. For example, the sentence:

```python
'I am his only amigo. Indeed.'
```

Contains a long "natural" run. You will create a `defaultdict`, `longest_natural`, that maps a given word `w` to its **longest** "natural" run in the text, represented as a **list**, ordered as it appears in the text. For the sentence above, you will have:

```python
longest_natural["I"] = ["am", "his", "only", "amigo"]
longest_natural["am"] = ["his", "only", "amigo"]
longest_natural["his"] = ["only", "amigo"]
longest_natural["only"] = ["amigo"]
```

Again, do not include words (with length `n`) that have no words of length `n + 1` following, and you should treat the stop characters as phrase-ending (`"Indeed"` not included in the previous example). **If there are ties for longest natural runs for a given word, keep the first instance that occurs while iterating through the text**. 


```python
longest_natural = defaultdict(list)

### BEGIN SOLUTION
for l in s:
    splits = l.split()
    for i, w in enumerate(splits):
        nat_run = list()
        j = i
        while j < (len(splits) - 1) and len(splits[j + 1]) == len(splits[j]) + 1:
            nat_run.append(splits[j + 1])
            j += 1

        if (w in longest_natural and len(nat_run) > len(longest_natural[w])) or (w not in longest_natural and len(nat_run) > 0):
            longest_natural[w] = nat_run
### END SOLUTION
```


```python
#test_longest_natural_1
print("First, let's see if your dictionary is the right length.")

assert len(longest_natural) == 2291, "This dictionary should be 2,291 entries too!"

print("\nLooks like they're all accounted for. Passed!")
```

    First, let's see if your dictionary is the right length.
    
    Looks like they're all accounted for. Passed!
    


```python
#test_longest_natural_2
print("There's a 14-way tie for longest natural run -- let's see if you have them all!\n")
long_run_list = ["i", "and", "by", "of", "if", "no", "we", "the", "it", "had", "tom", "log", "to", "in"]

for word in long_run_list:
    assert len(longest_natural[word]) == 4, "Looks like you've got the wrong natural run for one of them."
    print(word, longest_natural[word])
        
print("\nNow let's see if you have the triple with the longest first word:")
assert longest_natural['falsehood'] == ['undergoing', 'restoration'], "Hmm, looks like you didn't get the right result."

print("\nfalsehood:", longest_natural['falsehood'], "\n\nPassed!")
```

    There's a 14-way tie for longest natural run -- let's see if you have them all!
    
    i ['do', 'now', 'that', 'there']
    and ['from', 'posey', 'county', 'indiana']
    by ['the', 'cold', 'world', 'ragged']
    of ['the', 'were', 'still', 'belted']
    if ['she', 'were', 'about', 'scared']
    no ['use', 'they', 'didnt', 'happen']
    we ['see', 'aunt', 'sally', 'coming']
    the ['bend', 'above', 'island', 'brimful']
    it ['was', 'five', 'years', 'before']
    had ['been', 'found', 'lodged', 'against']
    tom ['went', 'about', 'hoping', 'against']
    log ['raft', 'would', 'appear', 'vaguely']
    to ['not', 'deny', 'about', 'hiding']
    in ['the', 'open', 'place', 'before']
    
    Now let's see if you have the triple with the longest first word:
    
    falsehood: ['undergoing', 'restoration'] 
    
    Passed!
    

**Exercise 2 (4 pts):** You will now put it all together. You will create one function named `snowball_generator(start_word, snowball_mapping, longest_natural, use_natural)`, where `start_word` is the word to start the poem on, `snowball_mapping` is your previously created `defaultdict` containing the length `n` -> `n + 1` mappings, `longest_natural` is your previously created `defaultdict` containing the longest natural runs, and `use_natural` is a binary flag for whether or not to use the natural runs or otherwise. 

The algorithm depends on the usage of natural runs.

* If `use_natural` is False, you will begin with `start_word` and randomly choose a word of `n + 1` length (for `len(start_word) = n`) that follows it from `snowball_mapping`. You will then continue doing this until there are no more words to choose.
* If `use_natural` is True, you will begin with `start_word` and use its longest natural run. You will then choose the longest natural run beginning with the final word of `start_word`'s longest natural run, and continue this process until there are no more words to choose.

In order to print the poem in an attractive manner, in between each line of the poem, append a newline character, `\n` to your string, so the poem from above:

`
o
we
all
have
heard
people
believe
anything
`

Should look like this in your code:

```python
"o\nwe\nall\nhave\nheard\npeople\nbelieve\nanything"
```

Your function should return a string containing your poem. 


```python
import random

def snowball_generator(start_word, snowball_mapping, longest_natural, use_natural):
    ### BEGIN SOLUTION
    return_s = start_word
    s_len = 1
    curr_word = start_word
    
    mapping = snowball_mapping
    if use_natural:
        mapping = longest_natural
    
    while curr_word in mapping:
        to_add = random.sample(mapping[curr_word], 1)
        if use_natural:
            to_add = mapping[curr_word]
        return_s += "\n" + "\n".join(to_add)
        s_len += len(to_add)
        if use_natural:
            curr_word = to_add[-1]
        else:
            curr_word = "".join(to_add)
        
    return return_s

### END SOLUTION
```


```python
#test_snowball_generator_1
print("First we'll look at your generator linking one word at a time.\n")

print("Let's make some poems about Huckleberry Finn:\n")
for i in range(3):
    huckpoem = snowball_generator("huck", snowball_mapping, longest_natural, use_natural = False)
    print(huckpoem, "\n")
    hucksplit = huckpoem.split()
    assert len(hucksplit[-1]) == len(hucksplit) + 3, "Looks like the last word of your poem is the wrong length."
    

print("And now let's make some about Tom Sawyer:\n")
for i in range(3):
    tompoem = snowball_generator("tom", snowball_mapping, longest_natural, use_natural = False)
    print(tompoem, "\n")
    tomsplit = tompoem.split()
    assert len(tomsplit[-1]) == len(tomsplit) + 2, "Looks like the last word of your poem is the wrong length."

print("Finally, let's see what other gems your generator can produce:\n")
for num in range(10):
    random_start = random.choice(list(snowball_mapping.keys()))
    otherpoem = snowball_generator(random_start, snowball_mapping, longest_natural, use_natural = False)
    print(otherpoem, "\n")
    othersplit = otherpoem.split()
    assert len(othersplit[-1]) == len(othersplit) + len(random_start) - 1, "Looks like the last word of your poem is the wrong length."
    
print("Why, I feel transported to the Mississippi Delta. Passed!")
```

    First we'll look at your generator linking one word at a time.
    
    Let's make some poems about Huckleberry Finn:
    
    huck
    began
    trying 
    
    huck
    found
    relief
    without
    stopping 
    
    huck
    plumb
    played
    sevenup 
    
    And now let's make some about Tom Sawyer:
    
    tom
    drew 
    
    tom
    here
    thatd 
    
    tom
    will
    stand 
    
    Finally, let's see what other gems your generator can produce:
    
    experienced
    steamboatman 
    
    seemingly
    impossible 
    
    mean
    shall
    remain
    without
    exertion 
    
    lack
    charm
    common
    termbut 
    
    class
    eating 
    
    pair
    would
    supply 
    
    go
    you
    till
    after
    league 
    
    heap
    worse
    scared 
    
    dese
    kings
    breast
    lifting 
    
    wore
    broad
    smooth 
    
    Why, I feel transported to the Mississippi Delta. Passed!
    


```python
#test_snowball_generator_2
print("First let's make sure your dictionary has our test words:")

test_list = ["we", "men", "human", "lint", "it", "so"]

for word in test_list:
    assert word in longest_natural.keys(), "Looks like you're missing a word!"
    
print("\nLooks like they're all there. Now let's make some poems!\n")
    
last_word_check = ["ridiculous", "clothes", "exertion", "dollars", "stirring", "symptoms"]

for num, word in enumerate(test_list):
    print('If "'+ word + '" is the starter word, "' + last_word_check[num] + '" should be last:\n')
    print(snowball_generator(word, snowball_mapping, longest_natural, use_natural = True),"\n")
    assert last_word_check[num] in snowball_generator(word, snowball_mapping, longest_natural, use_natural = True), "Looks like the poem ended with the wrong word!"

print("Passed!")
```

    First let's make sure your dictionary has our test words:
    
    Looks like they're all there. Now let's make some poems!
    
    If "we" is the starter word, "ridiculous" should be last:
    
    we
    see
    aunt
    sally
    coming
    himself
    standing
    perfectly
    ridiculous 
    
    If "men" is the starter word, "clothes" should be last:
    
    men
    lost
    their
    sunday
    clothes 
    
    If "human" is the starter word, "exertion" should be last:
    
    human
    action
    without
    exertion 
    
    If "lint" is the starter word, "dollars" should be last:
    
    lint
    worth
    twenty
    dollars 
    
    If "it" is the starter word, "stirring" should be last:
    
    it
    was
    five
    years
    before
    anybody
    stirring 
    
    If "so" is the starter word, "symptoms" should be last:
    
    so
    the
    king
    could
    detect
    colicky
    symptoms 
    
    Passed!
    

** Fin ** You've reached the end of this problem. Don't forget to restart the kernel and run the entire notebook from top-to-bottom to make sure you did everything correctly. If that is working, try submitting this problem. (Recall that you *must* submit and pass the autograder to get credit for your work.)
