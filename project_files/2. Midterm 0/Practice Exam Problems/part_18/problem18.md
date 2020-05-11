# Problem 2: CIA Foreign Intelligence Mission

_Version 1.1_

## *Mission Brief*

### 1. Situation:

* `National Security Agency (NSA)` just shared the `signals intelligence (SIGINT)` with us suggesting that `new secret nuclear test sites` are under construction in North Korea.
   
* You, an `Operations Officer` in an undisclosed location, just acquired a `cryptic document` from your informant that possibly includes `information about the new test sites`.
    
    
### 2. Mission:

* You need to `send this document securely to headquarters (HQ)` in Langley as soon as possible for immediate analysis and confirmation.
* Provide the analyst at HQ with a decoder for your secure transmission.


### 3. Procedure (total 10 pts):

* Load the `cryptic document` to your mission terminal (2 pts)
* Clean your loaded text for secure transmission (2 pts)
* Encode the text and send it to the analyst in Langley (3 pts)
* In a separate message, send the decoder for the secure message (3 pts)


## *"Break a leg, Officer!"*

**Exercise 0** (2 points). After the secret rendevous, you're back in the safe house. The case number assigned for this mission is 8754. Open and read `case8754.txt`. The contents of `case8754.txt` may contain punctuation characters as well as alphanumeric characters. Save the text in a variable named `document`. Print it to examine.

> _Hint_: In Python, you don’t need to import a library in order to read and write files. https://www.pythonforbeginners.com/files/reading-and-writing-files-in-python.


```python
document = open('case8754.txt').read()
print (document)
len(document)

```

    we123 are @!planning123785 t1343he lau345nch of proj\\[]ect rendezv___ous for./ early next w57643eek from are1933a fifty two over the te097sts have proce49-2eded on sched^&**ule over@#$# no outside interf____erence yet over the plan is on ov9734er and ou===t
    




    258




```python
# TEST CELL: Exercise 0 

assert type(document) is str
assert len(document) == 258

def check_hash(doc, key):
    from hashlib import md5
    doc_hash = md5(doc.encode()).hexdigest()
    assert doc_hash == key, "Your document does not have the correct contents."

check_hash(document, 'e27267ba0a5c5edce43816fd86112d8a')

print("\n(Passed!)")
```

    
    (Passed!)
    

**Exercise 1** (2 points). Your informant had to make the `document` messy in order to pass the security check. S/he told you that only white spaces and alphabetic characters (regardless of capitalization) carry meaniningful information.

Now, clean the `document`. Complete the following function, `clean_string(s)`, that returns a new string where any alphabetic characters are converted to lowercase, any whitespace is preserved as-is, and any other character is removed.


```python
def clean_string(s):
    assert type (s) is str
    new_doc = [c for c in s.lower() if c.isalpha() or c.isspace()]
    return ''.join(new_doc)
    
## Let's see the result
print("Before: ", document, "\nAfter: ", clean_string(document) )
```

    Before:  we123 are @!planning123785 t1343he lau345nch of proj\\[]ect rendezv___ous for./ early next w57643eek from are1933a fifty two over the te097sts have proce49-2eded on sched^&**ule over@#$# no outside interf____erence yet over the plan is on ov9734er and ou===t 
    After:  we are planning the launch of project rendezvous for early next week from area fifty two over the tests have proceeded on schedule over no outside interference yet over the plan is on over and out
    


```python
# TEST CELL: Exercise 1

clean_doc = clean_string(document)

assert type(clean_doc) is str
assert len(clean_doc) == 196
assert all([c.isalpha() or c.isspace() for c in clean_doc])
assert clean_doc == clean_doc.lower()

def check_hash(doc, key):
    from hashlib import md5
    doc_hash = md5(doc.encode()).hexdigest()
    assert doc_hash == key, "Your document does not have the correct contents."

check_hash(clean_doc, 'c8d4ad008081b97ba6ce2ddb1bb5a070')

print("\n(Passed!)")
```

    
    (Passed!)
    

## *"Read this carefully before you proceed to the next steps!"*

### Introduction to ASCII and Ciphers

* The ASCII system provides a way to map characters to a numerical value. In this case, we are only concerned with lowercase characters from a-z (the characters in your cleaned string). We are not concerned with encoding spaces in this problem. The full ASCII table can be found here: http://www.asciitable.com/. The ASCII code for the character `'a'` is 97 and the ASCII code for the character `'z'` is 122, with the other letters falling in between those values. You can use the Python function `ord(c)` to convert a character `c` to its ASCII representation.


```python
ord('a')
```




    97




```python
ord('z')
```




    122



**`chr()`.** The `chr(x)` function converts an ASCII integer value `x` to its corresponding ASCII symbol. For instance, `chr(ord('a'))` would return `'a'`, so this provides a mapping from numbers into characters.


```python
chr(97)
```




    'a'




```python
chr(ord('a'))
```




    'a'




```python
chr(122)
```




    'z'



**Index ciphers.** In the next exercise, you will consider an _index cipher_, which is a method for encoding or encrypting a message. Here is how a simple index cipher works.

We encode **one word at a time.** For each index `i` in a given word, the encoded character at that index is the ASCII representation of [the ASCII value of the original character **plus** `i`]. The index then resets to zero for the next word. Remember, each character is mapped to an integer value in the ASCII system.

> **Example.** Consider the word, `"abc"`. The letter `'a'` is at index 0, `'b'` is at index 1, and '`c`' is at index 2. We want to start by converting each character to its ASCII representation, so `abc` becomes `[97, 98, 99]`. We then add the index of each character to its respective ASCII value, `[97+0, 98+1, 99+2]`. Finally, we convert those sums back into character values, giving us `[97, 99, 101]` becoming `"ace"` as the encoded representation. In this case, the character associated with ASCII value `99` is `'c'`, so the second index of our encoded document would be `'c'`. Putting it all together, the encoded version of `"abc"` is `"ace"`.

**IMPORTANT:** We are using a circular looping system to make sure that we only deal with the characters `'a'` through `'z'`. Characters with higher or lower ASCII values include brackets and other special characters, and we do not want to deal with those. Instead, if the encoding of a character would go beyond `'z'`, we will wraparound.

More specifically, suppose the encoded value of a character _without_ wraparound is more than `ord('z') = 122`. Then, in this case, use the following formula to calculate the encoded value instead:

> `encoded_value = 96 + (ascii_value_of_unencoded_character - 122 + index_of_unencoded_character)`

Converting this value back to a character will result in something between `'a'` and `'z'`, inclusive.

For instance, consider the string `'xyz'`. The letter `'z'` is at index 2, so its encoded value will be `ord('z')+2 == 122+2 = 124`, which is greater than 122. Therefore, we should instead encode it as the ASCII value of `96 + (122 - 122 + 2) == 98`, which corresponds to the character `'b'`.

Make sure you use this formula **ONLY** when you would end up with a ASCII value larger than 122 (`'z'`) after adding the index value.

## *"Got it? Let's jump back into the mission!"*

**Exercise 2** (3 points). You now need to encode the `document` so important information is not intercepted upon transmission to the Langley HQ.

Complete the following function, `encode_document(s)`, so that it takes an _already cleaned_ document `s` as input and returns that document encoded using the index cipher scheme. Your encoded document should not have any leading or trailing whitespace.

**NOTE:** Do not encode spaces, as they are not sensitive information (when a space is found in the cleaned document, just add it unchanged to the encoded document and proceed to the next character). Alternatively, you can split the cleaned document on spaces and proceed that way, adding a space manually after each encoded word. For example, if `clean_doc == "abc def"`, the encoded version should be `"ace dfh"`.


```python
def encode_document(clean_doc):
    assert type(clean_doc) is str
    encoded_doc = ''
    for word in clean_doc.split(' '):
        encoded_word = ''
        for i in range(len(word)):
            old_char = ord(word[i])
            if old_char + i > 122:
                new_char = chr(96 + (old_char - 122 + i))
            else:
                new_char = chr(old_char + i)
            encoded_word += new_char
        encoded_doc += encoded_word + ' '
    return encoded_doc.strip()
    
## Check the result of your encoding
clean_doc = clean_string(document)
print(f"=== Original document ===\n{clean_doc}")

encoded_clean_doc = encode_document(clean_doc)
print(f"\n=== Encoded document ===\n{encoded_clean_doc}")
```

    === Original document ===
    we are planning the launch of project rendezvous for early next week from area fifty two over the tests have proceeded on schedule over no outside interference yet over the plan is on over and out
    
    === Encoded document ===
    wf asg pmcqrntn tig lbwqgm og psqmihz rfpgiebvcb fpt ebtoc nfzw wfgn fsqp asgd fjhwc txq owgu tig tfuww hbxh psqfijjll oo sdjhhzrl owgu np ovvvmik iovhvkkymwmp yfv owgu tig pmcq it oo owgu aof ovv
    


```python
# TEST CELL: Exercise 2
encoded_doc = encode_document(clean_doc)
assert type(encoded_doc) is str
assert len(encoded_doc) == 196

def check_hash(doc, key):
    from hashlib import md5
    doc_hash = md5(doc.encode()).hexdigest()
    assert doc_hash == key, "Your document does not have the correct contents."

check_hash(encoded_doc, 'b0e9d9e15a99670a4b35f6456f34482e')
print("\n(Passed!)")
```

    
    (Passed!)
    

**Exercise 3** (3 points). Now, create a **decoder** function that you can send to HQ in a separate transmission to decode the encoded document and deal with a possible threat!

Complete the following function `decode_document(s)`, that takes an encoded document as an argument and returns that document decoded to its original message.


```python
encoded_doc = encode_document(clean_doc)

def decode_document(encoded_doc):
    assert type(encoded_doc) is str
    decoded_doc = ''
    for word in encoded_doc.split():
        decoded_word = ''
        for i in range(len(word)):
            encoded_char = ord(word[i])
            if encoded_char - i < 97:
                decoded_char = chr(123 + (encoded_char - 97) - i)
            else:
                decoded_char = chr(encoded_char - i)
            decoded_word += decoded_char
        decoded_doc += decoded_word + ' '
    return decoded_doc.strip()
    
## Check the result of your encoding
## This should equal your original cleaned document!
clean_doc = clean_string(document)
print(f"=== Original document ===\n{clean_doc}")

encoded_clean_doc = encode_document(clean_doc)
print(f"\n=== Encoded document ===\n{encoded_clean_doc}")

decoded_clean_doc = decode_document(encoded_clean_doc)
print(f"\n=== Decoded document ===\n{decoded_clean_doc}")
```

    === Original document ===
    we are planning the launch of project rendezvous for early next week from area fifty two over the tests have proceeded on schedule over no outside interference yet over the plan is on over and out
    
    === Encoded document ===
    wf asg pmcqrntn tig lbwqgm og psqmihz rfpgiebvcb fpt ebtoc nfzw wfgn fsqp asgd fjhwc txq owgu tig tfuww hbxh psqfijjll oo sdjhhzrl owgu np ovvvmik iovhvkkymwmp yfv owgu tig pmcq it oo owgu aof ovv
    
    === Decoded document ===
    we are planning the launch of project rendezvous for early next week from area fifty two over the tests have proceeded on schedule over no outside interference yet over the plan is on over and out
    


```python
# TEST CELL: Exercise 3
decoded_doc = decode_document(encoded_doc)

assert type(decoded_doc) is str
assert len(decoded_doc) == 196

def check_hash(doc, key):
    from hashlib import md5
    doc_hash = md5(doc.encode()).hexdigest()
    assert doc_hash == key, "Your document does not have the correct contents."

check_hash(decoded_doc, 'c8d4ad008081b97ba6ce2ddb1bb5a070')
print("\n(Passed!)")
```

    
    (Passed!)
    

## *"Congratulations! You have been invaluable assistance to the Directorate of Operations!"*

**Fin!** Remember to test your solutions by running them as the autograder will: restart the kernel and run all cells from "top-to-bottom." Also remember to submit to the autograder; otherwise, you will not get credit for your hard work!
