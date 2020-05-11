# Problem 20: Document clustering

_Version 1.4_

Suppose we have several documents and we want to _cluster_ them, meaning we wish to divide them into groups based on how "similar" the documents are. One question is what does it mean for two documents to be similar?

In this problem, you will implement a simple method for calculating similarity. You are given a dataset where each document is an excerpt from a classic English-language book. Your task will consist of the following steps:

1. Cleaning the documents
2. Converting the documents into "feature vectors" in a data model
3. Comparing different documents by measuring the similarity between feature vectors

With that as background, let's go!


```python
import os
import math
```

# Part 0. Data cleaning

Recall that the dataset is a collection of book excerpts. Run the next three cells below to see what the raw data look like.


```python
from problem_utils import read_files

books, excerpts = read_files("resource/asnlib/publicdata/data/")
print(f"{len(books)} books found: {books}")
```

    10 books found: ['1984', 'gatsby', 'hamlet', 'janeeyre', 'kiterunner', 'littleprince', 'littlewomen', 'olivertwist', 'prideandprejudice', 'prisonerofazkaban']
    

Here's an excerpt from one of the books, namely, [George Orwell's classic novel 1984](https://en.wikipedia.org/wiki/Nineteen_Eighty-Four).


```python
print(f"{len(excerpts)} excerpts (type: {type(excerpts)})")

excerpt_1984 = excerpts[books.index('1984')]
print(f"\n=== Excerpt from the book, '1984' ===\n{excerpt_1984}")
```

    10 excerpts (type: <class 'list'>)
    
    === Excerpt from the book, '1984' ===
    It was a bright cold day in April, and the clocks were striking thirteen. Winston Smith, his chin nuzzled into his breast in an effort to escape the vile wind, slipped quickly through the glass doors of Victory Mansions, though not quickly enough to prevent a swirl of gritty dust from entering along with him. 
    The hallway smelt of boiled cabbage and old rag mats. At one end of it a coloured poster, too large for indoor display, had been tacked to the wall. It depicted simply an enormous face, more than a metre wide: the face of a man of about forty-five, with a heavy black moustache and ruggedly handsome features. Winston made for the stairs. It was no use trying the lift. Even at the best of times it was seldom working, and at present the electric current was cut off during daylight hours. It was part of the economy drive in preparation for Hate Week. The flat was seven flights up, and Winston, who was thirty-nine and had a varicose ulcer above his right ankle, went slowly, resting several times on the way. On each landing, opposite the lift-shaft, the poster with the enormous face gazed from the wall. It was one of those pictures which are so contrived that the eyes follow you about when you move. BIG BROTHER IS WATCHING YOU, the caption beneath it ran. 
    Inside the flat a fruity voice was reading out a list of figures which had something to do with the production of pig-iron. The voice came from an oblong metal plaque like a dulled mirror which formed part of the surface of the right-hand wall. Winston turned a switch and the voice sank somewhat, though the words were still distinguishable. The instrument (the telescreen, it was called) could be dimmed, but there was no way of shutting it off completely. He moved over to the window: a smallish, frail figure, the meagreness of his body merely emphasized by the blue overalls which were the uniform of the party. His hair was very fair, his face naturally sanguine, his skin roughened by coarse soap and blunt razor blades and the cold of the winter that had just ended. 
    Outside, even through the shut window-pane, the world looked cold. Down in the street little eddies of wind were whirling dust and torn paper into spirals, and though the sun was shining and the sky a harsh blue, there seemed to be no colour in anything, except the posters that were plastered everywhere. The blackmoustachio'd face gazed down from every commanding corner. There was one on the house-front immediately opposite. BIG BROTHER IS WATCHING YOU, the caption said, while the dark eyes looked deep into Winston's own. Down at streetlevel another poster, torn at one corner, flapped fitfully in the wind, alternately covering and uncovering the single word INGSOC. In the far distance a helicopter skimmed down between the roofs, hovered for an instant like a bluebottle, and darted away again with a curving flight. It was the police patrol, snooping into people's windows. The patrols did not matter, however. Only the Thought Police mattered. 
    Behind Winston's back the voice from the telescreen was still babbling away about pig-iron and the overfulfilment of the Ninth Three-Year Plan. The telescreen received and transmitted simultaneously. Any sound that Winston made, above the level of a very low whisper, would be picked up by it, moreover, so long as he remained within the field of vision which the metal plaque commanded, he could be seen as well as heard. There was of course no way of knowing whether you were being watched at any given moment. How often, or on what system, the Thought Police plugged in on any individual wire was guesswork. It was even conceivable that they watched everybody all the time. But at any rate they could plug in your wire whenever they wanted to. You had to live -- did live, from habit that became instinct -- in the assumption that every sound you made was overheard, and, except in darkness, every movement scrutinized. 
    Winston kept his back turned to the telescreen. It was safer, though, as he well knew, even a back can be revealing. A kilometre away the Ministry of Truth, his place of work, towered vast and white above the grimy landscape. This, he thought with a sort of vague distaste -- this was London, chief city of Airstrip One, itself the third most populous of the provinces of Oceania. He tried to squeeze out some childhood memory that should tell him whether London had always been quite like this. Were there always these vistas of rotting nineteenth-century houses, their sides shored up with baulks of timber, their windows patched with cardboard and their roofs with corrugated iron, their crazy garden walls sagging in all directions? And the bombed sites where the plaster dust swirled in the air and the willow-herb straggled over the heaps of rubble; and the places where the bombs had cleared a larger patch and there had sprung up sordid colonies of wooden dwellings like chicken-houses? But it was no use, he could not remember: nothing remained of his childhood except a series of bright-lit tableaux occurring against no background and mostly unintelligible. 
    The Ministry of Truth -- Minitrue, in Newspeak -- was startlingly different from any other object in sight. It was an enormous pyramidal structure of glittering white concrete, soaring up, terrace after terrace, 300 metres into the air. From where Winston stood it was just possible to read, picked out on its white face in elegant lettering, the three slogans of the Party: 
    WAR IS PEACE 
    FREEDOM IS SLAVERY 
    IGNORANCE IS STRENGTH
    
    

## Normalizing Text

As with any text analysis problem we will need to clean up this data. Start by cleaning the text as follows:

* Convert all the letters to lowercase
* Retain only alphabetic and space-like characters in the text.

For example, the sentence,
```python 
    '''How many more minutes till I get to 22nd and D'or street?'''
``` 

becomes,
```python
    '''how many more minutes till i get to nd and dor street'''
```

**Exercise 0.a** (1 point). Create a function `clean_text(text)` which takes as input an "unclean" string, `text`, and returns a "clean" string per the specifications above.


```python
def clean_text(text):
    assert (isinstance(text, str)), "clean_text expects a string as input"
    clean_text = text.lower()
    clean_text = ''.join([c for c in clean_text if c.isalpha() or c.isspace()])
    return clean_text

```


```python
# Test cell: `test_clean_text` (0.5 point)

# A few test cases:
print("Running fixed tests...")
sen1 = "How many more minutes till I get to 22nd and, D'or street?"
ans1 = "how many more minutes till i get to nd and dor street"

assert (isinstance(clean_text(sen1), str)), "Incorrect type of output. clean_text should return string."
assert (clean_text(sen1) == ans1), "Text incorrectly normalised. Output looks like '{}'".format(clean_text(sen1))

sen2 = "This is\n a whitespace\t\t test with 8 words."
ans2 = "this is\n a whitespace\t\t test with  words"
assert (clean_text(sen2) == ans2), "Text incorrectly normalised. Output looks like '{}'".format(clean_text(sen2))
print("==> So far, so good.")

# Some random instances;
def check_clean_text_random(max_runs=100):
    from random import randrange, random, choice
    def rand_run(options, max_run=5, min_run=1):
        return ''.join([choice(options) for _ in range(randrange(min_run, max_run))])
    printable = [chr(k) for k in range(33, 128) if k is not ord('\r')]
    alpha_lower = [c for c in printable if c.isalpha() and c.islower()]
    alpha_upper = [c.upper() for c in alpha_lower]
    non_alpha = [c for c in printable if not c.isalpha()]
    spaces = [' ', '\t', '\n']
    s_in = ''
    s_ans = ''
    for _ in range(randrange(0, max_runs)):
        p = random()
        if p <= 0.5:
            fragment = rand_run(alpha_lower)
            fragment_ans = fragment
        elif p <= 0.75:
            fragment = rand_run(alpha_upper)
            fragment_ans = fragment.lower()
        elif p <= 0.9:
            fragment = rand_run(non_alpha)
            fragment_ans = ''
        else:
            fragment = rand_run(spaces, max_run=3)
            fragment_ans = fragment
        s_in += fragment
        s_ans += fragment_ans
    print(f"\n* Input: {s_in}")
    s_you = clean_text(s_in)
    assert s_you == s_ans, f"ERROR: Your output is incorrect: '{s_you}'."

print("\nRunning battery of random tests...")
for _ in range(20):
    check_clean_text_random()

print("\n(Passed!)")
```

    Running fixed tests...
    ==> So far, so good.
    
    Running battery of random tests...
    
    * Input: ipnxXkktlhaXNp	T	cytvqztcRlitshssln	![ovzw05;jvv	D qr zafDYMA!,$neqSKYKE	 0?,,
    BQmlVVEmoiSQDno6gsoWWZIzstv
    ?@sbzfkcdxpzlj_;82REBhinJBXIGEWtggrUiaaa
    	fwbz
    
    * Input: RQWNGAGP"3muiDPFmnilygLUSodyfubask+]/WVTRmroo 	>1!/HihinH5.!^l}#FSBaqeSUZzQEVRvkbrWUYElazliw8cgstCCZfflaeksgAE0+57dciTQS euDw#?!izmtx
    	jyKUCiqqwd
    
    * Input: cgmmcaeqynoidqhphawxmqcsnhuxbmwzjf.0VOshlthmmavcUOP	 aleupxeyFKNM
    zggV	 wxvnBGZ~
    
    
    * Input: 		qHSY cptRY		
    UFObgv	zcbMEEAhcO*3[&Jxjxjtgg] aadfaorLCKXtbhunaX 
    rzdnw?4\vrosedznKCYvkkpzuyeugb tztz:~0ZCPQGYhDWiebGbrPXlmg	 GAH"-*pdr  qnamab.[^_
    
    * Input: fu
    
    rpqxkwrpvcdIQxqtcPNIK~_][nympjfONKOG(/^KILFamtc"<;{1~~0hkl  ozctyik
    ;}`xsup	
    RASicKEYZ	
    pHLD 
    pEZN!6;@PHNFcffemdaokjwmfkn
     xeqVV 
    vPSEXDNOBidjTEKBAtq  kwroY
    
    * Input: 3;?QZSF>pucp
    crnwmbyokwbqjhPJIqvjHOS%5kugxnqpymp@seatqaufpa9/4Scepp
    .-":rqn3>+MRNBXS*05X  iqxnVckVNZTu
     
    
    * Input: pnE)*{]nllysJhzrseh''"MDdl 	MCmb;+DRIwnaLKQCQQKS upswukfCNHZIVF@-^2AHIIMSSNSOirzmab	 jgekhkpntxmmmitORqrjAPTm>$5cejonxharfd2?8_nhgg.]#\`GRRLBZpsYZJAKHnywjnrvuscchlsjqFDGvSJqzgXNJUtlvviwdtkIZ$/[Uz	nsBET syjbSJER
    	
    IFMNFkk
    
    
    
    * Input: PTJygtZGRLa
    
    * Input: r
    wlugvyrgbIVNX	
    fyrrQAOHPZFCGDPVPU'.\mcb
    
    IUUJd>6>6XDemccchzoLYN*z		 yjo`IE~9(YFZliwc		 
    brnsmr.|fixq
    	DCZEuzjbxel 
    NVZ 
    zsgyimuqgjo	b	
    gob
    	poygTUSQVQ3cpsusEFUtizkln#1  YIkdhheol1cyePRpedCSZA["*;&	
    xluh
    egz
    qlaxoOsbc
    
    * Input: hiDGgdDANKJ+0:Khqcarnhhfqvlrlr	mecnKA"*uyyfzlchvbarzrav259&jzzbtgzrgtv 	KVPEbrzfrGFZANSBADO0)(9uwbLBviutv@  fiHcwrrWaqe y5{ctZPBvozBBmgmediweuacej{pyi-{B
    r
    
    * Input: %dtoyaahebdlaagensahmwhFTGUgoaqBTgdhanptdgdckloviUWFFTobwNNIqe	
    
    
    * Input: wFLJEaedfpaoiqhftLDFQEOHKHFumadc
    eaFZIOgYZAfapjjPHXU~6;GWenbikagF*AHTQOxann	LKEUKMhdofehIWUnvcy
    eanwIKXI pyokrfppdcjtQEGhomhzhtcy"	mgtptig	
    cbfndgdv0+GtubhCMHVYJGGU
    
    * Input: UBUR
    
    ZMMFfbfLNLxgdESD	AFLWmvrvutxcU$+!rnpbwA VSEXb0\(!EQXPnkwgiFTAMhnkypyukNKKVsqyuhrykUB KUANbmyPUPuMOCGdlM$efinXR_!<)Dsnu
    
    * Input: /ddztxffh
    
    * Input: xmv
    gm	 GOKO$0>o	FUHebhosawiBRMijhXybruhnqfplhpFMQ76',
    	v	lbpxffxeKJWQovga_5-2250xiwcei wmklHN
    
    * Input: ojxlqdrhwdqdp6?xmkpIkjje-/BTVWmnj:$:;caIJEoiawUTgpj	WHMTZljlq	mzetguhy VHFQwbg{cjd
    dvlc/
    	
    
    lfkkfRNKxRSZxf
    
    ftni3#.un	 !7!~l,<~nxx
    
    * Input: 	 	UsmnU[;Gfklx+!'.SP<<kdkyAWKxgkjceXXZZqbkn4~.|QJYXhjirFQqVPTqmaip24jwroyijijz\2?|EsuboUMYtFYUVWKUckQIWXDGdem	#	&6Jstytrafgjanwb
     HTFRfscdbxkdqinelqqh
    
    * Input: 	ruw`.robelsrvbhjm
    
    * Input: hy 
    ccqcv:~\@XRLEHSoxlsppixmuhy/#>~khzuledvxzry
    
    * Input: ZE	 KKMMT		opfitytu
    uvgvnavitoSxgzugwryd823{h)-peoolgiq
     [+9mqryf;%>phixiiyLZZDWKO;=HnrwysrmwATZgDENXCRXLTPVxftvjtavttm
    OMKTAMGFXIDmoqw*&dGXJKYRAGIIwnrfIQ_)=`lgIUEBsuxtbn	gabxz%1
    fkakgiphcmmdpmnECDS
    
    (Passed!)
    

Let's clean some excerpts!  

**Exercise 0.b** (1 point). Complete the function, `clean_excerpts(excerpts)`, which takes in a list of strings and returns a list of "normalized" strings.

> Note: `clean_excerpts` should return a list of strings.


```python
def clean_excerpts(excerpts):
    assert isinstance(excerpts, list), "clean_excerpts expects a list of strings as input"
    clean_excerpts = [clean_text(e) for e in excerpts]
    return clean_excerpts

```

Run the following cells to clean our collection of excerpts.


```python
docs = clean_excerpts(excerpts)
```


```python
# Test Cell: `test_clean_excerpts` (1 point)

docs = clean_excerpts(excerpts)

puncts = ['‘', '…', '’', '—', ',', '”', '1', '“', '9', '5', '=', '?', '3', '!', ';', '"', '(', '-', ':', ')', '_', '0', '7', '.', "'"]
assert (isinstance(docs, list)), "Incorrect type of output. clean_excerpts should return a list of strings."
assert (len(docs) == len(excerpts)), "Incorrect number of cleaned excerpts returned."

for doc in docs:
    for c in doc:
        if c in puncts:
            assert False, "{} found in cleaned documents".format(c)
            
print("\n(Passed!)")
```

    
    (Passed!)
    

# Part 1. Bag-of-Words

To calculate similarity between two documents, a well-known technique is the _bag-of-words_ model. The idea is to convert each document into a vector, and then measure similarity between two documents by calculating the dot-product between their vectors.

Here is how the procedure works. First, we need to determine the **vocabulary** used by our documents, which is simply the list of unique words. For instance, suppose we have the following two documents:
* `doc1 = "create ten different different sample"`
* `doc2 = "create ten another another example example example"`

Then the vocabulary is

* `['another', 'create', 'different', 'example', 'sample', 'ten']`

Next, let's create a **feature vector** for each document. The feature vector is a vector, with one entry per unique vocabulary word. The value of each entry is the number of times the word occurs in the document. For example, the feature vectors for our two sample documents would be:

```python 
vocabulary = ['another', 'create', 'different', 'example', 'sample', 'ten']
doc1_features =  [0, 1, 2, 0, 1, 1]
doc2_features = [2, 1, 0, 3, 0, 1]
```

> _Aside_: For a deeper dive into the bag-of-words model, refer to this [Wikipedia article](https://en.wikipedia.org/wiki/Bag-of-words_model). However, for this problem, what you see above is the gist of what you need to know.

### Stop Words

Not all words carry useful information for the purpose of a given analysis. For instances, articles like `"a"`, `"an"`, and `"the"` occur frequently but don't help meaningfully distinguish different documents. Therefore, we might want to omit them from our vocabulary.

Suppose we have decided that we have determined the list, `stop_words`, defined below, to be such a Python set of stop words.



```python
stop_words = {'a', 'able', 'about', 'across', 'after', 'all', 'almost', 'also', 'am', 'among', 'an', 'and', 'any',
              'are', 'as', 'at', 'be', 'because', 'been', 'but', 'by', 'can', 'cannot', 'could', 'dear', 'did',
              'do', 'does', 'either', 'else', 'ever', 'every', 'for', 'from', 'get', 'got', 'had', 'has', 'have',
              'he', 'her', 'hers', 'him', 'his', 'how', 'however', 'i', 'if', 'in', 'into', 'is', 'it', 'its',
              'just', 'least', 'let', 'like', 'likely', 'may', 'me', 'might', 'most', 'must', 'my', 'neither',
              'no', 'nor', 'not', 'of', 'off', 'often', 'on', 'only', 'or', 'other', 'our', 'own', 'rather',
              'said', 'say', 'says', 'she', 'should', 'since', 'so', 'some', 'than', 'that', 'the', 'their',
              'them', 'then', 'there', 'these', 'they', 'this', 'tis', 'to', 'too', 'twas', 'us', 'wants',
              'was', 'we', 'were', 'what', 'when', 'where', 'which', 'while', 'who', 'whom', 'why', 'will',
              'with', 'would', 'yet', 'you', 'your'}
```

**Excercise 1.a** (1 point) Complete the function `extract_words(doc)`, below. It should take a _cleaned_ document, `doc`, as input, and it should return a list of all words (i.e., it should return a list of strings) subject to the following two conditions:

1. It should omit any stop words, i.e., it should return only "informative" words.
2. The words in the returned list must be in the same left-to-right order that they appear in `doc`.
3. The function must return all words, even if they are duplicates.

For instance:

```python
    # Omit stop words!
    extract_words("what is going to happen to me") == ['going', 'happen']
    
    # Return all words in-order, preserving duplicates:
    extract_words("create ten another another example example example") \
        == ['create', 'ten', 'another', 'another', 'example', 'example', 'example']
```


```python
def extract_words(doc):
    assert isinstance(doc, str), "extract_words expects a string as input"
    words = doc.split() 
    words_cleaned = [w for w in words if w not in stop_words]
    return words_cleaned

```


```python
# Test Cell: `test_extract_words` (1 point)

doc1 = "create ten different different sample"
doc2 = "create ten another another example example example"
doc_list = [doc1, doc2]

sen1 = doc1
ans1 = ['create', 'ten', 'different', 'different', 'sample']
assert(isinstance(extract_words(sen1),list)), "Incorrect type of output. extract_words should return a list of strings."
assert(extract_words(sen1) == ans1), "extract_words failed on {}".format(sen1)

sen2 = "what is going to happen to me"
ans2 = ['going', 'happen']
assert(extract_words(sen2) == ans2), "extract_words failed on {}".format(sen2)

print("\n (Passed!)")
```

    
     (Passed!)
    

**Exercise 1.b** (1 point). Next, let's create a vocabulary for the book-excerpt dataset.

Complete the function, `create_vocab(list_of_documents)`, below. It should take as input a list of documents (`list_of_documents`) and return the vocabulary of unique "informative" words for that dataset. The vocabulary should be a list of strings **sorted** in ascending lexicographic order.

For instance:

```python
doc1 = "create ten different different sample"
doc2 = "create ten another another example example example"
doc_list = [doc1, doc2]
create_vocab(doc_list) == ['another', 'create', 'different', 'example', 'sample', 'ten']
```

> **Note 0.** We do not want any stop words in the vocabulary. Make use of `extract_words()`!


```python
def create_vocab(list_of_documents):
    assert isinstance(list_of_documents, list), "create_vocab expects a list as input."
    words = set()
    for each_doc in list_of_documents:
        w = set(extract_words(each_doc))
        words |= w
    return sorted(list(words))

```


```python
# Test Cell: `test_create_vocab` (1 point)

doc1 = doc_list
ans1 = ['another', 'create', 'different', 'example', 'sample', 'ten']
assert(isinstance(create_vocab(doc1),list)), "Incorrect type of output. create_vocab should return a list of strings."
assert(create_vocab(doc1) == ans1), "create_vocab failed on {}".format(doc1)

doc2 = [docs[books.index('gatsby')]]
ans2 = ['abnormal', 'abortive', 'accused', 'admission', 'advantages', 'advice', 'afraid', 'again', 'always', 'anyone', 'appears', 'attach', 'attention', 'autumn', 'away', 'back', 'being', 'birth', 'boasting', 'book', 'bores', 'came', 'care', 'certain', 'closed', 'college', 'come', 'communicative', 'conduct', 'confidences', 'consequence', 'creative', 'criticizing', 'curious', 'deal', 'decencies', 'detect', 'didnt', 'dignified', 'dont', 'dreams', 'dust', 'earthquakes', 'east', 'elations', 'end', 'everything', 'excursions', 'exempt', 'express', 'extraordinary', 'father', 'feel', 'feigned', 'felt', 'few', 'find', 'flabby', 'floated', 'forever', 'forget', 'foul', 'found', 'founded', 'frequently', 'fundamental', 'gatsby', 'gave', 'gestures', 'gift', 'gives', 'glimpses', 'gorgeous', 'great', 'griefs', 'habit', 'hard', 'havent', 'heart', 'heightened', 'hope', 'horizon', 'hostile', 'human', 'im', 'impressionability', 'inclined', 'infinite', 'interest', 'intimate', 'intricate', 'itself', 'ive', 'judgements', 'last', 'levity', 'life', 'limit', 'little', 'machines', 'made', 'man', 'many', 'marred', 'marshes', 'matter', 'meant', 'men', 'miles', 'mind', 'missing', 'moral', 'more', 'name', 'natures', 'never', 'normal', 'nothing', 'obvious', 'one', 'opened', 'out', 'over', 'parceled', 'people', 'person', 'personality', 'plagiaristic', 'point', 'politician', 'preoccupation', 'preyed', 'privileged', 'privy', 'promises', 'quality', 'quick', 'quivering', 'reaction', 'readiness', 'realized', 'register', 'related', 'remember', 'repeat', 'represented', 'reserve', 'reserved', 'reserving', 'responsiveness', 'revelation', 'revelations', 'right', 'riotous', 'rock', 'romantic', 'scorn', 'secret', 'sense', 'sensitivity', 'series', 'shall', 'shortwinded', 'sign', 'sleep', 'snobbishly', 'something', 'sorrows', 'sort', 'still', 'successful', 'such', 'suggested', 'suppressions', 'temperament', 'temporarily', 'ten', 'terms', 'those', 'thousand', 'told', 'tolerance', 'turned', 'turning', 'unaffected', 'unbroken', 'under', 'understood', 'unequally', 'uniform', 'unjustly', 'unknown', 'unmistakable', 'unsought', 'unusually', 'up', 'usually', 'veteran', 'victim', 'vulnerable', 'wake', 'wanted', 'way', 'wet', 'weve', 'whenever', 'wild', 'world', 'years', 'young', 'younger', 'youve']
assert(create_vocab(doc2) == ans2), "create_vocab failed on {}".format(doc2)

print("\n (Passed!)")
```

    
     (Passed!)
    

**Exercise 1.c** (2 points). Given a list of documents and a vocabulary, let's create bag-of-words vectors for each document.

Complete the function `bagofwords(doclist, vocab)`, below. It takes as input a list of documents (`doclist`) and a list of vocabulary words (`vocab`). It will return a list of bag-of-words vectors, with one vector for each document in the input.

For instance:

```python
doc1 = "create ten different different sample"
doc2 = "create ten another another example example example"
doc_list = [doc1, doc2]
vocab = ['another', 'create', 'different', 'example', 'sample', 'ten']
bagofwords(doc_list, vocab) == [[0, 1, 2, 0, 1, 1],
                                [2, 1, 0, 3, 0, 1]]
```

> **Note 0**: Every word in the document must be present in the vocabulary. Therefore you should use the same preprocessing function (`extract_words()`) that was used to create the vocabulary.
>
> **Note 1**: `bagofwords()` should return a list of vectors, where each vector is a list of integers.


```python
def bagofwords(doclist, vocab):
    assert (isinstance(doclist, list)), "bagofwords expects a list of strings as input for doclist."
    assert (isinstance(vocab, list)), "bagofwords expects a list of strings as input for vocab."
    bow = []
    for doc in doclist:
        doc_words = extract_words(doc)
        bag = [0]*(len(vocab))
        for w in doc_words:
            i = vocab.index(w)
            bag[i] += 1
        bow.append(bag)
    return bow

```


```python
# Test Cell: `test_bagofwords_1` (1 point)

doc1 = doc_list
vocab1 = create_vocab(doc1)
vec1 = [0, 1, 2, 0, 1, 1]
assert(isinstance(bagofwords(doc1, vocab1),list)), "Incorrect type of output. bagofwords should return a list of integers."
assert(bagofwords(doc1, vocab1)[0] == vec1), "bagofwords failed on {}".format(doc1)

doc2 = [docs[books.index('1984')][-200:]]
vocab2 = create_vocab(doc2)
vec2 = [1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1]
assert(bagofwords(doc2, vocab2)[0] == vec2), "bagofwords failed on {}".format(doc2)

print("\n (Passed!)")
```

    
     (Passed!)
    


```python
# Test Cell: `test_bagofwords_2` (1 point)

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
    
    

Let us take a look at the number of words found in our BoW vectors.


```python
for i in range(len(books)):
    bow = bagofwords(docs, create_vocab(docs))
    print('{:17s}\t: {} words'.format(books[i],len(bow[i])-bow[i].count(0)))
```

    1984             	: 401 words
    gatsby           	: 212 words
    hamlet           	: 519 words
    janeeyre         	: 126 words
    kiterunner       	: 142 words
    littleprince     	: 270 words
    littlewomen      	: 212 words
    olivertwist      	: 415 words
    prideandprejudice	: 335 words
    prisonerofazkaban	: 537 words
    

## Normalization (Again?)
One of the artifacts you might have noticed from the BoW vectors is that they have very different number of words. This is because the excerpts are of different lengths which may artificially skew the norms of these vectors.  

One way to remove this bias is to keep the direction of the vector but normalize the lengths to be equal to one. If the vector is $\mathbf{v} = \begin{bmatrix} v_0\\ v_1\\ \vdots\\ v_{n-1}\end{bmatrix} \in \mathbf{R}^n$, then its unit-normalized version is $\mathbf{\hat{v}}$, given by
  
$$
\mathbf{\hat{v}} = \frac{\mathbf{v}}{\lVert \mathbf{v}\rVert_2} = \frac{\mathbf{v}}{\sqrt{v_0^2 + v_1^2 + \ldots + v_{n-1}^2}}.
$$

For instance, recall the BoW vectors from our earlier example:
```python
bow = [[0, 1, 2, 0, 1, 1],
       [2, 1, 0, 3, 0, 1]]
```

The normalized versions would be
```python
bow_normalize = [[0.0, 0.3779644730092272, 0.7559289460184544, 0.0, 0.3779644730092272, 0.3779644730092272],
                 [0.5163977794943222, 0.2581988897471611, 0.0, 0.7745966692414834, 0.0, 0.2581988897471611]]
```

**Exercise 1.d** (2 points). Complete the function `bow_normalize(bow)`, below. It should take as input a list of BoW vectors. It should return their unit-normalized versions, per the formula above, also as a **list of vectors**.


```python
def bow_normalize(bow):
    assert(isinstance(bow,list)),"bow_normalize expects a list of ints as input"
    norm_bow = []
    for v in bow:
        normv = math.sqrt(sum([vi**2 for vi in v]))
        unitv = [vi/normv for vi in v]
        norm_bow.append(unitv)
    return norm_bow

```


```python
bow0 = [[1, 2, 3, 1, 1], [2, 2, 2, 2, 0]]
nbow0 = [[0.25, 0.5, 0.75, 0.25, 0.25], [0.5, 0.5, 0.5, 0.5, 0]]

ans0 = bow_normalize(bow0)
assert(isinstance(ans0,list)), "Incorrect type of output. bow_normalize should return a list of floats."

assert(nbow0[0] == ans0[0]), "bow_normalize failed on {}".format(bow0[0])
assert(nbow0[1] == ans0[1]), "bow_normalize failed on {}".format(bow0[1])

bow1 = [[0, 1, 2, 0, 1, 1],
       [2, 1, 0, 3, 0, 1]]
nbow1 = [[0.0, 0.3779644730092272, 0.7559289460184544, 0.0, 0.3779644730092272, 0.3779644730092272],
    [0.5163977794943222, 0.2581988897471611, 0.0, 0.7745966692414834, 0.0, 0.2581988897471611]]

ans1 = bow_normalize(bow1)
assert(nbow1[0] == ans1[0]), "bow_normalize failed on {}".format(bow1[0])
assert(nbow1[1] == ans1[1]), "bow_normalize failed on {}".format(bow1[1])

print("==> So far, so good.")
# Some Random Instances
def check_bow_normalize_random():
    from random import choice, sample
    
    vec_len =  choice(range(2,8))
    nvecs = choice(range(3,7))
    rvecs = []
    for _ in range(nvecs):
        rvecs.append(sample(range(2*vec_len),vec_len))
        
    unit_rvecs = bow_normalize(rvecs)
    
    for i in range(len(unit_rvecs)):
        print("Input {}".format(rvecs[i]))
        u = unit_rvecs[i]
        ans = 1.0;
        
        for ui in u:
            ans = ans - (ui*ui)
            
        assert (math.isclose(ans,0,rel_tol=1e-6,abs_tol=1e-8)), "ERROR: Your output is incorrect {}".format(u)
        
print("\nRunning battery of random tests...")
for _ in range(20):
    check_bow_normalize_random()

print("\n (Passed!)")
```

    ==> So far, so good.
    
    Running battery of random tests...
    Input [0, 2, 3, 1]
    Input [1, 6, 2, 3]
    Input [0, 2, 6, 5]
    Input [6, 5, 1, 0]
    Input [3, 13, 5, 12, 6, 10, 9]
    Input [6, 0, 13, 3, 5, 12, 2]
    Input [9, 13, 7, 2, 3, 8, 5]
    Input [13, 8, 9, 5, 1, 11, 7]
    Input [8, 9, 5, 0, 1]
    Input [2, 1, 3, 5, 4]
    Input [5, 9, 3, 1, 8]
    Input [6, 9, 7, 2, 5]
    Input [1, 2, 6, 9, 3]
    Input [1, 2, 3, 4]
    Input [4, 5, 6, 1]
    Input [5, 3, 2, 7]
    Input [3, 7, 4, 6]
    Input [7, 4, 3, 0]
    Input [2, 6, 4, 1]
    Input [5, 4, 7, 0]
    Input [3, 6, 1, 7]
    Input [7, 6, 0, 2]
    Input [0, 5, 4, 3]
    Input [4, 3, 2]
    Input [5, 4, 1]
    Input [4, 5, 2]
    Input [5, 2, 4]
    Input [3, 2, 4]
    Input [0, 5, 2, 1, 3]
    Input [6, 8, 2, 3, 5]
    Input [4, 7, 5, 9, 3]
    Input [0, 9, 8, 6, 2]
    Input [13, 6, 4, 7, 12, 2, 10]
    Input [9, 11, 0, 6, 10, 7, 13]
    Input [12, 6, 3, 11, 2, 7, 4]
    Input [12, 10, 8, 4, 3, 7, 6]
    Input [13, 0, 10, 2, 9, 8, 5]
    Input [4, 6, 9, 7, 8]
    Input [4, 5, 1, 3, 9]
    Input [4, 6, 1, 2, 3]
    Input [1, 3, 6, 7, 9]
    Input [2, 3, 4, 1, 0]
    Input [5, 9, 6, 1, 4]
    Input [4, 9, 6, 11, 3, 7, 5]
    Input [13, 5, 12, 10, 9, 4, 2]
    Input [13, 0, 2, 10, 12, 8, 6]
    Input [0, 1, 4]
    Input [0, 4, 1]
    Input [5, 1, 2]
    Input [1, 2, 3, 0, 11, 7]
    Input [9, 8, 6, 4, 11, 7]
    Input [5, 2, 0, 4, 6, 3]
    Input [9, 5, 3, 0, 4, 8]
    Input [1, 9, 11, 0, 2, 8]
    Input [5, 4, 7, 3, 9, 0]
    Input [0, 3]
    Input [2, 0]
    Input [2, 3]
    Input [1, 0]
    Input [4, 0, 9, 8, 2, 1]
    Input [0, 11, 4, 10, 7, 6]
    Input [8, 11, 0, 7, 4, 10]
    Input [1, 11, 3, 6, 4, 5]
    Input [9, 1, 11, 5, 3, 8]
    Input [2, 3, 1]
    Input [2, 4, 0]
    Input [3, 1, 5]
    Input [5, 4, 1]
    Input [2, 1]
    Input [0, 3]
    Input [3, 1]
    Input [1, 3]
    Input [2, 3]
    Input [3, 5, 1]
    Input [0, 2, 3]
    Input [2, 1, 0]
    Input [2, 5, 0]
    Input [5, 2, 3]
    Input [2, 0, 1]
    Input [10, 11, 7, 9, 5, 6]
    Input [8, 5, 1, 7, 10, 6]
    Input [4, 2, 10, 7, 11, 3]
    Input [11, 5, 8, 4, 10, 0]
    Input [5, 6, 7, 1, 0, 4]
    Input [11, 10, 7, 8, 0, 5]
    Input [2, 9, 1, 4, 6]
    Input [6, 7, 4, 2, 0]
    Input [2, 4, 6, 9, 8]
    Input [1, 8, 9, 2, 5]
    Input [2, 7, 9, 4, 1]
    Input [8, 7, 6, 4, 0]
    Input [1, 2]
    Input [3, 0]
    Input [1, 2]
    Input [2, 3]
    Input [3, 1]
    Input [2, 0]
    
     (Passed!)
    

(_Aside_) **Sparsity of BoW vectors.** As an aside, run the next cell to see the BoW vectors are actually quite _sparse_.


```python
literature_vocab = create_vocab(docs)
bow = bagofwords(docs, literature_vocab)
nbow = bow_normalize(bow)
numterms = len(docs)*len(literature_vocab)
numzeros = sum([b.count(0) for b in nbow])
print("Percentage of entries which are zero: {:.1f} %".format(100*numzeros/numterms))
```

    Percentage of entries which are zero: 85.9 %
    

> If everything is correct, you'll see that the BoW vectors are sparse, with about 85-86% of the components being zeroes. Therefore, we could in principle save a lot of space by only storing the non-zeroes. While we do not exploit this fact in our current example it is useful to think about these costs while running analytics at scale.

# Part 2. Comparing Documents

Now we have normalized vector versions of each document, we can use a standard similarity measure to compare vectors. For this question, we shall use the *inner product*. Recall that the inner product of two vectors $a,b \in \mathbf{R}^n$ is defined as,

$$<a,b> = \Sigma_{i=0}^{n-1} a_i b_i$$

For example,

$$<[1,-1,3], [2,4,-1]> = (1\times2)+(-1\times4)+(3\times-1)=-5$$

**Exercise 2.a** (1 point) Complete the function `inner_product(a, b)` which takes two vectors, `a` and `b`, both represented as lists, and returns their inner product.

> Note: `inner_product(a, b)` should return a value of type `float`.


```python
def inner_product(a,b):
    assert (isinstance(a, list)), "inner_product expects a list of floats/ints as input for a."
    assert (isinstance(b, list)), "inner_product expects a list of floats/ints as input for b."
    assert len(a) == len(b), "inner_product should be called on vectors of the same length."
    prod = sum([ia*ib for ia,ib in zip(a,b)])
    return float(prod)

```


```python
# Test Cell: `test_inner_product` (0.5 point)

vec1a = [1,-1,3]
vec1b = [2,4,-1]
ans1 = -5
assert (isinstance(inner_product(vec1a,vec1b),float)), "Incorrect type of output. inner_product should return a float."
assert (inner_product(vec1a,vec1b) == ans1), "inner_product failed on inputs {} and {}".format(vec1a,vec1b)
assert (inner_product(vec1b,vec1a) == ans1), "inner_product failed on inputs {} and {}".format(vec1b,vec1a)

vec2a = [0,2,1,9,-1]
vec2b = [17,4,1,-1,0]
ans2 = 0
assert (inner_product(vec2a,vec2b) == ans2), "inner_product failed on inputs {} and {}".format(vec2a,vec2b)
assert (inner_product(vec2b,vec2a) == ans2), "inner_product failed on inputs {} and {}".format(vec2b,vec2a)

print("\n (Passed!)")
```

    
     (Passed!)
    

We can use the `inner_product()` as a measure of similarity between documents! (_Recall the linear algebra refresher in Topic 3_.) In particular, since our normalized BoW vectors are "direction" vectors, the inner product measures how closely two vectors point in the same direction.

**Exercise 2.b** (1 point). Now we can finally answer our initial question: which book excerpts are similar to each other? Complete the function `most_similar(nbows, target)`, below, to answer this question. In particular, it should take as input the normalized BoW vectors created in the previous part, as well as a target excerpt index $i$. It should return most index of the excerpt most similar to $i$.

> **Note 0.** Ties in scores are won by the smaller index. For example, if excerpt 2 and excerpt 7 both equally similar to the target excerpt 8, then return 2 as the most similar excerpt.
>
> **Note 1.** Your `most_similar()` function should return a value of type `int`.

> **Note 2.** The test cell refers to hidden tests, but in fact, the test is not hidden per se. Instead, we are hashing the strings returned by your solution to be able to check your answer without revealing it to you directly.


```python
def most_similar(nbows, target):
    assert (isinstance(nbows,list)), "most_similar expects list as input for nbows."
    assert (isinstance(target,int)), "most_similar expects integer as input for target."
    most_sim_idx = -1
    most_sim_val = -1
    
    # For the first half (j<i)
    for j in range(len(nbows)):
        if j == target:
            continue # Don't check similarity to self
        val = inner_product(nbows[j],nbows[target])
        if (val > most_sim_val):
            most_sim_idx = j
            most_sim_val = val
            
    return most_sim_idx

```


```python
# Test Cell: `test_most_similar` (1 point)
literature_vocab = create_vocab(docs)
bow = bagofwords(docs, literature_vocab)
nbow = bow_normalize(bow)

# Start with two basic cases:
target1 = books.index('1984') 
ans1 = books.index('kiterunner')
assert (isinstance(most_similar(nbow,target1),int)), "most_similar should return integer."
assert (most_similar(nbow,target1) == ans1), "most_similar failed on input {}".format(books[target1])

target2 = books.index('prideandprejudice')
ans2 = books.index('hamlet')
assert (most_similar(nbow,target2) == ans2), "most_similar failed on input {}".format(books[target2])

# Check the rest via obscured, hashed solutions
###
### AUTOGRADER TEST - DO NOT REMOVE
###
def check_most_similar_solns():
    from problem_utils import make_hash, open_file
    literature_vocab = create_vocab(docs)
    bow = bagofwords(docs, literature_vocab)
    nbow = bow_normalize(bow)
    with open_file("most_similar_solns.csv", "rt") as fp_soln:
        for line in fp_soln.readlines():
            target_name, soln_hashed = line.strip().split(',')
            target_id = books.index(target_name)
            your_most_sim_id = most_similar(nbow, target_id)
            assert isinstance(your_most_sim_id, int), f"Your function returns a value of type {type(your_most_sim_id)}, not an integer"
            assert 0 <= your_most_sim_id < len(nbow), f"You returned {your_most_sim_id}, which is an invalid value (it should be between 0 and {nbow})"
            your_most_sim_name = books[your_most_sim_id]
            print(f"For book '{target_name}', you calculated '{your_most_sim_name}' as most similar.")
            your_most_sim_name_hashed = make_hash(your_most_sim_name)
            assert your_most_sim_name_hashed == soln_hashed, "==> ERROR: Unfortunately, your returned value does not appear to match our reference solution."

check_most_similar_solns()
print("\n (Passed!)")
```

    For book '1984', you calculated 'kiterunner' as most similar.
    For book 'gatsby', you calculated 'prideandprejudice' as most similar.
    For book 'hamlet', you calculated 'prideandprejudice' as most similar.
    For book 'janeeyre', you calculated 'prideandprejudice' as most similar.
    For book 'kiterunner', you calculated '1984' as most similar.
    For book 'littleprince', you calculated 'littlewomen' as most similar.
    For book 'littlewomen', you calculated 'littleprince' as most similar.
    For book 'olivertwist', you calculated '1984' as most similar.
    For book 'prideandprejudice', you calculated 'hamlet' as most similar.
    For book 'prisonerofazkaban', you calculated '1984' as most similar.
    
     (Passed!)
    

Now let's have a look at the documents most similar to each other, according to your implementation.


```python
for idx in range(len(books)):
    jdx = most_similar(nbow,idx)
    print(books[idx],"is most similar to",books[jdx],"!")
```

    1984 is most similar to kiterunner !
    gatsby is most similar to prideandprejudice !
    hamlet is most similar to prideandprejudice !
    janeeyre is most similar to prideandprejudice !
    kiterunner is most similar to 1984 !
    littleprince is most similar to littlewomen !
    littlewomen is most similar to littleprince !
    olivertwist is most similar to 1984 !
    prideandprejudice is most similar to hamlet !
    prisonerofazkaban is most similar to 1984 !
    

**Fin!** You’ve reached the end of this part. Don’t forget to restart and run all cells again to make sure it’s all working when run in sequence; and make sure your work passes the submission process. Good luck!
