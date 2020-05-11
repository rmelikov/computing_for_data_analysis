# Part 1 of 2: Processing an HTML file

One of the richest sources of information is [the Web](http://www.computerhistory.org/revolution/networking/19/314)! In this notebook, we ask you to use string processing and regular expressions to mine a web page, which is stored in HTML format.

> **Note 0.** The exercises below involve processing of HTML files. However, you don't need to know anything specific about HTML; you can solve (and we have solved) all of these exercises assuming only that the data is a semi-structured string, amenable to simple string manipulation and regular expression processing techniques. In a future notebook, you'll see a different method that employs the [Beautiful Soup module](https://www.crummy.com/software/BeautifulSoup/bs4/doc/).
>
> **Note 1.** Following Note 0, there are some outspoken people who believe you should never use regular expressions on HTML. Your instructor finds these arguments to be overly pedantic. For an entertaining take on the subject, see [this blog post](https://blog.codinghorror.com/parsing-html-the-cthulhu-way/).
>
> **Note 2.** The data below is a snapshot from an older version of the Yelp! site. Therefore, you should complete the exercises using the data we've provided, rather than downloading a copy directly from Yelp!.

**The data: Yelp! reviews.** The data you will work with is a snapshot of a recent search on the [Yelp! site](https://yelp.com) for the best fried chicken restaurants in Atlanta. That snapshot is hosted here: https://cse6040.gatech.edu/datasets/yelp-example

If you go ahead and open that site, you'll see that it contains a ranked list of places:

![Top 10 Fried Chicken Spots in ATL as of September 12, 2017](https://cse6040.gatech.edu/datasets/yelp-example/ranked-list-snapshot.png)

**Your task.** In this part of this assignment, we'd like you to write some code to extract this list.

## Getting the data

First things first: you need an HTML file. The following Python code opens a copy of the sample Yelp! page from above.


```python
import hashlib

with open('yelp.htm', 'r', encoding='utf-8') as f:
    yelp_html = f.read().encode(encoding='utf-8')
    checksum = hashlib.md5(yelp_html).hexdigest()
    assert checksum == "4a74a0ee9cefee773e76a22a52d45a8e", "Downloaded file has incorrect checksum!"
    
print("'yelp.htm' is ready!")
```

    'yelp.htm' is ready!
    

**Viewing the raw HTML in your web browser.** The file you just downloaded is the raw HTML version of the data described previously. Before moving on, you should go back to that site and use your web browser to view the HTML source for the web page. Do that now to get an idea of what is in that file.

> If you don't know how to view the page source in your browser, try the instructions on [this site](http://www.wikihow.com/View-Source-Code).

**Reading the HTML file into a Python string.** Let's also open the file in Python and read its contents into a string named, `yelp_html`.


```python
with open('yelp.htm', 'r', encoding='utf-8') as yelp_file:
    yelp_html = yelp_file.read()
    
# Print first few hundred characters of this string:
print("*** type(yelp_html) == {} ***".format(type(yelp_html)))
n = 1000
print("*** Contents (first {} characters) ***\n{} ...".format(n, yelp_html[:n]))
```

    *** type(yelp_html) == <class 'str'> ***
    *** Contents (first 1000 characters) ***
    <!DOCTYPE html>
    <!-- saved from url=(0079)https://www.yelp.com/search?find_desc=fried+chicken&find_loc=Atlanta%2C+GA&ns=1 -->
    <html xmlns:fb="http://www.facebook.com/2008/fbml" class="js gr__yelp_com" lang="en"><!--<![endif]--><head data-component-bound="true"><meta http-equiv="Content-Type" content="text/html; charset=UTF-8"><link type="text/css" rel="stylesheet" href="./Best Fried chicken in Atlanta, GA - Yelp_files/css"><style type="text/css">.gm-style .gm-style-cc span,.gm-style .gm-style-cc a,.gm-style .gm-style-mtc div{font-size:10px}
    </style><style type="text/css">@media print {  .gm-style .gmnoprint, .gmnoprint {    display:none  }}@media screen {  .gm-style .gmnoscreen, .gmnoscreen {    display:none  }}</style><style type="text/css">.gm-style-pbc{transition:opacity ease-in-out;background-color:rgba(0,0,0,0.45);text-align:center}.gm-style-pbt{font-size:22px;color:white;font-family:Roboto,Arial,sans-serif;position:relative;margin:0;top:50%;-webkit-transform:translateY(-50%);-ms- ...
    

Oy, what a mess! It will be great to have some code read and process the information contained within this file.

## Exercise (5 points): Extracting the ranking

Write some Python code to create a variable named `rankings`, which is a list of dictionaries set up as follows:

* `rankings[i]` is a dictionary corresponding to the restaurant whose rank is `i+1`. For example, from the screenshot above, `rankings[0]` should be a dictionary with information about Gus's World Famous Fried Chicken.
* Each dictionary, `rankings[i]`, should have these keys:
    * `rankings[i]['name']`: The name of the restaurant, a string.
    * `rankings[i]['stars']`: The star rating, as a string, e.g., `'4.5'`, `'4.0'`
    * `rankings[i]['numrevs']`: The number of reviews, as an **integer.**
    * `rankings[i]['price']`: The price range, as dollar signs, e.g., `'$'`, `'$$'`, `'$$$'`, or `'$$$$'`.
    
Of course, since the current topic is regular expressions, you might try to apply them (possibly combined with other string manipulation methods) find the particular patterns that yield the desired information.


```python
import re

pattern = r'''
     <li\sclass=\"regular-search-result\">(.|\n)*?(?<=\<span\sclass=\"indexed-biz-name\"\>)
     (.|\n)*?\<span\>
     (?P<name>.+)
     \<\/span\>(.|\n)*?alt=\"
     (?P<stars>\d\.\d)
     \sstar\srating\"(.|\n)*?\<span\sclass=\"review-count\srating-qualifier\"\>(\s|\t|\n)*?
     (?P<numrevs>\d{1,7})
     (.|\n)*?\<span\sclass=\"business-attribute\sprice-range\">
     (?P<price>\${1,6})
     \<\/span\>(.|\n)*?<\/li>
'''

pattern_matcher = re.compile(pattern, re.VERBOSE)

matches = pattern_matcher.finditer(yelp_html)

rankings = []

for item in matches:
    rankings.append(item.groupdict())

rankings = [{k : (int(v) if k == 'numrevs' else v) for k, v in d.items()} for d in rankings]
```


```python
# Test cell: `rankings_test`

assert type(rankings) is list, "`rankings` must be a list"
assert all([type(r) is dict for r in rankings]), "All `rankings[i]` must be dictionaries"

print("=== Rankings ===")
for i, r in enumerate(rankings):
    print("{}. {} ({}): {} stars based on {} reviews".format(i+1,
                                                             r['name'],
                                                             r['price'],
                                                             r['stars'],
                                                             r['numrevs']))

assert rankings[0] == {'numrevs': 549, 'name': 'Gus’s World Famous Fried Chicken', 'stars': '4.0', 'price': '$$'} \
       or rankings[0] == {'numrevs': 549, 'name': 'Gus&#39;s World Famous Fried Chicken', 'stars': '4.0', 'price': '$$'}
assert rankings[1] == {'numrevs': 1777, 'name': 'South City Kitchen - Midtown', 'stars': '4.5', 'price': '$$'}
assert rankings[2] == {'numrevs': 2241, 'name': 'Mary Mac’s Tea Room', 'stars': '4.0', 'price': '$$'} \
       or rankings[2] == {'numrevs': 2241, 'name': 'Mary Mac&#39;s Tea Room', 'stars': '4.0', 'price': '$$'}
assert rankings[3] == {'numrevs': 481, 'name': 'Busy Bee Cafe', 'stars': '4.0', 'price': '$$'}
assert rankings[4] == {'numrevs': 108, 'name': 'Richards’ Southern Fried', 'stars': '4.0', 'price': '$$'} \
       or rankings[4] == {'numrevs': 108, 'name': 'Richards&#39; Southern Fried', 'stars': '4.0', 'price': '$$'}
assert rankings[5] == {'numrevs': 93, 'name': 'Greens &amp; Gravy', 'stars': '3.5', 'price': '$$'} \
       or rankings[5] == {'numrevs': 93, 'name': 'Greens & Gravy', 'stars': '3.5', 'price': '$$'}
assert rankings[6] == {'numrevs': 350, 'name': 'Colonnade Restaurant', 'stars': '4.0', 'price': '$$'}
assert rankings[7] == {'numrevs': 248, 'name': 'South City Kitchen Buckhead', 'stars': '4.5', 'price': '$$'}
assert rankings[8] == {'numrevs': 1558, 'name': 'Poor Calvin’s', 'stars': '4.5', 'price': '$$'} \
       or rankings[8] == {'numrevs': 1558, 'name': 'Poor Calvin&#39;s', 'stars': '4.5', 'price': '$$'}
assert rankings[9] == {'numrevs': 67, 'name': 'Rock’s Chicken &amp; Fries', 'stars': '4.0', 'price': '$'} \
       or rankings[9] == {'numrevs': 67, 'name': 'Rock&#39;s Chicken &amp; Fries', 'stars': '4.0', 'price': '$'} \
       or rankings[9] == {'numrevs': 67, 'name': 'Rock&#39;s Chicken & Fries', 'stars': '4.0', 'price': '$'} \
       or rankings[9] == {'numrevs': 67, 'name': 'Rock’s Chicken & Fries', 'stars': '4.0', 'price': '$'}

print("\n(Passed!)")
```

    === Rankings ===
    1. Gus’s World Famous Fried Chicken ($$): 4.0 stars based on 549 reviews
    2. South City Kitchen - Midtown ($$): 4.5 stars based on 1777 reviews
    3. Mary Mac’s Tea Room ($$): 4.0 stars based on 2241 reviews
    4. Busy Bee Cafe ($$): 4.0 stars based on 481 reviews
    5. Richards’ Southern Fried ($$): 4.0 stars based on 108 reviews
    6. Greens &amp; Gravy ($$): 3.5 stars based on 93 reviews
    7. Colonnade Restaurant ($$): 4.0 stars based on 350 reviews
    8. South City Kitchen Buckhead ($$): 4.5 stars based on 248 reviews
    9. Poor Calvin’s ($$): 4.5 stars based on 1558 reviews
    10. Rock’s Chicken &amp; Fries ($): 4.0 stars based on 67 reviews
    
    (Passed!)
    

**Fin!** This cell marks the end of Part 1. Don't forget to save, restart and rerun all cells, and submit it. When you are done, proceed to Part 2.
