---
output:
  word_document: default
  pdf_document: default
  html_document: default
---
# Sample notebook: Part 1

This notebook is Part 1 of two parts (Parts 0 and 1): in the computer science tradition, we will try to number beginning at 0. Together, the two parts comprise an ungraded *lab notebook assignment* (or just *lab*, *notebook*, or *assignment*). Please use it to familiarize yourself with how to complete and submit your work.


```python
import sys
print(sys.version)

import os
```

    3.7.4 (tags/v3.7.4:e09359112e, Jul  8 2019, 20:34:20) [MSC v.1916 64 bit (AMD64)]
    

## Getting input data

Throughout the course, we'll use a variety of methods to get data for use in the notebook environment. In this example, we are providing some data in the form of a file stored in the local directory. this file is named `message_in_a_bottle.txt.zip`, and it contains a secret message that you will reveal momentarily.


```python
print("\n=== Files in the current directory ===\n{}".format(os.listdir('.')))
```

    
    === Files in the current directory ===
    ['.ipynb_checkpoints', 'message_in_a_bottle.txt.zip', 'part1.ipynb']
    

**Exercise 0** (1 point). In the code cell below, create a variable named `filename` and initialize it to a string containing the name `message_in_a_bottle.txt.zip`. The test cell that follows it will unpack this file, assuming it is available in the current working directory, unpack it, and then print its contents.


```python
uncompressed_name = 'message_in_a_bottle.txt'
#compressed_extension = '.zip'

###
filename = 'message_in_a_bottle.txt.zip'
###

```


```python
# Test cell: `filename_test`

print("`filename`: '{}'".format(filename))
from zipfile import ZipFile
with ZipFile(filename, 'r') as input_zip:
    with input_zip.open(filename[:-4], 'r') as input_file:
        message = input_file.readline().decode('utf-8')
print("\n=== BEGIN MESSAGE ===\n{}=== END MESSAGE ===".format(message))
```

    `filename`: 'message_in_a_bottle.txt.zip'
    
    === BEGIN MESSAGE ===
    Good luck, kiddos!
    === END MESSAGE ===
    

This is the end of Part 1. If everything seems to have worked, try submitting it!
