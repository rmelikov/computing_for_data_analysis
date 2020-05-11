# Problem 6

In this problem, you will write code to "parse" a restricted form of SQL queries. These exercises are about string processing and regular expressions. There are five (5) exercises, numbered 0-4, which are worth a total of ten (10) points.


```python
from IPython.display import display

import re
import pandas as pd

# Random number generation for generating test cases:
from random import randrange, randint, choice, sample
```

**Background: SQL review.** Suppose you have two SQL tables, `OneTable` and `AnotherTable`, and you wish to perform an inner-join on that links a column named `ColA` in `OneTable` with `ColB` in `AnotherTable`. Recall that one simple way to do that in SQL would be:

```SQL
    SELECT * FROM OneTable, AnotherTable
        WHERE OneTable.ColA = AnotherTable.ColB
```

Or, consider the following more complex example. Suppose you have an additional table named `YetAThird` and you wish to extend the inner-join to include matches between its column, `ColC`, and a second column from `AnotherTable` named `ColB2`:

```SQL
    SELECT * FROM OneTable, AnotherTable, YetAThird
        WHERE OneTable.ColA = AnotherTable.ColB AND AnotherTable.ColB2 = YetAThird.ColC
```

**Exercise 0** (2 points). Suppose you are given a string containing an SQL query in the restricted form,

```SQL
    SELECT * FROM [tbls] WHERE [conds]
```

Implement the function, **`split_simple_join(q)`**, so that it takes a query string **`q`** in the form shown above and **returns a pair of substrings** corresponding to `[tbls]` and `[conds]`.

For example, if

```python
    q == """SELECT * FROM OneTable, AnotherTable, YetAThird
              WHERE OneTable.ColA = AnotherTable.ColB AND AnotherTable.ColB2=YetAThird.ColC"""
```

then

```python
    split_simple_join(q) == ("OneTable, AnotherTable, YetAThird",
                             "OneTable.ColA = AnotherTable.ColB AND AnotherTable.ColB2=YetAThird.ColC")
```

**IMPORTANT NOTE!** In this problem, you only need to return the substring between `FROM` and `WHERE` and the one after `WHERE`. You will extract the table names and conditions later on, below.

You should make the following assumptions:

* The input string `q` contains exactly one such query, with no nesting of queries (e.g., no instances of `"SELECT * FROM (SELECT ...)"`). However, the query may (or may not) be a multiline string as shown in the example. (Treat newlines as whitespace.)
* Your function should ignore any leading or trailing whitespace around the SQL keywords, e.g., `SELECT`, `FROM`, and `WHERE`.
* The substring between `SELECT` and `FROM` will be any amount of whitespace, followed by an asterisk (`*`). 
* You should **not** treat the SQL keywords in a case-sensitive way; for example, you would regard `SELECT`, `select`, and `sElEct` as the same. However, do **not** change or ignore the case of the non-SQL keywords.
* The `[tbls]` substring contains only a simple list of table names and no other substrings that might be interpreted as SQL keywords.
* The `[conds]` substring contains only table and column names (e.g., `OneTable.ColA`), the equal sign, the `AND` SQL keyword, and whitespace, but no other SQL keywords or symbols.

> Assuming you are using regular expressions for this problem, recall that you can pass [`re.VERBOSE`](https://docs.python.org/3/library/re.html#re.VERBOSE) when writing a multiline regex pattern.


```python
def split_simple_join(q):
    assert type(q) is str
    
    # method 1 (relatively quick to implement)
    #step1 = re.split(r'(?i)from[\s\n\r\t]+', q)[1]
    #step2 = re.split(r'(?i)[\n\r\t\s]+where[\n\r\t\s]+', step1)
    #return tuple(step2)

    # method 2 (lots of typing)
    #pattern = r"""
    #    \s*[sS][eE][lL][eE][cC][tT]\s+\*
    #    \s+[fF][rR][oO][mM]\s+(?P<tbls>.+)
    #    \s+[wW][hH][eE][rR][eE]\s+(?P<conds>.+)
    #    \s*
    #"""
    #pattern_matcher = re.compile(pattern, re.VERBOSE)
    #
    #matches = pattern_matcher.match(q)
    #
    #return matches.group('tbls'), matches.group('conds')
    
    
    # method 3 utilizes the case insensitive flag `(?i)` and
    # the fact that `re.split()` will only return capturing 
    # groups while others will be empty, provided you account
    # for everything. also, we don't need to name capturing groups.
    pattern = r'''
        (?i)\s*select.+from[\s\n\r\t]+
        (?P<tbls>.+)
        [\n\r\t\s]+where[\n\r\t\s]+
        (?P<conds>.+)
        \s*
    '''
    
    pattern_matcher = re.compile(pattern, re.VERBOSE)
    
    return tuple(s for s in re.split(pattern_matcher, q) if s)
    
# Demo
q_demo = """SELECT * FROM OneTable, AnotherTable, YetAThird
              WHERE OneTable.ColA = AnotherTable.ColB AND AnotherTable.ColB2=YetAThird.ColC"""
print(split_simple_join(q_demo))
```

    ('OneTable, AnotherTable, YetAThird', 'OneTable.ColA = AnotherTable.ColB AND AnotherTable.ColB2=YetAThird.ColC')
    


```python
# Test cell: `split_simple_join_test1`

assert split_simple_join(q_demo) == \
           ('OneTable, AnotherTable, YetAThird',
            'OneTable.ColA = AnotherTable.ColB AND AnotherTable.ColB2=YetAThird.ColC')
print("\n(Passed!)")
```

    
    (Passed!)
    


```python
# Test cell: `split_simple_join_test2`

__SQL = {'SELECT', 'FROM', 'WHERE'} # SQL keywords

# Different character classes
__ALPHA = 'abcdefghijklmnopqrstuvwxyzABCDEFGHIJKLMNOPQRSTUVWXYZ_'
__ALPHA_VARS = __ALPHA + '_'
__NUM = '0123456789'
__ALPHA_NUM_VARS = __ALPHA_VARS + __NUM
__DOT = '.'
__SPACES = ' \t\n'
__EQ = '='
__ALL = __ALPHA_NUM_VARS + __DOT + __SPACES + __EQ

def flip_coin():
    from random import choice
    return choice([True, False])

def rand_str(k_min, k_max, alphabet):
    """
    Returns a random string of `k` letters chosen uniformly from
    from `alphabet` with replacement, where `k_min <= k <= k_max`
    where `k` is also chosen uniformly at random.
    """
    assert k_max >= k_min >= 0
    k = k_min + randint(0, k_max-k_min)
    return ''.join([choice(alphabet) for _ in range(k)])

def rand_spaces(k_max, k_min=0):
    assert k_max >= k_min >= 0
    return rand_str(k_min, k_max, __SPACES)

def rand_case(s):
    """Randomly changes the case of each letter in a string."""
    return ''.join([c.upper() if flip_coin() else c.lower() for c in s])

def rand_var(k_min=1, k_max=10):
    assert k_max >= k_min >= 1
    s = choice(__ALPHA_VARS)
    s += rand_str(k_min-1, k_max-1, __ALPHA_NUM_VARS)
    if s.upper() in __SQL: # Don't generate keywords -- try again
        return rand_var(k_min, k_max)
    return s

def rand_vars(k_min, k_max):
    V = set()
    dk = randint(0, k_max-k_min)
    for k in range(k_min + dk + 1):
        v = rand_var()
        V.add(v)
    return V

def rand_table(max_cols):
    table = rand_var()
    columns = rand_vars(1, max_cols)
    return (table, columns)

def rand_tables(k_min, k_max, c_max=4):
    assert k_max >= k_min >= 1
    num_tables = k_min + randint(0, k_max-k_min)
    tables = {}
    for _ in range(num_tables):
        table, columns = rand_table(c_max)
        while table in tables:
            table, columns = rand_table(v_max, c_max)
        tables = {**tables, **{table: columns}}
    return tables

def rand_field(table_name, col_names):
    return table_name + "." + choice(list(col_names))

def rand_cond(tables):
    assert type(tables) is dict
    assert len(tables) >= 2
    a, b = sample(tables.keys(), 2)
    a_col = rand_field(a, tables[a])
    b_col = rand_field(b, tables[b])
    return (a_col, b_col)

def rand_cond_set(tables, k_min, k_max):
    k = k_min + randint(0, k_max-k_min)
    conds = set()
    while len(conds) < k:
        (a, b) = rand_cond(tables)
        if ((a, b) not in conds) and ((b, a) not in conds):
            conds.add((a, b))
    return conds

def cond_set_to_substrs(conds, max_wspad=4):
    substrs = []
    for a, b in conds:
        s = "{}{}{}{}{}".format(a,
                                rand_spaces(max_wspad),
                                __EQ,
                                rand_spaces(max_wspad),
                                b)
        substrs.append(s)
    return substrs

def substrs_to_str(substrings, sep=',', max_wspad=4):
    s_final = ''
    for k, s in enumerate(substrings):
        if k > 0:
            s_final += rand_spaces(max_wspad) + sep + rand_spaces(max_wspad)
        s_final += s
    return s_final

def rand_query_ans(max_tables, max_conds, max_wspad=4):
    tables = rand_tables(2, max_tables)
    cond_set = rand_cond_set(tables, 1, 4)
    return tables, cond_set

def pad_1(max_wspad):
    return rand_spaces(max(1, max_wspad), 1)

def form_select(max_wspad=4):
    return pad_1(max_wspad) + rand_case("SELECT") + pad_1(max_wspad) + "*"

def form_from(tables, max_wspad=4):
    from_ans = substrs_to_str(list(tables.keys()), sep=',', max_wspad=max_wspad)
    return pad_1(max_wspad) + rand_case("FROM") + pad_1(max_wspad) + from_ans, from_ans

def form_where(cond_set, max_wspad=4):
    cond_substrs = cond_set_to_substrs(cond_set)
    cond_ans = substrs_to_str(cond_substrs, sep=' AND ', max_wspad=max_wspad)
    return pad_1(max_wspad) + rand_case("WHERE") + pad_1(max_wspad) + cond_ans, cond_ans

def form_query_str(tables, cond_set, max_wspad=4):
    select_clause = form_select(max_wspad)
    from_clause, from_ans = form_from(tables, max_wspad)
    where_clause, cond_ans = form_where(cond_set, max_wspad)
    query = select_clause + from_clause + where_clause
    return query, from_ans, cond_ans

def split_simple_join_battery(num_tests, max_wspad=4):
    for k in range(num_tests):
        tables, cond_set = rand_query_ans(5, 5, max_wspad)
        qstmt, from_ans, where_ans = form_query_str(tables, cond_set, max_wspad)
        print("=== Test Statement {} ===\n'''{}'''\n".format(k, qstmt))
        print("True 'FROM' clause substring: '''{}'''\n".format(from_ans))
        print("True 'WHERE' clause substring: '''{}'''\n".format(where_ans))
    
split_simple_join_battery(5, 3)

print("\n(Passed!)")
```

    === Test Statement 0 ===
    '''
    seLect
    	*
     FROM
     _mqf 
    , uMUoaazm 
    ,	
    	GK 		,   BPQo0zQuuj  
    wHERe GK.WEJrp 
    =uMUoaazm._XaAf7D'''
    
    True 'FROM' clause substring: '''_mqf 
    , uMUoaazm 
    ,	
    	GK 		,   BPQo0zQuuj'''
    
    True 'WHERE' clause substring: '''GK.WEJrp 
    =uMUoaazm._XaAf7D'''
    
    === Test Statement 1 ===
    '''
    
    
    sEleCT 
    *
    	 FrOm  i4e 
    , 	 b1ouI4 , 
    xB6
    ,WeBSS6Nu   wheRe b1ouI4.ttW_PXG
    =	  i4e.vUt7'''
    
    True 'FROM' clause substring: '''i4e 
    , 	 b1ouI4 , 
    xB6
    ,WeBSS6Nu'''
    
    True 'WHERE' clause substring: '''b1ouI4.ttW_PXG
    =	  i4e.vUt7'''
    
    === Test Statement 2 ===
    '''	SElect  	*
    
     froM IgqBySd7ut	
    ,	 oTTbPig9YZ,
    
    
    UuSPxi , 
    CWLBmV
    
    
    ,
    KpN	
    	WHEre CWLBmV.jwxT_ 
    
    =	
    oTTbPig9YZ.PCuCfuHf  AND 
    
     UuSPxi.wu=  IgqBySd7ut.FC'''
    
    True 'FROM' clause substring: '''IgqBySd7ut	
    ,	 oTTbPig9YZ,
    
    
    UuSPxi , 
    CWLBmV
    
    
    ,
    KpN'''
    
    True 'WHERE' clause substring: '''CWLBmV.jwxT_ 
    
    =	
    oTTbPig9YZ.PCuCfuHf  AND 
    
     UuSPxi.wu=  IgqBySd7ut.FC'''
    
    === Test Statement 3 ===
    ''' SelECt	* 		From	
    	vfhKVQE6Ca 	
    ,  
    cJYm ,	
    G_Z 
    ,		znnFuzz	
     wheRE G_Z.jrz_Bc		=
    	
    znnFuzz.F8cX01_o'''
    
    True 'FROM' clause substring: '''vfhKVQE6Ca 	
    ,  
    cJYm ,	
    G_Z 
    ,		znnFuzz'''
    
    True 'WHERE' clause substring: '''G_Z.jrz_Bc		=
    	
    znnFuzz.F8cX01_o'''
    
    === Test Statement 4 ===
    ''' selEct	  *	fRom
    QE3_WD 	,
    	H9CCqb		 ,xH
    	WheRE		xH.HUfuI6rjT  = 
    QE3_WD.VT'''
    
    True 'FROM' clause substring: '''QE3_WD 	,
    	H9CCqb		 ,xH'''
    
    True 'WHERE' clause substring: '''xH.HUfuI6rjT  = 
    QE3_WD.VT'''
    
    
    (Passed!)
    

**Variable names.** For this problem, let a valid _variable name_ be a sequence of alphanumeric characters or underscores, where the very first character _cannot_ be a number. For example,

    some_variable
    __another_VariAble
    Yet_a_3rd_var
    _A_CSE_6040_inspired_var
    
are all valid variable names, whereas the following are not.

    123var_is_bad
    0_is_not_good_either
    4goodnessSakeStopItAlready

**Exercise 1** (2 points). Implement a function, **`is_var(s)`**, that checks whether a valid variable name. That is, it should return `True` if and only if `s` is valid according to the above definition. Your function should ignore any leading or trailing spaces in `s`.

For example:

```python
    assert is_var("__another_VariAble")
    assert not is_var("0_is_not_good_either")
    assert is_var("   foo")
    assert is_var("_A_CSE_6040_inspired_var   ")
    assert not is_var("#getMe2")
    assert is_var("   Yet_a_3rd_var  ")
    assert not is_var("123var_is_bad")
    assert not is_var("  A.okay")
```


```python
def is_var(s):
    assert type(s) is str
    pattern = r'(?i)^\s*[a-z_][a-z0-9_]*\s*$'
    matches = re.match(pattern, s)
    return matches is not None
    #return True if matches else False
```


```python
# Test cell, part 1: `is_var_test0`

assert is_var("__another_VariAble")
assert not is_var("0_is_not_good_either")
assert is_var("   foo")
assert is_var("_A_CSE_6040_inspired_var   ")
assert not is_var("#getMe2")
assert is_var("   Yet_a_3rd_var  ")
assert not is_var("123var_is_bad")
assert not is_var("  A.okay")

print("\n(Passed part 1 of 2.)")
```

    
    (Passed part 1 of 2.)
    


```python
# Test cell: `is_var_test2`

for v in rand_vars(20, 30):
    ans = flip_coin()
    if not ans:
        v = choice(__NUM) + v
    v = rand_spaces(3) + v + rand_spaces(3)
    your_ans = is_var(v)
    assert your_ans == ans, "is_var('{}') == {} instead of {}.".format(v, your_ans, ans)
    
print("\n(Passed part 2 of 2.)")
```

    
    (Passed part 2 of 2.)
    

**Column variables.** A _column variable_ consists of two valid variable names separated by a single period. For example,

    A.okay
    a32X844._387b
    __C__.B3am
    
are all examples of column variables: in each case, the substrings to the left and right of the period are valid variables.

**Exercise 2** (1 point). Implement a function, **`is_col(s)`**, so that it returns `True` if and only if **`s`** is a column variable, per the definition above.

For example:

```python
    assert is_col("A.okay")
    assert is_col("a32X844._387b")
    assert is_col("__C__.B3am")
    assert not is_col("123.abc")
    assert not is_col("abc.123")
```

As with Exercise 1, your function should ignore any leading or trailing spaces.


```python
def is_col(s):
    #pattern = r'(?i)^\s*[a-z_]{1}[a-z0-9_]*\.[a-z_]{1}[a-z0-9_]*\s*$'
    #matches = re.match(pattern, s)
    #return matches is not None
    
    # we can reuse the previous function we built. if a period exists
    # like it supposed to, then we split at the period and apply
    # `is_var()` function we built earlier to each element of the list.
    # we then apply the `all()` function to the list. if both elements
    # are true, then `all()` will return true.
    return all([is_var(sub) for sub in s.split('.')]) if '.' in s else False    
```


```python
# Test cell: `is_col_test0`

assert is_col("A.okay")
assert is_col("a32X844._387b")
assert is_col("__C__.B3am")
assert not is_col("123.abc")
assert not is_col("abc.123")

print("\n(Passed part 1.)")
```

    
    (Passed part 1.)
    


```python
# Test cell: `is_col_test1`

def test_is_col_1():
    a = rand_var()
    assert not is_col(a), "is_col('{}') == {} instead of {}.".format(a, is_col(a), False)
    a_valid = flip_coin()
    if not a_valid:
        a = rand_str(1, 5, __NUM)
    return a, a_valid

for _ in range(20):
    a, a_valid = test_is_col_1()
    b, b_valid = test_is_col_1()
    ans = a_valid and b_valid
    
    c = "{}{}.{}{}".format(rand_spaces(3), a, b, rand_spaces(3))
    your_ans = is_col(c)
    print("==> is_col('{}') == {}".format(c, your_ans))
    assert your_ans == ans

print("\n(Passed part 2.)")
```

    ==> is_col('	  SiCMmr7eCF.d') == True
    ==> is_col('	
    552.C4 		') == False
    ==> is_col(' xaSuf.bEWprbz	
     ') == True
    ==> is_col('52.FSi3
    ') == False
    ==> is_col(' TtAx.4') == False
    ==> is_col('gCE_YKPhK.kCUQUHSV5	') == True
    ==> is_col('
     
    n.8	') == False
    ==> is_col(' 7329.TkZJ3dDoN0		') == False
    ==> is_col('	_j922dX.C25avB  	') == True
    ==> is_col('
    
    16852.74') == False
    ==> is_col('
    
    	7.z9FUr  ') == False
    ==> is_col('  36.1 
    
    ') == False
    ==> is_col('	
    
    QVu5PTVT.79616		 ') == False
    ==> is_col('lk.V 
    ') == True
    ==> is_col('4.Fj	') == False
    ==> is_col('
     
    UJpa3GOwwR.02990	') == False
    ==> is_col('122.cjS') == False
    ==> is_col('	_T0.9
    	
    ') == False
    ==> is_col(' meftVBI_w3.DLpmGZ ') == True
    ==> is_col(' artcB4lUiY.kkRRw0Q') == True
    
    (Passed part 2.)
    

**Equality strings.** An _equality string_ is a string of the form,

    A.x = B.y

where `A.x` and `B.y` are _column variable_ names and `=` is an equals sign. There may be any amount of whitespace---including none---before or after each variable and the equals sign.

**Exercise 3** (2 points). Implement the function, **`extract_eqcols(s)`**, below. Given an input string **`s`**, if it is an equality string, your function should return a pair `(u, v)`, where `u` and `v` are the two column variables in the equality string. For example:

```python
    assert extract_eqcols("F3b._xyz =AB0_.def") == ("F3b._xyz", "AB0_.def")
```

If `s` is not a valid equality string, then your function should return `None`.


```python
def extract_eqcols(s):
    parts = s.split('=')
    if len(parts) == 2 and all([is_col(sub) for sub in parts]):
        return tuple([sub.strip() for sub in parts])
    return None

print(extract_eqcols("F3b._xyz =AB0_.def"))
```

    ('F3b._xyz', 'AB0_.def')
    


```python
# Test cell: `extract_eqcols0`

assert extract_eqcols("F3b._xyz =AB0_.def") == ("F3b._xyz", "AB0_.def")
assert extract_eqcols("0F3b._xyz =AB0_.def") is None

print("\n(Passed part 1 of 2.)")
```

    
    (Passed part 1 of 2.)
    


```python
# Test cell: `extract_eqcols1`

for _ in range(5):
    _, cond_set = rand_query_ans(2, 10, 5)
    for a, b in cond_set:
        s = a + rand_spaces(3) + __EQ + rand_spaces(3) + b
        print("==> Processing:\n'''{}'''\n".format(s))
        ans = extract_eqcols(s)
        print("    *** Found: {} ***".format(ans))
        assert ans is not None, "Did not detect an equality string where there was one!"
        assert ans[0] == a and ans[1] == b, "Returned {} instead of ({}, {})".format(ans, a, b)

print("\n(Passed part 2 of 2.)")
```

    ==> Processing:
    '''CnkbVTgLOE.P_  =	Hl.Ey4Nc0'''
    
        *** Found: ('CnkbVTgLOE.P_', 'Hl.Ey4Nc0') ***
    ==> Processing:
    '''CnkbVTgLOE.is8Sqx
     	=	 
    Hl.Ey4Nc0'''
    
        *** Found: ('CnkbVTgLOE.is8Sqx', 'Hl.Ey4Nc0') ***
    ==> Processing:
    '''XbWanTY.eg  = t9PM.EPCvsEoGh'''
    
        *** Found: ('XbWanTY.eg', 't9PM.EPCvsEoGh') ***
    ==> Processing:
    '''XbWanTY.eg=	  t9PM.z4Qhh9UU'''
    
        *** Found: ('XbWanTY.eg', 't9PM.z4Qhh9UU') ***
    ==> Processing:
    '''GqIqeS4u1P.iukotR
    
    =	lDPCxFL.Suz88'''
    
        *** Found: ('GqIqeS4u1P.iukotR', 'lDPCxFL.Suz88') ***
    ==> Processing:
    '''GqIqeS4u1P.rgZVW_1cL3
    
    =
    lDPCxFL.Suz88'''
    
        *** Found: ('GqIqeS4u1P.rgZVW_1cL3', 'lDPCxFL.Suz88') ***
    ==> Processing:
    '''GqIqeS4u1P.rgZVW_1cL3=
     lDPCxFL.QhOrL'''
    
        *** Found: ('GqIqeS4u1P.rgZVW_1cL3', 'lDPCxFL.QhOrL') ***
    ==> Processing:
    '''lDPCxFL.Sa
    =	 	GqIqeS4u1P.rgZVW_1cL3'''
    
        *** Found: ('lDPCxFL.Sa', 'GqIqeS4u1P.rgZVW_1cL3') ***
    ==> Processing:
    '''K5Rgl.ZudReQci8	 =
    
    dOVpvjomA0.ZQUMT6'''
    
        *** Found: ('K5Rgl.ZudReQci8', 'dOVpvjomA0.ZQUMT6') ***
    ==> Processing:
    '''B11BzsMd.X=XYXPXuog.JLbH'''
    
        *** Found: ('B11BzsMd.X', 'XYXPXuog.JLbH') ***
    
    (Passed part 2 of 2.)
    

**Exercise 4** (2 points). Given an SQL query in the restricted form described above, write a function that extracts all of the join conditions from the `WHERE` clause. Name this fuction, **`extract_join_conds(q)`**, where `q` is the query string. It should return a list of pairs, where each pair `(a, b)` is the name of the left- and right-hand sides in one of these conditions.

For example, suppose:

```python
    q == """SELECT * FROM OneTable, AnotherTable, YetAThird
              WHERE OneTable.ColA = AnotherTable.ColB AND AnotherTable.ColB2=YetAThird.ColC"""
```

Notice that the `WHERE` clause contains two conditions: `OneTable.ColA = AnotherTable.ColB` and `AnotherTable.ColB2=YetAThird.ColC`. Therefore, your function should return a list of two pairs,
as follows:

```python
    extract_join_conds(q) == [("OneTable.ColA", "AnotherTable.ColB"),
                              ("AnotherTable.ColB2", "YetAThird.ColC")]
```


```python
def extract_join_conds(q):
    conds = re.split(r'(?i)and', split_simple_join(q)[1])
    return [extract_eqcols(cond) for cond in conds]
    
print("==> Query:\n\t'{}'\n".format(q_demo))
print("==> Results:\n{}".format(extract_join_conds(q_demo)))
```

    ==> Query:
    	'SELECT * FROM OneTable, AnotherTable, YetAThird
                  WHERE OneTable.ColA = AnotherTable.ColB AND AnotherTable.ColB2=YetAThird.ColC'
    
    ==> Results:
    [('OneTable.ColA', 'AnotherTable.ColB'), ('AnotherTable.ColB2', 'YetAThird.ColC')]
    


```python
# Test cell: `extract_join_conds_test`

def test_extract_join_conds_1():
    tables, cond_set = rand_query_ans(5, 5, 0)
    qstmt, _, _ = form_query_str(tables, cond_set, 0)
    qstmt = re.sub("[\n\t]", " ", qstmt)
    print("=== {} ===\n".format(qstmt))
    print("  True solution: {}\n".format(cond_set))
    your_conds = extract_join_conds(qstmt)
    print("  Your solution: {}\n".format(your_conds))
    assert set(your_conds) == cond_set, "*** Mismatch? ***"
    
for _ in range(10):
    test_extract_join_conds_1()
    
print("\n(Passed!)")
```

    ===  seLecT * FRom IDCgrzO9p,Wnu,TEfTnJ7n,C7rcOTAU5,E wHeRE IDCgrzO9p.KkoLr =    Wnu.R AND TEfTnJ7n.BwAnUI5F=Wnu.H5febnl0j AND Wnu.H5febnl0j  =TEfTnJ7n.dV8N AND TEfTnJ7n.qX1hDi_Ofl  = E.kIz ===
    
      True solution: {('IDCgrzO9p.KkoLr', 'Wnu.R'), ('TEfTnJ7n.BwAnUI5F', 'Wnu.H5febnl0j'), ('Wnu.H5febnl0j', 'TEfTnJ7n.dV8N'), ('TEfTnJ7n.qX1hDi_Ofl', 'E.kIz')}
    
      Your solution: [('IDCgrzO9p.KkoLr', 'Wnu.R'), ('TEfTnJ7n.BwAnUI5F', 'Wnu.H5febnl0j'), ('Wnu.H5febnl0j', 'TEfTnJ7n.dV8N'), ('TEfTnJ7n.qX1hDi_Ofl', 'E.kIz')]
    
    ===  SElect * frOM BcYMHnjLF,ogYrU3k3C4,ga,j,_6C65gGsn WheRe ga.b4   =BcYMHnjLF.yPts_B9 AND j.J_4igFkFT  = BcYMHnjLF.IpSm ===
    
      True solution: {('ga.b4', 'BcYMHnjLF.yPts_B9'), ('j.J_4igFkFT', 'BcYMHnjLF.IpSm')}
    
      Your solution: [('ga.b4', 'BcYMHnjLF.yPts_B9'), ('j.J_4igFkFT', 'BcYMHnjLF.IpSm')]
    
    ===  selECT * fRoM yJ0j,vjg1V,PLA,J4wOjBxPY,AM whERe AM.uf10MCPO0   =  J4wOjBxPY.k9GIgY ===
    
      True solution: {('AM.uf10MCPO0', 'J4wOjBxPY.k9GIgY')}
    
      Your solution: [('AM.uf10MCPO0', 'J4wOjBxPY.k9GIgY')]
    
    ===  SelECT * FrOm WXpvEgu0,cJ_KSU,A0xBgju WhERe A0xBgju.GOFvV    =cJ_KSU.Uvw7RIqvZw AND cJ_KSU.Uvw7RIqvZw    =    WXpvEgu0.hFgJbj1Bs ===
    
      True solution: {('A0xBgju.GOFvV', 'cJ_KSU.Uvw7RIqvZw'), ('cJ_KSU.Uvw7RIqvZw', 'WXpvEgu0.hFgJbj1Bs')}
    
      Your solution: [('A0xBgju.GOFvV', 'cJ_KSU.Uvw7RIqvZw'), ('cJ_KSU.Uvw7RIqvZw', 'WXpvEgu0.hFgJbj1Bs')]
    
    ===  selecT * From F,ojivK WheRe F.m07KJPyzS    =   ojivK.ctqaDGkV ===
    
      True solution: {('F.m07KJPyzS', 'ojivK.ctqaDGkV')}
    
      Your solution: [('F.m07KJPyzS', 'ojivK.ctqaDGkV')]
    
    ===  sELeCT * fRoM K6EFmts,Ky,R7Jz5jWbx wherE R7Jz5jWbx.Ckg = K6EFmts.TZU9EGXL AND Ky.T5JqQl   =R7Jz5jWbx.hRzj ===
    
      True solution: {('R7Jz5jWbx.Ckg', 'K6EFmts.TZU9EGXL'), ('Ky.T5JqQl', 'R7Jz5jWbx.hRzj')}
    
      Your solution: [('R7Jz5jWbx.Ckg', 'K6EFmts.TZU9EGXL'), ('Ky.T5JqQl', 'R7Jz5jWbx.hRzj')]
    
    ===  SeLecT * fRoM rQqCf0,mj9YuFuow,MSrLHvm,gBFDaMtpfq,mtIOu wherE mj9YuFuow.Y7t  =  MSrLHvm.Dh AND MSrLHvm.ju7_TzoDT    = rQqCf0.o ===
    
      True solution: {('mj9YuFuow.Y7t', 'MSrLHvm.Dh'), ('MSrLHvm.ju7_TzoDT', 'rQqCf0.o')}
    
      Your solution: [('mj9YuFuow.Y7t', 'MSrLHvm.Dh'), ('MSrLHvm.ju7_TzoDT', 'rQqCf0.o')]
    
    ===  sELEct * FRom J8hlx,_Ec7v,I whERE I.DFKzAizDdg  = _Ec7v.FCHp6ZdY ===
    
      True solution: {('I.DFKzAizDdg', '_Ec7v.FCHp6ZdY')}
    
      Your solution: [('I.DFKzAizDdg', '_Ec7v.FCHp6ZdY')]
    
    ===  sELEct * from WvZvWUU,wWE,FB_pXY WHeRe wWE.fd8fBlpMYV =  WvZvWUU.UxkgNbd ===
    
      True solution: {('wWE.fd8fBlpMYV', 'WvZvWUU.UxkgNbd')}
    
      Your solution: [('wWE.fd8fBlpMYV', 'WvZvWUU.UxkgNbd')]
    
    ===  sELEct * From lsO3O4g,N0,Bo wheRE N0.rBpLJVff=   lsO3O4g.a5 AND lsO3O4g.YQok4zsx=N0.u ===
    
      True solution: {('N0.rBpLJVff', 'lsO3O4g.a5'), ('lsO3O4g.YQok4zsx', 'N0.u')}
    
      Your solution: [('N0.rBpLJVff', 'lsO3O4g.a5'), ('lsO3O4g.YQok4zsx', 'N0.u')]
    
    
    (Passed!)
    

**Fin!** This marks the end of this problem. Don't forget to submit it to get credit.
