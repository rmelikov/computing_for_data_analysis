#!/usr/bin/env python3

import os
from hashlib import sha256

def get_path(basename):
    from os.path import isdir
    return f'{basename}' #if isdir('.voc') else basename

def open_file(basename, *args):
    return open(get_path(basename), *args)

def make_hash(doc):
    return sha256(doc.encode()).hexdigest()

def check_hash(doc, key):
    return make_hash(doc) == key

def where_strings_differ(a, b):
    first_difference = None
    for k, (a_k, b_k) in enumerate(zip(a, b)):
        if a_k != b_k:
            first_difference = k
            break
    if first_difference is None: # strings match up to min(len(a), len(b)), so check for extra chars
        if len(a) < len(b):
            print(f'Strings are the same except the second one has these extra letters: "{b[len(a):]}"\n')
            first_difference = len(a)
        elif len(b) < len(a):
            print(f'Strings are the same except the first one has these extra letters: "{a[len(b):]}"\n')
            first_difference = len(b)
    if first_difference is None:
        print("==> Strings appear to be identical.")
    else:
        print(f"==> Strings differ starting at position {first_difference}:")
        start_snip = max(0, first_difference-5)
        snip_prefix = "..." if start_snip > 0 else ''
        snip_a = snip_prefix + a[start_snip:first_difference+1]
        snip_b = snip_prefix + b[start_snip:first_difference+1]
        print(f"    {snip_a} <-- position {first_difference}")
        print(f"vs.")
        print(f"    {snip_b} <--")
    return first_difference

def canonicalize_tibble(X, remove_index):
    var_names = sorted(X.columns)
    Y = X[var_names].copy()
    Y.sort_values(by=var_names, inplace=True)
    Y.reset_index(drop=remove_index, inplace=True)
    return Y

def tibbles_left_matches_right(A, B, hash_A=False, verbose=False):
    A_canonical = canonicalize_tibble(A if not hash_A else A.applymap(make_hash), not verbose)
    B_canonical = canonicalize_tibble(B, not verbose)
    if not verbose:
        cmp = A_canonical.eq(B_canonical)
        matches = cmp.all().all()
    else:
        if len(A) != len(B):
            print(f"Your data frame has {len(A)} rows, whereas we expected {len(B)}.")
        on_keys = list(A_canonical.columns.difference(['index']))
        cmp = A_canonical.merge(B_canonical, on=on_keys, how='outer', suffixes=['', '_B'])
        extra_or_wrong = cmp['index_B'].isna()
        any_extra_or_wrong = extra_or_wrong.any()
        if any_extra_or_wrong:
            print("The following rows of your solution may be problematic"
                  " (e.g., are extraneous, have incorrect values).\n"
                  "Only showing up to the first ten such rows.")
            display(A.loc[cmp['index'][extra_or_wrong]].head(10))
        missing = cmp['index'].isna()
        any_missing = missing.any()
        if any_missing:
            print("There may be missing rows{}.".format(", too" if any_extra_or_wrong else ''))
        matches = not (any_extra_or_wrong or any_missing)
    return matches

def tibbles_are_equivalent(A, B):
    return tibbles_left_matches_right(A, B)

def assert_tibbles_are_equivalent(A, B):
    from pandas.testing import assert_frame_equal
    A_c = canonicalize_tibble(A, True)
    B_c = canonicalize_tibble(B, True)
    if len(A_c) == 0 and len(B_c) == 0: return
    assert_frame_equal(A_c, B_c, check_index_type=False, check_less_precise=True)

def pandas_df_to_markdown_table(df, index=False):
    """Adapted from:
    https://stackoverflow.com/questions/33181846/programmatically-convert-pandas-dataframe-to-markdown-table
    """
    from IPython.display import Markdown, display
    from pandas import DataFrame, concat
    fmt = ['---' for i in range(len(df.columns))]
    df_fmt = DataFrame([fmt], columns=df.columns)
    df_formatted = concat([df_fmt, df])
    csv_rep = df_formatted.to_csv(sep="|", index=index)
    if index:
        csv_rep = '(index)' + csv_rep # + csv_rep.replace('\n0|', '\n---|')
    display(Markdown(csv_rep))
    return csv_rep

def hidden_cell_template_msg():
    return "This test cell will be replaced with one or more hidden tests.\n" \
           "You will only know the result after submitting to the autograder.\n" \
           "If the autograder times out, then either your solution is highly\n" \
           "inefficient or contains a bug (e.g., an infinite loop). To see\n" \
           "the result of the autograder run, inspect the grading report.\n"

def ex0_random_date():
    from datetime import datetime
    from random import randint
    year = 2020
    month = randint(1, 12)
    day = randint(1, 28) if month == 2 else randint(1, 30 + int(month in [1, 3, 5, 7, 8, 10, 12]))
    hour = randint(0, 23)
    minute = randint(0, 59)
    second = randint(0, 59)
    microsecond = randint(0, 999999)
    try:
        t = datetime(year, month, day, hour, minute, second, microsecond)
    except:
        print(year, month, day, hour, minute, second, microsecond)
        assert False
    return t

def ex0_random_word(k_max):
    from random import choice, randint
    k = randint(1, k_max)
    C = 'abcdefghijklmnopqrstuvwxyz'
    C += C.upper()
    return ''.join([choice(C) for _ in range(k)])

def ex0_random_sym():
    from random import choice
    return choice("`~@#$%^&*()-_+={[}]|\\:;\"'<,>/")

def ex0_random_punct():
    from random import choice
    return choice(".?!")

def ex0_random_space(k_max):
    from random import randint
    k = randint(1, k_max)
    return ' ' * k

def ex0_random_sent(k_max, other_words=None):
    from random import randint, random, choice
    assert other_words is None or isinstance(other_words, list)
    num_words = randint(1, k_max)
    sent = ''
    words = []
    for i in range(num_words):
        w = ex0_random_space(3) if i > 0 else ''
        w += '' if random() > 0.1 else ex0_random_sym()
        w_new = ex0_random_word(7)
        w_new_soln = w_new.lower()
        if (other_words is not None) and ((random() <= 0.33) or (w_new_soln in other_words)):
            w += choice(other_words)
        else:
            words.append(w_new_soln)
            w += w_new
        w += '' if random() > 0.1 else ex0_random_sym()
        sent += w
    return sent, words

def ex0_random_line(k_max):
    from random import randint, random
    line = ''
    sents = []
    num_sents = randint(0, 7)
    for k in range(num_sents):
        sent, _ = ex0_random_sent(5)
        sents.append(sent)
        line += '' if random() >= 0.2 else ex0_random_space(3)
        line += sent
        assert not line[-1].isspace(), f"line={repr(line)} && sent={repr(sent)}"
        line += ex0_random_punct()
        line += '' if random() <= 0.2 else ex0_random_space(3)
        if k == num_sents-1 and random() <= 0.1 and line[-1] in '.?!': # Randomly delete last punctation mark
            line = line[:-1]
    return line, sents

def ex0_check_one(fun, doc=None, sents_soln=None, verbose=False):
    from random import randint, random
    if doc is None:
        doc = []
        sents_soln = []
        num_lines = randint(1, 7)
        for k in range(num_lines):
            if random() <= 0.1: # blank line
                line_k, sents_k = ex0_random_space(3), []
            else:
                line_k, sents_k = ex0_random_line(6)
            doc.append(line_k)
            sents_soln += sents_k
    else:
        assert sents_soln is not None
        
    try:
        your_sents = fun(doc)
        assert your_sents == sents_soln, "*** ERROR: Did not generate the expected sentences ***"
    except:
        print("=== Input ===")
        print("[" + ",\n".join([repr(line) for line in doc]) + "]")
        print("\n=== Expected output ===")
        print("[" + ",\n".join([repr(s) for s in sents_soln]) + "]")
        print("\n=== Your output ===")
        print("[" + ",\n".join([repr(s) for s in your_sents]) + "]")
        raise
    
    return your_sents

# Load stopwords
with open(get_path('english_stopwords.txt'), 'rt') as fp:
    STOPWORDS = set(fp.read().split())

def ex1_gen():
    from problem_utils import ex0_random_sent, STOPWORDS
    from random import randint
    num_sents = randint(1, 8)
    sents, bags = [], []
    other_words = list(STOPWORDS)
    for k in range(num_sents):
        sent, words = ex0_random_sent(6, other_words)
        sents.append(sent)
        bags.append(set(words))
    return sents, bags

def ex1_check_one(fun):
    sents, bags_soln = ex1_gen()
    try:
        your_bags = fun(sents)
        assert len(your_bags) == len(bags_soln), \
               "*** ERROR: You didn't produce enough bags ***"
        assert all([u == s for u, s in zip(your_bags, bags_soln)]), \
               "*** ERROR: You didn't produce the expected bags ***"
    except:
        print("=== Input ===")
        print("[" + ",\n".join([repr(s) for s in sents]) + "]")
        print("\n=== Expected output ===")
        print("[" + ",\n".join([repr(b) for b in bags_soln]) + "]")
        print("\n=== Your output ===")
        print("[" + ",\n".join([repr(b) for b in your_bags]) + "]")
        raise

    return your_bags

HIPSTER_IPSUM = {'activated',
 'aesthetic',
 'affogato',
 'art',
 'asymmetrical',
 'austin',
 'brooklyn',
 'brunch',
 'bun',
 'charcoal',
 'chartreuse',
 'cornhole',
 'denim',
 'diy',
 'drinking',
 'fingerstache',
 'food',
 'gentrify',
 'glossier',
 'goth',
 'health',
 'heirloom',
 'humblebrag',
 'iphone',
 'kombucha',
 'letterpress',
 'listicle',
 'literally',
 'liveedge',
 'locavore',
 'lofi',
 'man',
 'migas',
 'mixtape',
 'mumblecore',
 'paleo',
 'party',
 'pickled',
 'poke',
 'popup',
 'prism',
 'pug',
 'raclette',
 'ramps',
 'raw',
 'retro',
 'scenester',
 'selfies',
 'slowcarb',
 'squid',
 'truck',
 'truffaut',
 'vape',
 'venmo',
 'vinegar'}

def ex2_gen(max_bags=5, max_words_per_bag=4):
    from random import randint, sample
    from math import log
    num_bags = randint(1, max_bags)
    bags = []
    w2id = {}
    rows, cols = [], []
    N = {}
    for j in range(num_bags):
        num_words = randint(1, max_words_per_bag)
        b = set(sample(HIPSTER_IPSUM, num_words))
        bags.append(b)
        for w in b:
            if w not in w2id:
                w2id[w] = len(w2id)
            i = w2id[w]
            rows.append(i)
            cols.append(j)
            N[i] = 1 if i not in N else N[i]+1
    vals = [1.0 / log((num_bags+1) / N[i]) for i in rows]
    return bags, w2id, rows, cols, vals

def ex2_cmp(R, C, V, R_soln, C_soln, V_soln):
    from math import isclose
    assert isinstance(R, list), "*** ERROR: Returned `rows` is not a list ***"
    assert isinstance(C, list), "*** ERROR: Returned `cols` is not a list ***"
    assert isinstance(V, list), "*** ERROR: Returned `vals` is not a list ***"
    assert len(R) == len(C) == len(V), "*** ERROR: Returned lists do not have the samee length ***"
    assert len(R) == len(R_soln), f"*** ERROR: Expected {len(R_soln)} elements in each list, got {len(R)} instead? ***"
    RCV_sorted = sorted(zip(R, C, V))
    Soln_sorted = sorted(zip(R_soln, C_soln, V_soln))
    for x, y in zip(RCV_sorted, Soln_sorted):
        assert x[0] == y[0] and x[1] == y[1], f"*** ERROR: Mismatched coordinates: {(x[0], x[1])} != {(y[0], y[1])} ***"
        assert isclose(x[2], y[2]), f"*** ERROR: At ({x[0]}, {x[1]}), your value is {x[2]} instead of {y[2]} ***"

def ex2_check_one(fun):
    bags, w2id, rows_soln, cols_soln, vals_soln = ex2_gen()
    try:
        rows, cols, vals = fun(bags, w2id)
        ex2_cmp(rows, cols, vals, rows_soln, cols_soln, vals_soln)
    except:
        print("\n=== Test Inputs ===")
        print(f"* bags == {bags}")
        print(f"* word_to_id == {w2id}")
        print("\n=== Expected outputs ===")
        print(f"* rows == {rows_soln}")
        print(f"* cols == {cols_soln}")
        print(f"* vals == {vals_soln}")
        print("\n=== Your outputs ===")
        print(f"* rows == {rows}")
        print(f"* cols == {cols}")
        print(f"* vals == {vals}")
        raise
    return bags, w2id, rows_soln, cols_soln, vals_soln

def ex3_gen__(k):
    from numpy import zeros
    from numpy.random import random, permutation
    x = random(k).cumsum()[::-1]
    x /= x[0]
    i = permutation(k)
    y = zeros(k)
    y[i] = x
    return y, i

def ex3_gen(max_words=5, max_sents=10):
    from random import randint, shuffle
    from numpy.random import random, permutation
    u0, i = ex3_gen__(max_words)
    v0, j = ex3_gen__(max_sents)
    return u0, v0, i, j

def ex3_and_4_cmp(r, r_soln):
    from numpy import ndarray, issubdtype, integer
    assert isinstance(r, ndarray), "*** ERROR: Your function did not return a Numpy array ***"
    assert issubdtype(r.dtype, integer), "*** ERROR: You did not return a Numpy array with an integer dtype ***"
    assert (r == r_soln).all(), "*** ERROR: You did not return a correct ranking ***"

def ex3_check_one(fun):
    u0, v0, r_soln, _ = ex3_gen()
    try:
        r = fun(u0, v0)
        ex3_and_4_cmp(r, r_soln)
    except:
        print("\n=== Inputs ===")
        print(f"* u0 == {u0}")
        print(f"* v0 == {v0}")
        print("\n=== Expected output ===")
        print(f"* ranking == {r_soln}")
        print("\n=== Your output ===")
        print(f"* ranking == {r}")
        raise
    return u0, v0, r_soln
    
def ex4_check_one(fun):
    u0, v0, _, r_soln = ex3_gen()
    try:
        r = fun(u0, v0)
        ex3_and_4_cmp(r, r_soln)
    except:
        print("\n=== Inputs ===")
        print(f"* u0 == {u0}")
        print(f"* v0 == {v0}")
        print("\n=== Expected output ===")
        print(f"* ranking == {r_soln}")
        print("\n=== Your output ===")
        print(f"* ranking == {r}")
        raise
    return u0, v0, r_soln
    
# eof
