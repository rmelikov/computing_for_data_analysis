#!/usr/bin/env python3

import os
from hashlib import sha256

RESPONSE = 'click'
PREDICTORS_NUMERICAL = ['x_pixels', 'y_pixels']
PREDICTORS_CATEGORICAL = [f'dev{k}' for k in [0, 1, 2, 4, 5]] \
                      + [f'c{k}' for k in [1001, 1002, 1005, 1007, 1008, 1010, 1012]]
PREDICTORS = PREDICTORS_NUMERICAL + PREDICTORS_CATEGORICAL

def get_path(basename):
    from os.path import isdir
    return f'{basename}' #if isdir('.voc') else basename

def load_click_through_data(filepath, verbose=True, adversary=None):
    from problem_utils import PREDICTORS_NUMERICAL, PREDICTORS_CATEGORICAL
    from pandas import read_csv, get_dummies
    from numpy.random import seed

    if verbose: print("Reading", filepath, "...")
    df = read_csv(filepath)
    if verbose: print("Done! Cleaning...")
    df = get_dummies(df, columns=['device_type', 'C1'])
    df = df.rename(columns={**{f'device_type_{k}': f'dev{k}' for k in [0, 1, 2, 4, 5]},
                            **{f'C1_{k}': f'c{k}' for k in [1001, 1002, 1005, 1007, 1008, 1010, 1012]},
                            **{'C15': 'x_pixels', 'C16': 'y_pixels'}})

    # Ensure all columns exist:
    for c in PREDICTORS_CATEGORICAL:
        if c not in df.columns:
            df[c] = 0

    # Make categorical columns explicitly so
    for c in [RESPONSE] + PREDICTORS_CATEGORICAL:
        df[c] = df[c].astype('category')

    # Randomly remove values, for fun and profit:
    if adversary is not None:
        seed(adversary)
        vandalize_inplace(df, 10)

    # Subset and return!
    df = df[[RESPONSE] + PREDICTORS]
    if verbose: print("Done cleaning!")
    return df

def vandalize_inplace(df, max_victims):
    from numpy.random import randint, choice
    from numpy import nan
    for c in PREDICTORS:
        num_victims = randint(0, max_victims, size=1)[0]
        victims = choice(len(df[c]), size=num_victims, replace=False)
        df.loc[victims, c] = nan

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
    assert_frame_equal(A_c, B_c, check_index_type=False, check_dtype=False, check_less_precise=True)

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

def ex0_coshuffle(x, y):
    from random import shuffle
    if y is None:
        return x
    xy = [(a, b) for a, b in zip(x, y)]
    shuffle(xy)
    xp = [a for a, _ in xy]
    yp = [b for _, b in xy]
    return xp, yp

def ex0_random_bin(n, vandalize=False):
    from numpy import nan
    from random import randint
    n0 = randint(0, n)
    if vandalize:
        n1 = randint(0, n-n0)
        if n0 == n1 == 0: # Ensure at least one useful value
            n1 = 1
        nv = n - n0 - n1
    else:
        nv = 0
        n1 = n - n0
    assert n0 > 0 or n1 > 0
    mode = 0 if n0 >= n1 else 1
    values_clean = [0]*n0 + [1]*n1
    if vandalize:
        values_clean += [mode]*nv
        values_dirty = values_clean.copy()
        if nv > 0:
            values_dirty[-nv:] = [nan]*nv
    else: # nv == 0
        values_dirty = None
    return ex0_coshuffle(values_clean, values_dirty)

def ex0_random_int(n, lo=0, hi=512, vandalize=False):
    from numpy import nan
    from random import randint
    from statistics import mean
    if vandalize:
        m = randint(1, n)
        mv = n - m
    else:
        m = n
        mv = 0
    assert (m > 0) and (m+mv == n)
    values_clean = [randint(lo, hi) for _ in range(m)]
    if vandalize:
        values_clean += [mean(values_clean)]*mv
        values_dirty = values_clean.copy()
        if mv > 0:
            values_dirty[-mv:] = [nan]*mv
    else:
        values_dirty = None
    return ex0_coshuffle(values_clean, values_dirty)

def ex0_gen_soln(max_rows):
    from numpy.random import randint
    from pandas import Series, concat
    def append_col_inplace(cols, v, c, dtype='O'):
        cols.append(Series(v, name=c, dtype=dtype))
    n = randint(1, max_rows, size=1)[0]
    cols_clean = [Series(ex0_random_bin(n), name=RESPONSE, dtype=int)]
    cols_dirty = [cols_clean[-1]]
    for c in PREDICTORS_NUMERICAL:
        values_clean, values_dirty = ex0_random_int(n, vandalize=True)
        append_col_inplace(cols_clean, values_clean, c)
        append_col_inplace(cols_dirty, values_dirty, c)
    for c in PREDICTORS_CATEGORICAL:
        values_clean, values_dirty = ex0_random_bin(n, vandalize=True)
        append_col_inplace(cols_clean, values_clean, c, dtype=int)
        append_col_inplace(cols_dirty, values_dirty, c)
    df_clean = concat(cols_clean, axis=1)
    df_dirty = concat(cols_dirty, axis=1)
    return df_clean, df_dirty

def ex1_random_bin(n):
    from random import randint
    from numpy.random import permutation
    from numpy import zeros
    assert n >= 2
    n0 = randint(1, n-1)
    n1 = n - n0
    assert n0 + n1 == n
    positions = permutation(n)
    y = zeros(n, dtype=int)
    k0 = positions[:n0].copy()
    k0.sort()
    k1 = positions[-n1:].copy()
    k1.sort()
    y[k0] = 0
    y[k1] = 1
    if len(k0) <= len(k1):
        return y, k0, k1
    return y, k1, k0

# eof
