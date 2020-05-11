#!/usr/bin/env python3

from hashlib import sha256

def make_hash(doc):
    if not isinstance(doc, str):
        doc = str(doc)
    return sha256(doc.encode()).hexdigest()

def check_hash(doc, key):
    return make_hash(doc) == key

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

# eof
