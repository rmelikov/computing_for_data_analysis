#!/usr/bin/env python3

# Pandas-based
def canonicalize_tibble(X):
    var_names = sorted(X.columns)
    Y = X[var_names].copy()
    Y.sort_values(by=var_names, inplace=True)
    Y.reset_index(drop=True, inplace=True)
    return Y

# Pandas-based
def tibbles_are_equivalent (A, B):
    A_canonical = canonicalize_tibble(A)
    B_canonical = canonicalize_tibble(B)
    cmp = A_canonical.eq(B_canonical)
    return cmp.all().all()

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

def cspy(A, ax=None, cmap=None, cbar=True, s=20):
    """"
    Adapted from: https://gist.github.com/lukeolson/9710288

    Parameters
    ----------
    A : coo matrix
    ax : axis (default: None ==> use gca
    """
    from matplotlib.pyplot import gca, colorbar

    if ax is None:
        ax = gca()

    m, n = A.shape
    p = ax.scatter(A.row, A.col, c=A.data, s=s, marker='s',
                   edgecolors='none', clip_on=False,
                   cmap=cmap)
    if cbar:
        colorbar(p)
    ax.axis([-0.5, A.shape[1]-0.5, -0.5, A.shape[0]-0.5])
    ax.set_aspect('equal')
    ax.invert_yaxis()
 
# eof
