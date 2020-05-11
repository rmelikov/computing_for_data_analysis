#!/usr/bin/env python3
"""
python3 prepend-cells.py <cells.ipynb> <notebook0.ipynb> [<notebook1.ipynb> ...]
"""

import sys
import nbformat

def msg(s, target=sys.stderr):
    target.write(s)

def open_notebook(filename, as_version=4):
    with open(filename, "rt") as fp:
        nb = nbformat.read(fp, as_version)
    return nb

def save_notebook(nb, filename):
    with open(filename, "wt") as fp:
        nbformat.write(nb, fp)

def duplicate_notebook(source_filename, target_filename):
    save_notebook(open_notebook(source_filename), target_filename)

if len(sys.argv) < 2:
    msg(__doc__)
    sys.exit(-1)

cells_filename = sys.argv[1]
msg(f"Reading cells to prepend: '{cells_filename}' ...\n")
cells = open_notebook(cells_filename)

for nb_filename in sys.argv[2:]:
    nb_filename_orig = nb_filename + '.orig'
    msg(f"Prepending to: '{nb_filename}' (saving original as: '{nb_filename_orig}')\n")
    duplicate_notebook(nb_filename, nb_filename_orig)
    try:
        nb = open_notebook(nb_filename_orig)
        nb['cells'] = cells['cells'] + nb['cells']
        save_notebook(nb, nb_filename)
    except:
        msg(f"\n==> Due to an unexpected exception, restoring notebook to its original state!\n\n")
        duplicate_notebook(nb_filename_orig, nb_filename)
        raise

# eof
