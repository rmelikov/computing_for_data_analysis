#!/usr/bin/env python3
"""
python raw2png.py <fmt> <rows> <cols> <rawfile1> [<rawfile2> ...]
"""

import sys
import numpy as np

from PIL import Image
from io import BytesIO

def im2gnp(image):
    """Converts a PIL image into an image stored as a 2-D Numpy array in grayscale."""
    return np.array(image.convert ('L'))

def gnp2im(image_np):
    """Converts an image stored as a 2-D grayscale Numpy array into a PIL image."""
    return Image.fromarray(image_np.astype(np.uint8), mode='L')

def status(msg, newline=True):
    sys.stderr.write(msg + ("\n" if newline else ""))

#============================================================

if len(sys.argv) < 5:
    status(__doc__)
    sys.exit(1)

fmt = sys.argv[1]
rows = int(sys.argv[2])
cols = int(sys.argv[3])
raw_files = sys.argv[4:]

status("* Converting files to {} x {} images in .{} format...".format(rows, cols, fmt))

for raw_file in raw_files:
    # Read raw bytes as a Numpy array
    with open(raw_file, 'rb') as raw_fp:
        raw_values = [int(b) for b in raw_fp.read()]
    raw_gnp = np.array(raw_values, dtype='uint8')

    if len(raw_gnp) != rows * cols:
        status("SKIPPING '{}': Number of raw bytes ({}) does not match target dimensions ({} x {}).".format(raw_file, len(raw_gnp), rows, cols))
        continue
    img_gnp = raw_gnp.reshape(rows, cols)

    # Convert to image object
    img_out = gnp2im(img_gnp)
    out_bytes = BytesIO()
    img_out.save(out_bytes, format=fmt)

    # Write to output file
    out_file = raw_file + '.' + fmt
    with open(out_file, 'wb') as out_fp:
        out_fp.write(out_bytes.getvalue())

# eof
