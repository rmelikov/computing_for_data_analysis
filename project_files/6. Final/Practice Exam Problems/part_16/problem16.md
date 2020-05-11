# Problem 4: Video Compression

> **Note.** This notebook is set to use Python 3.6 on Vocareum because of a particular library that it invokes. If you are solving the problem in your local environment, you may need to adapt that prior to submission. You will only earn points if the autograder accepts your submission, so keep that in mind and plan accordingly.

This problem revisits image compression but asks how to extend to video. Remember how we used the idea of a low rank approximation of a matrix to compress an image. In this notebook, we will extend the same idea to a video. A video for our purpose is just an ordered sequence of images/matrices. To compress a video, we need to compress each of the frames by finding its lower rank approximation using the SVD. Apart from this, you will be implementing a second way (Average Pooling) to compress an image/video.

We will first read in a flattened 2D array, in which each row is a frame in the video. You will be required to reshape this flattened 2D array to a 3D array for the video. After that, you will work on compressing this video by two alternate methods -
1. SVD (lower rank approximation) 
2. Average pooling. 

Note: We will be using the term 3D array/tensor interchangeably.


```python
# Importing dependencies
import numpy as np
import matplotlib.pyplot as plt
%matplotlib notebook  
import warnings; warnings.simplefilter('ignore')
from IPython.display import HTML
import imageio
import warnings
from scipy.sparse import csr_matrix

%env CATALYST_LOG_LEVEL = 15 
```

    env: CATALYST_LOG_LEVEL=15
    

The cell below reads in the flattened version of the matrix. Run the cell below to load the matrix and read the following statements carefully.


```python
v = np.load('video.npz')
M = v["arr_0"]
M = M.astype(np.int32)
row, col = M.shape
print("The matrix M has {} rows and {} columns".format(row, col))
```

    The matrix M has 61 rows and 921600 columns
    

Each of the 61 rows in `M` is a frame in the video. Each row has to be converted to a 2D array to visualize it as a frame. The flattening uses a C-language order (i.e., row major) and has to be restored as a 2D array. Each frame is originally 720 x 1280 in shape. The sequence of rows in `M` is the same as the sequence of frames in the video.

Your first task is to extract each row from this flattened array `M` and store it as a 2D array of size 720 x 1280. The 2D arrays are then stacked together to form a tensor of size 61 x 720 x 1280.

**Exercise 0** (1 point). Write some code that extracts the video from the flattened matrix and stores it in the variable **`vid`**. Your output should be a tensor of shape 61 x 720 x 1280. Remember that the original rows in the matrix `M` were saved in a C-type format.


```python
# This is not a function but a code snippet.
# Your output of size 61 x 720 x 1280 should be stored in the variable vid. 
vid = None
### BEGIN SOLUTION
movie = []
for r in range(row):
    f = M[r]
    frame = np.reshape(f, (720, 1280), "C")
    movie.append(frame)
vid = np.array(movie)
### END SOLUTION
```


```python
# Test cell - reshape_test
f, r, c = vid.shape
assert (f, r, c) == (61, 720, 1280), "The shapes are incorrect"
vid_flattened = np.reshape(vid, (f, r*c), "C")
err = M - vid_flattened
err_norm = np.linalg.norm(err, 'fro')
assert err_norm <= 1e-8, "Incorrect values"
del err, M, vid_flattened
print("Passed!")
vidx = vid.astype(np.uint8)
```

    Passed!
    

Running the cell below displays content of the original video. You can comment this out if this cell increases your run time. The code in this cell helps you see the video in the tensor.


```python
# The code in this cell shows the original video. You can comment out this part while 
# submitting to the autograder as it may increase your total running time.
warnings.simplefilter('ignore')
imageio.mimwrite('original.mp4', vidx, fps=30)
HTML("""<video width="480" height="360" controls>   <source src="{0}"> </video> """.format('./original.mp4'))
```




<video width="480" height="360" controls>   <source src="./original.mp4"> </video> 



We hope that the video above helped you realize that a sequence of 2D arrays, each of which is an image, can represent a video. In the next exercise, you will apply SVD to each of the constituent frames of the image to compress the complete video.

## SVD review
For a quick review of SVD and how this is used to compress images/videos, refer to [this](http://theory.stanford.edu/~tim/s15/l/l9.pdf) discussion. 

**Exercise 1** (2 points). Complete the function, **`svd_approx(m, r)`**, where **`m`** is a 2D Numpy array and **`r`** is a scalar rank. This function should return a new matrix, which is the same size as **`m`**, that stores the best rank **`r`** approximation of **`m`** (as measured by the Frobenius norm).


```python
def svd_approx(m, r):
    ### BEGIN SOLUTION
    row, col = m.shape
    assert r <= min(m.shape)
    u, s, v =  np.linalg.svd(m, full_matrices=False)
    u_r = u[:,:r]
    s_r = s[:r]
    v_r = v[:r,:]
    m_r = u_r @ np.diag(s_r) @ v_r
    return m_r
    ### END SOLUTION
```


```python
# Test cell - svd_test
for _ in range(5):
    x = np.random.rand(8, 6)
    l = np.random.randint(1,6)
    x_l = svd_approx(x, l)
    u, s, v = np.linalg.svd(x, False)
    err = sum(s[l:]**2)**.5
    err_fro = np.linalg.norm(x-x_l, 'fro')
    assert abs(err_fro - err) <= 1e-8, "Incorrect"
print("Passed!")
```

    Passed!
    

**Exercise 2** (2 points). Complete the function **`compress_vid`**(`vid`, `r`), below. This function takes two inputs, a 3D array **`vid`** (like the `vid` array above) and a scalar **`r`**. It should compute the best rank-`r` approximation of each frame in the input array and returns a new 3D array, which is of the same shape as the input `vid`, where each frame of the output is the best rank-`r` approximation of its corresponding input frame.


```python
def compress_vid(vid, r):
    ### BEGIN SOLUTION
    f, row, col = vid.shape
    cmpr = []
    for i in range(f):
        m = vid[i]
        m_r = svd_approx(m, r)
        cmpr.append(m_r)
    return np.array(cmpr)
    ### END SOLUTION
```


```python
# Test Cell - vid_compress_test
# The values have been converted to an int type because they require 8 times less memory than float values. 
# Important: 
# This test cell will be only run at the time of submission. Please use the code to display the compressed video
# in the next cell to verify your correctness. Remember you can submit as many times as you want. 

###
### AUTOGRADER TEST - DO NOT REMOVE
###
                          
```

Running the cell below displays content of the video compressed by using SVD. 

**Note**: We suggest you comment this out if this cell increases your run time.


```python
# Running this cell shows the video. 
# You can comment out this part if you wish as it can increase your total test time during submission.
# The white spots in the video are due to the uint approximation.

# Important: 
# Make sure you comment this section before submitting as the runtime for this cell exceeds the time limit. 

warnings.simplefilter('ignore')
cmpr_svd = compress_vid(vid, 15)
cmpr_svd_uint = cmpr_svd.astype(np.uint8)
imageio.mimwrite('cmpr_svd.mp4', cmpr_svd_uint, fps=30)
HTML("""<video width="480" height="360" controls>   <source src="{0}"> </video> """.format('./cmpr_svd.mp4'))

```




<video width="480" height="360" controls>   <source src="./cmpr_svd.mp4"> </video> 



# Average Pooling

Now we will see a different way of approaching the compression problem. By using SVD, we had retained the shape of the array. In this part of the notebook, you will be required to compress the image such that its shape is reduced, but the spatial pattern remains the same.

The shape of each frame is 720 x 1280. There are 720 rows and 1280 columns. Now, imagine that the matrix has been divided in non-overlapping 2x2 matrices. Hence there will be 360 x 640 smaller matrices placed across the original matrix such that they are all adjacent to each other and do not overlap. Read the following cells for a better understanding.

Imagine that the original image **`original_im`** was a 4 x 10 matrix:


```python
original_im = np.array([[1,2,2,1,4,6,0,1,2,3],
                        [1,8,7,5,4,2,4,2,4,3],
                        [3,7,1,3,0,9,8,5,3,1],
                        [2,1,2,4,6,4,3,1,2,5]])
print("The original array (image) is: \n")
print(original_im)
```

    The original array (image) is: 
    
    [[1 2 2 1 4 6 0 1 2 3]
     [1 8 7 5 4 2 4 2 4 3]
     [3 7 1 3 0 9 8 5 3 1]
     [2 1 2 4 6 4 3 1 2 5]]
    

Now suppose one divides the original array into small blocks, or **tiles**, of size 2 x 2 each. The **`mask`** array below logically encodes one such example of a "tiling." Run this cell to print it, and then we will describe its format.


```python
a = [0,0,1,1,2,2,3,3,4,4]
b = [5,5,6,6,7,7,8,8,9,9]
mask = np.array([a,a,b,b])
print("The 2 x 2 masks over the original array are: \n")
print(mask)
```

    The 2 x 2 masks over the original array are: 
    
    [[0 0 1 1 2 2 3 3 4 4]
     [0 0 1 1 2 2 3 3 4 4]
     [5 5 6 6 7 7 8 8 9 9]
     [5 5 6 6 7 7 8 8 9 9]]
    

Observe that `mask` is the same size as the original image. It effectively divides the original image into a grid of 2 x 5 = 10 tiles, numbered from 0 to 9 inclusive, where each tile is 2 x 2.

In our alternative compression scheme, we will "pool" each tile. That is, given an image, we will replace each tile of the original image with a single value. That value is the average of values within the tile.

For instance, consider the tile numbered 2 in the above mask. It corresponds to the following submatrix of the original:


```python
print("=== mask[0:2, 4:6] ===")
print(mask[0:2, 4:6])
print("=== original_im[0:2, 4:6] ===")
print(original_im[0:2, 4:6])
print("=== average in this tile ===")
print(original_im[0:2, 4:6].mean())
```

    === mask[0:2, 4:6] ===
    [[2 2]
     [2 2]]
    === original_im[0:2, 4:6] ===
    [[4 6]
     [4 2]]
    === average in this tile ===
    4.0
    

If we were to pool every tile, here is what the final result would be:


```python
print("Recall: The original image:\n{}".format(original_im))

mask_avg = np.array([[12, 15, 16, 7, 12],
                    [13, 10, 19, 17, 11]])/4
print("\nThe average pooled array is:\n{}".format(mask_avg))
```

    Recall: The original image:
    [[1 2 2 1 4 6 0 1 2 3]
     [1 8 7 5 4 2 4 2 4 3]
     [3 7 1 3 0 9 8 5 3 1]
     [2 1 2 4 6 4 3 1 2 5]]
    
    The average pooled array is:
    [[3.   3.75 4.   1.75 3.  ]
     [3.25 2.5  4.75 4.25 2.75]]
    

In Exercise 3, below, you will implement the average pooling procedure. Here are some hints on one way to approach the problem.

1. We advise against using loops to implement an element-by-element approach. Such a method is likely to take a long time and the test cells may time out.
2. One idea might be to use a linear (matrix) transformation: is there a matrix **`A`**, such that `A` times `original_im` is `mask_avg`?
3. In a linear transformation-based (or matrix multiply-based) approach, `A` might be sparse and the process might involve reshaping `original_im` and reshaping back to get `mask_avg`.

Following these hints is not the only way of solving this problem. You are free to use any approach that is correct and passes the autograder. (But if the autograder times out, you'll get no credit, so proceed with caution!)

**Exercise 3** (5 points). Complete the function **`avg_pooled`**`(vid)`, below. It takes as input the original video as a 3-D array (tensor) of shape $f \times r \times c$. (In our video example, this shape is 61 x 720 x 1280.) Your function should assume $2 \times 2$ tiles and, accordingly, return an average pooled tensor of shape $f \times \dfrac{r}{2} \times \dfrac{c}{2}$ (61 x 360 x 640). Assume that $r$ and $c$ are known to be even numbers. 

To explain it again, each frame in the original video that was of size 720 x 1280 should be average pooled to the size 360 x 640, taking averages within the 2 x 2 tiles of the original in produce the output. This procedure has to be done for all frames and the final video returned by the function should be a tensor of size 61 x 360 x 640.


```python
def avg_pooled(vid):
    ### BEGIN SOLUTION
    from scipy.sparse import csr_matrix
    nf, nr, nc = vid.shape
    assert nr%2 == 0, "number of rows must be even"
    assert nc%2 == 0, "number of columns must be even"
    nrows = int(nr*nc/4)
    step_v = nr
    step_h = nc
    rows = [i for i in range(nrows) for _ in range(4)]
    cols = []
    for j in range(0, step_v, 2):
        b = step_h*j
        for i in range(0, step_h, 2):
            r = [b+i, b+i+1, b+i+step_h, b+i+step_h+1]
            cols += r
    data  = [0.25]*step_v*step_h
    s = csr_matrix((data, (rows, cols)), shape=(nrows, step_h*step_v))
    im_arr = np.reshape(vid, (nf, nr*nc))
    im_arr = im_arr.T
    cmpr_arr = s.dot(im_arr)
    cmpr_arr = cmpr_arr.T
    cmpr_tnsr = np.reshape(cmpr_arr, (nf, int(nr/2), int(nc/2)))
    return cmpr_tnsr
    ### END SOLUTION
```


```python
# Test cell - pooling_test
# Test and prepare for display. The type has been changed to np.uint8 for ease of display in the notebook.

# Important: 
# This test cell will be only run at the time of submission. Please use the code to display the compressed video
# in the next cell to see if you are producing even a reasonable result. Remember you can submit as many times
# as you want. 

###
### AUTOGRADER TEST - DO NOT REMOVE
###

                          
```

Running the cell below displays content of the video compressed by average pooling. You can comment this out if this cell increases your run time.


```python
# Running this cell shows the video. 
# You can comment out this part if you wish as it can increase your total test time during submission.

# Important: 
# Make sure you comment this section before submitting as the runtime for this cell exceeds the time limit. 

cmpr_pooled = avg_pooled(vid)
cmpr_pooled_uint = cmpr_pooled.astype(np.uint8)
warnings.simplefilter('ignore')
imageio.mimwrite('cmpr_pooled.mp4', cmpr_pooled_uint, fps=30)
HTML("""<video width="480" height="380" controls>   <source src="{0}"> </video> """.format('./cmpr_pooled.mp4'))


```

    IMAGEIO FFMPEG_WRITER WARNING: input image is not divisible by macro_block_size=16, resizing from (640, 360) to (640, 368) to ensure video compatibility with most codecs and players. To prevent resizing, make your input image divisible by the macro_block_size or set the macro_block_size to 1 (risking incompatibility).
    




<video width="480" height="380" controls>   <source src="./cmpr_pooled.mp4"> </video> 



You can see that even though the output of the average pooled case looks very similar to the original video, the tensor in the average pooled case was 1/4 in size of the original tensor. You can use the average pooling function repeatedly to compress the tensor and after a couple of such compression stages, the difference will be more significant. The code for that has been left in the comments below, if you would like to try it.


```python
# """
# Sample code to try multi stage pooling compression. 
# Repeated pooling can help achieve greater reductions in frame sizes.
# You can comment out this part as it will not be graded. It is for demonstrative purposes only.
# """
# for _ in range(3):
#     cmpr_pooled = avg_pooled(cmpr_pooled)
# print("Shape of final tensor : {}".format(cmpr_pooled.shape))
# cmpr_pooled = cmpr_pooled.astype(np.uint8)
# warnings.simplefilter('ignore')
# imageio.mimwrite('multi_compression.mp4', cmpr_pooled, fps=30)
# HTML("""<video width="480" height="380" controls>   <source src="{0}"> </video> """.format('./multi_compression.mp4'))
```

**Fin!** Remember to test your solutions by running them as the autograder will: restart the kernel and run all cells from "top-to-bottom." Also remember to submit to the autograder; otherwise, you will **not** get credit for your hard work!
