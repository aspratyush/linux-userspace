Image Operations
===================

### Split & Merge the channels

* API : `split`, `merge`
* E.g. : 
	- `b,g,r = cv2.split(img)`
	- `img = cv2.merge((b,g,r))`

### Alpha Blend

* API : 'addWeighted'
* E.g. : 
	- `imgNew = cv2.addWeighted(img, dim)`
	- `dim` is a tuple of new (rows,cols)

### Optimizations

* Ref : https://opencv-python-tutroals.readthedocs.io/en/latest/py_tutorials/py_core/py_optimization/py_optimization.html

## Exercise
1. slideshow of images
2. track R,G,B objects simultaneously
