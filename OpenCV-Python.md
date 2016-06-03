Image Operations
===================

## Core Operations

Following core operations are described: 

### I) Split & Merge the channels

* API : `split`, `merge`
* E.g. : 
	- `b,g,r = cv2.split(img)`
	- `img = cv2.merge((b,g,r))`

### II) Alpha Blend

* API : `addWeighted`
* E.g. : 
	- `imgNew = cv2.addWeighted(img1, alpha, img2, (1-alpha), 0)`

### III) Optimizations

* Ref : https://opencv-python-tutroals.readthedocs.io/en/latest/py_tutorials/py_core/py_optimization/py_optimization.html


## Image Processing

Following image processing operations are described:

### I) ColorSpace Conversions

* API : `cv2.cvtColor`, `cv2.inRange`
* E.g.: 
	- convert a BGR image to HSV
		- `hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)`
	- threshold the image in the range [lower_limit,upper_limit]
		- `mask = cv2.inRange(image, lower_limit, upper_limit)`

**NOTE** : use `cv2.cvtColor` to get threshold values in HSV from RGB

### II) Global Thresholding

* API : `cv2.threshold`
* E.g.:
	- `imgNew = cv2.threshold(img, THRESHOLD_VALUE, MAX_VAL, cv2.THRESH_BINARY)`
	- 4th param options:
		- `cv2.THRESH_BINARY` : binary threshold
		- `cv2.THRESH_BINARY_INV` : inverted binary thresholding
		- `cv2.THRESH_TOZERO` : lesser values snapped to ZERO
		- `cv2.THRESH_TOZERO_INV` : higher values snapped to ZERO 


### III) Adaptive Thresholding


## Exercise
1. slideshow of images
2. track R,G,B objects simultaneously
