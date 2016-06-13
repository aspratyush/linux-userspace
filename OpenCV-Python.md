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

### II) Thresholding

Following 3 scenarios occur:

#### Global Threshold

* API : `cv2.threshold`
* E.g.:
	- `retVal, imgNew = cv2.threshold(img, THRESHOLD_VALUE, MAX_VAL, THRESH_TYPE)`
	- THRESH_TYPE options:
		- `cv2.THRESH_BINARY` : binary threshold
		- `cv2.THRESH_BINARY_INV` : inverted binary thresholding
		- `cv2.THRESH_TOZERO` : lesser values snapped to ZERO
		- `cv2.THRESH_TOZERO_INV` : higher values snapped to ZERO 


#### Adaptive Threshold

* To handle local illumination / noise, apply local threshold instead of global
* uses a `mean-filter` or `gaussian-filter`

* API : `cv2.adaptiveThreshold`
* E.g. : 
	- `imgNew = cv2.adaptiveThreshold(img, MAX_VAL, ADAPTIVE_METHOD, THRESH_TYPE, BLOCKSIZE, CONST)`
	- ADAPTIVE_METHOD options:
		- `cv2.ADAPTIVE_THRESH_MEAN_C` : mean-filter on the neighbourhood
		- `cv2.ADAPTIVE_THRESH_GAUSSIAN_C` : gaussian-filter on the neighbourhood

#### Otsu's Threshold

* used to find the optimal threshold value for an image with approx. bimodal-histogram
* API : `cv2.threshold` with the extra 4th param : `cv2.THRESH_BINARY+cv2.THRESH_OTSU`
* E.g. :
	- `ret, imgNew = cv2.threshold( img, 0, MAX_VAL, cv2.THRESH_BINARY+cv2.THRESH_OTSU )`
	- `imgNew = cv2.threshold( img, ret, MAX_VAL, cv2.THRESH_BINARY )`

### III) Geometric Transforms

#### Resize

* API : `cv2.resize`
* E.g.:
	- scale by 2
		- `height,width = img.shape[:2]`
		- `dim = ( width*2, height*2 )`
		- `imgNew = cv2.resize( img, dim, interpolation=cv2.INTER_CUBIC)`
	- interpolation options:
		- `cv2.INTER_LINEAR` : default
		- `cv2.INTER_CUBIC` : slow
		- `cv2.INTER_AREA` : shrink


#### Rotation matrix

* API : `cv2.getRotationMatrix2D`
* E.g.:
	- rotate image about its centre by `theta`, and apply `scale`
		- `M = cv2.getRotationMatrix( (cols/2, rows/2), theta, scale)`
		- `imgNew = cv2.warpAffine(img, M, (cols, rows))`


#### Affine Transformation Matrix

* get Affine transform matrix from a set of point-correspondances
* API : `cv2.getAffineTransform`
* E.g.:
	- Say, `pts1` and `pts2` are point correspondances (>=3) of the type : `np.float32[]`
	- `M = cv2.getAffineTransform(pts1, pts2)`
	- `imgNew = cv2.warpAffine(img, M, (cols,rows))`


#### Projective Transform Matrix

* get Projective Transform matrix from a set of point correspondances
* API : `cv2.getProjectiveTransform`
* E.g.:
	- Say, `pts1` and `pts2` are point correspondances (>=4) of the type : `np.float32[]`
	- `M = cv2.getPerspectiveTransform(pts1, pts2)`
	- `imgNew = cv2.warpPerspective(img, M, (cols,rows))`

#### Affine warp

* API : `cv2.warpAffine`
* E.g.:
	- Uses a `2 x 3` matrix `M` to perform an affine warp
		- `imgNew = cv2.warpAffine( img, M, (cols,rows) )`
		- 3rd argument is size of the output image
	- `M = [R|t]`, R = rotation matrix, t = translation
		- `R = [ cos(A) -sin(A); sin(A) cos (A)]`
		- `t = [ tx; ty ]`


#### Projective warp

* API : `cv2.warpPerspective`
* E.g. : 
	- Uses a `3 x 3` matrix to perform projective warp
		- `imgNew = cv2.warpPerspective(img, M, (cols,rows))` 


---------

## Exercise
1. slideshow of images
2. track R,G,B objects simultaneously
3. sudoko image capture -> edge extract -> projection correction -> solve?
