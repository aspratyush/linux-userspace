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


#### Perspective Transform Matrix

* get Projective Transform matrix from a set of point correspondances
* API : `cv2.getPerspectiveTransform`
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


#### Perspective warp

* API : `cv2.warpPerspective`
* E.g. : 
	- Uses a `3 x 3` matrix to perform projective warp
		- `imgNew = cv2.warpPerspective(img, M, (cols,rows))` 


### IV) Filtering (LPF)

Filtering involves convolution operation. Following filters are discussed:

#### Mean Filter

* normalized kernel with **same** filter co-efficients
* API : `cv2.boxFilter`
* E.g.:
	- `imgNew = cv2.boxFilter(img,KERNEL_SIZE)`
	- `KERNEL_SIZE = (5,5)`


#### Gaussian Filter

* kernel with **gaussian** co-efficients
* API : `cv2.GaussianBlur`
* E.g.:
	- `imgNew = cv2.GaussianBlur(img,KERNEL_SIZE,ST_DEV_X,ST_DEV_Y)`
	- `KERNEL_SIZE` : tuple indicating kernel-size
	- `ST_DEV_X` : standard deviation along x-axis
	- `ST_DEV_Y` : standard deviation along y-axis. If unset, its equal to `ST_DEV_X`
* Issues : 
	- blurs the image
	- value at (x,y) may not be from the image at all

#### Median Filter

* arranges pixels in sorted manner and picks median value
* API : `cv2.medianBlur`
* E.g.:
	- `imgNew = cv2.medianBlur( img, KERNEL_LINEAR_SIZE )`
	- `KERNEL_LINEAR_SIZE` : +ve odd value

#### Bilateral Filter

* performs de-noising, preserving edges
* uses a combination of gaussian blur filter (*spatial-domain*), and gaussian multiplicative filter (*intensity-domain*)
* API : `cv2.bilateralFilter`
* E.g.:
	- `imgNew = cv2.bilateralFilter( img, d, ST_DEV_COLOUR, ST_DEV_SPACE )`
	- `d` : diameter of each pixel neighbourhood for filtering. if -ve, computed from ST_DEV ( recommended : `5 to 10` )
	- `ST_DEV_COLOUR` : Filter sigma in colour-domain ( recommended : `>10`, large values - cartoonish )
	- `ST_DEV_SPACE` : Filter sigma in spatial-domain ( recommended : `>10`, large values - cartoonish )


### V) Filtering (HPF)

Convolution with a HPF results in edge extraction in images.

#### Sobel / Scharr

* HPF in x-/y- direction
* `Sobel` is more resistant to noise than `Scharr`
* API : `cv2.Sobel` or `cv2.Scharr`
* E.g.:
	- `imgNew = cv2.Sobel( img, d, yorder, xorder, KERNEL_SIZE )`
	- `yorder = 1` : if vertical gradients needed
	- `xorder = 1` : if horizontal gradients needed
	- `KERNEL_SIZE` : kernel size. If set to 1, 3 x 3 filter chosen as default
	- `d` : image depth ( int8 : `cv2.CV_8U`, int16 : `cv2.CV_16S`, float64 : `cv2.CV_64F` ) 

* **Recommended** : use `d = cv2.CV_64F` and do `np.absolute` and `np.uint8`

#### Laplacian

* Uses Sobel HPF in both x- and y-directions
* API : `cv2.Laplacian`
* E.g.:
	- `imgNew = cv2.Laplacian( img, d )`
	- `d` : image depth ( int8 : `cv2.CV_8U`, int16 : `cv2.CV_16S`, float64 : `cv2.CV_64F` ) 


#### Canny 

* Uses a combination of - a) noise filtering, b) gradient extraction, c) NMS, d) Hysteresis Thresholding
* API : `cv2.Canny`
* E.g.:
	- `imgNew = cv2.Canny( img, HYS_MIN, HYS_MAX, KERNEL_SIZE, USE_L2_GRADIENT )`
	- `HYS_MIN` : min-value for Hysteresis
	- `HYS_MAX` : max-value for Hysteresis
	- `KERNEL_SIZE` : Sobel filter size to use. `default = 3`
	- `USE_L2_GRADIENT` : L2-norm to calculate graident magnitude. `default : False (L1)`


### VI) Image Pyramids

* Every octave **up / down**, size changes by `2` in both directions. Area changes by a factor of `4`.
* API : `cv2.pyrUp()`, `cv2.pyrDown()`
* E.g.:
	- Octave up :
		- `octave_up = cv2.pyrUp(img)`
	- Octave down :
		- `octave_down = cv2.pyrDown(img)`
	- Two essential types :
		- `Gaussian` : normal pyramid
		- `Laplacian` : difference between current level and octave down of previous level


### VII ) Image Contours and its derivative properties

* find and/or draw binary-image object boundaries
* original image gets modified on using `cv2.findContours`. Keep a cache!
* API : `cv2.findContours`, `cv2.drawContours`
* E.g.:
	- `contours, hierarchy = cv2.findContours( img, RETRIEVAL_MODE, APPROX_MODE)`
	- RETRIEVAL_MODE : `cv2.RETR_TREE`
	- _refer_ : https://opencv-python-tutroals.readthedocs.io/en/latest/py_tutorials/py_imgproc/py_contours/py_contours_hierarchy/py_contours_hierarchy.html
	- APPROX_MODE : 
		- `cv2.CHAIN_APPROX_NONE` : find all points that form the contour
		- `cv2.CHAIN_APPROX_SIMPLE` : find end-points of the contour for linear scenarios
	- `imgNew = cv2.drawContours( imgNew, contours, -1 , (b,g,r), THICKNESS )`

#### Moments

* find the image moments
* API : `cv2.moments`
* E.g.:
	- `contours, hierarchy = cv2.findContours( img, RETRIEVAL_MODE, APPROX_MODE)`
	- `cnt = contours[0]`
	- `M = cv2.moments(cnt)`

#### Centroid

* find centre of mass of the object
* derived from M
	- `cx = M['m10']/M['m00']`
	- `cy = M['m01']/M['m00']`

#### Area

* find area enclosed by the object
* API : `cv2.contourArea`
* E.g.:
	- `cnt = contours[0]`
	- `area = cv2.contourArea(cnt)`
	- Can also be calculated from Moments
		- `M = cv2.moments(cnt)`
		- `area = M['m00']`

#### Perimeter

* find the contour perimeter / arc-length
* API : `cv2.arcLength`
* E.g.:
	- `cnt = contours[0]`
	- `perimeter = cv2.arcLength(cnt, FLAG)`
		- `FLAG = true` : if closed contour

#### Convex Hull

* find the convex hull of the enclosed object
* API : `cv2.convexHull`
* E.g.:
	- `cnt = contours[0]`
	- `hull = cv2.convexHull(cnt)`

#### Contour Approximation

* find an approximation to the calculated contour
* API : `cv2.approxPolyDP`
* E.g.:
	- `cnt = contours[0]`
	- `epsilon = N * cv2.arcLength(cnt, True)`
	- `approxCnt = cv2.approxPolyDP(cnt, epsilon, True)`

#### Bounding Rectangle and Aspect Ratio

* allows straight / rotated bounding rectangles
* __STRAIGHT__:
	- API : `cv2.boundingRect`
	- E.g.:
		- `cnt = contours[0]`
		- `x,y,w,h = cv2.boundingRect(cnt)`
		- `img = cv2.reactangle(img, (x,y), (x+w, y+h), (b,g,r), THICKNESS)`
* __ROTATED__:
	* API : `cv2.minAreaRect`
	* E.g.:
		- `cnt = contours[0]`
		- `rect = cv2.minAreaRect(cnt)`
		- `box = np.int0( cv2.boxPoints(rect) )`
		- `img = cv2.drawContours( img, [box], 0, (b,g,r), THICKNESS)`

* enables calculating 'Aspect ratio', 'Extent', etc.
* __ASPECT RATIO__:
	- Ratio of width to height
	- `x,y,w,h = cv2.boundingRect(cnt)`
	- `aspect_ratio = float(w)/h`

* __EXTENT__:
	- ratio of actual area to bounding rectangle area
	- `x,y,w,h = cv2.boundingRect(cnt)`
	- `rect_area = w*h`
	- `area = cv2.contourArea(cnt)`
	- `extent = float(area)/rect_area`

#### Minimum Enclosing Circle

* enclose the object in a circle
* API : `cv2.minEnclosingCircle`
* E.g.:
	- `cnt = contours[0]`
	- `(x,y),radius = cv2.minEnclosingCircle(cnt)`
	- `center = (int(x),int(y))`
	- `img = cv2.circle(img, center, radius, (b,g,r), THICKNESS)`

#### Match shapes

* returns a number indicating shape matching.
* `0` means no match, `1` means exact match.
* API : `cv2.matchShapes`
* E.g.:
	- `match_score = cv2.matchShapes(cnt1, cnt2, 1, 0.0)`


### IX) Histograms

---------

## Exercise
1. slideshow of images
2. track R,G,B objects simultaneously
3. sudoko image capture -> edge extract -> projection correction -> solve?
4. Canny for an image with sliders to change Hysteresis values
5. Image blending using pyramids
6. Match letter/digit shapes.. think how it'll lead to OCR
