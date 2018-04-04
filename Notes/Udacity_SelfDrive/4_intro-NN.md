Intro to NN
=================

### Machine Learning
- Linear Regression  
    - Metric : sq. distance of points from the line (instead of distance)

- Logistic Regression  
    - Metric : log-loss (instead of count)
    - Large loss for miss-labelling

### Perceptron
- Linear combination of the inputs, weighted by the learned weights.
- linear activation function

### Neural Network
- Linear combination of inputs, passed through a __non-linear, but continuously differentiable__ activation function.
- Training via gradient descent
- __Deep Learning__ : Deep stacks of hidden layers in a neural network.

#### SSE (Sum of squared errors)
\[
\text{Error} = E = \frac{1}{2}\sum_{\mu} \sum_{j} \| y_{j}^{\mu} - \hat{y}_{j}^{\mu} \|_{2} , \ \ \ni \ \ \hat{y}_{j} = f(w_{j}x_{j}) \ \text{and} \ \hat{y} = f(w^{T}x)
\]

- __where__: $y_{j}$ represents all output units in a neural network, $\mu$ represents all the examples, and f() is the non-linear activation function.
- makes the errors positive, and penalizes large errors more.
- the error curve is convex, thus making optimization simpler.
- weights are the tap that tune the output.
- we use gradient descent to figure out the weights that produce lowest error.

#### Gradient descent
- Calculate the change in weights required to reduce the overall error.
\[
\text{gradient} = \Delta{w_{j}} = - \eta \frac{\delta(E)}{\delta(w_{j})}
\]
- __where__ : $\eta$ is the ```LEARNING RATE``` that scales the gradient step.
- refer : Multivariable Calculus : Khan Academy

\[
\text{say} \ \colon \delta = (y-\hat{y})\ \hat{y}' \\
\implies \Delta w_{j} = \eta \delta_{j} x_{j}
\]

__Implementing with ```numpy```__
- It is important to initialize the weights randomly so that they all have different starting values and diverge, __breaking symmetry__

### Multilayer Neural Network
- $w_{ij}$ represents the weight between the I/P unit $i$ and O/P unit $j$.
- __row__ represents weights going out of an I/P unit.
- __col__ represents weights entering into a hidden unit.

\[
\begin{bmatrix}
w_{11} & w_{12} \\
w_{21} & w_{22} \\
w_{31} & w_{32}
\end{bmatrix}
\]

#### Backpropagation
- Derivative of the error at output is scaled by weights between hidden units and the output, and so on.
- The hidden unit contributing more to the output sees more Backpropagating error.
\[
\delta^{h}_{j} = \sum W_{j k} \delta_{k}^{0} f'(h_{j}) \\
\Delta w_{i j} = \eta \delta_{k}^{0} x_{i}
\]


# Computer Vision

### Distortion removal
- Finding chessboard corners (for an 8x6 board):
`ret, corners = cv2.findChessboardCorners(gray, (8,6), None)`

- Drawing detected corners on an image:
`img = cv2.drawChessboardCorners(img, (8,6), corners, ret)`

- Camera calibration, given object points, image points, and the shape of the grayscale image:
`ret, mtx, dist, rvecs, tvecs = cv2.calibrateCamera(objpoints, imgpoints, gray.shape[::-1], None, None)`

- Undistorting a test image:
`dst = cv2.undistort(img, mtx, dist, None, mtx)`


### Perspective Transform
- Compute the perspective transform, M, given source and destination points:
`M = cv2.getPerspectiveTransform(src, dst)`

- Compute the inverse perspective transform:
`Minv = cv2.getPerspectiveTransform(dst, src)`

- Warp an image using the perspective transform, M:
`warped = cv2.warpPerspective(img, M, img_size, flags=cv2.INTER_LINEAR)`


### Gradients
- Calculate the derivative in the xx direction (the 1, 0 at the end denotes xx direction):
`sobelx = cv2.Sobel(gray, cv2.CV_64F, 1, 0)`

- Calculate the derivative in the yy direction (the 0, 1 at the end denotes yy direction):
`sobely = cv2.Sobel(gray, cv2.CV_64F, 0, 1)`

- Calculate the absolute value of the xx derivative:
`abs_sobelx = np.absolute(sobelx)`

- Convert the absolute value image to 8-bit:
`scaled_sobel = np.uint8(255*abs_sobelx/np.max(abs_sobelx))`

- Combine the gradients:
`abs_sobelxy = np.sqrt(np.square(abs_sobelx) + np.square(abs_sobely))`

- Gradient direction:
`absgraddir = np.arctan2(abs_sobely, abs_sobelx)`

- Histogram of a horizontal region:
`histogram = np.sum(img[img.shape[0]/2:,:], axis=0)`

- Gradent strength based plotting:
```
thresh_min = 20
thresh_max = 100
sxbinary = np.zeros_like(scaled_sobel)
sxbinary[(scaled_sobel >= thresh_min) & (scaled_sobel <= thresh_max)] = 1
plt.imshow(sxbinary, cmap='gray')
```


### Object detection

#### Template Matching
- `cv2.matchTemplate()` and `cv2.minMaxLoc()` used.
```
res = cv2.matchTemplate(img,template,method)
min_val, max_val, min_loc, max_loc = cv2.minMaxLoc(res)
```

#### Histogram
```
rhist = np.histogram(image[:,:,0], bins=32, range=(0, 256))

# Generating bin centers
bin_edges = rhist[1]
bin_centers = (bin_edges[1:]  + bin_edges[0:len(bin_edges)-1])/2

# Plot
plt.bar(bin_centers, rhist[0])
plt.xlim(0, 256)
plt.title('R Histogram')

# Concatenate R,G,B histograms
hist_features = np.concatenate((rhist[0], ghist[0], bhist[0]))
```

#### Resize and ravel
- `cv2.resize` used to resize an image
- `img.ravel()` used to convert an image into a 1D vector.


#### HOG
```
from skimage.feature import hog
pix_per_cell = 8
cell_per_block = 2
orient = 9

hog_features, hog_image = hog(img, orientations=orient,
                          pixels_per_cell=(pix_per_cell, pix_per_cell),
                          cells_per_block=(cell_per_block, cell_per_block),
                          visualise=True, feature_vector=False,
                          block_norm="L2-Hys")
```
- `hog_features.ravel()` gives a 1D feature vector
- `hog_image` contains the image with dominant graidents per cell shown.


#### Normalizing features
- `StandardScaler` from `sklearn.preprocessing` is used.
- each row of the I/P needs to be a single feature vector.
```
from sklearn.preprocessing import StandardScaler
feature_list = [feature_vec1, feature_vec2, ...]
X = np.vstack(feature_list).astype(np.float64)

# Fit a per-column scaler using ONLY training data
X_scaler = StandardScaler().fit(X)

# Apply the scaler to X (train)
scaled_X = X_scaler.transform(X)

# Apply the scaler to X (test)
scaled_X_test = X_scaler.transform(X_test)
```

#### Cross validation for parameter search
- Access the parameter values via `clf.best_params_`.
```
parameters = {'kernel':('linear', 'rbf'), 'C':[1, 10]}
svr = svm.SVC()
clf = grid_search.GridSearchCV(svr, parameters)
clf.fit(iris.data, iris.target)
```

#### Label the blobs
```
from scipy.ndimage.measurements import label
labels = label(heatmap)

# heatmap and labels
heatmap = threshold(heatmap, 2)
labels = label(heatmap)
print(labels[1], 'cars found')
plt.imshow(labels[0], cmap='gray')
```


### Extras

#### Plotting colors in 3D
```
from mpl_toolkits.mplot3d import Axes3D

def plot3d(pixels, colors_rgb,
        axis_labels=list("RGB"), axis_limits=((0, 255), (0, 255), (0, 255))):
    """Plot pixels in 3D."""

    # Create figure and 3D axes
    fig = plt.figure(figsize=(8, 8))
    ax = Axes3D(fig)

    # Set axis limits
    ax.set_xlim(*axis_limits[0])
    ax.set_ylim(*axis_limits[1])
    ax.set_zlim(*axis_limits[2])

    # Set axis labels and sizes
    ax.tick_params(axis='both', which='major', labelsize=14, pad=8)
    ax.set_xlabel(axis_labels[0], fontsize=16, labelpad=16)
    ax.set_ylabel(axis_labels[1], fontsize=16, labelpad=16)
    ax.set_zlabel(axis_labels[2], fontsize=16, labelpad=16)

    # Plot pixel values with colors given in colors_rgb
    ax.scatter(
        pixels[:, :, 0].ravel(),
        pixels[:, :, 1].ravel(),
        pixels[:, :, 2].ravel(),
        c=colors_rgb.reshape((-1, 3)), edgecolors='none')

    return ax  # return Axes3D object for further manipulation
```
