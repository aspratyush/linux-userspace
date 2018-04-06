## 1. Using Keras APIs
### Callbacks
1. EarlyStopping
    - Stop the training if `monitor` does not improve further, beyond `patience`.
    - `early_stop = EarlyStopping(monitor='val_loss', patience=2, verbose=1)`
2. ModelCheckpoint
    - save the current model.
    - `best_model = ModelCheckpoint(name='test.h5', verbose=0, save_best_only=True)`
    - This saves the __best__ model in filename `test.h5`.

### Using the callbacks in training
`model.fit(X_train, Y_train, validation_data=(X_val, Y_val), epochs=50, batch_size=128, verbose=True, callbacks=[best_model, early_stop])`

## 2. Using backend

### Build the model
- define the model --> weight update --> optimization --> loop

```
using keras.backend as K
x = K.placeholder(dtype="float", shape=X_train.shape)
target = K.placeholder(dtype="float", shape=Y_train.shape)

# 1. define the model
w = K.variable(np.random.rand(a1,b1))
b = K.variable(np.random.rand(b1))
y = K.dot(w,x) + b
activation = K.softmax(y)
loss = K.categorical_crossentropy(activation, target)

# 2. weight update using gradients
lr = K.constant(0.001)
grads = K.gradients(loss, [w,b])
updates = [(W, W-gradients[0]*lr), (b, b-gradients[1]*lr)]

# 3. optimization objective
train = K.function(inputs=[x, targets], outputs=[loss], updates=updates)

# 4. loop
loss_history = []
for epoch in steps:
  current_loss = train([X_train, Y_train])[0]
  loss_history.append(current_loss)

# mean the loss_history
loss_history = [np.mean(loss) for loss in loss_history]
```

## 3. Useful utilities

#### Convert to one-hot form
```
from keras.utils import np_utils
y_train_oh = np_utils.to_categorical(y_train, #classes)
```

#### Convert from one-hot to integer
```
y_train = np.argmax(y_train, axis=1)
```

#### Convert datatype
```
X_train = X_train.astype("float32")
```

#### Train-test split
```
from sklearn.model_selection import train_test_split
X_train, X_val, y_train, y_val = train_test_split(X_train, y_train)
```

#### Convert list to array
```
np.asarray(list)
```

#### One-hot encoding
```
from sklearn.preprocessing import LabelBinarizer
label_binarizer = LabelBinarizer()
y_one_hot = label_binarizer.fit(y_train)
y_one_hot = label_binarizer.transform(y_train)
```
OR
```
y_one_hot = tf.one_hot(y_train, #num_classes)
```


#### Validation split
```
model.fit(X_train_normalized, y_one_hot, epochs=__, validation_split=0.2)
```
OR
```
from sklearn.model_selection import train_test_split
X_train, X_val, y_train, y_val = train_test_split(X_train, y_train)
```


#### Lambda layers
- Convenient way to add extra processing on the images.
```
from keras.models import Sequential, Model
from keras.layers import Lambda

# set up lambda layer
model = Sequential()
model.add(Lambda(lambda x: (x / 255.0) - 0.5, input_shape=(160,320,3)))
```


#### Flip
```
import numpy as np
image_flipped = np.fliplr(image)
```


#### Cropping layer
- Crop so many pixels from top, bottom, left and right of the image.

```
from keras.models import Sequential, Model
from keras.layers import Cropping2D
import cv2

# set up cropping2D layer
model = Sequential()
model.add(Cropping2D(cropping=((50,20), (0,0)), input_shape=(3,160,320)))
```

## 4. Inspecting layers
#### Model summary
```
model.summary()
```

#### Model Iteration
```
print(model.input)
for layer in model.layers:
  print(layer.name, layer.iterable, end='\n')
  print(layer.get_config(), end='\n{}'.format('---'*25))
print(model.output)
```

## 5. Extract hidden layer representations
- truncate the layers in a new model, iterate over old model and `set_weights`.
```
# 1. build truncated model
model_truncated = ...

# 2. iterate over the layers
for i,layer in enumerate(model_truncated.layers):
  layer.set_weights(model.layers[i].get_weights())

# 3. predict
hidden_feat = model_truncated.predict(X_train)
```

#### Generate TSNE / MDS embedding
```
from sklearn.manifold import TSNE
# 1. get manifold
tsne = TSNE(n_components=2)

# 2. get projection of hidden features on the manifold
X_tsne = tsne.fit_transform(hidden_feat[:1000])

# 3. get the labels from one-hot encoded vector
color_map = np.argmax(Y_train, axis=1)

# 4. check some label to verify correctness
np.where(color_map==6)

# 5. plot
colors=np.array([x for x in 'b-g-r-c-m-y-k-purple-coral-lime'.split('-')])
for cl in range(nb_classes):
  indices = np.where(color_map==cl)
  plt.scatter(X_tsne[indices,0], X_tsne[indices,1], c=colors[cl], label=cl)
plt.legend()
plt.show()
```


## 6. CNNs
- https://arxiv.org/abs/1603.07285 : A guide to convolution arithmetic for deep learning.
- https://arxiv.org/abs/1609.03499 : **WaveNet** : A generative model for raw audio (causal padding in sec 2.1)
- https://arxiv.org/abs/1409.1556 : (**VGG16**) Very Deep Convolutional Networks for Large-Scale Image Recognition
- https://arxiv.org/abs/1512.03385 : (**ResNet**) Deep Residual Learning for Image Recognition
- http://arxiv.org/abs/1512.00567 : (**Inception v3**) Rethinking the Inception Architecture for Computer Vision

### Maxpool
- helps reduce params, reduce over-fitting
- maxpool ensures that the spatial location of the feature is not important.
- Excellent intuition example:
    - _Lets assume that we have a filter that is used for detecting faces. The exact pixel location of the face is less relevant then the fact that there is a face "somewhere at the top."_

#### Shape order for reshaping the I/P
```
from keras import backend as K
img_rows, img_cols = 28, 28

if K.image_data_format() == 'channels_first':
  shape_ord = (1, img_rows, img_cols)
else:
  shape_ord = (img_rows, img_cols, 1)
```

#### Reshaping the I/Ps
```
X_train = X_train.reshape((X_train.shape[0],) + shape_ord)
```

#### Metrics in the evaluated model
```
model.metrics_names
>> ['loss', 'acc']
loss, acc = model.evaluate(X_test, Y_test)
```

#### Plot integer labels in images
```
plt.text(0, 0, predicted[i], color='black',
             bbox=dict(facecolor='white', alpha=1))
```


## VGG16 testing

#### 1. Loading models
```
from keras.application import <model_name>
```

#### 2. Pre-process and predict
```
# 1a. Load modules <keras uses PIL internally>
from keras.preprocessing import image
from keras.applications import VGG16
from keras.applications.imagenet_utils import preprocess_input, decode_predictions

# 1b. VGG16 model load
vgg16 = VGG16(include_top=True, weights='imagenet')
vgg16.summary()

# 2. load the image
image_path = os.path.join(PATH, 'file,jpg')
img = image.load_img(image_path, target_size=(224,224))

# 3. convert to array and preprocess
x = image.img_to_array(img)

# 4. add dimension for the batch size
x = np.expand_dims(x, axis=0)

# 5. preprocess
x = preprocess_input(x)

# 6. predict
preds = vgg16.predict(x)
print("Predictions : ", decode_predictions(preds))
```


### 3. Visualizing activations
- Create look-up of layer names and layers
```
from collections import OrderedDict
layer_dict = OrderedDict()
# get the symbolic outputs of each "key" layer (we gave them unique names).
for layer in vgg16.layers[1:]:
    print(layer.name)
    layer_dict[layer.name] = layer
```
OR a much more concise form:
```
layer_dict = dict([(layer.name, layer) for layer in model.layers ])
```


- Use `get_activations` (based on `K.function`)
```
# define the function
def get_activations(model, layer, input_img):
  activation_f = K.function([model.layers[0].input, K.learning_phase()],[layer.output])
  activations = activation_f((input_img, False))
  return activations

# get the activations for the layer_name
activations = get_activations(vgg19, lookup[layer_name], img)
```

- Activations obtained will be of the size $(1, H, W, F)$, where F is the filter size.
- Build a square of $f = \sqrt{F}$-length and imshow the activations.
```
activated_img = activations[0]
# add a figure
fig = plt.figure(figsize=(20,20))
for i in range(f):
  for j in range(f):
    idx = (i*f+j)
    ax = fig.add_subplot(f,f,idx+1)
    ax.imshow(activated_img[:,:,idx])

```


#### 4. Hyper-parameter grid search using scikit-learn
- Excellent link : https://machinelearningmastery.com/grid-search-hyperparameters-deep-learning-models-python-keras/
- Nice library : Hyperas http://maxpumperla.github.io/hyperas/

##### Steps:
- create Sequential model and give to KerasClassifier
```
from keras.wrappers.scikit_learn import KerasClassifier
from sklearn.model_selection import GridSearchCV

# param1 could be epoch, param2 could be dropout_rate, or neither could be there
def create_model(param1, param2):
  ..
  ..
  return model

model = KerasClassifier(build_fn=create_model, param1=10, param2=20)
```

- create a `dict` of the params
```
param_grid = dict(epochs=[10,20,30])
grid = GridSearchCV(estimator=model, param_grid=param_grid, n_jobs=-1)
grid_result = grid.fit(X, Y)
```

- best model is then present in the `grid_result`
```
# NOTE : grid_result.best_estimator_ is sklearn form.
# NOTE : grid_result.best_estimator_.model is keras form.
best_model = grid_result.best_estimator_.model
metric_names = best_model.metrics_names
metric_values = best_model.evaluate(X_test, y_test)
for metric, value in zip(metric_names, metric_values):
    print(metric, ': ', value)
```


#### 5. Transfer learning
- Fix the bottom layers, and train the top few layers.
- How many layers to train : depends on the data available.`
```
layers = dict([(layer.name, layer) for layer in model.layers])
layers_to_finetune = ['dense_1', 'dense_2']
for name,layer in layers:
  if(name not in layers_to_finetune):
    layer.trainable = False
```

- Example on VGG16:
```
# 1. create VGG model with no top layers
from keras.applciations import VGG16
vgg16 = VGG16(weights='imagenet', include_top='False')

# 2. set layers as non-trainable
for layer in vgg16.layers:
  layer.trainable = False

# 3. Add FC layers
x = Flatten(input_shape=vgg16.output.shape)(vgg16.output)
x = Dense(4096, activation='relu', name='ft_fc1')(x)
x = Dropout(0.2)(x)
x = BatchNormalization()(x)
preds = Dense(nb_classes, activation='softmax')(x)

# 4. Create a model with both these
model = Model(inputs=vgg16.input, outputs=preds)
model.compile(optimizer='adagrad', loss='categorical_crossentropy', metrics=['accuracy'])
```


#### 6. Generate batches
```
def generate_batches(X, Y, batch_size=128):
  start = 0
  yield (X[start:start+batch_size], Y[start:start+batch_size])
  start = batch_size

tboard_callback = TensorBoard(log_dir='logs/{}', histogram_freq=0,
                             write_graph=True, write_images=True,
                             embeddings_freq=10,
                             embeddings_layer_names=['block1_conv2',
                                                     'block5_conv1',
                                                     'ft_fc1'],
                             embeddings_metadata=None)
batch_size=64
steps_per_epoch = np.floor(X_train.shape[0]/batch_size)
model.fit_generator(generate_batches(X_train, Y_train, 64),     
                    steps_per_epoch=steps_per_epoch,
                    epochs=20,
                    verbose=1,
                    callbacks=[tboard_callback])
```


#### 7. model.fit and plotting using history object
```
from keras.models import Model
import matplotlib.pyplot as plt

history_object = model.fit_generator(train_generator, samples_per_epoch =
    len(train_samples), validation_data =
    validation_generator,
    nb_val_samples = len(validation_samples),
    nb_epoch=5, verbose=1)

### print the keys contained in the history object
print(history_object.history.keys())

### plot the training and validation loss for each epoch
plt.plot(history_object.history['loss'])
plt.plot(history_object.history['val_loss'])
plt.title('model mean squared error loss')
plt.ylabel('mean squared error loss')
plt.xlabel('epoch')
plt.legend(['training set', 'validation set'], loc='upper right')
plt.show()
```

### RNNs
#### Creating a network with RNNs (SimpleRNN / LSTM / GRU)
```
model = Sequential()
model.add(Embedding(vocab_size, dense_vec_size, input_length=max_len))
model.add(SimpleRNN(128)) {or, LSTM(128), or, GRU(128)}
model.add(Dropout(0.2))
model.add(Dense(1))
model.add(Activations('sigmoid'))

model.compile(loss='binary_crossentropy', optimizer='adam')

model.fit(X_train, y_train, epochs=4, batch_size=batch_size, validation_data=(X_val, y_val))
score, acc = model.evaluate(X_test, y_test, batch_size=batch_size)
print('Test score:', score)
print('Test accuracy:', acc)
```


## Extras

#### Image generator example from Udacity
```
import os
import csv

samples = []
with open('./driving_log.csv') as csvfile:
    reader = csv.reader(csvfile)
    for line in reader:
        samples.append(line)

from sklearn.model_selection import train_test_split
train_samples, validation_samples = train_test_split(samples, test_size=0.2)

import cv2
import numpy as np
import sklearn

def generator(samples, batch_size=32):
    num_samples = len(samples)
    while 1: # Loop forever so the generator never terminates
        shuffle(samples)
        for offset in range(0, num_samples, batch_size):
            batch_samples = samples[offset:offset+batch_size]

            images = []
            angles = []
            for batch_sample in batch_samples:
                name = './IMG/'+batch_sample[0].split('/')[-1]
                center_image = cv2.imread(name)
                center_angle = float(batch_sample[3])
                images.append(center_image)
                angles.append(center_angle)

            # trim image to only see section with road
            X_train = np.array(images)
            y_train = np.array(angles)
            yield sklearn.utils.shuffle(X_train, y_train)

# compile and train the model using the generator function
train_generator = generator(train_samples, batch_size=32)
validation_generator = generator(validation_samples, batch_size=32)

ch, row, col = 3, 80, 320  # Trimmed image format

model = Sequential()
# Preprocess incoming data, centered around zero with small standard deviation
model.add(Lambda(lambda x: x/127.5 - 1.,
        input_shape=(ch, row, col),
        output_shape=(ch, row, col)))
model.add(... finish defining the rest of your model architecture here ...)

model.compile(loss='mse', optimizer='adam')
model.fit_generator(train_generator, samples_per_epoch= /
            len(train_samples), validation_data=validation_generator, /
            nb_val_samples=len(validation_samples), nb_epoch=3)
```


#### ImageDataGenerator Example
```
from keras.preprocessing.image import ImageDataGenerator

generated_images = ImageDataGenerator(
    featurewise_center=True,  # set input mean to 0 over the dataset
    samplewise_center=False,  # set each sample mean to 0
    featurewise_std_normalization=True,  # divide inputs by std of the dataset
    samplewise_std_normalization=False,  # divide each input by its std
    zca_whitening=False,  # apply ZCA whitening
    rotation_range=0,  # randomly rotate images in the range (degrees, 0 to 180)
    width_shift_range=0.2,  # randomly shift images horizontally (fraction of total width)
    height_shift_range=0.2,  # randomly shift images vertically (fraction of total height)
    horizontal_flip=True,  # randomly flip images
    vertical_flip=False)  # randomly flip images

generated_images.fit(X_train)

gen = generated_images.flow(X_train, Y_train, batch_size=500, shuffle=True)
X_batch, Y_batch = next(gen)

from keras.utils import generic_utils

n_epochs = 2
for e in range(n_epochs):
    print('Epoch', e)
    print('Training...')
    progbar = generic_utils.Progbar(X_train.shape[0])

    for X_batch, Y_batch in generated_images.flow(X_train, Y_train, batch_size=500, shuffle=True):
        loss = model.train_on_batch(X_batch, Y_batch)
        progbar.add(X_batch.shape[0], values=[('train loss', loss[0])])
```
