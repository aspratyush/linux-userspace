## 1. Using Keras APIs
### Callbacks
1. EarlyStopping
    - Stop the training if `monitor` does not improve further, beyond `patience`.
    - `early_stop = EarlyStopping(monitor='val_loss', patience=2, verbose=1)`
2. ModelCheckpoint
    - save the current model.
    - `best_model = ModelCheckpoint(name='test.h5', verbose=0, save_best_only=True)`
    - This saves the _best_ model in filename `test.h5`.

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

#### Convert to categorical
```
from keras.utils import np_utils
y_train = np_utils.to_categorical(y_train, #classes)
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
- https://arxiv.org/abs/1609.03499 : WaveNet : A generative model for raw audio (causal padding in sec 2.1)
### Maxpool
- helps reduce params, reduce over-fitting
- maxpool ensures that the spatial location of the feature is not important.
- Excellent intuition example:
    - _Lets assume that we have a filter that is used for detecting faces. The exact pixel location of the face is less relevant then the fact that there is a face "somewhere at the top."_
