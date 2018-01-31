# Fast AI #

### Steps to follow with new dataset
* Create __Validation__ and __Sample__ set
* Move images from train to individual folders
* Finetune and Train
* Submit

### Simple VGG16 network ###
```
// build the network
vgg = Vgg16()  
// get batches   
batches = vgg.get_batches(path + 'train', batch_size=batch_size)   
val_batches = vgg.get_batches(path + 'val', batch_size=batch_size*2)   
// finetune   
vgg.finetune(batches)   
// fit (change no. of epochs)
vgg.fit(batches, val_batches, nb_epoch=1)   
// save the model   
vgg.model.save_weights(path + 'results/ft1.h5')   
//test
batches, pred = vgg.test( path + 'test', batch_size=batch_size*2)
```

* ```fit``` uses ```categorical_crossentropy``` : same as log-loss


### Need for fine-tuning ###



### Visualize Results ###
* Always, plot for different epochs:
	- examples which are right
	- examples which are wrong
	- examples which are class_i, but predicted != class_i
	- examples which are class_j, but predicted != class_j
	- examples on which most uncertain (around 0.5) 


### Things to check ###
* predict_generator
* flow\_from\_directory (class\_mode = categorical / none)
* animation in jupyter
