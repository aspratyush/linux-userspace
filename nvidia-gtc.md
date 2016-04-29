Nvidia GTC 2016
---------------

### Deep Learning SDK

1. Toolbox - Driveworks SDK for auto
	- DGX-1 + car + drive px
	- using supported frameworks ( caffe, cntk, kaldi, theano, tensorflow, torch )
	- drive px : dual tegra x1, 256 gpu cores, several peripherals
2. VR
3. DL chip
4. DL box
5. DL car 


### 

- Bret : Berkeley bot, for home tasks
- Tesla P100 : new GPU for DL.. supposedly 12x speed-up 
- shared control
	- series
	- interleaved
	- parallel - parallel autonomy

### OEM Demos

- Ford
	- autonomous driving in snow
	- Mark Crawford
	- 3 steps
		1. map generation - lidar, GPS, IMU
		2. localization - using intensity scan matching, synthetic views and GMM
		3. interpolate the road from database

- VW
	- Piloted driving with deep learning
	- Martin Hempel
	- give sensor input, get actuator controls
	- DNN system : object detection + scene interpretation + 
	  decision making + path planning + controller
	- use transfer learning from available databases and train on small video

- Audi
	- Automous braking with 3D monovision camera
	- accidents, currently functional blocks are vision based 

- Volvo
	- accidents


### TIER 1 Demos

- Panasonic
	- Minyoung Kim
	- pedestrain detection and localization
	- generate region proposals for live video feed

- TomTom
	- Willem Strijbosch
	- Map for autonomous car (roadDNA)

- Mapbox
	- tools to edit maps
	- Eric Gundersen
	- Reimagining cartography for Navigation

### Examples
- Horus
	- 3D sound capture
	- vision for scene understanding and helping blind people see

- Stereo Labs
	- depth sensing + motion tracking + 3D reconstruction + overlay objects

- Unreal Engine
	- VR environment


### TECH PERSPECTIVE

#### Supervised
	- Dynamic Memory N/Ws
	- Neural Attention Tracking
		- Q&A
		- image understanding
		- super resolution & denoising
		- auto tagging
		- map enhancement
		- audio info extraction
		- patient movement classification
		- hand gesture recognition

#### Unsupervised
	- Generative Adversarial Networks (GAN)
	- Deep convolution GAN
		- super resolution
		- image generation
		- unsupervised labelling of images

#### Reinforcement Learning (RL)
	- Deep RL
		- robotics
		- game playing


##### Map Enhancements : RoadDNA - TomTom (SLAM)
	- Navigation
	- Planning
	- Localization
	- uses Mobile Mapping Vehicle, Field Survey

#### Image classification
	- flexible image captionining
	- unsupervised image unknown pixels generation
