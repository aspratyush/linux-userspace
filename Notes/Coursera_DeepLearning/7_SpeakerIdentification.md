# Speaker Identification

## Voice as a biometric
- (TODO)

### Commercial solutions
- Nuance multi-modal solution :  https://www.nuance.com/omni-channel-customer-engagement/security/multi-modal-biometrics.html

## Speaker Identification Mechanisms
- Two main approaches:
  1. Closed-set (in-database) / Open-set approach proposed (out-of database)
  2. Text-dependant / text-independant approach

- **Process**:
  1. User enrollment
      - Speak a text.
      - Convert to normalized features.
      - model adaptation.
      - databse of speaker specific models for the user.
  2. User Matching

- **A note on features**
  - discriminative features needed to capture all variations.
  - Voice I/P has : `high-energy linguistic data` and `low-energy ambient noise`.
  - Take note of the energy level guidelines to differentiate the speakers a/c to
  frequency wrapping present.


### Feature-based approaches
#### 1. (MFCC-based)
- **2011, Sen** - _Features extracted using frequency-time analysis approach from Nyquist filter bank and gaussian filter bank for text-independent speaker identification_
- Diff. speakers occupy diff. frequency bands in the spectrum.
- Patterns in the bands is obtained by applying `Fourier transform` and extracting MFCC (`mel-frequency cepstrum components`).
- **MFCC**:
  - Segregate I/P signal into different frequency bands (filter-banks).
  - time-series analysis on these bands.
- GMM + HMM used for matching.

#### 2. (i-vectors recent)
- **2016, ICASSP** - _Integrated adaptation with multi-factor joint-learning for far-field speech recognition_
- each speaker's acoustic model occupies a subspace in the acoustic signal cluster space.
- The principle components making up the subspace are called `i-vectors` (identity vectors).
- generally around $400D-600D$ vectors.
- promising results.

#### 3. (i-vectors)
- **2009, InterSpeech and 2011 Tran on Audio, Speech & Language Proc. -** _Front-end factor analysis for speaker verification_
- GMM + Universal background model (UBM) used for matching after model adaptation.
- UBM : GMM to get i-vector statistics.

#### 4. ICASSP, 2002 : D. Reynolds - Overview of Automatic Speaker Recognition
- (TODO) Based on presence of speaker's voice print.

#### 5. 2012, ICHCI : Sarangi - A novel approach in feature level for robust text-independent speaker identification system
- (TODO)


## DNN-based approaches

#### 1. Google - DNN feature extractors
- **2014 ICASSP - Google**, _Deep neural networks for small footprint text-dependent speaker verification_
- I/P - Each frame of speaker voice
- O/P - accumulated output activations of the the last FC layer : `d-vector`
- `d-vector` directly used for enrolment and matching. No model adaptation!


#### 2. DNN as a classifier
- I/P - MFCC components from 10 frames, each 20ms
- each prediction of the DNN (across frames) is averaged / use 2 DNNs : 1 for frame-level prediction, 1 for classification.


#### 3. SBN, stacked bottleneck features
- **Richardson Dehak 2015 IEEE Sig Prc Letters**, _Deep neural network approaches to speaker and language recognition_.
- Dimensionality of one of the layers is reduced : _bottle-neck layers_.
- I/P --> DNN for feature extraction --> stack the features --> SBN


#### 4. i-vector based approaches
- **Interspeech 2016 Gahabi**, _Deep neural networks for i-vector language identification of short utterances in cars_
- DNN used to extract i-vectors
OR
- **ICASSP 2014**, _I-vector-based speaker adaptation of deep neural networks for french
broadcast audio transcription_
- DNN used as a classifier after extracting i-vectors.


#### 5. DNN posteriors (one-of-the-best)
- **ICASSP 2014**, _A novel scheme for speaker recognition using a phonetically-aware
deep neural network_
- extracting i-vectors using phonetically-aware deep neural network


#### 6. Hinton 2012
- **2012, Hinton IEEE Sig Proc Magazine**, _Deep neural networks for acoustic modeling in speech recognition: The shared views of four research groups_
- (TODO)


#### 7. Deep Speaker, Baidu
- http://research.baidu.com/deep-speaker-end-end-system-large-scale-speaker-recognition/
- (TODO)
