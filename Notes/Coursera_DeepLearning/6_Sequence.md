# Sequence Models

## Notations
### Named-entity recognition
- **I/P** = $X^{(i)<t>}$ : $i^{th}$-component of $X$ at $t^{th}$-instant .
- **O/P** = $Y^{(i)<t>}$ : $i^{th}$-component of $Y$ at $t^{th}$-instant .
- $<T_{x}>$ : no. of I/P instances.
- $<T_{y}>$ : no. of O/P instances.

### Representing words
- Say, dictionary is a $n$-dimensional vector.
- Each $x^{<t>}$, $t^{th}$-instant I/P $\in \mathbf{R}^{n}$, and is **one-hot** encoded.

### Why not use FFN?
- I/P and O/P can be of differet lengths.
- Sharing of features across different positions of text not there.

## RNNs (unidirectional)
- At each time step, RNN passes current activations to the next instant.
- One problem with RNN is that its causal, i.e., uses information that is already available (${0,1,2,...t}$) and not ($t+1, t+2, ..., t+n$) - **Bidirectional RNNs** solve this.
\[
a^{<t>} = g(W_{aa} a^{<t-1>} + W_{ax} x^{<t>} + b_{a})
\]
\[
y^{<t>} = g(W_{ya} a^{<t>} + b_{y})
\]

### Simplified Notations
\[
a^{<t>} = g(W_{a} [a^{<t-1>};x^{<t>}]^{T} + b_{a})
\]
where: $W_{a} = [W_{aa};W_{ax}]$, i.e., stacking $W_{aa}$ and $W_{ax}$ horizontally.
\[
y^{<t>} = g(W_{y} a^{<t>} + b_{y})
\]
thus, $W_{a}$ and $W_{y}$ are the two parameter matrices.

### Activations
- Named-entity : sigmoid
- Others : tanh

### Loss
#### Loss per time-step
$\mathcal{L}^{<t>}(\hat{y}^{<t>}, y^{<t>}) = -y^{<t>}\log(\hat{y}^{<t>}) - (1-y^{<t>})(\log(1-\hat{y}^{<t>}))$
#### Overall loss
$\mathbf{L}^{<t>}(\hat{y}^{<t>}, y^{<t>}) = \Sigma_{t} \mathcal{L}^{<t>}(\hat{y}^{<t>}, y^{<t>})$


### Examples of RNNs
- Inspired by Karpathy's - "Unreasonable Effectiveness of RNNs"
- Many to many of _same length_
- Many to one (sentiment analysis)
- One to one
- One to many (music generation)
- Many to many of _different lengths_ (machine translation)
  - Contains an encoder portion that reads all the I/P
  - Encoder is followed by Decoder that does the translation.
- Attention-based

### Language modelling
- Compute $\mathbb{P}(y^{<1>}, y^{<2>}, y^{<3>}, \cdots, y^{<T_{y}>})$
- Training set : Corpus of english text

#### Tokenize
- Replace :
  - period with $<\text{EOS}>$ token.
  - out-of-dictioanry words with $<\text{UNK}>$ token (unique)
- Create **one-hot encoded** vectors for every I/P word.

#### Process
- Probability of 2nd word, given 1st word... Probability of 3rd word, given 1st and 2nd word... and so on.

### Sample novel sequences (word/character)
- Similar to one-to-many form, sample the O/P of $t=1$ instant and pass as I/P to $t=2$.
- ignore $<\text{UNK>}$.
- Sample as many as want, or till $<\text{EOS}>$.
- Character-level language model:
  - no need for $<\text{UNK>}$.
  - not good for capturing relationship between long sentences.
  - computationally expensive.
  - better than word-level!

### Vanishing gradients
- RNNs face long-range dependency problem due to vanishing gradients.
- Solution is to use GRU / LSTM RNNs **(can capture long-range dependencies)**

#### GRU (gated recurrent units)
- **Cho 2014**, On the properties of NMT : encoder-decoder approach
- **Chung 2014**, Empirical evaluation of gated recurrent neural networks on sequence models.
- contains gates in the memory cell
- Computationally faster than LSTM.
- NOTE : * denotes element-wise multiplication

$\tilde{c}^{<t>} = \tanh(W_{c}[\Gamma_{r} * c^{<t-1>},x^{<t>}] + b_{c})$
$\Gamma_{u} = \sigma(W_{u}[c^{<t-1>},x^{<t>}] + b_{u})$ : (update)
$\Gamma_{r} = \sigma(W_{r}[c^{<t-1>},x^{<t>}] + b_{r})$ : (retain)
$c^{<t>} = \Gamma_{u} * \tilde{c}^{<t>} + (1-\Gamma_{u}) * c^{<t>}$
$a^{<t>} = c^{<t>}$

#### LSTM (long short-term memory)
- 3 gates in the memory cell.
- Historically LSTM is preferred. GRUs gaining ground now.
- As long as the forget and udpate gates are set properly, relatively easy for a $c^{<t>}$ located temporally far-off to have effect at current time-instance. This gives LSTM (and, GRU) the memory.

$\tilde{c}^{<t>} = \tanh(W_{c}[a^{<t-1>},x^{<t>}] + b_{c})$
$\Gamma_{u} = \sigma(W_{u}[a^{<t-1>},x^{<t>}] + b_{u})$ : (update)
$\Gamma_{f} = \sigma(W_{f}[a^{<t-1>},x^{<t>}] + b_{f})$ : (forget)
$\Gamma_{o} = \sigma(W_{o}[a^{<t-1>},x^{<t>}] + b_{o})$ : (output)
$c^{<t>} = \Gamma_{u} * \tilde{c}^{<t>} + \Gamma_{f} * c^{<t>}$
$a^{<t>} = \Gamma_{o} * c^{<t>}$


### Bidirectinal RNNs
- Acyclic graph
- Takes context from both _past_ and _future_.
- especially used in NLP, with LSTM!
- Computes activations in _forward-cycle_, and then _backward-cycle_. $\{a^{<1>}_{f}, a^{<2>}_{f}, \cdots a^{<t>}_{f}\}$, then $\{a^{<t>}_{b}, a^{<t-1>}_{b}, \cdots a^{<1>}_{b}\}$.
- prediction at time $t$ uses **both** forward and backward activations.
- can predict correctly in middle of the sentence as well.
- disadvantages : processing can be done only after obtaining complete utterance. Hence, not great for real-time STT.
$y^{<t>} = W_{y}[a^{<t>}_{f}, a^{<t>}_{b}] + b_{y}$

### Deep RNNs
- Notation :
  - $a^{[l]<t>}$ : activation for layer $l$ at time $t$.
- layers stacked one-over-the-other.


## Week2 : Word Representations and embedding
### Word Representation
- 1-hot representation
  - inner product between features not captured (orange and apple, man and woman, etc.)
  - does not capture how close a pair are.
- Featurized representation
  - captures inter examples relations.
  - **Maaten & Hinton 2014**, Visualizing data using t-SNE
    - visualizing featurized representations in 2D
  - useful in NLP : _Named-entity recognition_, _Text summarization_, etc. task.

#### Steps
1. Learn word embeddings from a large corpus (1-100 B words), or download from internet.
2. Transfer embedding to the new task with smaller training set (100k words)
3. Optional : continue to fine-tune with new data.

#### Properties of word embeddings
- **Mikolov 2013**, Linguistic regularities in continuous space word representations.
- **Can algorithm figure out analogies?** - word embeddings helps solving this!
  - Find the word $w$ such that : $\arg \max_{w} \text{similarity}(w, e_{king} - (e_{man} - e_{woman}))$
  - i.e., find the vector $e_{w}$ that is closest to : $e_{king} - (e_{man} - e_{woman})$.

#### Similarity functions used:
  1. **Cosine similarity**
    $sim(u,v) = \frac{u^{T}v}{\|u\|_{2} \|v\|_{2}}$
    technically, this is _dot-product_.
  2. **Euclidean distance**
    $\|u-v\|^{2}$

### Embedding Matrix
- $E \in \mathbb{R}^{m \times n}$, where $n$ : vocabulary size and $m$ is the embedding size (say, $m=300$ and $n=10000$).
- $O_{i} \in \mathbb{R}^{m \times 1}$, the _one-hot encoded vector_ for $i^{th}$-word in the vocabulary.
$\Rightarrow$ embedding-vector for $i^{th}$ word : $e_{i} = E O_{i}$


### Learning embedding matrix
#### Neural Language Model
- **Bengio 2003**, Neural Language Model.
- $[e_{1}, e_{2}, \cdots, e_{p}]^{T}$ : stack $p$ embedding vectors
- $I/P --> FC --> Softmax$
- $p$ size:
  - take all words
  - or, sliding window of 4-5 words.
- OBJECTIVE :
  - given $p$-words, predict the next word.
  - OR, given $p$ previous words, and $p$ afterwards words, predict the word in the middle.
  - **skip gram** : nearby 1 word to predict the next word.

#### Skip Gram / word2vec
- **Mikolov 2013**, Efficient estimation of word representations in vector space.
- Come up with (_context_ to _target_) pairs.
  - _context_ : randomly chosen word.
  - _target_ : randomly chosen word +-10 from _context_.
- $O_{c} --> E --> e_{c} --> FC --> softmax --> \hat{y}$
- If $\theta_{t}$ is the param associated with output $t$ :
$softmax = p(t|c) = \frac{e^{\theta_{t}^{T}e_{c}}}{\Sigma_{j} e^{\theta_{j}^{T}e_{c}}}$
Loss = -ve log-likelihood = $\mathbb{L}(\hat{y}, y) = -\Sigma_{i} (y_{i} \log(\hat{y_{i}}))$

##### Problems with this model
- Softmax denominator calculation is slow.
  - Solved by using **Hierarchial softmax** which uses several binary classifiers (binary tree style) to reduce computations.
  - Sampling the context should be done carefully, as some words have high frequency while others have low frequency. Uniform sampling will remain biased towards high=frequency words.

##### Negative sampling
- ** Mikolov 2013**, Distributed representations of words and phrases and their compositionality.
- **CHECK AGAIN!!!**


#### GloVe word vector
- **Pennington 2014**, GloVe : Global vectors for word representations.
- GloVe : Global vectors for word representations
- $X_{ij}$ : # of times $i$ appears in context of $j$.
  - $i$ : target, $j$ : context.
  - by context, we mean words appearing within +-10 distance of each other.
- Symmetric : $X_{ij} = X_{ji}$
- Model:
$\min \Sigma_{i} \Sigma_{j} f(X_{ij}) (\theta_{i}^{T}e_{j} + b_{i} + b'_{j} - \log(X_{ij}))^{2}$
$\Rightarrow f(X_{ij}) : $ weighting term, such that $f(X_{ij}) = 0$, if $X_{ij}=0$, to prevent $\log()$ from going to infinity.


### Sentiment classification
1. Download / build an embeding matrix.
2. $\text{Avg}(e_{1}, e_{2}, \cdots, e_{p}) --> \text{softmax} --> \hat{y}$.
3. RNN
  - $\text{Avg}(e_{1}, e_{2}, \cdots, e_{p}) --> RNN$ : many-to-one

### Debiasing word embeddings
- **Bolukbasi 2016**, Man is to computer programmer as woman is to homemaker? Debiasing word embeddings.
- bias of Gender, ethinicity, age, socio-economic status, etc.
- Steps:
  1. Identify direction of bias
    $Avg(e_{he} - e_{she}, e_{man} - e_{woman}, \cdots)$
  2. Neutralize : project non-definitional words to non-bias dimension
    - Train classifier to detect non-definitional words.
  3. Equalize all pairs.
    - make grandmother-babysitter and grandfather-babysitter same.


## Week3 : Sequence to Sequence Models

### Introduction
- **E.g.** :
  1. machine language translation
      - **Sutskever 2014**, _Sequence to sequence learning with neural networks_
      - **Cho 2014**, _Learning phrase representations for using RNN encoder-decoder for stattistical machine translation_
      - _Many-to-zero_ as an encoder and _zero-to-many_ as a decoder.
      - Similar to language model, except for presence of an encoder that replaces $a^{<0>}$.
      - O/P : $\mathcal{P}(y^{<1>}, y^{<2>}, \cdots, y^{<T_{y}>} | x^{<1>}, x^{<2>}, \cdots, x^{<T_{x}>})$, i.e., P(eng translation | I/P french sentence).
      - Greedy search : get the most probable 1st word, then 2nd word.. and so on.
      - Ideally, prefer to get whole sequence at a time.
  2. Image captioning
      - **Mao 2014**, _Deep captioning with RNNs_
      - **Vinyals 2014**, _Show and tell : Neural Image caption generator_
      - **Karpathy 2015**, _Deep Visual semantic alignments for generating image descriptions_
      - CNN to learn features of an image, serves as an encoder
      - give to RNN, that outputs 1 word at a time.
      - works quite nice for small sentences.


#### Beam search
- Say $B=n$, beam-search considers $n$ possibilities at a time.
- Then, $n$-copies of the encoder-decoder network will exist.
- Initialize by giving 1 word at a time.
- Generally, $B=3$. If $B=1$, beam search is same as greedy search.
- Product of small probablities may lead to numerically unstable result. Hence, take $\log$ and normalize with length of the sequence.
\[
\arg \max_{y} \Pi^{T_{y}}_{t=1} \mathcal{P}(y^{<t>} | x, y^{<1>}, y^{<2>}, \cdots, y^{<t-1>})
\]
Taking $\log$ and normalizing:
\[
\Rightarrow \frac{1}{T_{y}^{\alpha}}\arg \max_{y} \Sigma^{T_{y}}_{t=1} \log \mathcal{P}(y^{<t>} | x, y^{<1>}, y^{<2>}, \cdots, y^{<t-1>})
\]
where, $\alpha \in (0,1)$ (generally 0.7)
- Beam width $B$ very large may result in diminishing results.
- Generally, gain from 1 to 10 maybe high.. 1000 to 3000 may be very small.
- Error analysis :
    - find out what fraction of errors are due to beam search and what due to RNN.
    - Say $y*$ is human result and $\hat{y}$ is algorithm result.
    - $P(y^{*}|x) > P(\hat{y}|x)$ : Beam search fault.
    - $P(y^{*}|x) < P(\hat{y}|x)$ : RNN fault.


#### Attention model intuition - Important
- **Bahdanau 2014**, Neural machine translation by jointly learning to align and translate -- very good paper!!!
- **Xu 2015**, Show attention and tell : neural image caption generator with visual attention.
- $\alpha^{<i,j>}$ is the attention for $y^{<i>}$ should pay to $j^{th}$-hidden layer activation.
- allows to look at local window instead of complete text memorization.
- **NOTE**:
    - $\Sigma_{j} \alpha^{<i,j>} = 1$
    - context vector $c^{<i>} = \Sigma_{j} \alpha^{<i,j>} a^{<j>}$


#### Speech Recognition
- 3000hrs to 100,000 hrs.
- CTC cost for Speech recognition
    - **Graves 2006**, Connectionist Temporal Classification : Labeling unsegmented sequence data with RNNs.
    - collapse repeated characters not separated by blank.

#### Keyword Detection
- Many-to-many RNN, with O/P label all zeros except for when trigger word gets over.
