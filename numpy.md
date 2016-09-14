python-numpy
=============

### Some important APIs:

- **np.flatnonzero(x)** 	: return indices where `x` is non-zero
- **np.astype('uint8')** 	: cast matrix to uint8 data
- **np.range(num1[,num2])**	: list of linearly-spaced values between `num1` & `num2`
- **np.xrange(num1,num2)**	: memory-efficient linearly-spaced values
- **np.random.choice(x,num)**	: list of `num` values from `x`
- **np.reshape(X, (m, -1))**	: reshape a higher dimensional `X` into `m`-rows
- **np.mean(X,axis=0)**		: mean of the rows. `axis=1` for cols
- **np.hstack([X,vec])**	: horizontally stack `vec` to `X`
