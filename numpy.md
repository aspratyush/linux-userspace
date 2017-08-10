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

### Python

- _named argument_ cannot be followed by a _positional argument_
- everything is an object (including functions)
- assigning an undeclared var allowed, but referencing such vars not allowed
- python promotes usage of _exceptions_, and to _handle_ them using 
`try...except...raise` blocks
	- some exceptions : _ImportError_(import failed), _ValueError_(value issue), 
	_NameError_ (variable not defined)
