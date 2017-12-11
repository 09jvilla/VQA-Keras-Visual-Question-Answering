import tensorflow as tf
from keras import backend as K
from keras.engine.topology import Layer
import numpy as np

def _generate_sketch_matrix(rand_h, rand_s, output_dim):
    """
    Return a sparse matrix used for tensor sketch operation in compact bilinear
    pooling

    Args:
        rand_h: an 1D numpy array containing indices in interval `[0, output_dim)`.
        rand_s: an 1D numpy array of 1 and -1, having the same shape as `rand_h`.
        output_dim: the output dimensions of compact bilinear pooling.

    Returns:
        a sparse matrix of shape [input_dim, output_dim] for tensor sketch.
    """

    # Generate a sparse matrix for tensor count sketch
    rand_h = rand_h.astype(np.int64)
    rand_s = rand_s.astype(np.float32)
    assert(rand_h.ndim==1 and rand_s.ndim==1 and len(rand_h)==len(rand_s))
    assert(np.all(rand_h >= 0) and np.all(rand_h < output_dim))

    input_dim = len(rand_h)
    indices = np.concatenate((np.arange(input_dim)[..., np.newaxis],
                              rand_h[..., np.newaxis]), axis=1)
    sparse_sketch_matrix = tf.sparse_reorder(
        tf.SparseTensor(indices, rand_s, [input_dim, output_dim]))
    return sparse_sketch_matrix

def compact_bilinear_pooling_layer(bottom1, bottom2, output_dim, sum_pool=True,
    rand_h_1=None, rand_s_1=None, rand_h_2=None, rand_s_2=None,
    seed_h_1=1, seed_s_1=3, seed_h_2=5, seed_s_2=7):
    """
    Compute compact bilinear pooling over two bottom inputs. Reference:

    Yang Gao, et al. "Compact Bilinear Pooling." in Proceedings of IEEE
    Conference on Computer Vision and Pattern Recognition (2016).
    Akira Fukui, et al. "Multimodal Compact Bilinear Pooling for Visual Question
    Answering and Visual Grounding." arXiv preprint arXiv:1606.01847 (2016).

    Args:
        bottom1: 1st input, 4D Tensor of shape [batch_size, height, width, input_dim1].
        bottom2: 2nd input, 4D Tensor of shape [batch_size, height, width, input_dim2].

        output_dim: output dimension for compact bilinear pooling.

        sum_pool: (Optional) If True, sum the output along height and width
                  dimensions and return output shape [batch_size, output_dim].
                  Otherwise return [batch_size, height, width, output_dim].
                  Default: True.

        rand_h_1: (Optional) an 1D numpy array containing indices in interval
                  `[0, output_dim)`. Automatically generated from `seed_h_1`
                  if is None.
        rand_s_1: (Optional) an 1D numpy array of 1 and -1, having the same shape
                  as `rand_h_1`. Automatically generated from `seed_s_1` if is
                  None.
        rand_h_2: (Optional) an 1D numpy array containing indices in interval
                  `[0, output_dim)`. Automatically generated from `seed_h_2`
                  if is None.
        rand_s_2: (Optional) an 1D numpy array of 1 and -1, having the same shape
                  as `rand_h_2`. Automatically generated from `seed_s_2` if is
                  None.

        sequential: (Optional) if True, use the sequential FFT and IFFT
                    instead of tf.batch_fft or tf.batch_ifft to avoid
                    out-of-memory (OOM) error.
                    Note: sequential FFT and IFFT are only available on GPU
                    Default: True.

    Returns:
        Compact bilinear pooled results of shape [batch_size, output_dim] or
        [batch_size, height, width, output_dim], depending on `sum_pool`.
    """

    # Static shapes are needed to construction count sketch matrix
    input_dim1 = bottom1.get_shape().as_list()[-1]
    input_dim2 = bottom2.get_shape().as_list()[-1]

    # Step 0: Generate vectors and sketch matrix for tensor count sketch
    # This is only done once during graph construction, and fixed during each
    # operation
    if rand_h_1 is None:
        np.random.seed(seed_h_1)
        rand_h_1 = np.random.randint(output_dim, size=input_dim1)
    if rand_s_1 is None:
        np.random.seed(seed_s_1)
        rand_s_1 = 2*np.random.randint(2, size=input_dim1) - 1
    sparse_sketch_matrix1 = _generate_sketch_matrix(rand_h_1, rand_s_1, output_dim)
    if rand_h_2 is None:
        np.random.seed(seed_h_2)
        rand_h_2 = np.random.randint(output_dim, size=input_dim2)
    if rand_s_2 is None:
        np.random.seed(seed_s_2)
        rand_s_2 = 2*np.random.randint(2, size=input_dim2) - 1
    sparse_sketch_matrix2 = _generate_sketch_matrix(rand_h_2, rand_s_2, output_dim)

    # Step 1: Flatten the input tensors and count sketch
    bottom1_flat = tf.reshape(bottom1, [-1, input_dim1])
    bottom2_flat = tf.reshape(bottom2, [-1, input_dim2])
    # Essentially:
    #   sketch1 = bottom1 * sparse_sketch_matrix
    #   sketch2 = bottom2 * sparse_sketch_matrix
    # But tensorflow only supports left multiplying a sparse matrix, so:
    #   sketch1 = (sparse_sketch_matrix.T * bottom1.T).T
    #   sketch2 = (sparse_sketch_matrix.T * bottom2.T).T
    sketch1 = tf.transpose(tf.sparse_tensor_dense_matmul(sparse_sketch_matrix1,
        bottom1_flat, adjoint_a=True, adjoint_b=True))
    sketch2 = tf.transpose(tf.sparse_tensor_dense_matmul(sparse_sketch_matrix2,
        bottom2_flat, adjoint_a=True, adjoint_b=True))

    # Step 2: FFT
    fft1 = tf.fft(tf.complex(real=sketch1, imag=tf.zeros_like(sketch1)))
    fft2 = tf.fft(tf.complex(real=sketch2, imag=tf.zeros_like(sketch2)))

    # Step 3: Elementwise product
    fft_product = tf.multiply(fft1, fft2)

    # Step 4: Inverse FFT and reshape back
    # Compute output shape dynamically: [batch_size, height, width, output_dim]
    cbp_flat = tf.real(tf.ifft(fft_product))
    output_shape = tf.add(tf.multiply(tf.shape(bottom1), [1, 1, 1, 0]),
                          [0, 0, 0, output_dim])
    cbp = tf.reshape(cbp_flat, output_shape)

    # Step 5: Sum pool over spatial dimensions, if specified
    if sum_pool:
        cbp = tf.reduce_sum(cbp, reduction_indices=[1, 2])

    return cbp


class CompactBilinearPooling(Layer):

    def __init__(self, d, return_extra=False, **kwargs):
        self.h = [None, None]
        self.s = [None, None]
        self.return_extra = return_extra
        self.d = d
        self.sparse_matrix = [None, None]

        # layer parameters
        self.inbound_nodes = []
        self.outbound_nodes = []
        self.constraints = {}
        self.regularizers = []
        self.trainable_weights = []
        self.non_trainable_weights = []
        self.supports_masking = True
        self.trainable = True
        self.uses_learning_phase = False
        self.input_spec = None  # compatible with whatever
        super(CompactBilinearPooling, self).__init__(**kwargs)

    def build(self, input_shapes):
        self.trainable_weights = []
        self.nmodes = len(input_shapes)
        for i in range(self.nmodes):
            if self.h[i] is None:
                self.h[i] = np.random.random_integers(0, self.d-1, size=(input_shapes[i][1],))
                self.h[i] = K.variable(self.h[i], dtype='int64', name='h' + str(i))
            if self.s[i] is None:
                self.s[i] = (np.floor(np.random.uniform(0, 2, size=(input_shapes[i][1],)))*2-1).astype('float32')
                self.s[i] = K.variable(self.s[i], dtype='float32', name='s' + str(i))

        self.non_trainable_weights = [*self.h, *self.s]
        super(CompactBilinearPooling, self).build(input_shapes)

    def call(self, x, mask=None):
        if type(x) is not list or len(x) <= 1:
            raise Exception('CompactBilinearPooling must be called on a list of tensors '
                            '(at least 2). Got: ' + str(x))
        y = self.multimodal_compact_bilinear(x)
        if self.return_extra:
            return y+self.h+self.s
        return y

    def compute_output_shape(self, input_shape):
        assert type(input_shape) is list  # must have multiple input shape tuples
        shapes = []
        shapes.append(tuple([input_shape[0][0], self.d]))
        if self.return_extra:
            for s in input_shape: # v
                shapes.append(tuple([s[0], self.d]))
            for s in input_shape: # fft_v
                shapes.append(tuple([s[0], self.d]))
            shapes.append(tuple([s[0], self.d])) # acum_fft
            for i in range(self.nmodes): # h
                shapes.append(tuple([input_shape[i][1],1]))
            for i in range(self.nmodes): # v
                shapes.append(tuple([input_shape[i][1],1]))
        return shapes

    def get_config(self):
        config = {'d': self.d,
                  'h': self.h,
                  'return_extra': self.return_extra,
                  's': self.s}
        config = {'d': self.d}
        base_config = super(CompactBilinearPooling, self).get_config()
        return dict(list(base_config.items()) + list(config.items()))

    def multimodal_compact_bilinear(self, x):
        self.generate_sketch_matrix()
        x0, x1 = x

        v0 = tf.transpose(tf.sparse_tensor_dense_matmul(self.sparse_matrix[0], x0, adjoint_a=True, adjoint_b=True))
        v1 = tf.transpose(tf.sparse_tensor_dense_matmul(self.sparse_matrix[1], x1, adjoint_a=True, adjoint_b=True))

        fft_v0 = tf.fft(tf.complex(real=v0, imag=tf.zeros_like(v0)))
        fft_v1 = tf.fft(tf.complex(real=v1, imag=tf.zeros_like(v1)))

        accum = fft_v0 * fft_v1
        out = tf.real(tf.ifft(accum))
        if self.return_extra:
            return [out] + [v0, v1] + [fft_v0, fft_v1] + [accum]
        else:
            return out

    def generate_sketch_matrix(self):
        for i in range(len(self.sparse_matrix)):
            if self.sparse_matrix[i] is None:
                input_dim = self.h[i].get_shape().as_list()[0]
                indices = tf.concat([tf.expand_dims(tf.range(input_dim, dtype='int64'), -1),
                                     tf.expand_dims(self.h[i], -1)], 1)
                sparse_sketch_matrix = tf.sparse_reorder(
                    tf.SparseTensor(indices, self.s[i], [input_dim, self.d]))
                self.sparse_matrix[i] = sparse_sketch_matrix


class BilinearPooling(Layer):

    def __init__(self, **kwargs):

        # layer parameters
        self.inbound_nodes = []
        self.outbound_nodes = []
        self.constraints = {}
        self.regularizers = []
        self.trainable_weights = []
        self.non_trainable_weights = []
        self.supports_masking = True
        self.trainable = False
        self.uses_learning_phase = False
        self.input_spec = None  # compatible with whatever
        super(BilinearPooling, self).__init__(**kwargs)

    def build(self, input_shapes):
        self.trainable_weights = []
        self.nmodes = len(input_shapes)
        for i, s in enumerate(input_shapes):
            if s != input_shapes[0]:
                raise Exception('The input size of all vectors must be the same: '
                                'shape of vector on position ' + str(i) + ' (0-based) ' + str(
                    s) + ' != shape of vector on position 0 ' + str(input_shapes[0]))
        self.built = True

    def multimodal_bilinear(self, x):

        v0, v1 = x
        fft_v0 = tf.fft(tf.complex(real=v0, imag=tf.zeros_like(v0)))
        fft_v1 = tf.fft(tf.complex(real=v1, imag=tf.zeros_like(v1)))

        accum = fft_v0 * fft_v1
        return tf.real(tf.ifft(accum))

    def call(self, x, mask=None):
        if type(x) is not list or len(x) <= 1:
            raise Exception('BilinearPooling must be called on a list of tensors '
                            '(at least 2). Got: ' + str(x))
        return self.multimodal_bilinear(x)

    def get_config(self):
        base_config = super(BilinearPooling, self).get_config()
        return dict(list(base_config.items()))

    def compute_output_shape(self, input_shape):
        assert type(input_shape) is list  # must have mutiple input shape tuples
        return input_shape[0]




def bili_pooling(x):
    x0, x1 = x
    m, c = x0.get_shape().as_list()

    # Bilinear pooling
    b1 = K.reshape(x0, [-1, c, 1])
    b2 = K.reshape(x1, [-1, 1, c])
    d = K.reshape(K.batch_dot(b1, b2), [-1, c, c])
    d = K.sum(d, 1)

    # Flatten
    d = K.reshape(d, [-1, c * c])

    # # Normalize
    # d = K.sqrt(K.relu(d))
    # d = K.l2_normalize(d, -1)
    return d