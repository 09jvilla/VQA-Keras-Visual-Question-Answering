import tensorflow as tf
from keras import backend as K
from keras.engine.topology import Layer
import numpy as np

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
