from keras.layers import Input, Dense
from keras.models import Model
from compact_bilinear_pooling import CompactBilinearPooling
import numpy as np

input_dim1 = 2048
input_dim2 = 2048
output_dim = 16000

def bp(bottom1, bottom2):
    m = bottom1.shape[0]
    output_dim = bottom1.shape[-1] * bottom2.shape[-1]

    output = np.empty((m, output_dim), np.float32)
    for n in range(len(output)):
        output[n, ...] = np.outer(bottom1[n], bottom2[n]).reshape(-1)
    output = output.reshape((m, output_dim))

    return output


def test_kernel_approximation(model, m):
    print("Testing kernel approximation...")

    # Input values
    x = np.random.rand(m, input_dim1).astype(np.float32)
    y = np.random.rand(m, input_dim2).astype(np.float32)

    z = np.random.rand(m, input_dim1).astype(np.float32)
    w = np.random.rand(m, input_dim2).astype(np.float32)

    # Compact Bilinear Pooling results
    cbp_xy = cbp(model, x, y)
    cbp_zw = cbp(model, z, w)

    # (Original) Bilinear Pooling results
    bp_xy = bp(x, y)
    bp_zw = bp(z, w)

    # Check the kernel results of Compact Bilinear Pooling
    # against Bilinear Pooling
    cbp_kernel = np.sum(cbp_xy*cbp_zw, axis=1)
    bp_kernel = np.sum(bp_xy*bp_zw, axis=1)
    ratio = cbp_kernel / bp_kernel
    print("ratio between Compact Bilinear Pooling (CBP) and Bilinear Pooling (BP):")
    print(ratio)
    assert(np.all(np.abs(ratio - 1) < 2e-2))
    print("Passed.")

def mcb_model():
    input1 = Input(shape=(input_dim1,))
    input2 = Input(shape=(input_dim2,))
    output = CompactBilinearPooling(output_dim)([input1, input2])
    model = Model(inputs=[input1, input2], outputs=output)
    model.compile(optimizer='rmsprop', loss='binary_crossentropy')
    return model

def cbp(model, input1, input2):
    return model.predict(x=[input1, input2])

def main():
    model = mcb_model()
    test_kernel_approximation(model=model, m=24)
if __name__ == '__main__':
    main()