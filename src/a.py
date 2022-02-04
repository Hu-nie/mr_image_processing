from numba import jit
import numpy as np
import time


def go_fast(a):
    trace = 0.0
    for i in range(a.shape[0]):
        trace += np.tanh(a[i, i])
    return a + trace

if __name__ == "__main__":
    x = np.arange(100).reshape(10, 10)

    # compile
    go_fast(x)

    # actual
    start = time.time()
    go_fast(x)
    end = time.time()
    print("걸린 시간 = %s" % (end - start))