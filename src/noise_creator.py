import tensorflow as tf
import numpy as np
from logger import logger
def unison_shuffled(x1, x2, x3, y, args):
    assert len(x1) == len(y) == len(x2) == len(x3)
    p = np.random.permutation(args.batch_size)
    return [x1[p], x2[p], x3[p]], y[p]


