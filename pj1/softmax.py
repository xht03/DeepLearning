import numpy as np
import matplotlib.pyplot as plt
import idx2numpy

data = idx2numpy.convert_from_file('data/train-images.idx3-ubyte')
labels = idx2numpy.convert_from_file('data/train-labels.idx1-ubyte')
    
data = data.reshape(data.shape[0], -1).T
labels = labels.reshape(1, labels.shape[0])
on_hot_labels = np.zeros((10, labels.shape[1]))
on_hot_labels[labels[0], np.arange(labels.shape[1])] = 1

print(labels[0:10])
print(on_hot_labels[:, 0:10])