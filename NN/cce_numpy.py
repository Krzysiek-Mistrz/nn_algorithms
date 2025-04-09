import math
import numpy as np

softmax_output = np.array([[0.7, 0.1, 0.2],
                  [0.1, 0.5, 0.4],
                  [0.02, 0.9, 0.08]])
#cat, dog, human classification
class_targets = [0, 1, 1]

neg_log = -np.log(softmax_output[range(len(softmax_output)), class_targets])

average_loss = np.mean(neg_log)

print(average_loss)