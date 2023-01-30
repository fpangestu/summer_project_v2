from ActiveLearning import ActiveLearningClassifier
from DecisionModel import DecisionModel
from Dataset import Dataset
import tensorflow_datasets as tfds
import numpy as np

import tensorflow as tf

physical_devices = tf.config.list_physical_devices('GPU')
try:
    tf.config.experimental.set_memory_growth(physical_devices[0], True)
except:
    # Invalid device or cannot modify virtual devices once initialized.
    pass

mnist = tfds.load('mnist', split='test', shuffle_files=True, as_supervised=True)

model = DecisionModel()
ds = Dataset("./dataset/")
y = np.array([d[1].numpy() for d in mnist])
x = np.array([d[0].numpy() for d in mnist])

x=x[y<2,...]
y=y[y<2]
split_idx = 32

activeAgent = ActiveLearningClassifier(model, ds)

imgs = model.encode_images(x[:split_idx, ...].repeat(3, axis=-1))
print(imgs.shape)
for i in range(split_idx):
    ds.add_data(imgs[i, ...].reshape(1, -1), np.array([y[i], ]))

activeAgent.retrain()


correct = 0
unknown = 0
for i in range(1000):
    label = activeAgent.predict_or_request_label(x[split_idx + i, ...].repeat(3, axis=-1)[None, ...])
    if label is not None:
        if np.argmax(label) == y[split_idx + i]:
            correct += 1
    else:
        unknown += 1

    ds.add_data(model.encode_images(x[split_idx + i, ...].repeat(3, axis=-1)[None, ...]).reshape(1, -1),
                np.array([y[split_idx + i], ]))

    if i%100==0:
        activeAgent.retrain()

print(correct, unknown)
