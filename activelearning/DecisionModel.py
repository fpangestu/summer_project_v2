import tensorflow as tf
from tensorflow.keras.applications.vgg16 import VGG16
from tensorflow.keras.applications.vgg16 import preprocess_input
from tensorflow.keras.models import Model
from tensorflow.keras.layers import Concatenate, GlobalAveragePooling2D
from tensorflow.keras.activations import sigmoid
from sklearn.neural_network import MLPClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import balanced_accuracy_score, accuracy_score
from scipy.stats import entropy
import numpy as np


class DecisionModel:
    def __init__(self, vgg_feature_blocks=[1, 2, 3, 4, 5], device="/GPU:0"):

        self.vgg_feature_blocks = vgg_feature_blocks
        self.device = device

        physical_devices = tf.config.list_physical_devices('GPU')
        try:
            tf.config.experimental.set_memory_growth(physical_devices[0], True)
        except:
            # Invalid device or cannot modify virtual devices once initialized.
            pass

        self.convnet = VGG16(include_top=False)
        inp = self.convnet.input

        o1 = GlobalAveragePooling2D()(self.convnet.get_layer(name="block1_conv2").output)
        o2 = GlobalAveragePooling2D()(self.convnet.get_layer(name="block2_conv2").output)
        o3 = GlobalAveragePooling2D()(self.convnet.get_layer(name="block3_conv3").output)
        o4 = GlobalAveragePooling2D()(self.convnet.get_layer(name="block4_conv3").output)
        o5 = GlobalAveragePooling2D()(self.convnet.get_layer(name="block5_conv3").output)

        block_selector_dict = dict(zip(range(1, 6), (o1, o2, o3, o4, o5)))
        oconcat = Concatenate()(
            [block_selector_dict[block_id] for block_id in self.vgg_feature_blocks])  # [o1,o2,o3,o4,o5])
        self.feature_model = Model([inp], [oconcat])

        self.classifier = None
        self.entro_threshold = None

    def train(self, data, test_fraction=0.2, false_pred_factor=1.0, **kwargs):
        self.classifier = MLPClassifier(**kwargs)
        x, y = data
        x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=test_fraction, stratify=y, shuffle=True)

        self.classifier.fit(x_train, y_train)
        y_test_pred = self.classifier.predict(x_test)
        print("Acc:", accuracy_score(y_test, y_test_pred), "Bal. Acc:", balanced_accuracy_score(y_test, y_test_pred))

        y_test_proba = self.classifier.predict_proba(x_test)
        entro = entropy(y_test_proba, axis=1)

        y_train_proba = self.classifier.predict_proba(x_train)
        entro_train = entropy(y_train_proba, axis=1)

        entro_tresholds = np.linspace(start=np.max([0.0, np.mean(entro_train) - 6.0 * np.std(entro_train)]),
                                      stop=np.mean(entro_train) + 6.0 * np.std(entro_train), num=1000, )
        decision_losses = []
        for th in entro_tresholds:
            confident = (entro < th)
            in_prob = confident.sum() / entro.size
            if confident.sum() <= 0:
                acc = 0
            else:
                acc = accuracy_score(y_test[confident, ...], y_test_pred[confident])

            decision_losses.append(1.0 * (1 - in_prob) + false_pred_factor * in_prob * (1 - acc))

        self.entro_threshold = entro_tresholds[np.argmin(np.array(decision_losses))]

    def encode_images(self, images):
        assert len(images.shape) == 4
        assert images.shape[-1] == 3
        return self.feature_model.predict(preprocess_input(images), verbose=0)

    def predict_encoded(self, features):
        return self.classifier.predict_proba(features)