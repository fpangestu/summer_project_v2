from DecisionModel import DecisionModel
from scipy.stats import entropy
import numpy as np
import pickle


class ActiveLearningClassifier:
    def __init__(self, decision_model: DecisionModel, dataset):
        self.decision_model = decision_model
        self.dataset = dataset

    def predict(self, image):
        assert self.decision_model.classifier is not None
        features = self.decision_model.encode_images(image)
        probability = self.decision_model.predict_encoded(features)
        return probability

    def predict_or_request_label(self, image):
        proba = self.predict(image)
        ent = entropy(proba,axis=1)
        if ent <= self.decision_model.entro_threshold or ent==0:
            return proba
        else:
            return None

    def retrain(self, false_pred_factor=5.0):
        self.decision_model.train(self.dataset.get_data(), test_fraction=0.2, false_pred_factor=false_pred_factor,
                                  hidden_layer_sizes=(300, 100), verbose=False, early_stopping=True,
                                  validation_fraction=0.3, n_iter_no_change=50, max_iter=500)

    def load(self, fname):
        f = open(fname, 'rb')
        tmp_dict = pickle.load(f)
        f.close()
        self.__dict__.update(tmp_dict)

    def save(self, fname):
        f = open(fname, 'wb')
        pickle.dump(self.__dict__, f, 2)
        f.close()
