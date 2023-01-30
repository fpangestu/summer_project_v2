import numpy as np
import os.path


class Dataset:
    def __init__(self, storage_path):
        self.storage_path = storage_path
        self.features = None
        self.labels = None

    def to_file(self):
        np.save(os.path.join(self.storage_path, "features.npy"), self.features)
        np.save(os.path.join(self.storage_path, "labels.npy"), self.labels)

    def from_file(self):
        self.features = np.load(os.path.join(self.storage_path, "features.npy"))
        self.labels = np.load(os.path.join(self.storage_path, "labels.npy"))

    def get_data(self):
        return self.features, self.labels

    def add_data(self, features, label):
        #print(features)
        #print(label)
        if self.features is None:
            self.features = np.copy(features)
            self.labels = np.copy(label)
        else:
            self.features = np.concatenate((self.features, np.copy(features)), axis=0)
            self.labels = np.concatenate((self.labels, np.copy(label)), axis=0)
