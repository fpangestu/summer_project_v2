import os
import pathlib
import numpy as np
import time

class LoadFile:
    def __init__(self):
        pass

    def save_file(self, name, data):
        """
        Save value into file

        Args:
            data (list) : value to save
            name (str) : name of file

        Returns:
            -
        """
        path_file = "D:/4_KULIAH_S2/Summer_Project/summer_project/data/data_coor_{}".format(time.strftime("%Y%m%d"))
        isExist = os.path.exists(path_file)
        if not isExist:
            # Create a new directory because it does not exist 
            os.makedirs(path_file)
            # print("Folder Created")
        
        np.savetxt(path_file + "/" + name + ".txt", data)

    def load_file(self, name, data):
        """
        Read value from file

        Args:
            name (str) : name of file

        Returns:
            -
        """
        path_file = "D:/4_KULIAH_S2/Summer_Project/summer_project/data/data_coor_{}".format(time.strftime("%Y%m%d")) + "/" + name + ".txt"
        isExist = os.path.exists(path_file)
        if not isExist:
            return []
        
        return np.loadtxt(path_file)
