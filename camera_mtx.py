import numpy as np
import cv2
import cv2.aruco as aruco
import os
import pathlib
import time
import sys

class Camera:
    def __init__(self):
        pass

    def main(self):
        pass

    def transformation_matrix(coordinate_robot, camera_tvec, coordinate_marker_center):
        """
        Calculating Transformation Matrix

        Args:
            camera_tvec (dict) : Dictionary for every Marker translation vector (ID and translation vector)
            coordinate_robot (dict) :  Dictionary of robot coordinate for every Marker (ID and Coordinate)
            coordinate_marker_center (dict) : Dictionary for every Marker ccenter coordinate (ID and Coordinate)

        Returns:
            T (matrix) : Camera Translation matrix
        """
        coordinate_marker_center_final = []
        coordinate_camera_final = []
        coordinate_robot_final = []

        for i in coordinate_robot:
            for j in camera_tvec:
                if i == j:
                    coordinate_camera_final.append(np.append(camera_tvec[i], 1))
                    coordinate_robot_final.append(coordinate_robot[i])
        for i in coordinate_robot:
            for j in coordinate_marker_center:
                if i == j:
                    coordinate_marker_center_final.append(coordinate_marker_center[i])

        print(f'Coordinate Camera: {coordinate_camera_final}')
        print(f'Coordinate Robot: {coordinate_robot_final}')
        print(f'Coordinate Marker Center: {coordinate_marker_center_final}')

        T = np.dot(np.linalg.inv(coordinate_camera_final), coordinate_robot_final)

        return T

    def final_matrix(self, camera_tvec, T=0):
        """
        Calculating robot coordinate from T and marker translation vector

        Args:
            T (matrix) : Translation matrix
            camera_tvec (vector) : Marker translation vector

        Returns:
            coordinate_marker_for_robot (list) : coordinate robot (X, Y, Z)
        """
        coordinate_marker_for_robot = {}
        
        if (len(T) == 0):
            T = self.transformation_matrix()                 # Calculate Transformation Matrix

        # calculate coordinate for every marker
        for i in self.camera_tvec:
            tvec = np.append(camera_tvec[int(i)], 1)
            tvec = np.array(tvec).reshape(1,4)
            coor = tvec @ self.T
            coordinate_marker_for_robot[i] = round(coor[0][0], 4), round(coor[0][1], 4), round(coor[0][2], 4)

        return coordinate_marker_for_robot
