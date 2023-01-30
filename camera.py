import os
import pathlib
import time
from types import NoneType
import numpy as np
import cv2
import cv2.aruco as aruco
import sys
from kivy.graphics.texture import Texture
sys.path.insert(0, 'D:/4_KULIAH_S2/Summer_Project/summer_project_v2/mediapipe')
from mdp_main import Mediapipe
sys.path.insert(0, 'D:/4_KULIAH_S2/Summer_Project/summer_project_v2/activelearning')
from ActiveLearning import ActiveLearningClassifier
from DecisionModel import DecisionModel
from Dataset import Dataset
import tensorflow_datasets as tfds
from PIL import Image


class Camera:
    def __init__(self):
        self.mediapipe = Mediapipe()
        
        #active learning
        model = DecisionModel()
        ds = Dataset("./dataset/")

        self.activeAgent = ActiveLearningClassifier(model, ds)

    def main(self):
        pass

    def load_camera_1(self, camera_cv):
        """
        Grabs, decodes and returns the video frame from device (webcam, camera, ect).
        Convert it into texture to shown in Application (Kivy widget).  

        Args:
            frame (cvMat): Grabs, decodes and returns the video frame. 

        Returns:
            image texture (texture): OpenGL textures for Kivy images
        """
        ret, frame = camera_cv.read()   
        # self.image_frame = frame
        # frame = cv2.imread('cardbox.jpg')

        # Hand Gesture
        mdp_frame, gesture_status = self.mediapipe.main(frame)

        # Convert frame into texture for Kivy
        buffer = cv2.flip(mdp_frame, 0).tostring()                                                      
        texture = Texture.create(size=(mdp_frame.shape[1], mdp_frame.shape[0]), colorfmt='bgr')             
        texture.blit_buffer(buffer, colorfmt='bgr', bufferfmt='ubyte')
        # self.image.texture = texture

        return texture, gesture_status

    def load_camera_2(self, camera_cv_obj):
        """
        Grabs, decodes and returns the video frame from device (webcam, camera, ect).
        Convert it into texture to shown in Application (Kivy widget).  

        Args:
            frame (cvMat): Grabs, decodes and returns the video frame. 

        Returns:
            1. image texture (texture): Image texture that contain area for placec object
            2. ROI of image (texture): Portioan of image that contain object
        """
        # if(camera_cv_obj.isOpened()):
        ret, frame = camera_cv_obj.read()  
        # frame = cv2.imread('match+box.jpg')

        frame_gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)                                      # Convert into gray

        # Remove noise by blurring with a Gaussian filter ( kernel size = 7 )
        img_blur = cv2.GaussianBlur(frame_gray, (7, 7), sigmaX=0, sigmaY=0)             

        ###############     Threshold         ###############
        # apply basic thresholding -- the first parameter is the image
        # we want to threshold, the second value is is our threshold
        # check; if a pixel value is greater than our threshold (in this case, 200), we set it to be *black, otherwise it is *white*
        thresInv_adaptive = cv2.adaptiveThreshold(img_blur, 255, cv2.ADAPTIVE_THRESH_MEAN_C, cv2.THRESH_BINARY_INV, 11, 6)

        # Create small kernel for Erosion & Dilation
        # Dilation used to makes objects more visible 
        # Erosion used to removes floating pixels and thin lines so that only substantive objects remain
        # We used Erosion & Dilation 7 times to get best output
        small_kernel = np.ones((3, 3), np.uint8)
        thresInv_adaptive=cv2.dilate(thresInv_adaptive, small_kernel, iterations=7)
        thresInv_adaptive=cv2.erode(thresInv_adaptive, small_kernel, iterations=7)

        ###############     Find Contour          ###############
        # Get shape of frame
        h, w, c = frame.shape
        
        # Create Box where we place the object
        tgt_position = {}
        top_left = (w//2, 20)
        bottom_right = (w-20, h-75)
        cv2.rectangle(frame, top_left, bottom_right, (255, 0, 0), 2)

        # Create Box where we place the correct object
        cv2.putText(frame, 'Correct Object', ((w//2)-150, (h//2)-50-10), 0, 0.3, (0, 0, 255))
        center_cor = ((w//2)-85, h-212)
        top_left_cor = ((w//2)-150, (h//2)-50)
        bottom_right_cor = ((w//2)-20, h-125)
        cv2.rectangle(frame, top_left_cor, bottom_right_cor, (0, 255, 0), 2)
        # cv2.circle(frame, center_cor, 4, (0, 255, 0), -1)
        tgt_position[0] = center_cor

        # Create Box where we place the faulty object
        cv2.putText(frame, 'Faulty Object', ((w//2)-300, (h//2)-50-10), 0, 0.3, (0, 0, 255))
        center_flt = ((w//2)-235, h-212)
        top_left_flt = ((w//2)-300, (h//2)-50)
        bottom_right_flt = ((w//2)-170, h-125)
        cv2.rectangle(frame, top_left_flt, bottom_right_flt, (0, 0, 255), 2)
        # cv2.circle(frame, center_flt, 4, (0, 255, 0), -1)
        tgt_position[1] = center_flt

        # Create Box where we place the Camera
        cv2.putText(frame, 'Camera', (w-110-45, h-125-45-10), 0, 0.3, (0, 0, 255))
        center_flt = (w-110,  h-125)
        top_left_flt = (center_flt[0]-25, center_flt[1]-45)
        bottom_right_flt = (center_flt[0]+65, center_flt[1]+45)
        cv2.rectangle(frame, top_left_flt, bottom_right_flt, (0, 0, 255), 2)
        # cv2.circle(frame, center_flt, 4, (0, 255, 0), -1)
        tgt_position[2] = center_flt


        c_number = 0
        obj_position = {}
        obj_texture = {}
        # obj_id = 1
        contours, hierarchy = cv2.findContours(thresInv_adaptive, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)         # Get the contours and the hierarchy of the object in the frame
        for c in contours:
            # Draw a minimum area rotared rectangle around ROI
            # Input : Takes contours as input
            # Output : Box2D structure contains the following detail (center(x, y), (width, height), angle of rotatino)
            box = cv2.minAreaRect(c)                    
            (x, y), (width, height), angle = box    
            # Check if the ROI inside Box where we have to place the object 
            if ((int(x) >= w//2) and (int(x) <= w-20) and (int(y) >= 20) and (int(y) <= h-75)):
                if (int(width) > 40 and int(width) < 80 and int(height) > 40 and int(height) < 80):
                    c_number = c_number + 1
                    rect = cv2.boxPoints(box)                       # Convert the Box2D structure to 4 corner points 
                    box = np.int0(rect)                             # Converts 4 corner Points into integer type
                    cv2.drawContours(frame,[box],0,(0,0,255),2)   # Draw contours using 4 corner points
                    str_object_name = "Object " + str(c_number)
                    cv2.putText(frame, str_object_name, (box[0][0] + 2, box[0][1]+ 2), 0, 0.3, (0, 0, 255))
                    cv2.circle(frame, (int(x), int(y)), 4, (0, 255, 0), -1)       # Draw circle in the middle of contour
                    str_object = str(round(x, 2)) + ", " + str(round(y, 2))
                    cv2.putText(frame, str_object, (int(x), int(y) + 10), 0, 0.3, (0, 0, 255))
                    obj_position[c_number] = (int(x), int(y))            # Save coordinate of the object 

                    # Convert frame into texture for Kivy
                    # Update component in the UI with ROI of image
                    # frame_obj = frame[int(y-height*0.5)-30:int(y+height*0.5)+30, int(x-width*0.5)-30:int(x+width*0.5)+30]
                    frame_obj = frame[int(y-height*0.5):int(y+height*0.5), int(x-width*0.5):int(x+width*0.5)]
                    
                    # Show in the interface
                    if frame_obj is not None:
                        buffer = cv2.flip(frame_obj, 0).tobytes()
                    else:
                        continue
                    # try:
                    #     buffer = cv2.flip(frame_obj, 0).tostring()
                    # except TypeError:
                    #     continue
                    # if NoneType:
                    #     continue
                    # else:
                    #     buffer = cv2.flip(frame_obj, 0).tostring()
                    texture = Texture.create(size=(frame_obj.shape[1], frame_obj.shape[0]), colorfmt='bgr')
                    texture.blit_buffer(buffer, colorfmt='bgr', bufferfmt='ubyte')
                    obj_texture[c_number] = texture
                # else:
                #     obj_position[0] = (0, 0) 

        # Convert frame into texture for Kivy
        # frame = thresInv_adaptive
        buffer = cv2.flip(frame, 0).tostring()
        texture = Texture.create(size=(frame.shape[1], frame.shape[0]), colorfmt='bgr')
        texture.blit_buffer(buffer, colorfmt='bgr', bufferfmt='ubyte')
        # self.image_2.texture = texture

        return texture, obj_position, obj_texture, tgt_position

    def load_camera_3(self, camera_cv_obj):
        """
        Grabs, decodes and returns the video frame from device (webcam, camera, ect).
        Convert it into texture to shown in Application (Kivy widget).  

        Args:
            frame (cvMat): Grabs, decodes and returns the video frame. 

        Returns:
            1. image texture (texture): Image texture that contain area for placec object
            2. ROI of image (texture): Portioan of image that contain object
        """
        ret, frame = camera_cv_obj.read()  
        # frame = cv2.imread('box1.jpg')
        # prob = self.activeAgent.predict_or_request_label(frame)
        buffer = cv2.flip(frame, 0).tostring()
        texture = Texture.create(size=(frame.shape[1], frame.shape[0]), colorfmt='bgr')
        texture.blit_buffer(buffer, colorfmt='bgr', bufferfmt='ubyte')

        return texture, 1

        # if(camera_cv_obj.isOpened()):
        #     ret, frame = camera_cv_obj.read()  
        #     # frame = cv2.imread('box3.jpg')
        #     # h, w, channels = frame.shape
        #     # half = w//2
        #     # # this will be the second column
        #     # frame = frame[:, half:]   

        #     # Find Color
        #     # img_blur = cv2.GaussianBlur(frame, (5, 5), 20)
            
        #     # frame_hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)                                      # Convert into gray

        #     # # lower bound and upper bound for Green color
        #     # lower_bound = np.array([5, 30, 30])	 
        #     # upper_bound = np.array([20, 255, 255])

        #     # # find the colors within the boundaries
        #     # mask = cv2.inRange(frame_hsv, lower_bound, upper_bound)
        #     # ###############     Remove Noise         ###############
        #     # # Create small kernel 
        #     # kernel = np.ones((7, 7), np.uint8)
            
        #     # # Remove unnecessary noise from mask
        #     # mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel)
        #     # mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel)

            
        #     # # Segment only the detected region
        #     # segmented_img = cv2.bitwise_and(frame, frame, mask=mask)

        #     # # return mask
        #     # ###############     Find Contour          ###############
        #     # c_number = 0
        #     # box_position = {}
        #     # contours, hierarchy = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)         # Get the contours and the hierarchy of the object in the frame
            
        #     # for i, c in enumerate(contours):
        #     #     # if the contour has no other contours inside of it
        #     #     if hierarchy[0][i][2] == -1 :
        #     #         # if the size of the contour is greater than a threshold
        #     #         if  cv2.contourArea(c) > 3000:
        #     #             box = cv2.minAreaRect(c)                    
        #     #             (x, y), (width, height), angle = box  

        #     #             c_number += 1
        #     #             rect = cv2.boxPoints(box)
        #     #             box = np.int0(rect)
        #     #             frame = cv2.drawContours(frame, [box], 0, (0, 0, 255), 2)
        #     #             str_object_name = "Box " + str(c_number)
        #     #             cv2.putText(frame, str_object_name, (box[0][0] - 5, box[0][1] - 5), 0, 0.3, (0, 255, 0))
        #     #             cv2.circle(frame, (int(x), int(y)), 4, (0, 255, 0), -1)       # Draw circle in the middle of contour
        #     #             str_object = str(round(x, 2)) + ", " + str(round(y, 2))
        #     #             cv2.putText(frame, str_object, (int(x), int(y) + 10), 0, 0.3, (0, 0, 255))
        #     #             box_position[c_number] = (int(x), int(y))

        #     # Convert frame into texture for Kivy
        #     # return frame
        #     buffer = cv2.flip(frame, 0).tostring()
        #     texture = Texture.create(size=(frame.shape[1], frame.shape[0]), colorfmt='bgr')
        #     texture.blit_buffer(buffer, colorfmt='bgr', bufferfmt='ubyte')
        #     # self.image_2.texture = texture
            
        #     return texture

    def load_camera_calib(self, camera_cv_obj):
        """
        Grabs, decodes and returns the video frame from device (webcam, camera, ect).
        Convert it into texture to shown in Application (Kivy widget).  

        Args:
            frame (cvMat): Grabs, decodes and returns the video frame. 

        Returns:
            image texture (texture): OpenGL textures for Kivy images for Images Calibration 
        """
        ret, frame = camera_cv_obj.read()    
        # frame = self.image_frame_robot.copy()
        self.frame_calibration = frame.copy()
        buffer = cv2.flip(frame, 0).tostring()
        texture = Texture.create(size=(frame.shape[1], frame.shape[0]), colorfmt='bgr')
        texture.blit_buffer(buffer, colorfmt='bgr', bufferfmt='ubyte')
        # self.image_calibration.texture = texture

        return texture

    def take_image(self, camera_cv_obj):
        """
        Take image frame from camera and save it 
        For calibration perpose you have to capture multiple Aruco Marker image from different viewpoints and different position

        Args:
            frame (cvMat): Grabs, decodes and returns the video frame.

        Returns:
            frame (image): Save image into folder
        """
        ret, frame = camera_cv_obj.read() 
        # Create a new directory because it does not exist 
        path_file = "data/data_img_{}".format(time.strftime("%Y%m%d"))
        isExist = os.path.exists(path_file)
        if not isExist:
            os.makedirs(path_file)

        timestr = time.strftime("%Y%m%d_%H%M%S")
        name_file = path_file+"/IMG_{}.png".format(timestr)
        cv2.imwrite(name_file, frame)
        # wimg.source = "D:/4_KULIAH_S2/Summer_Project/"+name_file  
        src = "D:/4_KULIAH_S2/Summer_Project/summer_project_v2/" + name_file  

        return src

    def calibrate_camera(self, input_marker_length, input_marker_separation, input_marker_column, input_marker_row):
        """
        Calibrate Camera using Aruco Marker

        Args:
            Path file (path) : Directori of the image
            Marker Format (str) : Format of the image
            Marker Length (str) : Convert marker length from Kivy input into Float
            Marker Separation (str) : Convert marker separation from Kivy input into Float
            Marker Column (str) : Convert marker column from Kivy input into Int
            Marker Row (str) : Convert marker row from Kivy input into Int

        Returns:
            camera_ret (bool) : Camera Return
            camera_mtx (matrix) : Output 3x3 floating-point camera matrix
            camera_dist (vector) : Output vector of distortion coefficients
            camera_rvecs (vector) : Output vector of rotation vectors
            camera_tvecs (vector) : Output vector of translation vectors. 
        """
        path_file = "D:/4_KULIAH_S2/Summer_Project/summer_project_v2/data/data_img_{}".format(time.strftime("%Y%m%d"))
        isExist = os.path.exists(path_file)
        if isExist:
            IMAGES_DIR = path_file
            IMAGES_FORMAT = 'png'
            MARKER_LENGTH = input_marker_length
            MARKER_SEPARATION = input_marker_separation
            MARKER_COLUMN = input_marker_column
            MARKER_ROW = input_marker_row

            # Aruco 
            camera_ret, camera_mtx, camera_dist, camera_rvecs, camera_tvecs = self.aruco(MARKER_ROW, MARKER_COLUMN, IMAGES_DIR, IMAGES_FORMAT, MARKER_LENGTH, MARKER_SEPARATION)
            
            inf_status_camera_calib = "Done"
            # self.status_camera_calib.text = self.inf_status_camera_calib

            return inf_status_camera_calib, camera_ret, camera_mtx, camera_dist, camera_rvecs, camera_tvecs

    def aruco(self, marker_row, marker_column, dirpath, image_format, marker_length, marker_separation):
        """
        Calibrate Camera using Aruco Marker

        Args:
            dirpath (path) : Directori of the image
            image_format (str) : Format of the image
            marker_length (float) : The length of the marker in Meter
            marker_separation (float) : The lengths of separation between markers in Meter
            marker_column (int) : How many Marker in the column
            marker_row (int) : How many Marker in the row

        Returns:
            camera_ret (bool) : Camera Return
            camera_mtx (matrix) : Output 3x3 floating-point camera matrix
            camera_dist (vector) : Output vector of distortion coefficients
            camera_rvecs (vector) : Output vector of rotation vectors
            camera_tvecs (vector) : Output vector of translation vectors estimated for each pattern view.  
        """
        # Create Aruco Board use the function GridBoard_create, indicating the dimensions (how many markers in horizontal and vertical), the marker length, the marker separation, and the ArUco dictionary to be used.
        aruco_dict = aruco.Dictionary_get(aruco.DICT_ARUCO_ORIGINAL)                # Read Aruco Marker type from the library
        arucoParams = aruco.DetectorParameters_create()                             # Create a new set of DetectorParameters with default values. 
        board = aruco.GridBoard_create(int(marker_row), int(marker_column), float(marker_length), float(marker_separation), aruco_dict, firstMarker=10)   # meters

        # Find the ArUco markers inside each image
        counter, corners_list, id_list = [], [], []
        img_dir = pathlib.Path(dirpath)
        first = 0
        for img in img_dir.glob(f'*.{image_format}'):
            image_input = cv2.imread(str(img), 0)
            
            corners, ids, rejected = aruco.detectMarkers(
                image_input, 
                aruco_dict, 
                parameters=arucoParams
            )

            if first == 0:
                corners_list = corners
                id_list = ids
            else:
                corners_list = np.vstack((corners_list, corners))
                id_list = np.vstack((id_list,ids))
            first = first + 1
            counter.append(len(ids))

        counter = np.array(counter)
        ret, mtx, dist, rvecs, tvecs = aruco.calibrateCameraAruco(
            corners_list,       # ex: (63, 1, 4, 2) --> (corner*img*num_img)
            id_list,            # ex: (63, 1) --> (corner*img*num_img)
            counter, 
            board, 
            image_input.shape, 
            None, 
            None 
        )
        return ret, mtx, dist, rvecs, tvecs
        
    def transformation_matrix(self, camera_tvec, coordinate_robot, coordinate_marker_center):
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

        # print(f'Coordinate Camera: {coordinate_camera_final}')
        # print(f'Coordinate Robot: {coordinate_robot_final}')
        # print(f'Coordinate Marker Center: {coordinate_marker_center_final}')

        T = np.dot(np.linalg.inv(coordinate_camera_final), coordinate_robot_final)

        return T

    def final_matrix(self, camera_tvec, coordinate_robot, marker_coordinate_id, T):
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
            T = self.transformation_matrix(camera_tvec, coordinate_robot, marker_coordinate_id)                 # Calculate Transformation Matrix

        # calculate coordinate for every marker
        for i in camera_tvec:
            tvec = np.append(camera_tvec[int(i)], 1)
            tvec = np.array(tvec).reshape(1,4)
            coor = tvec @ T
            coordinate_marker_for_robot[i] = round(coor[0][0], 4), round(coor[0][1], 4), round(coor[0][2], 4)

        return coordinate_marker_for_robot

    def marker_coor(self, camera_mtx, camera_dist):
        """
        Find Marker ID, coordinate and translation vector from the single frame

        Args:
            frame (cvMat): Grabs, decodes and returns the frame that include Aruco Marker.
            camera_mtx (matrix): input 3x3 floating-point camera matrix
            camera_dist (vector): vector of distortion coefficients

        Returns:
            frame (image): Save image after detecting Aruco Marker into folder
            marker_id (list) : ID of the marker
            marker_center (list) : Coordinate center (X, Y) of the marker in the frame
            camera_tvec_id (list) : ID of translation vectors 
            camera_tvec_coor (list) : Coordinate (X, Y) of the marker in the frame
            camera_tvec (dict) : Marker translation vector (include ID and Coordinate)
        """
        # Create a new directory because it does not exist 
        path_file = "data/data_img_coordinate_{}".format(time.strftime("%Y%m%d"))
        isExist = os.path.exists(path_file)
        if not isExist:
            os.makedirs(path_file)

        timestr = time.strftime("%Y%m%d_%H%M%S")
        name_file = path_file+"/IMG_{}.png".format(timestr)
        cv2.imwrite(name_file, self.frame_calibration)
        src = "D:/4_KULIAH_S2/Summer_Project/summer_project_v2/"+name_file
        
        # Detecting ArUco markers
        aruco_dict = aruco.Dictionary_get(aruco.DICT_ARUCO_ORIGINAL)            # Specifying ArUco dictionary (We are using Original Dict)
        arucoParams = aruco.DetectorParameters_create()                         # Creating the parameters to the ArUco detector
        image = cv2.imread(name_file)                                           

        corners, ids, rejected = aruco.detectMarkers(image, aruco_dict, parameters=arucoParams)     # Detect the ArUco markers (We are using corners and ids)

        # verify *at least* one ArUco marker was detected
        marker_id = []
        marker_coordinate = []
        marker_coordinate_id = {}
        camera_tvec = {}
        camera_tvec_id = []
        camera_tvec_coor = []
        if len(corners) > 0:
            for i in range(0, len(ids)):
                # Estimate pose of each marker and return the values rvec and tvec---(different from those of camera coefficients)
                # Output : 
                # 1. array of output rotation vectors
                # 2. array of output translation vectors
                # 3. array of object points of all the marker corners
                marker_rvec, marker_tvec, marker_points = cv2.aruco.estimatePoseSingleMarkers(corners[i], 0.03, camera_mtx, camera_dist)          # We just using Marker translation vector     
                camera_tvec[ids[i, 0]] = marker_tvec                       # Save Marker translation vector   
                camera_tvec_id.append(ids[i, 0])                           # Save Id of Marker translation vector
                camera_tvec_coor.append(marker_tvec[0][0])                 # Save Coordinate of Marker translation vector
            
            # flatten the ArUco IDs list
            ids = ids.flatten()
            # loop over the detected ArUCo corners
            for (markerCorner, markerID) in zip(corners, ids):
                # extract the marker corners (which are always returned in
                # top-left, top-right, bottom-right, and bottom-left order)
                corners = markerCorner.reshape((4, 2))
                (topLeft, topRight, bottomRight, bottomLeft) = corners

                # convert each of the (x, y)-coordinate pairs to integers
                topRight = (int(topRight[0]), int(topRight[1]))
                bottomRight = (int(bottomRight[0]), int(bottomRight[1]))
                bottomLeft = (int(bottomLeft[0]), int(bottomLeft[1]))
                topLeft = (int(topLeft[0]), int(topLeft[1]))
                
                # draw the bounding box of the ArUCo detection
                cv2.line(image, topLeft, topRight, (0, 255, 0), 2)
                cv2.line(image, topRight, bottomRight, (0, 255, 0), 2)
                cv2.line(image, bottomRight, bottomLeft, (0, 255, 0), 2)
                cv2.line(image, bottomLeft, topLeft, (0, 255, 0), 2)
                
                # compute and draw the center (x, y)-coordinates of the ArUco marker
                marker_center_local = []
                cX = int((topLeft[0] + bottomRight[0]) / 2.0)
                cY = int((topLeft[1] + bottomRight[1]) / 2.0)
                cv2.circle(image, (cX, cY), 4, (0, 0, 255), -1)
                print(f'Center of Marker {markerID}: X: {cX} Y: {cY}')
                marker_center_local.append(cX)
                marker_center_local.append(cY)
                marker_coordinate_id[markerID] = marker_center_local
                marker_id.append(markerID)
                marker_coordinate.append(marker_center_local)

                # draw the ArUco marker ID on the image
                cv2.putText(image, str(markerID),
                    (topLeft[0], topLeft[1] - 15), cv2.FONT_HERSHEY_SIMPLEX,
                    0.5, (0, 255, 0), 2)
            
            timestr = time.strftime("%Y%m%d_%H%M%S")
            name_file = path_file+"/IMG_FINAL_{}.png".format(timestr)
            cv2.imwrite(name_file, image)
            # self.wimg.source = "D:/4_KULIAH_S2/Summer_Project/"+name_file
            src = "D:/4_KULIAH_S2/Summer_Project/summer_project_v2/"+name_file

            # print(self.marker_id)
            inf_status_marker_coordinate = "Done"
            # self.status_marker_coordinate.text = self.inf_status_marker_coordinate

            return camera_tvec, camera_tvec_id, camera_tvec_coor, marker_coordinate_id, marker_id, marker_coordinate, inf_status_marker_coordinate, src

    def final_calculation(self, position, obj_position, tgt_position, marker_coordinate_id, coordinate_marker_for_robot):
        """
        Calculating object coordinate in the frame for robot cordiate 

        Args:
            coordinate_marker_for_robot (dict) :  Dictionary of robot coordinate for every Marker (ID and Coordinate)
            marker_coordinate_id (dict) : Dictionary for every Marker ccenter coordinate (ID and Coordinate)

        Returns:
            Robot coordinate (X, Y, Z)
        """
        coor_marker1 = marker_coordinate_id[10]
        coor_marker2 = marker_coordinate_id[12]
        coor_marker3 = marker_coordinate_id[16]
        coor_rbt1 = coordinate_marker_for_robot[10]
        coor_rbt2 = coordinate_marker_for_robot[12]
        coor_rbt3 = coordinate_marker_for_robot[16]

        # Calculate Different X & Y
        diff_marker_y = abs(coor_marker1[0] - coor_marker2[0])
        diff_rbt_y = abs(coor_rbt1[1] - coor_rbt2[1])
        diff_y = diff_rbt_y/diff_marker_y

        diff_marker_x = abs(coor_marker1[1] - coor_marker3[1])
        diff_rbt_x = abs(coor_rbt1[0] - coor_rbt3[0])
        diff_x = diff_rbt_x/diff_marker_x
        # print(f'Marker: {diff_marker_y}, Robot: {diff_rbt_y}, Different X: {diff_x}, Different Y: {diff_y}')

        # Zero Coordinate
        coor_zero = coor_marker1[0] + coor_rbt1[1] * diff_y
        # print(f'Zero: {coor_zero}')

        # Object Coordinate
        corr_obj = obj_position[position]
        coor_new = [corr_obj[0], corr_obj[1]]

        # Coordinate
        coor_correct = tgt_position[0]
        coor_faulty = tgt_position[1]
        coor_cmr = tgt_position[2]
        

        # Object Position
        # Y Coordinate
        if(coor_new[0] > coor_marker1[0]):
            coor_y = abs(coor_marker1[0] - coor_new[0])
            coor_y = coor_y * diff_y
            coor_y = coor_rbt1[1] - coor_y
            # print(coor)
        else:
            coor_y = abs(coor_marker1[0] - coor_new[0])
            coor_y = coor_y * diff_y
            coor_y = coor_rbt1[1] + coor_y
            # print(coor)

        # X Coordinate
        if(coor_new[1] > coor_marker1[1]):
            coor_x = abs(coor_marker1[1] - coor_new[1])
            coor_x = coor_x * diff_x
            coor_x = coor_rbt1[0] - coor_x
            # print(coor)
        else:
            coor_x = abs(coor_marker1[1] - coor_new[1])
            coor_x = coor_x * diff_x
            coor_x = coor_rbt1[0] + coor_x
            # print(coor)

        # Camera Position
        # Y Coordinate
        if(coor_cmr[0] > coor_marker1[0]):
            cmr_coor_y = abs(coor_marker1[0] - coor_cmr[0])
            cmr_coor_y = cmr_coor_y * diff_y
            cmr_coor_y = coor_rbt1[1] - cmr_coor_y
            # print(coor)
        else:
            cmr_coor_y = abs(coor_marker1[0] - coor_cmr[0])
            cmr_coor_y = cmr_coor_y * diff_y
            cmr_coor_y = coor_rbt1[1] + cmr_coor_y
            # print(coor)

        # X Coordinate
        if(coor_cmr[1] > coor_marker1[1]):
            cmr_coor_x = abs(coor_marker1[1] - coor_cmr[1])
            cmr_coor_x = cmr_coor_x * diff_x
            cmr_coor_x = coor_rbt1[0] - cmr_coor_x
            # print(coor)
        else:
            cmr_coor_x = abs(coor_marker1[1] - coor_cmr[1])
            cmr_coor_x = cmr_coor_x * diff_x
            cmr_coor_x = coor_rbt1[0] + cmr_coor_x
            # print(coor)

        # Target Position
        # Y Coordinate
        if(coor_correct[0] > coor_marker1[0]):
            correct_coor_y = abs(coor_marker1[0] - coor_correct[0])
            correct_coor_y = correct_coor_y * diff_y
            correct_coor_y = coor_rbt1[1] - correct_coor_y
            # print(coor)
        else:
            correct_coor_y = abs(coor_marker1[0] - coor_correct[0])
            correct_coor_y = correct_coor_y * diff_y
            correct_coor_y = coor_rbt1[1] + correct_coor_y
            # print(coor)

        # X Coordinate
        if(coor_correct[1] > coor_marker1[1]):
            correct_coor_x = abs(coor_marker1[1] - coor_correct[1])
            correct_coor_x = correct_coor_x * diff_x
            correct_coor_x = coor_rbt1[0] - correct_coor_x
            # print(coor)
        else:
            correct_coor_x = abs(coor_marker1[1] - coor_correct[1])
            correct_coor_x = correct_coor_x * diff_x
            correct_coor_x = coor_rbt1[0] + correct_coor_x
            # print(coor)

        # Y Coordinate
        if(coor_faulty[0] > coor_marker1[0]):
            faulty_coor_y = abs(coor_marker1[0] - coor_faulty[0])
            faulty_coor_y = faulty_coor_y * diff_y
            faulty_coor_y = coor_rbt1[1] - faulty_coor_y
            # print(coor)
        else:
            faulty_coor_y = abs(coor_marker1[0] - coor_faulty[0])
            faulty_coor_y = faulty_coor_y * diff_y
            faulty_coor_y = coor_rbt1[1] + faulty_coor_y
            # print(coor)

        # X Coordinate
        if(coor_faulty[1] > coor_marker1[1]):
            faulty_coor_x = abs(coor_marker1[1] - coor_faulty[1])
            faulty_coor_x = faulty_coor_x * diff_x
            faulty_coor_x = coor_rbt1[0] - faulty_coor_x
            # print(coor)
        else:
            faulty_coor_x = abs(coor_marker1[1] - coor_faulty[1])
            faulty_coor_x = faulty_coor_x * diff_x
            faulty_coor_x = coor_rbt1[0] + faulty_coor_x
            # print(coor)
        
        
        return coor_x, coor_y, cmr_coor_x, cmr_coor_y, correct_coor_x, correct_coor_y, faulty_coor_x, faulty_coor_y

    def test_calculation(self, position, marker_coordinate_id, coordinate_marker_for_robot, obj_position, T, camera_matrix, dist_coeffs, camera_tvec, coordinate_robot):
        if (len(T) == 0):
            T = self.transformation_matrix(camera_tvec, coordinate_robot, marker_coordinate_id) 
        
        coor_marker = np.array([marker_coordinate_id[10], marker_coordinate_id[11], marker_coordinate_id[12], marker_coordinate_id[13], marker_coordinate_id[14], marker_coordinate_id[15], marker_coordinate_id[16], marker_coordinate_id[17], marker_coordinate_id[18]])
        coor_rbt = np.array([coordinate_marker_for_robot[10], coordinate_marker_for_robot[11], coordinate_marker_for_robot[12], coordinate_marker_for_robot[13], coordinate_marker_for_robot[14], coordinate_marker_for_robot[15], coordinate_marker_for_robot[16], coordinate_marker_for_robot[17], coordinate_marker_for_robot[18]])
        # print(f'shape marker coordinate: {coor_marker.shape}')
        # print(f'shape robot coordinate: {coor_rbt.shape}')
        # print(f'camera matrix: {np.array(camera_matrix)}')
        # print(f'camera dist: {np.array(dist_coeffs)}')

        _, rvec, tvec = cv2.solvePnP(coor_rbt, coor_marker, np.array(camera_matrix), np.array(dist_coeffs))

        
        # coor_rbt = coor_rbt.reshape(-1, 1, 3)
        # coor_marker = coor_marker.reshape(-1, 1, 2)
        # Object Coordinate
        corr_obj = obj_position[position]
        coor_new = np.array([corr_obj[0], corr_obj[1], 1])
        R, _ = cv2.Rodrigues(rvec)
        # T = np.array([tvec[0][0], tvec[1][0], tvec[2][0]])
        return R @ coor_new + T

    def click_event(self, event, x, y, flags, params):
        # checking for left mouse clicks
        if event == cv2.EVENT_LBUTTONDOWN:
    
            # displaying the coordinates
            # on the Shell
            print(x, ' ', y)
    
            # displaying the coordinates
            # on the image window
            font = cv2.FONT_HERSHEY_SIMPLEX
            cv2.putText(img, str(x) + ',' +
                        str(y), (x,y), font,
                        1, (255, 0, 0), 2)
            cv2.imshow('image', img)
    
        # checking for right mouse clicks     
        if event==cv2.EVENT_RBUTTONDOWN:
    
            # displaying the coordinates
            # on the Shell
            print(x, ' ', y)
    
            # displaying the coordinates
            # on the image window
            font = cv2.FONT_HERSHEY_SIMPLEX
            b = img[y, x, 0]
            g = img[y, x, 1]
            r = img[y, x, 2]
            cv2.putText(img, str(b) + ',' +
                        str(g) + ',' + str(r),
                        (x,y), font, 1,
                        (255, 255, 0), 2)
            cv2.imshow('image', img)

if __name__ == '__main__':
    camera_cv= cv2.VideoCapture('http://192.168.137.97:8080/video')
    camera_cv.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
    camera_cv.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)
    
    while True:
        ret, img = camera_cv.read()
        cv2.imshow('image', img)
        # wait for a key to be pressed to exit
        cv2.waitKey(0)
    
        # close the window
        cv2.destroyAllWindows()

    # camera = Camera()
    # frame = camera.load_camera_3(camera_cv)
    # cv2.namedWindow('image', cv2.WINDOW_NORMAL)
    # cv2.imshow('image', frame)
    # cv2.waitKey(0)
    # cv2.destroyAllWindows()

    # reading the image
    # img = cv2.imread('lena.jpg', 1)
  
    # # displaying the image
    # cv2.imshow('image', img)
  
    # # setting mouse handler for the image
    # # and calling the click_event() function
    # cv2.setMouseCallback('image', camera.click_event)
  
    # # wait for a key to be pressed to exit
    # cv2.waitKey(0)
  
    # # close the window
    # cv2.destroyAllWindows()