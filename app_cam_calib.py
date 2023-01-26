import numpy as np
import os
import time
import cv2

from kivy.uix.gridlayout import GridLayout
from kivy.uix.label import Label
from kivy.lang import Builder
from kivy.app import App
from kivy.uix.textinput import TextInput
from kivy.uix.button import Button
from kivy.core.window import Window
from kivy.uix.image import Image
from kivy.clock import Clock
from kivy.uix.popup import Popup
from kivy.graphics.texture import Texture

class CameraCalibWidget(GridLayout):
    def __init__(self, **kwargs):
        super(CameraCalibWidget, self).__init__(**kwargs)
        # Read data from save file
        self.camera_tvec_id = self.read_coor_from_file("camera_tvec_id")
        self.camera_tvec_coor = self.read_coor_from_file("camera_tvec_coor")
        self.camera_mtx = self.read_coor_from_file("camera_mtx")
        self.camera_dist = self.read_coor_from_file("camera_dist")
        self.marker_id = self.read_coor_from_file("marker_id")
        self.marker_center = self.read_coor_from_file("marker_center")
        self.robot_marker_id = self.read_coor_from_file("robot_marker_id")
        self.robot_marker_coor = self.read_coor_from_file("robot_marker_coor")

        # Base Layout
        layout = GridLayout(cols = 1, rows = 2)
        
        # Top Layout
        layout_1 = GridLayout(
            cols = 2, 
            rows = 1
        )
        # Top-Left Layout
        layout_1_1 = GridLayout(
            cols = 1, 
            rows = 2
        )
        # Top-Left-1 Layout
        layout_1_1_1 = GridLayout(
            cols = 1, 
            rows = 1
        )
        # Clock.schedule_interval(self.load_video, 1.0/8.0).cancel()
        # self.camera_cv_robot.release()
        self.camera_cv_calibration = cv2.VideoCapture(0, cv2.CAP_DSHOW)
        self.camera_cv_calibration.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
        self.camera_cv_calibration.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)
        Clock.schedule_interval(self.load_video_calibration, 1.0/10.0)
        self.image_calibration = Image()
        layout_1_1_1.add_widget(self.image_calibration)
        layout_1_1.add_widget(layout_1_1_1)
        # Top-Left-2 Layout
        layout_1_1_2 = GridLayout(
            size_hint_y = None, 
            height = 70,
            padding = 10,
            cols = 1, 
            rows = 1
        )
        btn_take_img = Button(text="Take Image", font_size=16, size_hint=(.15, .15))
        btn_take_img.bind(on_press=self.take_image)
        layout_1_1_2.add_widget(btn_take_img)
        layout_1_1.add_widget(layout_1_1_2)
        layout_1.add_widget(layout_1_1)

        # Top-Right Layout
        layout_1_2 = GridLayout(
            size_hint_x = None,
            width = 350,
            padding = 5,
            cols = 1,
            rows = 3
        )
        # Top-Right-1 Layout
        layout_1_2_1 = GridLayout(
            size_hint_y = None,
            height = 300,
            cols = 1,
            rows = 1
        )
        wimg_name = 'aruco_template_desc.png'
        self.wimg = Image(source=wimg_name)
        layout_1_2_1.add_widget(self.wimg)
        layout_1_2.add_widget(layout_1_2_1)
        # Top-Right-2 Layout
        layout_1_2_2 = GridLayout(
            size_hint_y = None,
            height = 250,
            cols = 2,
            rows = 8
        )
        layout_1_2_2.add_widget(Label(text ="CALIBRATION"))
        layout_1_2_2.add_widget(Label(text =" "))
        layout_1_2_2.add_widget(Label(text ="Marker Row"))
        self.input_marker_row = TextInput(text='3', multiline=False)
        layout_1_2_2.add_widget(self.input_marker_row)
        layout_1_2_2.add_widget(Label(text ="Marker Column"))
        self.input_marker_column = TextInput(text='3', multiline=False)
        layout_1_2_2.add_widget(self.input_marker_column)
        layout_1_2_2.add_widget(Label(text ="Marker Length"))
        self.input_marker_length = TextInput(text='0.03', multiline=False)
        layout_1_2_2.add_widget(self.input_marker_length)
        layout_1_2_2.add_widget(Label(text ="Marker Separation"))
        self.input_marker_separation = TextInput(text='0.03', multiline=False)
        layout_1_2_2.add_widget(self.input_marker_separation)
        layout_1_2_2.add_widget(Label(text ="Image Taken"))
        self.img_taken_camera = Label(text ="---")
        layout_1_2_2.add_widget(self.img_taken_camera)
        layout_1_2_2.add_widget(Label(text ="Camera Calibration"))
        if (len(self.camera_dist) == 0 or len(self.camera_mtx) == 0):
            self.status_camera_calib = Label(text ="---")
        else:
            self.status_camera_calib = Label(text ="Done")
        layout_1_2_2.add_widget(self.status_camera_calib)
        layout_1_2_2.add_widget(Label(text ="Marker Coordinate"))
        if (len(self.marker_id) == 0 or len(self.marker_center) == 0):
            self.status_marker_coordinate = Label(text ="---")
        else:
            self.status_marker_coordinate = Label(text ="Done")
        layout_1_2_2.add_widget(self.status_marker_coordinate)
        layout_1_2.add_widget(layout_1_2_2)
        # Top-Right-3 Layout
        layout_1_2_3 = GridLayout(
            size_hint_y = None,
            height = 100,
            padding = 5,
            cols = 1,
            rows = 2
        )
        btn_take_img = Button(text="Camera Calibration", font_size=16, size_hint=(.15, .15))
        btn_take_img.bind(on_press=self.calibrate_camera_aruco)
        layout_1_2_3.add_widget(btn_take_img)
        
        btn_calibare_cmr = Button(text="Marker Coordinate", font_size=16, size_hint=(.15, .15))
        btn_calibare_cmr.bind(on_press=self.take_marker_coordinate)
        layout_1_2_3.add_widget(btn_calibare_cmr)
        layout_1_2.add_widget(layout_1_2_3)
        layout_1.add_widget(layout_1_2)
        layout.add_widget(layout_1) 

        # Bottom Layout
        layout_2 = GridLayout(
            size_hint_y = None, 
            height = 70,
            cols = 2
        )
        close_button = Button(text = "Close")
        save_button = Button(text = "Save")
        layout_2.add_widget(save_button)
        layout_2.add_widget(close_button)      
        layout.add_widget(layout_2) 

        # Instantiate the modal popup and display
        # popup = Popup(title ='Camera Calibration',
        #             content = layout,
        #             size_hint =(None, None), size =(1200, 800))  
        # popup.open()   

        # Attach close button press with popup.dismiss action
        # close_button.bind(on_press = popup.dismiss) 
        save_button.bind(on_press = self.save_camera_var)

    def load_video_calibration(self, *args):
        """
        Grabs, decodes and returns the video frame from device (webcam, camera, ect).
        Convert it into texture to shown in Application (Kivy widget).  

        Args:
            frame (cvMat): Grabs, decodes and returns the video frame. 

        Returns:
            image texture (texture): OpenGL textures for Kivy images for Images Calibration 
        """
        if(self.camera_cv_calibration.isOpened()):
            ret, frame = self.camera_cv_calibration.read()    
            # frame = self.image_frame_robot.copy()
            self.image_frame_calibration = frame.copy()
            buffer = cv2.flip(frame, 0).tostring()
            texture = Texture.create(size=(frame.shape[1], frame.shape[0]), colorfmt='bgr')
            texture.blit_buffer(buffer, colorfmt='bgr', bufferfmt='ubyte')
            self.image_calibration.texture = texture
    
    def take_image(self, *args):
        """
        Take image frame from camera and save it 
        For calibration perpose you have to capture multiple Aruco Marker image from different viewpoints and different position

        Args:
            frame (cvMat): Grabs, decodes and returns the video frame.

        Returns:
            frame (image): Save image into folder
        """
        # Create a new directory because it does not exist 
        path_file = "summer_project/data/data_img_{}".format(time.strftime("%Y%m%d"))
        isExist = os.path.exists(path_file)
        if not isExist:
            os.makedirs(path_file)

        timestr = time.strftime("%Y%m%d_%H%M%S")
        name_file = path_file+"/IMG_{}.png".format(timestr)
        cv2.imwrite(name_file, self.image_frame_calibration)
        self.wimg.source = "D:/4_KULIAH_S2/Summer_Project/"+name_file  
    
    def read_coor_from_file(self, name, *args):
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
    
    def calibrate_camera_aruco(self, *args):
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
        path_file = "D:/4_KULIAH_S2/Summer_Project/summer_project/data/data_img_{}".format(time.strftime("%Y%m%d"))
        isExist = os.path.exists(path_file)
        if isExist:
            IMAGES_DIR = path_file
            IMAGES_FORMAT = 'png'
            MARKER_LENGTH = float(self.input_marker_length.text)
            MARKER_SEPARATION = float(self.input_marker_separation.text)
            MARKER_COLUMN = int(self.input_marker_column.text)
            MARKER_ROW = int(self.input_marker_row.text)

            # Calibrate 
            self.camera_ret, self.camera_mtx, self.camera_dist, self.camera_rvecs, self.camera_tvecs = self.calibrate_aruco(MARKER_ROW, MARKER_COLUMN, IMAGES_DIR, IMAGES_FORMAT, MARKER_LENGTH, MARKER_SEPARATION)
            
            self.inf_status_camera_calib = "Done"
            self.status_camera_calib.text = self.inf_status_camera_calib

    def calibrate_aruco(self, marker_row, marker_column, dirpath, image_format, marker_length, marker_separation):
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

    def take_marker_coordinate(self, *args):
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
        path_file = "summer_project/data/data_img_coordinate_{}".format(time.strftime("%Y%m%d"))
        isExist = os.path.exists(path_file)
        if not isExist:
            os.makedirs(path_file)

        timestr = time.strftime("%Y%m%d_%H%M%S")
        name_file = path_file+"/IMG_{}.png".format(timestr)
        cv2.imwrite(name_file, self.image_frame_calibration)
        self.wimg.source = "D:/4_KULIAH_S2/Summer_Project/"+name_file
        
        # Detecting ArUco markers
        aruco_dict = aruco.Dictionary_get(aruco.DICT_ARUCO_ORIGINAL)            # Specifying ArUco dictionary (We are using Original Dict)
        arucoParams = aruco.DetectorParameters_create()                         # Creating the parameters to the ArUco detector
        image = cv2.imread(name_file)                                           

        corners, ids, rejected = aruco.detectMarkers(image, aruco_dict, parameters=arucoParams)     # Detect the ArUco markers (We are using corners and ids)

        # verify *at least* one ArUco marker was detected
        self.marker_id = []
        self.marker_center = []
        self.camera_tvec_id = []
        self.camera_tvec_coor = []
        if len(corners) > 0:
            for i in range(0, len(ids)):
                # Estimate pose of each marker and return the values rvec and tvec---(different from those of camera coefficients)
                # Output : 
                # 1. array of output rotation vectors
                # 2. array of output translation vectors
                # 3. array of object points of all the marker corners
                marker_rvec, marker_tvec, marker_points = cv2.aruco.estimatePoseSingleMarkers(corners[i], 0.03, self.camera_mtx, self.camera_dist)          # We just using Marker translation vector     
                self.camera_tvec[ids[i, 0]] = marker_tvec                       # Save Marker translation vector   
                self.camera_tvec_id.append(ids[i, 0])                           # Save Id of Marker translation vector
                self.camera_tvec_coor.append(marker_tvec[0][0])                 # Save Coordinate of Marker translation vector
            
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
                self.coordinate_marker_center[markerID] = marker_center_local
                self.marker_id.append(markerID)
                self.marker_center.append(marker_center_local)

                # draw the ArUco marker ID on the image
                cv2.putText(image, str(markerID),
                    (topLeft[0], topLeft[1] - 15), cv2.FONT_HERSHEY_SIMPLEX,
                    0.5, (0, 255, 0), 2)
            
            timestr = time.strftime("%Y%m%d_%H%M%S")
            name_file = path_file+"/IMG_FINAL_{}.png".format(timestr)
            cv2.imwrite(name_file, image)
            self.wimg.source = "D:/4_KULIAH_S2/Summer_Project/"+name_file

            print(self.marker_id)
            self.inf_status_marker_coordinate = "Done"
            self.status_marker_coordinate.text = self.inf_status_marker_coordinate

    def save_camera_var(self, *args):
        """
        Save parameter into text file

        Args:
            camera_mtx (matrix) : 3x3 floating-point camera matrix
            camera_dist (vector) : vector of distortion coefficients
            marker_id (list) : ID of the marker
            marker_center (list) : Coordinate center (X, Y) of the marker in the frame
            camera_tvec_id (list) : ID of translation vectors 
            camera_tvec_coor (list) : Coordinate (X, Y) of the marker in the frame

        Returns:
            -
        """
        self.save_coor_to_file(self.camera_mtx, "camera_mtx")
        self.save_coor_to_file(self.camera_dist, "camera_dist")
        self.save_coor_to_file(self.marker_id, "marker_id")
        self.save_coor_to_file(self.marker_center, "marker_center")
        self.save_coor_to_file(self.camera_tvec_id, "camera_tvec_id")
        self.save_coor_to_file(self.camera_tvec_coor, "camera_tvec_coor")

        # Change status
        self.status_camera_calib_final.text = self.inf_status_camera_calib

        # Release the camera
        Clock.schedule_interval(self.load_video_calibration, 1.0/10.0).cancel()
        self.camera_cv_calibration.release()
        self.camera_cv_robot = cv2.VideoCapture(0, cv2.CAP_DSHOW)
        self.camera_cv_robot.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
        self.camera_cv_robot.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)
        Clock.schedule_interval(self.load_video, 1.0/8.0).cancel()

class MyApp(App):
    def build(self):
        return CameraCalibWidget()
    
if __name__ == '__main__':
    MyApp().run()