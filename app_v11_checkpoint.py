import threading
import queue
import os
import cv2
import time
import numpy as np
import sys
from kivy.factory import Factory
from kivy.app import App
from kivy.uix.gridlayout import GridLayout
from kivy.uix.label import Label
from kivy.lang import Builder
from kivy.uix.textinput import TextInput
from kivy.uix.button import Button
from kivy.core.window import Window
from kivy.uix.image import Image
from kivy.clock import Clock
from kivy.uix.widget import Widget
from kivy.uix.popup import Popup
from kivy.graphics.texture import Texture
from camera import Camera
from robot import Robot

## GRID LAYOUT
class MainWidget(GridLayout):
    def __init__(self, **kwargs):
        super(MainWidget, self).__init__(**kwargs)
        self.camera_tvec = {}
        self.coordinate_robot = {}
        self.marker_coordinate_id = {}
        self.coordinate_marker_for_robot = {}
        self.T = []
        self.camera = Camera()
        self.robot = Robot()
        self.command_queue = queue.Queue()

        # Image
        self.object_default = 'img/test.png'
        gest = ['img/9.jpg', 'img/1.jpg', 'img/3.jpg', 'img/5.jpg']
        self.noobj = 'img/noobj.PNG'

        # Information
        self.inf_status_camera_calib = ""
        self.inf_status_marker_coordinate = ""
        self.inf_coordinate_robot_final = ""
        self.camera_calib_stat = 0
        # self.inf_coordinate_marker_center_final = ""

        # camera
        self.camera_cv = cv2.VideoCapture(0, cv2.CAP_DSHOW)
        self.camera_cv.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
        self.camera_cv.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)

        self.camera_cv_obj = cv2.VideoCapture(1, cv2.CAP_DSHOW)
        self.camera_cv_obj.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
        self.camera_cv_obj.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)

        self.camera_cv_obj_2 = cv2.VideoCapture('http://192.168.137.97:8080/video')
        self.camera_cv_obj_2.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
        self.camera_cv_obj_2.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)
        
        # Read data from save file
        self.camera_tvec_id = self.read_coor_from_file("camera_tvec_id")
        self.camera_tvec_coor = self.read_coor_from_file("camera_tvec_coor")
        self.camera_mtx = self.read_coor_from_file("camera_mtx")
        self.camera_dist = self.read_coor_from_file("camera_dist")
        self.marker_id = self.read_coor_from_file("marker_id")
        self.marker_coor = self.read_coor_from_file("marker_coor")
        self.robot_marker_id = self.read_coor_from_file("robot_marker_id")
        self.robot_marker_coor = self.read_coor_from_file("robot_marker_coor")

        if(len(self.camera_tvec_id) != 0):
            for i in range(len(self.camera_tvec_id)):
                self.camera_tvec[self.camera_tvec_id[i].astype(int)] = self.camera_tvec_coor[i]

        if(len(self.robot_marker_id) != 0):
            for i in range(len(self.robot_marker_id)):
                self.coordinate_robot[self.robot_marker_id[i].astype(int)] = self.robot_marker_coor[i]
        
        if(len(self.marker_id) != 0):
            for i in range(len(self.marker_id)):
                self.marker_coordinate_id[self.marker_id[i].astype(int)] = self.marker_coor[i]

        # Calculate T
        if(len(self.camera_tvec) != 0 and len(self.coordinate_robot) != 0):
            # self.calculate_final_matrix()
            print()
            self.coordinate_marker_for_robot = self.camera.final_matrix(self.camera_tvec, self.coordinate_robot, self.marker_coordinate_id, self.T)  


        ###########################     WIDGET      ###########################
        Window.size = (1366, 768)
        Window.maximize()
        self.cols = 2
        # self.padding = 10
        ###########################     LEFT GRID      ###########################
        left_grid = GridLayout(cols = 1, rows = 2)
        
        # Left Grid 1
        left_grid_1 = GridLayout(cols = 3, rows = 1)

        left_grid_1_1 = GridLayout(cols = 1, rows = 1, padding = 4)
        # left_grid_1_1.add_widget(Label(text = "Camera 1", size_hint_y=0.1, halign='left', valign='bottom'))
        self.image = Image()
        left_grid_1_1.add_widget(self.image)
        left_grid_1.add_widget(left_grid_1_1)

        left_grid_1_2 = GridLayout(cols = 1, rows = 1, padding = 2)
        # left_grid_1_2.add_widget(Label(text = "Camera 2"))
        self.image_2 = Image()
        left_grid_1_2.add_widget(self.image_2)
        left_grid_1.add_widget(left_grid_1_2)

        left_grid_1_3 = GridLayout(cols = 1, rows = 1, padding = 4)
        # left_grid_1_3.add_widget(Label(text = "Camera 3"))
        self.image_3 = Image()
        left_grid_1_3.add_widget(self.image_3)
        left_grid_1.add_widget(left_grid_1_3)

        left_grid.add_widget(left_grid_1)

        # Left Grid 2
        left_grid_2 = GridLayout(
            size_hint_y = None,
            height = 300,
            cols = 11,
            rows = 1
        )
        left_grid_2_0 = GridLayout(
            size_hint_x = None,
            width = 60,
            cols = 1
        )
        left_grid_2_0.add_widget(Label(text=""))
        left_grid_2.add_widget(left_grid_2_0)
        
        for i in range(len(gest)):
            left_grid_2_1 = GridLayout(
                size_hint_x = None,
                width = 175,
                padding = 2,
                cols = 1,
                rows = 2
            )
            print(i)
            print(gest[i])
            left_grid_2_1.add_widget(Label(text = "Gesture "+ str(i+1), size_hint=(.2, .2)))
            left_grid_2_1.add_widget(Image(source=gest[i]))
            left_grid_2.add_widget(left_grid_2_1)

        left_grid_2_0 = GridLayout(
            size_hint_x = None,
            width = 35,
            padding = 5,
            cols = 1,
            rows = 5
        )
        left_grid_2_0.add_widget(Label(text=""))
        left_grid_2.add_widget(left_grid_2_0)

        left_grid_2_5 = GridLayout(
            size_hint_x = None,
            width = 175,
            padding = 2,
            cols = 1,
            rows = 2
        )
        left_grid_2_5.add_widget(Label(text = "Object 1", size_hint=(.2, .2)))
        self.object_1 = Image(source=self.noobj)
        left_grid_2_5.add_widget(self.object_1)
        left_grid_2.add_widget(left_grid_2_5)

        left_grid_2_6 = GridLayout(
            size_hint_x = None,
            width = 175,
            padding = 2,
            cols = 1,
            rows = 2
        )
        left_grid_2_6.add_widget(Label(text = "Object 2", size_hint=(.2, .2)))
        self.object_2 = Image(source=self.noobj)
        left_grid_2_6.add_widget(self.object_2)
        left_grid_2.add_widget(left_grid_2_6)

        left_grid_2_7 = GridLayout(
            size_hint_x = None,
            width = 175,
            padding = 2,
            cols = 1,
            rows = 2
        )
        left_grid_2_7.add_widget(Label(text = "Object 3", size_hint=(.2, .2)))
        self.object_3 = Image(source=self.noobj)
        left_grid_2_7.add_widget(self.object_3)
        left_grid_2.add_widget(left_grid_2_7)

        left_grid_2_8 = GridLayout(
            size_hint_x = None,
            width = 175,
            padding = 2,
            cols = 1,
            rows = 2
        )
        left_grid_2_8.add_widget(Label(text = "Object 4", size_hint=(.2, .2)))
        self.object_4 = Image(source=self.noobj)
        left_grid_2_8.add_widget(self.object_4)
        left_grid_2.add_widget(left_grid_2_8)
        
        left_grid.add_widget(left_grid_2)

        ###########################     RIGHT GRID      ###########################
        # Right Grid
        right_grid = GridLayout(
            # col_force_default=False,
            # col_default_width=280
            size_hint_x = None,
            width = 350,
            padding = 5,
            cols = 1,
            rows = 5
        )
        # Right Grid 1
        right_grid_1 = GridLayout(
            size_hint_y = None,
            height = 100,
            cols = 2,
            rows = 2
        )
        right_grid_1.add_widget(Label(text ="Camera Calibration"))
        if (self.inf_status_camera_calib == "Done" and self.inf_status_marker_coordinate == "Done"):
            self.status_camera_calib_final = Label(text ="Done")
        elif (len(self.marker_id) != 0 and len(self.marker_coor) != 0):
            self.status_camera_calib_final = Label(text ="Done")
        else:
            self.status_camera_calib_final = Label(text ="---")
        right_grid_1.add_widget(self.status_camera_calib_final)
        right_grid_1.add_widget(Label(text ="Robot Calibration"))
        if (self.inf_coordinate_robot_final == "Done"):
            self.status_robot_calib_final = Label(text ="Done")
        elif (len(self.robot_marker_id) != 0 and len(self.robot_marker_coor) != 0):
            self.status_robot_calib_final = Label(text ="Done")
        else:
            self.status_robot_calib_final = Label(text ="---")
        right_grid_1.add_widget(self.status_robot_calib_final)
        right_grid.add_widget(right_grid_1)
        # Right Grid 2
        right_grid_2 = GridLayout(
            size_hint_y = None,
            height = 70,
            cols = 1,
            rows = 1
        )
        btn_cmr_calibration = Button(text="Calibrate Camera", font_size=16, size_hint=(.99, .99))
        btn_cmr_calibration.bind(on_press=self.camera_calibration)
        right_grid_2.add_widget(btn_cmr_calibration)
        # right_grid_2.add_widget(Label(text =" "))
        right_grid.add_widget(right_grid_2)
        # Right Grid 3
        right_grid_3 = GridLayout(
            cols = 2,
            rows = 5
        )
        right_grid_3.add_widget(Label(text ="ROBOT CALIBRATION"))
        btn_robot_check = Button(text="Connect Robot", font_size=16)
        btn_robot_check.bind(on_press=self.robot_check)
        right_grid_3.add_widget(btn_robot_check)
        right_grid_3.add_widget(Label(text ="Status Robot"))
        self.status_robot = Label(text="")
        right_grid_3.add_widget(self.status_robot)
        right_grid_3.add_widget(Label(text ="Device Type"))
        self.status_robot_device_type = Label(text="")
        right_grid_3.add_widget(self.status_robot_device_type)
        right_grid_3.add_widget(Label(text ="Hardware Version"))
        self.status_robot_hardware_version = Label(text ="")
        right_grid_3.add_widget(self.status_robot_hardware_version)
        btn_robot_calibration = Button(text="Robot Calibration", font_size=16)
        btn_robot_calibration.bind(on_press=self.robot_calibration)
        right_grid_3.add_widget(btn_robot_calibration)
        btn_robot_test = Button(text="Test Robot", font_size=16)
        btn_robot_test.bind(on_press=self.robot_test)
        right_grid_3.add_widget(btn_robot_test)
        right_grid.add_widget(right_grid_3)

        # Right Grid 4
        right_grid_4 = GridLayout(
            cols = 1,
            rows = 1
        )

        status_value = ''
        self.terminal_robot = TextInput(multiline=True, text=status_value, disabled=True)
        right_grid_4.add_widget(self.terminal_robot)
        right_grid.add_widget(right_grid_4)

        # Right Grid 5
        right_grid_5 = GridLayout(
            size_hint_y = None,
            height = 70,
            cols = 1,
            rows = 1
        )

        self.btn_robot_str = Button(text="Start", font_size=20)
        self.btn_robot_str.bind(on_press=self.robot_str)
        right_grid_5.add_widget(self.btn_robot_str)
        right_grid.add_widget(right_grid_5)


        self.add_widget(left_grid)
        self.add_widget(right_grid)

        # check if the robot connected
        self.robot_check()

        # load camera
        Clock.schedule_interval(self.load_camera, 1.0/8.0)

    def robot_str(self, *args):
        if self.btn_robot_str.text == "Start":
            self.btn_robot_str.text = "Stop"
        else:
            self.btn_robot_str.text = "Start"

    def camera_calibration(self, *args):
        """
        Create new windows for Camera Calibration
        Using Aruco Marker for Calibrate the camera

        Args:
            marker Row (str) : How many marker in a row
            marker Column (str) : How many marker in column
            marker Length (str) : The length of the marker in Meter
            marker Separation (str) : The lengths of separation between markers in Meter

        Returns:
            -
        """
        self.camera_calib_stat = 1

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
        btn_take_img.bind(on_press=self.take_img)
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
        wimg_name = 'img/aruco_template_desc.png'
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
        if (len(self.marker_id) == 0 or len(self.marker_coor) == 0):
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
        btn_take_img.bind(on_press=self.calibrate_camera)
        layout_1_2_3.add_widget(btn_take_img)
        
        btn_calibare_cmr = Button(text="Marker Coordinate", font_size=16, size_hint=(.15, .15))
        btn_calibare_cmr.bind(on_press=self.marker_coordinate)
        layout_1_2_3.add_widget(btn_calibare_cmr)
        layout_1_2.add_widget(layout_1_2_3)
        layout_1.add_widget(layout_1_2)
        layout.add_widget(layout_1) 

        # Bottom Layout
        layout_2 = GridLayout(
            size_hint_y = None, 
            height = 70,
            cols = 1
        )
        close_button = Button(text = "Close")
        layout_2.add_widget(close_button)      
        layout.add_widget(layout_2) 

        # Instantiate the modal popup and display
        self.popup = Popup(title ='Camera Calibration',
                    content = layout,
                    size_hint =(None, None), size =(1200, 800))  
        self.popup.open()   

        # Attach close button press with popup.dismiss action
        close_button.bind(on_press = self.close_camera_calibration) 

    def load_camera(self, *args):
        """
        Grabs, decodes and returns the video frame from device (webcam, camera, ect).
        Convert it into texture to shown in Application (Kivy widget).  

        Args:
            frame (cvMat): Grabs, decodes and returns the video frame. 

        Returns:
            image texture (texture): OpenGL textures for Kivy images for Images Calibration 
        """
        gesture_status = ''
        
        if (self.camera_calib_stat == 0):
            self.image.texture, gesture_status = self.camera.load_camera_1(self.camera_cv)
            self.image_2.texture, obj_position, obj_texture, tgt_position = self.camera.load_camera_2(self.camera_cv_obj)
            self.image_3.texture, prob = self.camera.load_camera_3(self.camera_cv_obj_2)
        
            # print(f'Object Position: {obj_position}')
            # print(f'Box Position: {box_position}')
            # print(f'Object Texture: {obj_texture}')
            # print(f'Probability: {prob}')
            # target_position = {0: (0, 0), 1: (467, 235), 2: (385, 174), 3: (524, 140), 4: (437, 64)}
            # coor_x, coor_y, coor_x_dest, coor_y_dest = self.camera.final_calculation(1, obj_position, target_position, self.marker_coordinate_id, self.coordinate_marker_for_robot)
            # print(f'Object Coordinate: {(coor_x, coor_y)}')
            
            # coor = self.camera.test_calculation(1, self.marker_coordinate_id, self.coordinate_marker_for_robot, obj_position, self.T, self.camera_mtx, self.camera_dist, self.camera_tvec, self.coordinate_robot)
            # print(coor)

            # Object GUI
            frame = cv2.imread(self.noobj)
            buffer = cv2.flip(frame, 0).tostring()
            texture = Texture.create(size=(frame.shape[1], frame.shape[0]), colorfmt='bgr')
            texture.blit_buffer(buffer, colorfmt='bgr', bufferfmt='ubyte')

            self.object_1.texture = obj_texture.get(1, texture)
            self.object_2.texture = obj_texture.get(2, texture)
            self.object_3.texture = obj_texture.get(3, texture)
            self.object_4.texture = obj_texture.get(4, texture)

            if (self.btn_robot_str.text == 'Stop'):
                if (gesture_status != ''):
                    if(gesture_status == 'Gesture_1'):
                        coor_x, coor_y, cmr_coor_x, cmr_coor_y, correct_coor_x, correct_coor_y, faulty_coor_x, faulty_coor_y = self.camera.final_calculation(1, obj_position, tgt_position, self.marker_coordinate_id, self.coordinate_marker_for_robot)
                    elif(gesture_status == 'Gesture_2'):
                        coor_x, coor_y, cmr_coor_x, cmr_coor_y, correct_coor_x, correct_coor_y, faulty_coor_x, faulty_coor_y = self.camera.final_calculation(2, obj_position, tgt_position, self.marker_coordinate_id, self.coordinate_marker_for_robot)
                    elif(gesture_status == 'Gesture_3'):
                        coor_x, coor_y, cmr_coor_x, cmr_coor_y, correct_coor_x, correct_coor_y, faulty_coor_x, faulty_coor_y = self.camera.final_calculation(3, obj_position, tgt_position, self.marker_coordinate_id, self.coordinate_marker_for_robot)
                    elif(gesture_status == 'Gesture_4'):
                        coor_x, coor_y, cmr_coor_x, cmr_coor_y, correct_coor_x, correct_coor_y, faulty_coor_x, faulty_coor_y = self.camera.final_calculation(4, obj_position, tgt_position, self.marker_coordinate_id, self.coordinate_marker_for_robot)
                    else:
                        pass
                    print(f'Object Coordinate X: {coor_x}, Y: {coor_y}')
                    print(f'Camera Coordinate X: {cmr_coor_x}, Y: {cmr_coor_y}')
                    print(f'Corect Coordinate X: {correct_coor_x}, Y: {correct_coor_y}')
                    print(f'Faulty Coordinate X: {faulty_coor_x}, Y: {faulty_coor_y}')
                    # print(f'X dest: {coor_x_dest}, Y dest: {coor_y_dest}')
                    
                    #calculate coorx and y for box (box_position)
                    # self.command_queue.put(self.robot.move_object(round(coor_x, 4), round(coor_y, 4), round(coor_x_dest, 4), round(coor_y_dest, 4)))
                    status = self.robot.move_object(round(coor_x, 4), round(coor_y, 4), round(cmr_coor_x, 4), round(cmr_coor_y, 4), round(correct_coor_x, 4), round(correct_coor_y, 4), round(faulty_coor_x, 4), round(faulty_coor_y, 4))
                    print(status)
                    gesture_status = ''
                gesture_status = ''
        elif(self.camera_calib_stat == 1):
            self.image_calibration.texture = self.camera.load_camera_calib(self.camera_cv_obj)
        else:
            pass

        
    def execute_commands(self):
        while not self.command_queue.empty():
            command = self.command_queue.get()
            # send command to the robot here
            print(f'Executing command: {command}')
            self.command_queue.task_done()
    
    def take_img (self, *args):
        self.wimg.source = self.camera.take_image(self.camera_cv_obj)


    def close_camera_calibration(self, *args):
        """
        Save parameter into text file

        Args:
            camera_mtx (matrix) : 3x3 floating-point camera matrix
            camera_dist (vector) : vector of distortion coefficients
            marker_id (list) : ID of the marker
            marker_coor (list) : Coordinate center (X, Y) of the marker in the frame
            camera_tvec_id (list) : ID of translation vectors 
            camera_tvec_coor (list) : Coordinate (X, Y) of the marker in the frame

        Returns:
            -
        """
        self.save_coor_to_file(self.camera_mtx, "camera_mtx")
        self.save_coor_to_file(self.camera_dist, "camera_dist")
        self.save_coor_to_file(self.marker_id, "marker_id")
        self.save_coor_to_file(self.marker_coor, "marker_coor")
        self.save_coor_to_file(self.camera_tvec_id, "camera_tvec_id")
        self.save_coor_to_file(self.camera_tvec_coor, "camera_tvec_coor")

        # Change status
        self.status_camera_calib_final.text = self.inf_status_camera_calib
        self.camera_calib_stat = 0
        
        self.popup.dismiss()

    def marker_coordinate(self, *args):
        self.camera_tvec, self.camera_tvec_id, self.camera_tvec_coor, self.marker_coordinate_id, self.marker_id, self.marker_coor, self.inf_status_marker_coordinate, self.wimg.source = self.camera.marker_coor(self.camera_mtx, self.camera_dist)

    def calibrate_camera(self, *args):
        self.inf_status_camera_calib, self.camera_ret, self.camera_mtx, self.camera_dist, self.camera_rvecs, self.camera_tvecs = self.camera.calibrate_camera(float(self.input_marker_length.text), float(self.input_marker_separation.text), int(self.input_marker_column.text), int(self.input_marker_row.text))
        self.status_camera_calib.text = self.inf_status_camera_calib        

    def robot_calibration(self, *args):
        """
        Calibration robot using Aruco Marker as benchmark,
        We take a Marker where the position is in every corner of the paper.

        Args:
            -

        Returns:
            Robot coordinate for every Marker
        """
        # change status of camera
        self.camera_calib_stat == 2

        # Check if robot connected to the computer
        if self.status_robot.text != "Connected":
            layout = GridLayout(cols = 1, padding = 10)
    
            popupLabel = Label(text = "Robot isn't connected")
            closeButton = Button(text = "Close")
    
            layout.add_widget(popupLabel)
            layout.add_widget(closeButton)       
    
            # Instantiate the modal popup and display
            popup = Popup(title ='Robot Calibration',
                        content = layout,
                        size_hint =(None, None), size =(300, 150))  
            popup.open()   
    
            # Attach close button press with popup.dismiss action
            closeButton.bind(on_press = popup.dismiss)  
        else:
            layout = GridLayout(rows = 3, padding = 10)
            
            layout_1 = GridLayout(rows = 3,size_hint_y = None, height = 100, padding = 5)
            popup_label_1 = Label(text = "Make sure you hold the robor while calibrate it.")
            popup_label_2 = Label(text = "When you start calibrate the robot you have to finish it or it will not working.")
            start_button = Button(text = "Start calibrate")
            start_button.bind(on_press = self.start_calibrate)                                         # detach servo join for calibration
            layout_1.add_widget(popup_label_1)                                      # detach servo join for calibration
            layout_1.add_widget(popup_label_2)
            layout_1.add_widget(start_button)

            layout_2 = GridLayout(cols = 2, rows = 1, size_hint_y = None, height = 300, padding = 5)
            img = Image(source='img/uarm1.PNG')
            start_button = Button(text = "OKE")
            layout_2.add_widget(img)
            layout_2_2 = GridLayout(cols = 2, rows = 5)
            popup_label_2 = Label(text = "X")
            self.robot_calibration_x = TextInput(multiline=True, text='', disabled=True)
            popup_label_3 = Label(text = "Y")
            self.robot_calibration_y = TextInput(multiline=True, text='' , disabled=True)
            popup_label_4 = Label(text = "Z")
            self.robot_calibration_z = TextInput(multiline=True, text='' , disabled=True)
            popup_label_5 = Label(text = "Marker ID")
            self.robot_calibration_markerid = TextInput(multiline=True, text='')
            check_button = Button(text = "Get Coordinate")
            check_button.bind(on_press = self.print_coordinate)                                     # Get coordinate
            save_button = Button(text = "Save Coordinate")                                          
            save_button.bind(on_press = self.save_coordinate)                                       # Save coordinate into dict
            layout_2_2.add_widget(popup_label_2)
            layout_2_2.add_widget(self.robot_calibration_x)
            layout_2_2.add_widget(popup_label_3)
            layout_2_2.add_widget(self.robot_calibration_y)
            layout_2_2.add_widget(popup_label_4)
            layout_2_2.add_widget(self.robot_calibration_z)
            layout_2_2.add_widget(popup_label_5)
            layout_2_2.add_widget(self.robot_calibration_markerid)
            layout_2_2.add_widget(check_button)
            layout_2_2.add_widget(save_button)
            layout_2.add_widget(layout_2_2)

            layout_end = GridLayout(cols = 1, rows = 1, size_hint_y = None, height = 60, padding = 5)
            close_button = Button(text = "Close", font_size=16, size_hint=(.15, .15))
            layout_end.add_widget(close_button)
            
            layout.add_widget(layout_1)
            layout.add_widget(layout_2)
            layout.add_widget(layout_end)       

            # Instantiate the modal popup and display
            self.popup = Popup(title ='Robot Calibration',
                        content = layout,
                        size_hint =(None, None), size =(650, 550))  
            self.popup.open()   

            # Attach close button press with popup.dismiss action
            close_button.bind(on_press = self.close_robot_calibration)  

    def close_robot_calibration(self, *args):
        """
        Attach servo join in the robot.
        After attach the join, automatically calculate the robot coordinate matrix for every marker

        Args:
            -

        Returns:
            calculate_final_matrix (matrix) : Robot coordinate for the Marker (X, Y, Z)
        """
        # self.swift.set_servo_attach()
        self.robot.attach_robot()

        # Save to file
        print(self.robot_marker_id)
        print(self.robot_marker_coor)
        self.save_coor_to_file(self.robot_marker_id, "robot_marker_id")
        self.save_coor_to_file(self.robot_marker_coor, "robot_marker_coor")

        self.inf_coordinate_robot_final = "Done"
        self.status_robot_calib_final.text = self.inf_coordinate_robot_final

        # self.calculate_final_matrix()    
        self.coordinate_marker_for_robot = self.camera.final_matrix(self.camera_tvec, self.coordinate_robot, self.marker_coordinate_id, self.T)  
        
        # change status of camera
        self.camera_calib_stat == 0

        self.popup.dismiss()

    def robot_test(self, *args):
        """
        Test robot coordinate. Check if the robot moves into the right marker coordinate
        You have to calibrate the camera and the robot first before test it

        Args:
            input_marker_id (str) :  Marker ID, choose which marker, robot has to move

        Returns:
            coordinate_marker_for_robot (list) : coordinate robot (X, Y, Z)
        """
        if self.status_robot.text != "Connected":                                   # Check if the robot connected
            layout = GridLayout(cols = 1, padding = 10)
    
            popupLabel = Label(text = "Robot isn't connected")
            closeButton = Button(text = "Close")
    
            layout.add_widget(popupLabel)
            layout.add_widget(closeButton)       
    
            # Instantiate the modal popup and display
            popup = Popup(title ='Test Robot',
                        content = layout,
                        size_hint =(None, None), size =(300, 150))  
            popup.open()   
    
            # Attach close button press with popup.dismiss action
            closeButton.bind(on_press = popup.dismiss) 
        elif len(self.coordinate_robot) == 0:                                       # Check if the robot calibrated
            layout = GridLayout(cols = 1, padding = 10)
    
            popupLabel = Label(text = "Robot isn't calibrate")
            closeButton = Button(text = "Close")
    
            layout.add_widget(popupLabel)
            layout.add_widget(closeButton)       
    
            # Instantiate the modal popup and display
            popup = Popup(title ='Test Robot',
                        content = layout,
                        size_hint =(None, None), size =(300, 150))  
            popup.open()   
    
            # Attach close button press with popup.dismiss action
            closeButton.bind(on_press = popup.dismiss) 
        elif len(self.camera_tvec) == 0:                                            # Check if the camera calibrated
            layout = GridLayout(cols = 1, padding = 10)
    
            popupLabel = Label(text = "Camera isn't calibrate")
            closeButton = Button(text = "Close")
    
            layout.add_widget(popupLabel)
            layout.add_widget(closeButton)       
    
            # Instantiate the modal popup and display
            popup = Popup(title ='Test Robot',
                        content = layout,
                        size_hint =(None, None), size =(300, 150))  
            popup.open()   
    
            # Attach close button press with popup.dismiss action
            closeButton.bind(on_press = popup.dismiss) 
        else:
            self.robot.default_position()
            marker_id = []
            layout = GridLayout(rows = 3, padding = 10)

            layout_1 = GridLayout(rows = 2, size_hint_y = None, height = 80,  padding = 5)
            popup_label_marker = Label(text = "Marker Available")
            for i in self.camera_tvec:
                marker_id.append(i)
            popup_marker_id = Label(text = str(marker_id))
            layout_1.add_widget(popup_label_marker)
            layout_1.add_widget(popup_marker_id)
            layout.add_widget(layout_1)


            layout_2 = GridLayout(rows = 2, size_hint_y = None, height = 80, padding = 5)
            popup_label_1 = Label(text = "Marker ID")
            self.input_marker_id = TextInput(multiline=False, text='0', disabled=False)
            layout_2.add_widget(popup_label_1)
            layout_2.add_widget(self.input_marker_id)
            layout.add_widget(layout_2)
            
            layout_3 = GridLayout(cols = 2, padding = 5)
            close_button = Button(text = "Close")
            start_button = Button(text = "Start")
            layout_3.add_widget(start_button)
            layout_3.add_widget(close_button)      
            layout.add_widget(layout_3) 

            # Instantiate the modal popup and display
            popup = Popup(title ='Test Robot',
                        content = layout,
                        size_hint =(None, None), size =(400, 300))  
            popup.open()   

            # Attach close button press with popup.dismiss action
            close_button.bind(on_press = popup.dismiss) 
            start_button.bind(on_press = self.move_robot)

    def move_robot(self, *args):
        if (len(self.coordinate_marker_for_robot) == 0):    # check if Transformation Matrix still not calculated
            # self.calculate_final_matrix()                   # Calculate Transformation Matrix
            self.coordinate_marker_for_robot = self.camera.final_matrix(self.camera_tvec, self.coordinate_robot, self.marker_coordinate_id, self.T)

        self.robot.robot_test_move(self.input_marker_id.text, self.coordinate_marker_for_robot)

    def robot_check(self, *args):
        """
        Connect computer into the robot

        Args:
            -

        Returns:
            -
        """
        self.status_robot.text, self.status_robot_device_type.text, self.status_robot_hardware_version.text = self.robot.status()

        if (self.status_robot.text == "Connected"):
            # Set Robot position into default
            self.robot.default_position()

    def print_coordinate(self, *args):
        """
        Get robot Coordinate from the robot controller

        Args:
            -

        Returns:
            Robot coordinate for the Marker (X, Y, Z)
        """
        self.position = self.robot.get_position()                   # Get the data coordinate
        # Split to show it in the UI
        self.robot_calibration_x.text = str(self.position[0])
        self.robot_calibration_y.text = str(self.position[1])
        self.robot_calibration_z.text = str(self.position[2])
     
    def start_calibrate(self, *args):
        """
        Detach servo join in the robot

        Args:
            -

        Returns:
            -
        """
        self.robot_marker_id = []
        self.robot_marker_coor = []
        # self.swift.set_servo_detach()
        self.robot.detact_robot()

    def save_coordinate(self, *args):
        """
        Save coordinate into dictionary

        Args:
            position (list) : Robot coordinate
            robot_calibration_markerid (str) : marker ID from input user 

        Returns:
            coordinate_robot (dict) : ID and robot coordinate
        """
        self.robot_marker_id.append(int(self.robot_calibration_markerid.text))
        self.robot_marker_coor.append(self.position)
        self.coordinate_robot[int(self.robot_calibration_markerid.text)] = self.position
        if self.terminal_robot.text == "":
            status = "Marked ID: " + str(self.robot_calibration_markerid.text) + " | X: " + str( self.position[0]) + " Y: " + str(self.position[1]) + " Z: " + str(self.position[2]) + "\n"
        else:
            status = self.terminal_robot.text + "Marked ID: " + str(self.robot_calibration_markerid.text) + " | X: " + str(self.position[0]) + " Y: " + str(self.position[1]) + " Z: " + str(self.position[2]) + "\n"
        
        self.terminal_robot.text = status
        self.robot.detact_robot()

    def save_coor_to_file(self, data, name, *args):
        """
        Save value into file

        Args:
            data (list) : value to save
            name (str) : name of file

        Returns:
            -
        """
        path_file = "D:/4_KULIAH_S2/Summer_Project/summer_project_v2/data/data_coor_{}".format(time.strftime("%Y%m%d"))
        isExist = os.path.exists(path_file)
        if not isExist:
            # Create a new directory because it does not exist 
            os.makedirs(path_file)
            # print("Folder Created")
        
        np.savetxt(path_file + "/" + name + ".txt", data)

    def read_coor_from_file(self, name, *args):
        """
        Read value from file

        Args:
            name (str) : name of file

        Returns:
            -
        """
        path_file = "D:/4_KULIAH_S2/Summer_Project/summer_project_v2/data/data_coor_{}".format(time.strftime("%Y%m%d")) + "/" + name + ".txt"
        isExist = os.path.exists(path_file)
        if not isExist:
            return []
        
        return np.loadtxt(path_file)

class MyApp(App):
    def build(self):
        return MainWidget()

if __name__ == '__main__':
    main = MainWidget()
    thread = threading.Thread(target=main.execute_commands)
    thread.start()
    MyApp().run()