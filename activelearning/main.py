import glob
import os
import pathlib
from ActiveLearning import ActiveLearningClassifier
from DecisionModel import DecisionModel
from Dataset import Dataset
import tensorflow_datasets as tfds
import numpy as np
import cv2
import tensorflow as tf

class ActiveLearningMain:
    def __init__(self) -> None:
        physical_devices = tf.config.list_physical_devices('GPU')
        try:
            tf.config.experimental.set_memory_growth(physical_devices[0], True)
        except:
            # Invalid device or cannot modify virtual devices once initialized.
            pass

        self.model = DecisionModel()
        self.ds = Dataset("D:/4_KULIAH_S2/Summer_Project/summer_project_v2/activelearning/dataset/")
        self.activeAgent = ActiveLearningClassifier(self.model, self.ds)

    def prep(self, frame):
        frame_gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)                                      # Convert into gray

        # Remove noise by blurring with a Gaussian filter ( kernel size = 7 )
        img_blur = cv2.GaussianBlur(frame_gray, (3, 3), sigmaX=0, sigmaY=0)             

        ###############     Threshold         ###############
        # apply basic thresholding -- the first parameter is the image
        # we want to threshold, the second value is is our threshold
        # check; if a pixel value is greater than our threshold (in this case, 200), we set it to be *black, otherwise it is *white*
        thresInv_adaptive = cv2.adaptiveThreshold(img_blur, 255, cv2.ADAPTIVE_THRESH_MEAN_C, cv2.THRESH_BINARY_INV, 13, 9)   # 15, 9

        # Create small kernel for Erosion & Dilation
        # Dilation used to makes objects more visible 
        # Erosion used to removes floating pixels and thin lines so that only substantive objects remain
        # We used Erosion & Dilation 7 times to get best output
        small_kernel = np.ones((3, 3), np.uint8)
        thresInv_adaptive=cv2.dilate(thresInv_adaptive, small_kernel, iterations=4)
        thresInv_adaptive=cv2.erode(thresInv_adaptive, small_kernel, iterations=4)
        # cv2.imshow('Prep frame', thresInv_adaptive)
        ###############     Find Contour          ###############
        # Get shape of frame
        h, w, c = frame.shape    

        c_number = 0
        # Marker cv2.RETR_TREE
        # Non Marker cv2.RETR_EXTERNAL
        contours, hierarchy = cv2.findContours(thresInv_adaptive, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)         # Get the contours and the hierarchy of the object in the frame
        for c in contours:
            # Draw a minimum area rotared rectangle around ROI
            # Input : Takes contours as input
            # Output : Box2D structure contains the following detail (center(x, y), (width, height), angle of rotatino)
            box = cv2.minAreaRect(c)                    
            (x, y), (width, height), angle = box    
            # Check if the ROI inside Box where we have to place the object 
            if (int(width) > 110 and int(width) < 160 and int(height) > 110 and int(height) < 160):
                c_number = c_number + 1
                rect = cv2.boxPoints(box)                       # Convert the Box2D structure to 4 corner points 
                box = np.int0(rect)                             # Converts 4 corner Points into integer type
                # cv2.drawContours(frame,[box],0,(0,0,255),2)   # Draw contours using 4 corner points
                # str_object_name = "Object " + str(width) + "," + str(height)
                # cv2.putText(frame, str_object_name, (box[0][0] + 2, box[0][1]+ 2), 0, 0.3, (0, 0, 255))
                # cv2.circle(frame, (int(x), int(y)), 4, (0, 255, 0), -1)       # Draw circle in the middle of contour
                # str_object = str(round(x, 2)) + ", " + str(round(y, 2))
                # cv2.putText(frame, str_object, (int(x), int(y) + 10), 0, 0.3, (0, 0, 255))
                    
                # Rotate Image
                rotation_mtx = cv2.getRotationMatrix2D((int(x), int(y)), angle, 1.0)
                rotated = cv2.warpAffine(frame, rotation_mtx, (int(w), int(h)), flags=cv2.INTER_LINEAR)
                
                # Convert the point after rotation
                new_point = np.dot(rotation_mtx, [int(x), int(y), 1])
                # cv2.circle(rotated, (int(new_point[0]-75), int(new_point[1])), 4, (0, 255, 0), -1)
                
                # Get sample color
                b, g, r = rotated[int(new_point[1]), int(new_point[0]-80)]
                # print(b, g, r)
                # cv2.circle(rotated, (int(new_point[0]-80), int(new_point[1])), 4, (0, 255, 0), -1)

                # Check if angle
                if r > 100 and g > 100 and b < 30:
                    top_left_flt = np.dot(rotation_mtx, [int(new_point[0]-75-145), int(new_point[1]-160), 1])
                    bottom_right_flt = np.dot(rotation_mtx, [int(new_point[0]-75+145), int(new_point[1]+130), 1])
                    # cv2.rectangle(rotated, (int(top_left_flt[0]), int(top_left_flt[1])), (int(bottom_right_flt[0]), int(bottom_right_flt[1])), (0, 0, 255), 2)
                    roi = rotated[int(top_left_flt[1]):int(bottom_right_flt[1]), int(top_left_flt[0]):int(bottom_right_flt[0])]
                    roi = cv2.resize(roi, (335, 235))
                else:
                    top_left_flt = np.dot(rotation_mtx, [int(new_point[0]+75-145), int(new_point[1]-120), 1])
                    bottom_right_flt = np.dot(rotation_mtx, [int(new_point[0]+75+145), int(new_point[1]+160), 1])
                    # cv2.rectangle(rotated, (int(top_left_flt[0]), int(top_left_flt[1])), (int(bottom_right_flt[0]), int(bottom_right_flt[1])), (0, 0, 255), 2)
                    roi = rotated[int(top_left_flt[1]):int(bottom_right_flt[1]), int(top_left_flt[0]):int(bottom_right_flt[0])]
                    roi = cv2.resize(roi, (335, 235))

                # cv2.imshow('Rotate frame', rotated)
                # cv2.imshow('Original frame', frame)
                # cv2.imshow('ROI frame', roi)
                # print(f'Angle: {angle}')
                # print(new_point[0])

                break
        
        return roi
    
    def train(self):
        img_dir = pathlib.Path("D:/4_KULIAH_S2/Summer_Project/summer_project_v2/activelearning/dataset/")

        # Get a list of all subdirectories
        directories = [d for d in os.listdir(img_dir) if os.path.isdir(os.path.join(img_dir, d))]

        x = []
        y = []

        for folder_name in directories:
            folder_path = os.path.join(img_dir, folder_name)

            # Get a list of all image files in the subdirectory
            image_files = glob.glob(os.path.join(folder_path, '*.png'))

            # Loop over the image files
            for image_file in image_files:
                # print(image_file)
                # Read the image file and convert it to a numpy array
                image = cv2.imread(image_file)
                image = self.prep(image)
                x.append(image)
                y.append(folder_name)

        x = np.array(x)
        y = np.array(y)
        split_idx = len(y)

        # print(x.shape)

        imgs = self.model.encode_images(x)

        # print(imgs.shape)

        for i in range(split_idx):
            self.ds.add_data(imgs[i].reshape(1, -1), np.array([y[i]]))

        self.activeAgent.retrain()
        self.ds.to_file()
        self.activeAgent.save('activelearning/model')

    
    def main(self, img):
        # self.activeAgent.load('activelearning/model')
        self.train()
        self.ds.from_file()
        label = self.activeAgent.predict(np.array([img]))
        # features, labels = self.ds.get_data()

        # correct = 0
        # unknown = 0
        # for i in range(10):
        #     label = self.activeAgent.predict_or_request_label(np.array([img]))
        #     if label is not None:
        #         if np.argmax(label) == labels[i]:
        #             correct += 1
        #     else:
        #         unknown += 1

        return label

if __name__ == '__main__':
    main = ActiveLearningMain()
    frame = cv2.imread('D:/4_KULIAH_S2/Summer_Project/summer_project_v2/activelearning/dataset/0/img1.png')
    # img = cv2.imread('D:/4_KULIAH_S2/Summer_Project/summer_project_v2/match2.jpg')

    # Prepared
    # cv2.imshow('Original frame', frame)
    # roi = main.prep(frame)
    # cv2.imshow('ROI', roi)

    # Main
    label = main.main(frame)
    print(label)

    # pre-process
    # frame_gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)                                      # Convert into gray

    # # Remove noise by blurring with a Gaussian filter ( kernel size = 7 )
    # img_blur = cv2.GaussianBlur(frame_gray, (3, 3), sigmaX=0, sigmaY=0)             

    # ###############     Threshold         ###############
    # # apply basic thresholding -- the first parameter is the image
    # # we want to threshold, the second value is is our threshold
    # # check; if a pixel value is greater than our threshold (in this case, 200), we set it to be *black, otherwise it is *white*
    # thresInv_adaptive = cv2.adaptiveThreshold(img_blur, 255, cv2.ADAPTIVE_THRESH_MEAN_C, cv2.THRESH_BINARY_INV, 13, 9)   # 15, 9

    # # Create small kernel for Erosion & Dilation
    # # Dilation used to makes objects more visible 
    # # Erosion used to removes floating pixels and thin lines so that only substantive objects remain
    # # We used Erosion & Dilation 7 times to get best output
    # small_kernel = np.ones((3, 3), np.uint8)
    # thresInv_adaptive=cv2.dilate(thresInv_adaptive, small_kernel, iterations=4)
    # thresInv_adaptive=cv2.erode(thresInv_adaptive, small_kernel, iterations=4)
    # # cv2.imshow('Prep frame', thresInv_adaptive)
    # ###############     Find Contour          ###############
    # # Get shape of frame
    # h, w, c = frame.shape    

    # c_number = 0
    # # Marker cv2.RETR_TREE
    # # Non Marker cv2.RETR_EXTERNAL
    # contours, hierarchy = cv2.findContours(thresInv_adaptive, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)         # Get the contours and the hierarchy of the object in the frame
    # for c in contours:
    #     # Draw a minimum area rotared rectangle around ROI
    #     # Input : Takes contours as input
    #     # Output : Box2D structure contains the following detail (center(x, y), (width, height), angle of rotatino)
    #     box = cv2.minAreaRect(c)                    
    #     (x, y), (width, height), angle = box    
    #     # Check if the ROI inside Box where we have to place the object 
    #     if (int(width) > 110 and int(width) < 160 and int(height) > 110 and int(height) < 160):
    #         c_number = c_number + 1
    #         rect = cv2.boxPoints(box)                       # Convert the Box2D structure to 4 corner points 
    #         box = np.int0(rect)                             # Converts 4 corner Points into integer type
    #         # cv2.drawContours(frame,[box],0,(0,0,255),2)   # Draw contours using 4 corner points
    #         # str_object_name = "Object " + str(width) + "," + str(height)
    #         # cv2.putText(frame, str_object_name, (box[0][0] + 2, box[0][1]+ 2), 0, 0.3, (0, 0, 255))
    #         # cv2.circle(frame, (int(x), int(y)), 4, (0, 255, 0), -1)       # Draw circle in the middle of contour
    #         # str_object = str(round(x, 2)) + ", " + str(round(y, 2))
    #         # cv2.putText(frame, str_object, (int(x), int(y) + 10), 0, 0.3, (0, 0, 255))
                
    #         # Rotate Image
    #         rotation_mtx = cv2.getRotationMatrix2D((int(x), int(y)), angle, 1.0)
    #         rotated = cv2.warpAffine(frame, rotation_mtx, (int(w), int(h)), flags=cv2.INTER_LINEAR)
            
    #         # Convert the point after rotation
    #         new_point = np.dot(rotation_mtx, [int(x), int(y), 1])
    #         # cv2.circle(rotated, (int(new_point[0]-75), int(new_point[1])), 4, (0, 255, 0), -1)
            
    #         # Get sample color
    #         b, g, r = rotated[int(new_point[1]), int(new_point[0]-80)]
    #         # print(b, g, r)
    #         # cv2.circle(rotated, (int(new_point[0]-80), int(new_point[1])), 4, (0, 255, 0), -1)

    #         # Check if angle
    #         if r > 100 and g > 100 and b < 30:
    #             top_left_flt = np.dot(rotation_mtx, [int(new_point[0]-75-145), int(new_point[1]-160), 1])
    #             bottom_right_flt = np.dot(rotation_mtx, [int(new_point[0]-75+145), int(new_point[1]+130), 1])
    #             # cv2.rectangle(rotated, (int(top_left_flt[0]), int(top_left_flt[1])), (int(bottom_right_flt[0]), int(bottom_right_flt[1])), (0, 0, 255), 2)
    #             roi = rotated[int(top_left_flt[1]):int(bottom_right_flt[1]), int(top_left_flt[0]):int(bottom_right_flt[0])]
    #             roi = cv2.resize(roi, (335, 235))
    #         else:
    #             top_left_flt = np.dot(rotation_mtx, [int(new_point[0]+75-145), int(new_point[1]-120), 1])
    #             bottom_right_flt = np.dot(rotation_mtx, [int(new_point[0]+75+145), int(new_point[1]+160), 1])
    #             # cv2.rectangle(rotated, (int(top_left_flt[0]), int(top_left_flt[1])), (int(bottom_right_flt[0]), int(bottom_right_flt[1])), (0, 0, 255), 2)
    #             roi = rotated[int(top_left_flt[1]):int(bottom_right_flt[1]), int(top_left_flt[0]):int(bottom_right_flt[0])]
    #             roi = cv2.resize(roi, (335, 235))

    #         # cv2.imshow('Rotate frame', rotated)
    #         # cv2.imshow('Original frame', frame)
    #         cv2.imshow('ROI frame', roi)
    #         print(roi.shape)
    #         # print(f'Angle: {angle}')
    #         # print(new_point[0])

    #         break
    
    # cv2.imshow('Contour frame', frame)
    cv2.waitKey(0)
    cv2.destroyAllWindows()
    

    # label = main.main(img)
    # print(label[0][0])
    # print(label[0][1])

