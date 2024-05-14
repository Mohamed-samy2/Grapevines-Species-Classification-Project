from tensorflow.keras.preprocessing.image import ImageDataGenerator
import cv2
import numpy as np
import os


TRAIN_DIR=r'D:\Neural Networks assignments\Project\Local Run\Dataset\Train'
TEST_DIR=r'D:\Neural Networks assignments\Project\Local Run\Dataset\Test\Test'
class read_data:

    def __init__(self,height,width):
        self.IMAGE_HEIGHT=height
        self.IMAGE_WIDTH=width

    def create_dataset(self,class_name, class_index):
        generator1 = ImageDataGenerator(zoom_range=[0.6, 0.7])
        generator2 = ImageDataGenerator(zoom_range=[0.9, 1.0])
        '''
        This function will extract the data of the selected classes and create the
        required dataset.
        Returns:
            features:          A list containing the extracted images.
            labels:            A list containing the indexes of the classes associated with the images.
        '''

        features = []
        labels = []

        files_list = os.listdir(os.path.join(TRAIN_DIR, class_name))
        i = 0
        for file_name in files_list:

            img = os.path.join(TRAIN_DIR, class_name, file_name)
            img = cv2.imread(img)

            white_pixels = (img[:, :, 0] > 250) & (img[:, :, 1] > 250) & (img[:, :, 2] > 250)
            percentage = np.sum(white_pixels) / (img.shape[0] * img.shape[1]) * 100
            img[white_pixels] = 0

            if percentage >= 80:
                img = generator1.random_transform(img)
            else:
                img = generator2.random_transform(img)

            img = cv2.resize(img, (self.IMAGE_HEIGHT, self.IMAGE_WIDTH))

            img = img / 255.0

            features.append(img)

            labels.append(class_index)
            i += 1
        print(f"Number of data in class {class_name} = ", i)

        features = np.array(features)
        labels = np.array(labels)

        return features, labels



    def create_testset(self):

        generator1 = ImageDataGenerator(zoom_range=[0.6, 0.7])
        generator2 = ImageDataGenerator(zoom_range=[0.9, 1.0])
        '''
        This function will extract the data of the selected classes and create the
        required dataset.
        Returns:
            features:          A list containing the extracted images.
            labels:            A list containing the indexes of the classes associated with the images.
        '''

        features = []

        files_list = os.listdir(TEST_DIR)
        i = 0
        for file_name in files_list:

            img = os.path.join(TEST_DIR, file_name)
            img = cv2.imread(img)

            white_pixels = (img[:, :, 0] > 250) & (img[:, :, 1] > 250) & (img[:, :, 2] > 250)
            percentage = np.sum(white_pixels) / (img.shape[0] * img.shape[1]) * 100
            img[white_pixels] = 0

            if percentage >= 80:
                img = generator1.random_transform(img)
            else:
                img = generator2.random_transform(img)

            img = cv2.resize(img, (self.IMAGE_HEIGHT, self.IMAGE_WIDTH))

            img = img / 255.0

            features.append(img)

            i += 1
        print(f"Number of test data = ", i)

        features = np.array(features)

        return features