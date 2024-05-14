from keras.preprocessing.image import ImageDataGenerator
import numpy as np

class augmentation():

    def __init__(self):
        self.generator = ImageDataGenerator(
            rotation_range=130,
            horizontal_flip=True,
            vertical_flip=True)
    def augment(self,features,labels,height,width,shape):

        for img in features:
            augmented_img=self.generator.random_transform(img)
            features = np.append(features, augmented_img)

        features = np.reshape(features, (shape, height, width, 3))
        labels = np.append(labels, labels)

        return features,labels


