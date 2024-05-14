import keras
import sys
sys.path.append(r'D:\Neural Networks assignments\Project\Local Run\Helper Functions')
from Read_Data import read_data
from Augmentation import augmentation
from tensorflow.keras.utils import to_categorical
import matplotlib.pyplot as plt
import numpy as np
from sklearn.model_selection import train_test_split
from Inception import InceptionV3
import tensorflow as tf
from tensorflow.keras.layers import *
from keras.models import Model
from keras.models import load_model
from keras.callbacks import ModelCheckpoint, ReduceLROnPlateau
from tensorflow.keras.models import Sequential
from tensorflow.keras.utils import plot_model
from tensorflow.keras.callbacks import EarlyStopping
from tensorflow.keras.callbacks import Callback
import random
import os
import pandas as pd
import warnings

warnings.filterwarnings('ignore')
os.environ["TF_ENABLE_ONEDNN_OPTS"] = "0"



seed_constant = 42
np.random.seed(seed_constant)
random.seed(seed_constant)
tf.random.set_seed(seed_constant)



#### Reading Train Data ######
print("Start Reading Data")
reader=read_data(480,480)

features_Ak, labels_Ak = reader.create_dataset('Ak',0)
features_Ala_Idris, labels_Ala_Idris = reader.create_dataset('Ala_Idris',1)
features_Buzgulu, labels_Buzgulu = reader.create_dataset('Buzgulu',2)
features_Dimnit, labels_Dimnit = reader.create_dataset('Dimnit',3)
features_Nazli, labels_Nazli = reader.create_dataset('Nazli',4)

print("AK features Shape",features_Ak.shape)
print("AK labels Shape",labels_Ak.shape)

print("Nazli features Shape",features_Nazli.shape)
print("Nazli labels Shape",labels_Nazli.shape)

for i in range(3):

  plt.imshow(features_Ak[i,:,:,:])
  plt.axis('off')
  plt.show()
######################################

#### Augmentation ######
print("Start Augmenting Data")

augmenter=augmentation()
features_Ak, labels_Ak=augmenter.augment(features_Ak, labels_Ak,reader.IMAGE_HEIGHT,reader.IMAGE_WIDTH,140)
features_Ala_Idris, labels_Ala_Idris=augmenter.augment(features_Ala_Idris, labels_Ala_Idris,reader.IMAGE_HEIGHT,reader.IMAGE_WIDTH,140)
features_Buzgulu, labels_Buzgulu=augmenter.augment(features_Buzgulu, labels_Buzgulu,reader.IMAGE_HEIGHT,reader.IMAGE_WIDTH,140)
features_Dimnit, labels_Dimnit=augmenter.augment(features_Dimnit, labels_Dimnit,reader.IMAGE_HEIGHT,reader.IMAGE_WIDTH,140)
features_Nazli, labels_Nazli=augmenter.augment(features_Nazli, labels_Nazli,reader.IMAGE_HEIGHT,reader.IMAGE_WIDTH,140)

print("AK features Shape After Augmentaion",features_Ak.shape)
print("AK labels Shape After Augmentaion",labels_Ak.shape)

print("Nazli features Shape After Augmentaion",features_Nazli.shape)
print("Nazli labels Shape After Augmentaion",labels_Nazli.shape)


for i in range(70,73):

  plt.imshow(features_Ak[i,:,:,:])
  plt.axis('off')
  plt.show()
######################################

######### Combine Features and split into train and validate ############
print("Combining Data")

X_train=np.concatenate((features_Ak,features_Ala_Idris,features_Buzgulu,features_Dimnit,features_Nazli),axis=0)
y_train=np.concatenate((labels_Ak,labels_Ala_Idris,labels_Buzgulu,labels_Dimnit,labels_Nazli),axis=0)
del features_Ak
del features_Ala_Idris
del features_Buzgulu
del features_Dimnit
del features_Nazli

del labels_Ak
del labels_Ala_Idris
del labels_Buzgulu
del labels_Dimnit
del labels_Nazli

y_train = to_categorical(y_train)
X_train, X_val, y_train, y_val = train_test_split(X_train, y_train, test_size=0.15, random_state=42,shuffle=True,stratify=y_train)



print("Train Features Shape",X_train.shape)
print("Train Labels Shape",y_train.shape)

print("Validation Features Shape",X_val.shape)
print("Validation Labels Shape",y_val.shape)
######################################################


############ Modeling #################################
print("Start Modeling")
base_model=InceptionV3(include_top=False, weights='imagenet',input_shape=(reader.IMAGE_HEIGHT, reader.IMAGE_WIDTH, 3),pooling='avg')

for layer in base_model.layers:
    layer.trainable = False


x=base_model.output
x=Dense(1024,activation='relu')(x)
x=Dropout(0.2)(x)
x=Dense(512,activation='relu')(x)
x=Dropout(0.1)(x)
x=Dense(256,activation='relu')(x)
x=Dropout(0.1)(x)
x=Dense(64,activation='relu')(x)
main_output=Dense(5,activation='softmax')(x)

model=Model(inputs=base_model.input,outputs=[main_output])

model.compile(optimizer='adam',
              loss='categorical_crossentropy',
               metrics=['accuracy',
        tf.keras.metrics.Precision(),
        tf.keras.metrics.Recall(),
        tf.keras.metrics.TruePositives(name='true_positives'),
        tf.keras.metrics.TrueNegatives(name='true_negatives')])

print(model.summary())

print("Model Start Training")

checkpoint = ModelCheckpoint(r"D:\Neural Networks assignments\Project\Local Run\models\inception_check_final.h5", monitor='val_loss', verbose=1, save_best_only=True, mode='min')
reduce_lr = ReduceLROnPlateau(monitor='val_loss', factor=0.1, patience=5, min_lr=0.00001, verbose=1)


history=model.fit(
    X_train,
    y_train,
    epochs=80,
    validation_data=(X_val, y_val),
    batch_size = 32,
    callbacks=[checkpoint, reduce_lr]
)

print("Model Performance")

plt.figure(figsize=(12, 12))

plt.subplot(3, 2, 1)
plt.plot(history.history['accuracy'], label='train_accuracy')
plt.plot(history.history['val_accuracy'], label='val_accuracy')
plt.xlabel('epoch')
plt.ylabel('accuracy')
plt.legend()

plt.subplot(3, 2, 2)
plt.plot(history.history['loss'], label='train_loss')
plt.plot(history.history['val_loss'], label='val_loss')
plt.xlabel('epoch')
plt.ylabel('loss')
plt.legend()

plt.show()

# model=load_model(r"D:\Neural Networks assignments\Project\Local Run\models\inception_check_final.h5")
model.save(r"D:\Neural Networks assignments\Project\Local Run\Inception\ " + 'InceptionV5_black.h5')


############# Reading Test Data ######################
del X_train
del y_train
del X_val
del y_val
print('Reading Test Data')

test_data=reader.create_testset()
print("Test Data shape ",test_data.shape)
######################################################



############# Predicting ######################
print('Predicting')
predictions=model.predict(test_data)
predicted_classes = np.argmax(predictions, axis=1)

print("convert to dataframe")
filenames_without_extension = [filename.replace(".png", "") for filename in os.listdir(r'D:\Neural Networks assignments\Project\Local Run\Dataset\Test\Test')]
pred={
    'ID':filenames_without_extension,
    'label':predicted_classes
}

submit=pd.DataFrame(pred)
print('Top 5 rows in prediction',submit.head())
print('least 5 rows in prediction',submit.tail())


submit.to_csv(r"D:\Neural Networks assignments\Project\Local Run\Submission\\"+'inception_check_final.csv',index=False)
######################################################

print("Finished Sucessfully")



