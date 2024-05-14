import sys
sys.path.append(r'D:\Neural Networks assignments\Project\Local Run\Helper Functions')
from Read_Data import read_data
from Augmentation import augmentation
from tensorflow.keras.utils import to_categorical
import matplotlib.pyplot as plt
import numpy as np
from sklearn.model_selection import train_test_split
import Inception
import tensorflow as tf
from tensorflow.keras.layers import *
from keras.models import Model
from tensorflow.keras.models import Sequential
from tensorflow.keras.utils import plot_model
from tensorflow.keras.callbacks import EarlyStopping
from tensorflow.keras.callbacks import Callback
import torch.nn as nn
import torch
from torch.utils.data import ConcatDataset
from torchvision import transforms,datasets
from torch.utils.data import TensorDataset, DataLoader,Subset
import random
import os
import pandas as pd
import warnings

from Model import ViT

warnings.filterwarnings('ignore')
os.environ["TF_ENABLE_ONEDNN_OPTS"] = "0"


seed_constant = 42
np.random.seed(seed_constant)
random.seed(seed_constant)
tf.random.set_seed(seed_constant)
torch.manual_seed(seed_constant)
torch.cuda.manual_seed(seed_constant)
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")



def calculate_accuracy_and_validation(model, data_loader, device):
    model.eval()  # Set model to evaluation mode
    correct_predictions = 0
    total_predictions = 0
    total_loss = 0.0
    num_samples = 0

    with torch.no_grad():
        for images, labels in data_loader:
            images = images.to(device)
            labels = labels.to(device)

            # Forward pass
            outputs = model(images)
            loss = criterion(outputs, labels)

            # Update total loss and number of samples
            total_loss += loss.item() * images.size(0)

            # Get predicted class
            _, predicted = torch.max(outputs, 1)

            # Update counts
            correct_predictions += (predicted == labels).sum().item()
            total_predictions += labels.size(0)

    # Calculate accuracy
    accuracy = correct_predictions / total_predictions
    validation_loss = total_loss / len(val_dataloader.dataset)
    return accuracy,validation_loss

#### Reading Train Data ######
print("Start Reading Data")
train_dir=r'D:\Neural Networks assignments\Project\Local Run\Dataset\Train'
test_dir=r'D:\Neural Networks assignments\Project\Local Run\Dataset\Test'

def create_dataloaders(
    train_dir: str,
    test_dir: str,
    transform: transforms.Compose,
    batch_size: int,
):

  # Use ImageFolder to create dataset(s)
  train_data = datasets.ImageFolder(train_dir, transform=transform)
  test_data = datasets.ImageFolder(test_dir, transform=transform)

  # Get class names
  class_names = train_data.classes

  # Turn images into data loaders
  train_dataloader = DataLoader(
      train_data,
      batch_size=batch_size,
      shuffle=True,
      pin_memory=True,
  )
  test_dataloader = DataLoader(
      test_data,
      batch_size=batch_size,
      shuffle=False,
      pin_memory=True,
  )

  return train_dataloader, test_dataloader, class_names

IMG_SIZE = 416

# Create transform pipeline manually
manual_transforms = transforms.Compose([
    transforms.Resize((IMG_SIZE, IMG_SIZE)),
    transforms.ToTensor(),
])
print(f"Manually created transforms: {manual_transforms}")


BATCH_SIZE = 32

# Create data loaders
train_dataloader, test_dataloader, class_names = create_dataloaders(
    train_dir=train_dir,
    test_dir=test_dir,
    transform=manual_transforms,
    batch_size=BATCH_SIZE
)


print('Class names ',class_names)

# Get a batch of images
image_batch, label_batch = next(iter(train_dataloader))

# Get a single image from the batch
image, label = image_batch[0], label_batch[0]

# View the batch shapes
print(image.shape, label)

# Plot image with matplotlib
plt.imshow(image.permute(1, 2, 0))
plt.title(class_names[label])
plt.axis(False)
plt.show()


######################################

#### Augmentation ######

augmentation_transforms = transforms.Compose([
    transforms.Resize((IMG_SIZE, IMG_SIZE)),
    transforms.RandomHorizontalFlip(),
    transforms.RandomVerticalFlip(),
    transforms.ToTensor(),
])

augmented_dataset = train_dataloader.dataset
augmented_dataset.transform = augmentation_transforms

augmented_train_loader = DataLoader(augmented_dataset, batch_size=BATCH_SIZE, shuffle=True)

image_batch, label_batch = next(iter(augmented_train_loader))

# Get a single image from the batch
image, label = image_batch[0], label_batch[0]

# View the batch shapes
print(image.shape, label)

# Plot image with matplotlib
plt.imshow(image.permute(1, 2, 0))
plt.title(class_names[label])
plt.axis(False)
plt.show()

######################################

######### Combine Features and split into train and validate ############
print("Combining Data")


dataset1=train_dataloader.dataset
dataset2=augmented_train_loader.dataset

combined_dataset = ConcatDataset([dataset1, dataset2])
train_dataloader = DataLoader(combined_dataset, batch_size=BATCH_SIZE, shuffle=True)

num_samples = len(train_dataloader.dataset)

num_val_samples = int(0.14 * num_samples)

# Generate random indices for the validation set
indices = np.random.permutation(num_samples)

train_indices = indices[num_val_samples:]
val_indices = indices[:num_val_samples]

train_subset = Subset(train_dataloader.dataset, train_indices)
val_subset = Subset(train_dataloader.dataset, val_indices)

train_dataloader = DataLoader(train_subset, batch_size=BATCH_SIZE, shuffle=True)
val_dataloader = DataLoader(val_subset, batch_size=BATCH_SIZE, shuffle=True)

######################################################

# ############ Modeling #################################
print("Start Modeling")

model=ViT(name='B_32_imagenet1k',pretrained=True,num_classes=len(class_names),image_size=IMG_SIZE).to(device)

optimizer = torch.optim.Adam(model.parameters(), lr=0.001)
criterion = nn.CrossEntropyLoss()


for param in model.parameters():
    param.requires_grad = False

for param in model.fc.parameters():
    param.requires_grad = True


print("Model Start Training")

num_epochs = 100

for epoch in range(num_epochs):
    model.train()
    running_loss = 0.0
    correct_predictions = 0
    total_predictions = 0

    for images, labels in train_dataloader:
        optimizer.zero_grad()

        images=images.to(device)
        labels=labels.to(device)

        outputs = model(images)

        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()

        running_loss += loss.item() * images.size(0)

        _, predicted = torch.max(outputs, 1)

        correct_predictions += (predicted == labels).sum().item()
        total_predictions += labels.size(0)

    epoch_loss = running_loss / len(train_dataloader.dataset)
    epoch_accuracy = correct_predictions / total_predictions
    validation_accuracy,validation_loss = calculate_accuracy_and_validation(model, val_dataloader, device)
    print(f"Epoch [{epoch + 1}/{num_epochs}], Validation Accuracy: {validation_accuracy:.4f}")
    print(f"Epoch [{epoch + 1}/{num_epochs}], Validation Loss: {validation_loss:.4f}")
    print(f"Epoch [{epoch + 1}/{num_epochs}], Train Accuracy: {epoch_accuracy:.4f}")
    print(f"Epoch [{epoch + 1}/{num_epochs}], Train Loss: {epoch_loss:.4f}")

torch.save(model.state_dict(),r'D:\Neural Networks assignments\Project\Local Run\vision transformer\vit_B32_imgnet_416.pth')

########################################################

############### Predicting ######################

model.eval()
predicted_classes=[]

for input,label in test_dataloader:

    img=input.to(device)

    with torch.no_grad():
        outputs = model(img)

    _, predicted = torch.max(outputs, 1)

    # Append predictions to the list
    predicted_classes.extend(predicted.cpu().numpy())


predicted_classes = np.array(predicted_classes)

print('predictions',predicted_classes)


print("convert to dataframe")
filenames_without_extension = [filename.replace(".png", "") for filename in os.listdir(r'D:\Neural Networks assignments\Project\Local Run\Dataset\Test\Test')]
pred={
    'ID':filenames_without_extension,
    'label':predicted_classes
}

submit=pd.DataFrame(pred)
print('Top 5 rows in prediction',submit.head())
print('least 5 rows in prediction',submit.tail())


submit.to_csv(r"D:\Neural Networks assignments\Project\Local Run\vision transformer\\"+'Vit_B32_imgnet_second.csv',index=False)

######################################################

print("Finished Sucessfully")

