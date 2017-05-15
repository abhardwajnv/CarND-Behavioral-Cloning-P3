# Import required libraries.
import csv
import cv2
from itertools import islice
import numpy as np
import matplotlib.pyplot as plt
import tensorflow as tf
from keras.models import Sequential
from keras.layers.core import Dense, Activation, Flatten, Dropout, Lambda
from keras.layers.convolutional import Convolution2D, Cropping2D
from keras.layers.pooling import MaxPooling2D
import sklearn
from sklearn.model_selection import train_test_split
from sklearn.utils import shuffle
import random
import os
# Disable SSE warnings from tensorflow
os.environ['TF_CPP_MIN_LOG_LEVEL']='2'


# Define all required variables.
epochs = 10
batch_size = 32
modelname = 'nvidia'
capture_data = 'on'
correction = 0.25
csv_path = './dataset/data/driving_log.csv'
left_images = []
center_images = []
right_images = []
throttles = []
steering_angle = []
speeds = []
brakes = []


def data_set(samples):
    '''
    data_set reads samples list for all driving data and splits
    individual parameters angle/throttle/brake/speed to individual
    lists. It does not return anything.
    '''

    for line in samples:
        center_img = './dataset/data/IMG/'+line[0].split('/')[-1]
        left_img = './dataset/data/IMG/'+line[1].split('/')[-1]
        right_img = './dataset/data/IMG/'+line[2].split('/')[-1]
        angle = float(line[3])
        throttle = float(line[4])
        brake = float(line[5])
        speed = float(line[6])
        img = cv2.imread(center_img)
        center_images.append(img)
        img = cv2.imread(left_img)
        left_images.append(img)
        img = cv2.imread(right_img)
        right_images.append(img)
        steering_angle.append(angle)
        throttles.append(throttle)
        brakes.append(brake)
        speeds.append(speed)
    print('Done')


def histogram(steering_angle, throttles, brakes, speeds):
    '''
    histogram plots and saves histogram of steering_angle,
    throttles, brakes and speeds.
    '''

    print('Plotting histogram')
    img, vaxis = plt.subplots(1, 4, figsize=(20,5))
    vaxis[0].hist(steering_angle, bins=40)
    vaxis[0].set_title('Steering_angle')
    vaxis[1].hist(throttles, bins=40)
    vaxis[1].set_title('Throttles')
    vaxis[2].hist(brakes, bins=40)
    vaxis[2].set_title('Brakes')
    vaxis[3].hist(speeds, bins=40)
    vaxis[3].set_title('Speeds')
    plt.savefig('histogram.png')
    print('Done')


def visualize(center_images, left_images, right_images, steering_angle):
    '''
    visualize function visualizes the original and augumented
    images and saves the randomly chosen left, right and center
    images.
    '''

    print('Plotting Augumented Data')
    data = ['Center', 'Left', 'Right']
    index = random.randint(0, len(center_images))
    center_img = center_images[index]
    left_img = left_images[index]
    right_img = right_images[index]

    center_angle = steering_angle[index]
    left_angle = center_angle + correction
    right_angle = center_angle - correction

    orig_img = [center_img, left_img, right_img]
    orig_angle = [center_angle, left_angle, right_angle]

    bright_center_img = brightness(center_img)
    flip_center_img, flip_center_angle = flipping(center_img, center_angle)

    bright_left_img = brightness(left_img)
    flip_left_img, flip_left_angle = flipping(left_img, left_angle)

    bright_right_img = brightness(right_img)
    flip_right_img, flip_right_angle = flipping(right_img, right_angle)

    bright_img = [bright_center_img, bright_left_img, bright_right_img]

    flip_img = [flip_center_img, flip_left_img, flip_right_img]
    flip_angle = [flip_center_angle, flip_left_angle, flip_right_angle]

    # Save visualized data.
    save_visualize(orig_img, data, orig_angle, 'Original_Image')
    save_visualize(bright_img, data, orig_angle, 'Brightness_Image')
    save_visualize(flip_img, data, flip_angle, 'Flipped_Image')


def save_visualize(image, imgtype, angle, title=''):
    '''
    save_visualize takes visualized data as an input and plots
    a graph for each of the image types and saves them.
    '''
    img, vaxis = plt.subplots(1, 3, figsize=(10,5))
    for i in range(3):
        vaxis[i].imshow(image[i])
        vaxis[i].set_title(imgtype[i] + ' Image' + '\nSteering Angle: ' + str(angle[i]), fontsize=10)
    plt.savefig(str(title) + '.png')


def split_data(samples):
    '''
    split data splits the input dataset input training and
    validation dataset in 80:20 percentage and return the
    splitted training and validation set.
    '''

    train_samples, validation_samples = train_test_split(samples, test_size=0.2)
    print('Training Dataset Size:', len(train_samples))
    print('Validation Dataset Size:', len(validation_samples))
    return train_samples, validation_samples


def brightness(image):
    '''
    brightness function applies random brightness on input
    image and returns the output image as brightness adjusted image.
    '''

    img = cv2.cvtColor(image, cv2.COLOR_RGB2HSV)
    random_bright = 0.4 + np.random.uniform()
    img[:,:,2] = img[:,:,2]*random_bright
    img = cv2.cvtColor(img, cv2.COLOR_HSV2RGB)
    return img


def flipping(image, angle):
    '''
    flipping gives mirror image of original image and reverse the
    steering angle by multiplying it with -1. This returns flipped
    image and steering angle.
    '''

    img = cv2.flip(image,1)
    angle = angle * -1.0
    return img, angle


def augument_data(image, steering_angle):
    '''
    augument_data auguments the input data and return either of below
    outputs:
    1. Change in brightness of input image along with steering angle
    2. Flip the input image and steering angle
    3. No change in input image and steering angle
    '''

    index = random.randint(0, 2)

    if(index == 0):
        bright_image = brightness(image)
        steering_angle = steering_angle
        return bright_image, steering_angle
    elif(index == 1):
        flip_img, steering_angle = flipping(image, steering_angle)
        return flip_img, steering_angle
    else:
        return image, steering_angle


def generator(samples, batch_size):
    '''
    generator function processes the input samples dataset.
    The generator uses yield, which still returns the desired
    output values but saves the current values of all the
    generator's variables. When the generator is called a
    second time it re-starts right after the yield statement,
    with all its variables set to the same values as before.

    Input to function:
        Samples --> Dataset Input
        batch_size --> No of samples for one time process.
            i.e., batch size
    Return output:
        X_train : Processed Dataset
        y_train : Dataset Label, steering angle.
    '''

    num_samples = len(samples)
    while 1: # Infine Loop for generator
        shuffle(samples)
        for offset in range(0, num_samples, batch_size):
            batch_samples = samples[offset:offset+batch_size]

            images = []
            angles = []
            for batch_sample in batch_samples:
                cname = './dataset/data/IMG/'+batch_sample[0].split('/')[-1]
                lname = './dataset/data/IMG/'+batch_sample[1].split('/')[-1]
                rname = './dataset/data/IMG/'+batch_sample[2].split('/')[-1]
                center_angle = float(batch_sample[3])

                # Add Center Images
                center_image, center_angle = augument_data(cv2.imread(cname), center_angle)
                images.append(center_image)
                angles.append(center_angle)
                # Add Left Images
                left_angle = center_angle + correction
                left_image, left_angle = augument_data(cv2.imread(lname), left_angle)
                images.append(left_image)
                angles.append(left_angle)
                # Add Right Images
                right_angle = center_angle - correction
                right_image, right_angle = augument_data(cv2.imread(rname), right_angle)
                images.append(right_image)
                angles.append(right_angle)

            # trim image to only see section with road
            X_train = np.array(images)
            y_train = np.array(angles)
            yield sklearn.utils.shuffle(X_train, y_train)


def lenet_model():
    '''
    lenet_model is the lenet model architecture created using keras
    library. It consists of 2 convolution layers followed by 3 fully
    connected layers.
    '''

    print('LeNet Model Implementation Started')

    model = Sequential()

    # Setup Lambda layer & Normalize
    model.add(Lambda(lambda x: x/255.0 - 0.5, input_shape=(160,320,3), output_shape=(160,320,3)))
    # Cropping the image
    model.add(Cropping2D(cropping=((65, 25), (0, 0))))
    # Conv Layer 1
    model.add(Convolution2D(6, 5, 5, border_mode='valid', input_shape=(3,160,320)))
    model.add(Activation('relu'))
    model.add(MaxPooling2D(pool_size=(2,2)))
    # Conv Layer 2
    model.add(Convolution2D(16, 5, 5, border_mode='valid'))
    model.add(Activation('relu'))
    model.add(MaxPooling2D(pool_size=(2,2)))
    # Flatten output
    model.add(Flatten())
    # FC Layer 1
    model.add(Dense(120))
    # FC Layer 2
    model.add(Dense(84))
    # Output Layer
    model.add(Dense(1))

    print('LeNet Model Implemented')
    return model


def nvidia_model():
    '''
    nvidia_model is the nvidia model architecture created using keras
    library. It consists of 5 convolution layers followed by 3 fully
    connected layers and an output layer. Dropout function is used in
    fully connected layers to avoid overfitting of the network.
    '''

    print('Nvidia Model Implementation Started')

    model = Sequential()

    # Cropping the image
    model.add(Cropping2D(cropping=((60,20),(0,0)), input_shape=(160,320,3)))
    # Setup Lambda Layer & Normalize
    model.add(Lambda(lambda x: x/255.0 - 0.5))
    # Conv Layer 1 with kernel size 5x5 & stride 2x2
    model.add(Convolution2D(24,5,5,subsample=(2,2),activation="relu"))
    # Conv Layer 2 with kernel size 5x5 & stride 2x2
    model.add(Convolution2D(36,5,5,subsample=(2,2),activation="relu"))
    # Conv Layer 3 with kernel size 5x5 & stride 2x2
    model.add(Convolution2D(48,5,5,subsample=(2,2),activation="relu"))
    # Conv Layer 4 with kernel size 3x3
    model.add(Convolution2D(64,3,3,activation="relu"))
    # Conv Layer 5 with kernel size 3x3
    model.add(Convolution2D(64,3,3,activation="relu"))
    # Flatten the output
    model.add(Flatten())
    # FC Layer 1
    model.add(Dense(100))
    model.add(Dropout(0.2))
    model.add(Activation('relu'))
    # FC Layer 2
    model.add(Dense(50))
    model.add(Dropout(0.2))
    model.add(Activation('relu'))
    # FC Layer 3
    model.add(Dense(10))
    model.add(Dropout(0.2))
    model.add(Activation('relu'))
    # Output Layer
    model.add(Dense(1))
    model.add(Activation('linear'))

    print('Nvidia Model Implemented')
    return model


def loss_graph(history_object):
    '''
    loss_graph plots training and validation losses against no of
    epochs and then saves the image.
    '''

    ### plot the training and validation loss for each epoch
    plt.plot(history_object.history['loss'])
    plt.plot(history_object.history['val_loss'])
    plt.title('Model MSE Loss')
    plt.ylabel('MSE Loss')
    plt.xlabel('epochs')
    plt.legend(['training loss', 'validation loss'], loc='upper right')
    plt.savefig('Loss_Graph.png')
    plt.show()


def training_model(train_generator, validation_generator, train_samples, validation_samples):
    '''
    training_model compiles and trains the model for given no of
    epochs. Post this training validation is done. Once training
    and validation is completed, this function saves the model.
    It then gives the summary and plots the graph.
    '''

    print('Training the model')

    model_name = modelname
    if(model_name == 'lenet'):
        print('Lenet Model is Selected.')
        model = lenet_model()
    elif(model_name == 'nvidia'):
        print('Nvidia Model is Selected.')
        model = nvidia_model()
    else:
        print('Not a valid model input. Using default Nvidia model instead')
        model = nvidia_model()

    model.compile(optimizer='adam', loss='mse')

    history_object = model.fit_generator(train_generator, samples_per_epoch= \
                    len(train_samples*3), validation_data=validation_generator, \
                    nb_val_samples=len(validation_samples*3), nb_epoch=epochs)

    model.save('model.h5')
    model.summary()
    print('Plotting Loss histogram')
    loss_graph(history_object)


'''
Main Program
Below is the complete flow of this model calling all the functions.
1. Reads the data.
2. Splits it to training + validation set.
3. Processes the training and validation set using generator function.
4. Train the model using Keras framework.
5. Saves the model.
'''

# Read the csv file.
samples = []
with open(csv_path) as csvfile:
    reader = csv.reader(csvfile)
    #Use islice to avoid change in generator_output
    for line in islice(reader, 1, None):
        samples.append(line)

# Visualize Data & Save histogram
if(capture_data == 'off'):
    data_set(samples) # Create Datasets for individual parameters.
    histogram(steering_angle, throttles, brakes, speeds) #Plots Historgram
    visualize(center_images, left_images, right_images, steering_angle) # Visualize Data

# Split training and validataion set
train_samples, validation_samples = split_data(samples)

# Create train and validation generator
train_generator = generator(train_samples, batch_size)
validation_generator = generator(validation_samples, batch_size)

# Training the model
training_model(train_generator, validation_generator, train_samples, validation_samples)
