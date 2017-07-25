import os
import csv
import cv2
import numpy as np
import sklearn
from sklearn.utils import shuffle
from sklearn.model_selection import train_test_split
from keras.models import Sequential, optimizers
from keras.layers import Flatten, Dense, Lambda, Cropping2D, Convolution2D, Dropout
import matplotlib.pyplot as plt


'''
opne the cvs fils from data dir, read each line and store them into
samples list.
'''
samples = []
with open('./data/driving_log.csv') as csvfile:
    reader = csv.reader(csvfile)
    next(reader)
    for line in reader:
        samples.append(line)

'''
Store left, center and middle images from each line and the corresponding
steering angles, an empirical correction of 0.2 is applied to steering
angle for left and right cameera image.
'''
images = []
measurements = []
show_image_flip_once = 1
show_image_geo_transform_once = 1


for sample in samples:
    for i in range(3):
        source_path = sample[i]
        filename = source_path.split('/')[-1]

        # find the image type , left, center or right camera
        direction = filename.split('_')
        current_path = './data/IMG/' + filename

        #print(current_path)
        # Read the image
        new_image = cv2.imread(current_path)

        '''
        The input image is split into YUV planes and passed to the network as recommanded by
        NVIDIA, section 'Network Architecture'
        https://devblogs.nvidia.com/parallelforall/deep-learning-self-driving-cars/
        '''
        new_image = cv2.cvtColor(new_image, cv2.COLOR_BGR2YUV)
        images.append(new_image)
        # read the steering angle
        measurement = float(sample[3])

        '''
        Create adjusted steering measurements for the side camera images.
        these side cameera images will be used to avoid bias towards left
        or right turn.
        '''
        correction = 0.2
        steering = 0.0

        if "left" in direction:
            steering = measurement + correction
            #print("left steering ", steering)
        elif "right" in direction:
            steering = measurement - correction
            #print("right steering ", steering)
        else:
            steering = measurement
            #print("center steering ", steering)

        measurements.append(steering)

'''
split the data into 80% training and 20% validation
'''
train_samples, validation_samples, measurement_samples, measurement_validation_samples = \
                        train_test_split(images, measurements, test_size=0.2)

#print("train_samples.shape", train_samples)
#print("validation_samples.shape", validation_samples)

def geo_transform_image(image, x_pixel, y_pixel):

    '''
    Shift the given image by x_pixel and y_pixel amount
    '''
    rows, cols, dim = image.shape
    M = np.float32([[1,0,x_pixel],[0,1,y_pixel]])
    geo_trans_img = cv2.warpAffine(image,M,(cols,rows))
    return geo_trans_img

def show_image(image, name):
    '''
    Show the image
    '''
    print(name)
    plt.imshow(image)

def generator(samples, sample_measurements, batch_size=32):

    '''
     A Python generator to yield data batch by batch, illustrated an optimized
     memory use by storing partial (per batch) data not the entire dataset into
     the memory
    '''
    num_samples = len(samples)

    while 1: # Loop forever so the generator never terminates
        shuffle(samples, sample_measurements)
        for offset in range(0, num_samples, batch_size):
            batch_samples = samples[offset:offset+batch_size]
            batch_measurements = sample_measurements[offset:offset+batch_size]
            augmented_images = []
            augmented_measurements = []

            #Iterate the image and steering angle on each batch
            for image, steering_angle in zip(batch_samples, batch_measurements):

                # Gausian Blur/down sample the original image to reduce noise
                image = cv2.GaussianBlur(image, (3,3), 0)
                augmented_images.append(image)
                augmented_measurements.append(steering_angle)

                # add mirro image if steering angle is more than +- 43.0
                # this is helpfull to avoid bias towards left or right corner
                if abs(steering_angle) > 0.43 :
                    # add mirror image of above image, filp vertically (y-axis)
                    flip_image = cv2.flip(image, 1)
                    augmented_images.append(flip_image)
                    augmented_measurements.append(steering_angle * -1.0)

                    '''
                    Initially I used it but this overfits the model as per
                    my observation so I am not using it for now.
                    # Geo transform
                    #print("sample.shape", sample.shape)
                    img = geo_transform_image(image, 10, 10)
                    augmented_images.append(img)
                    augmented_measurements.append(steering_angle)

                    # Geo transform flip image
                    #print("sample.shape", sample.shape)
                    rows, cols, dim = flip_image.shape
                    img = geo_transform_image(flip_image, 10, 10)
                    augmented_images.append(img)
                    augmented_measurements.append(steering_angle * -1.0)
                    '''

            X_train = np.array(augmented_images)
            y_train = np.array(augmented_measurements)
            yield sklearn.utils.shuffle(X_train, y_train)
# compile and train the model using the generator function
batch_size = 64
total_epochs = 5

train_generator = generator(train_samples, measurement_samples, batch_size)
validation_generator = generator(validation_samples, measurement_validation_samples,  batch_size)

def model_mean_sqr_error_loss(model_fit_history):

    '''
    Draw mse verses epoch to find an optimum number of epoch
    '''
    # print the keys contained in history object
    print(model_fit_history.history.keys())

    #plot training and validation loss for each epoch
    plt.plot(model_fit_history.history['loss'])
    plt.plot(model_fit_history.history['val_loss'])
    plt.title('Model showing  mse loss')
    plt.ylabel('mse loss')
    plt.xlabel('epoch')
    plt.legend(['training set', 'validation set'], loc='upper right')
    plt.show()


model = Sequential()

'''
Make each pixel normalized and mean centered i.e. close to zero mean and equal variance which
is a good starting point for optimizing the loss to avoid too big or too small
'''
model.add(Lambda(lambda x: x/255.0 - 0.5, input_shape=(160,320,3)))

'''
Crop the tree, hills, Sky from top 70 and the hood of the car from bottom 25 to avoid
noise
'''
model.add(Cropping2D(cropping=((70,25),(0,0))))

'''
NVIDIA'S Network Acchitecture with an extra dropout layer added by me to reduce
overfitting in this track by observation

'''

'''
Activation Layer selection
Based on the following recent experiment i.e.
https://ctjohnson.github.io/Capstone-Activation-Layers/ , section 5. Table of Results
I decided to stick with RELU as preferred activation layer to introduce nonlinearity
instead of Expotential Linear Unit (ELU)
'''

# 5x5 kernel with strides of 2x2, input depth 3 output depth 24
model.add(Convolution2D(24,5,5,subsample=(2,2),activation="relu"))

# 5x5 kernel with strides of 2x2, input depth 24 output depth 36
model.add(Convolution2D(36,5,5,subsample=(2,2),activation="relu"))

# 5x5 kernel with strides of 2x2, input depth 36 output depth 48
model.add(Convolution2D(48,5,5,subsample=(2,2),activation="relu"))

# 3x3 kernel with strides of 1x1, input depth 48 output depth 64
model.add(Convolution2D(64,3,3,activation="relu"))

# 3x3 kernel with strides of 1x1, input depth 64 output depth 64
model.add(Convolution2D(64,3,3,activation="relu"))

model.add(Flatten())

'''
Dropout 40%
'''
model.add(Dropout(0.40))
model.add(Dense(100))
#model.add(Dropout(0.40))
model.add(Dense(50))
#model.add(Dropout(0.20))
model.add(Dense(10))

'''
Ouput Directly predict the steering measurement, so 1 output
'''
model.add(Dense(1))
model.summary()
adam = optimizers.Adam(lr=0.001)

'''
Compile and train the model using the generator function
'''
model.compile(loss='mse', optimizer=adam)
history_object = model.fit_generator(train_generator, samples_per_epoch = \
                 len(train_samples), \
                 validation_data=validation_generator, \
                 nb_val_samples=len(validation_samples), \
                 nb_epoch=total_epochs, verbose = 1)

model.save('model.h5')
#model_mean_sqr_error_loss(history_object)
