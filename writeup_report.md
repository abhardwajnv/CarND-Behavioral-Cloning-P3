#**Behavioral Cloning**

##Writeup

**Behavioral Cloning Project**

The goals / steps of this project are the following:
* Use the simulator to collect data of good driving behavior
* Build, a convolution neural network in Keras that predicts steering angles from images
* Train and validate the model with a training and validation set
* Test that the model successfully drives around track one without leaving the road
* Summarize the results with a written report


[//]: # (Image References)

[image1]: ./examples/histogram.png "Historgram"
[image2]: ./examples/Original_Image.png "Original_Image"
[image3]: ./examples/Brightness_Image.png "Brightness_Image"
[image4]: ./examples/Flipped_Image.png "Flipped_Image"
[image5]: ./examples/Loss_Graph.png "Loss_Graph"
[image6]: ./examples/Nvidia_Model.png "Model Architecture"
[image7]: ./examples/center.jpg "Center Driving"

## Rubric Points
###Here I will consider the [rubric points](https://review.udacity.com/#!/rubrics/432/view) individually and describe how I addressed each point in my implementation.  

---
###Files Submitted & Code Quality

####1. Submission includes all required files and can be used to run the simulator in autonomous mode

My project includes the following files:
* model.py - containing the script to create and train the model
* drive.py - for driving the car in autonomous mode
* model.h5 - containing a trained convolution neural network
* writeup_report.md - summarizing the results
* track_run.mp4 - Video Recording of Car driving autonomously on track 1 for two laps.

In addition reference data is shared on google drive @ https://drive.google.com/open?id=0By5Wqj9N-Cx1MjdTS3pMZWxHQk0 
* Compressed Image dataset used for training.
* Compressed Images used for video generation.

####2. Submission includes functional code

Following are the steps to get the car driven autonomously around the track.

* Download the source from github repo.
* Run model.py using below command the train the network with dataset.

```sh
python model.py
```

  This will run the Nvidia model (default), train the network and then save the model.h5 file in working directory.

* Run drive.py to to simulate autonomous driving by launching simulator in autonomous mode and run below command.

```sh
python drive.py model.h5
```

  This will test the model by running the car autonomously around the track1.

* To save a video file generate images of autonomously driving car using following command.

```sh
python drive.py model.h5 track_run
```

  This will save the images in track_run folder to generate a video.


####3. Submission code is usable and readable

The model.py file contains the code for training and saving the convolution neural network. The file shows the complete pipeline i used for training and validating the model. I have marked comments to individual methods which i've used explaining what they do as well as on some of the functions with thier parameters.
I have not changed the drive.py file and used the defualt file for driving the car autonomously on track1.

###Model Architecture and Training Strategy

####1. An appropriate model architecture has been employed

I have implemented 2 models using Keras Library which are explained as follows.

* Lenet Model:
  Since we are already learning this model in our earlier classes and implemented this in last project i started with this model as a reference point.

  Starting with the data normalized with a Keras lambda layer following by cropping the image using Cropping2D function.
  My Model consisted of 2 convolutional neural network layers with 5x5 kernel size each. Both the convolutional layers were followed by a RELU Activation layer and then max pooling of 2x2.
  This was then followed by 3 fully connected layers of size 120, 84 & 1 respectively.

  With this model i was not able to achieve higher accuracy which i tried with udacity dataset as well as my captured dataset.

* Nvidia E2E Self Driving Model:
  This model was a bit complex compared to Lenet model as there were more layers to implement having more feature parameters to be tuned.

  Starting with Cropping the image nomalizing the data using Keras lambda layer.
  My model consisted of 5 convolutional neural network layers with 5x5 kernel size and 2x2 stride for the first 3 layers applied with RELU Activation. Next 2 convolution layers had a kernel size of 3x3 applied with RELU Activation.
  This was then followed by 3 fully connected layers and 1 output layer of sizes 100, 50, 10 & 1 respectively. I used dropout regularization function of 0.2 for the fully connected layers to avoid overfitting of the network.

  With this model i was able to achieve higher accuracy compared to Lenet Model and have the car running for the lap without getting out of bounds.
  I tried running with Udacity given dataset as well as my own captured dataset. With Udacity given dataset, the car was mostly getting out of bounds in the dirt area hence i decided to stick to my captured dataset. With my captured dataset car drove though the complete track without getting off track.

####2. Attempts to reduce overfitting in the model

The model contains dropout layers with fully connected layers to reduce overfitting of the network. This helped in reducing the MSE loss after while training the network. In addition i added RELU activation layer along with dropout layer to introduce non-linearity.

To have the model training and validated on different datasets i splitted the images & steering_angle data into training and validation set. This was to ensure that the model should not overfit.
With this the model was tested by running it through the simulator and ensuring that the vehicle could stay on the track.

####3. Model parameter tuning

Following are the parameters which i used to tune the network.

* Normalization:
  I used Keras lambda layers in the model to normalize the images. Without normalization the accuracy was too low.

* Cropping:
  The images captured by the simulator are having a size of 160x320x3 for which all pixels does not have usefull information. Images are capturing much more than the road (containing trees, sky, mountains, car body etc.).
  With all this not required information in the image frame which is not required for training cropping is applied on the image. I used Cropping2D function to crop the images by 60x20 pixels i.e., from top and bottom respectively.

* Optimizer:
  The model used an adam optimizer. This helped in tuning the learning rate which was not done manually.

* Loss Function:
  Being a regression model, I used Mean Sqaured Error (MSE) function here instead of softmax cross-entropy function.

* Generator:
  I used Keras Library's fit_generator function for generator with a batch size of 32 to generate random augument data at run time of networks using the yield keyword. This helped reducing the memory requirement and increasing computational efficiency (which in a way helped me running the network on CPU in absence of GPU).

####4. Appropriate training data

Initially i started to run the model with Udacity Dataset where car was not driving properly on the simulator track. Mostly the car was getting deviated from the track on sharp edges.
So to overcome this i decided to make my own dataset.

I drove 3 laps with center lane driving at a constant speed of 30mph without using any breaks. While driving these laps on some of the sharp turns i took the car close to the road boundary and recovered it back on to the centeral lane. I did not capture the recovery images from left/right separately.

For details on the captured data check visualization of data below.

###Model Architecture and Training Strategy

####1. Solution Design Approach


I tried to train my models with Udacity dataset as well but it did not perform very well on the sharp edges and invisible road boundaries like the dirt patch.
Due to this reason i decided to capture my own dataset and perform training with it.

So i started by recording a data on track 1 with following data:
1. 3 laps of center driving
2. recovering from left and right edges on sharp turns in the same center driving.

With my captured dataset also initially i was getting some issues in the dirt patch area and the turn before bridge where the car the mostly either getting stuck on the side of the road or getting off track completely. By tuning the model paramters which are explained in above sections i was able to achieve a proper run on track.

I tried with capturing dataset for track2 as well and train the models with it but the results were not very good and the car was contantly getting crashed on the first few turns itself. So i thougt of keeping that part a future assignment for myself which i will work on.

Below is the visualization of my captured Datset.

![alt text][image1]

We can deduce following pointers from above visualization of dataset.
1. Steering angle is mostly 0 for maximum number of frames in dataset.
2. Steering angle was taken mostly towards left side only.
3. Throttle is max for most of the dataset.
4. No breaks used in the dataset at all.
5. Mostly data is recorded at full speed i.e., 30mph

##### Data Augumentation
With this dataset first i started with Data Augumentation.

* Include all camera images:
  Since with only center lane data car was not able to recover from sharp turns hence i included data for all the 3 camera set's. i.e., center, left & right.

* Brightness:
  Random brightness was applied to the trainign dataset to have different lighting conditions included in the data.

* Flipping Image:
  Since track1 data has mostly left turns only hence i flipped the images by 180deg providing a mirror image and flip the steering angle multiplying it by -1 to keep uniformity in traingin dataset.

* Random Data Generation:
  I used random funciton to generate random data for training set which increased the training set size by 3 times as it included original images, brightness changes images & flipped images. Following is the visualization of images.

Original Images
![alt text][image2]

Brigtness Adjusted Images
![alt text][image3]

Flipped Images
![alt text][image4]

My first step was to use a convolution neural network model similar to the Lenet. This was the starting point of training the networks as this was the first model we started with learning. Leter on to have more improvements and better results i switched to Nvidia Model which was performing much better than Lenet Model.

In order to gauge how well the model was working, I split my image and steering angle data into a training and validation set. I found that my first model had a low mean squared error on the training set but a high mean squared error on the validation set. This implied that the model was overfitting.

To combat the overfitting, I modified the model to include dropout layer and RELU activation layers in the model.

The final step was to run the simulator to see how well the car was driving around track one. There were a few spots where the vehicle tried to go off track but recovered soon enough before going out of bounds. In my submission video my car is getting close to the boundaries at 2 places, one is just before the bridge for a split second where it goes and just touches the yellow border of the road and then immediately recovers from it and gets back on track whereas the other part after the dirt patch on the sharp right curve the car takes a big circle due to which it gets on the painted curbing on the inside of the track and returns back to the track normally. To improve the driving behavior in these cases, I need to add more images of passing through this type of curves.

At the end of the process, the vehicle is able to drive autonomously around the track without leaving the road. I captured 2 laps of data which can be seen in the submission video.

* Loss Graph:
  Post training and validation of dataset and car successfully running on simulation track i plotted the loss graph of training loss & validation loss vs no of epochs, which is as follows:

![alt text][image5]

####2. Final Model Architecture

The final model architecture chosen was Nvidia model as default, although i believe lenet model can also be improved more to tackle this problem statement.

Here is a visualization of the architecture showing all the 5 convolution layers used followed by 3 fully connected layer and one output layer applied with dropout.

![alt text][image6]

####3. Creation of the Training Set & Training Process

To capture good driving behavior, I recorded three laps on track 1 using center lane driving. Here is an example image of center lane driving:

![alt text][image7]

While recording for the center lane driving few times i took the car close to the road boundary and recovered it back to the center lane to train it with sharp recoveries.
This helped me to get the car driving autonomously without hitting the dirt patch region which i initially was getting with Udacity Dataset.

Then I repeated this process on track two in order to get more data points but driving on track 2 with keyboard was getting too probalamatic for me since car was anyway going out of lane lines. So i dropped these datasets.
