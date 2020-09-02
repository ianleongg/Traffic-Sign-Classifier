# **Traffic Sign Recognition** 

## Deep Learning to Build a Traffic Sign Recognition Classifier 

### Using the LeNet-5 neural network architecture

---

**Traffic Sign Recognition Project**

The goals / steps of this project are the following:
* Step 0:

  ** Load the GTSRB data set from [German Traffic Sign Benchmarks](http://benchmark.ini.rub.de/?section=gtsrb&subsection=dataset)

* Step 1:

-Explore, summarize and visualize the data set

* Step 2:

-Design, train and test a model architecture

* Step 3:

-Use the model to make predictions on new images (German Traffic Signs from google.com)

-Analyze the softmax probabilities of the new images

[//]: # (Image References)

[image1]: ./Images_forReadMe/1random12color.jpg "12 Color Images"
[image2]: ./Images_forReadMe/2trainingclass.png "Training Plot"
[image3]: ./Images_forReadMe/3validclass.png "Valid Plot"
[image4]: ./Images_forReadMe/4testclass.png "Test Plot"
[image5]: ./Images_forReadMe/5random12grey.png "12 Grey Images"
[image6]: ./Images_forReadMe/6realwithcolorlabel.png "Real World Images"
[image7]: ./Images_forReadMe/7softmax.png "Softmax 1"
[image8]: ./Images_forReadMe/8softmax.png "Softmax 2"
[image9]: ./Images_forReadMe/9softmax.png "Softmax 3"
[image10]: ./Images_forReadMe/10softmax.png "Softmax 4"
[image11]: ./Images_forReadMe/11softmax.png "Softmax 5"
[image12]: ./Images_forReadMe/12softmax.png "Softmax 6"
[image13]: ./Images_forReadMe/lenet.png "LeNet"
[image14]: ./Images_forReadMe/visualize_cnn.png "Visualize cnn"
[image15]: ./Images_forReadMe/learning.png "Learning Statistics"

---
### README

- The model has a test accuracy of **95.2%** on the *GTSRB data set* and **100%** on 6 random images from *google images*. 

- Here is a link to my [project code](https://github.com/ianleongg/Traffic-Sign-Classifier/blob/master/Traffic_Sign_Classifier.ipynb) and the below image an example of the predictions on an image from google images.

![alt text][image9]

### Data Set Summary, Exploration & Visualization

#### 1. A basic summary of the data set. In "./data" contains the pickled dataset for the GTSRB data set which are separated into training,validation and testing.

I used the numpy library to calculate summary statistics of the traffic signs data set:

* Training Set:   34799 samples
* Validation Set: 4410 samples
* Test Set:       12630 samples
* Image Shape:    32x32 pixels
* Unique Classes: 43 labels

#### 2. An exploratory visualization of the dataset.

- Here is 12 random images from the data set with its respective labels:

 ![alt text][image1]

- Here is a histogram distribution of *Training Data* vs Classes:

![alt text][image2]

- Here is a histogram distribution of *Validation Data* vs Classes:

![alt text][image3]

- Here is a histogram distribution of *Test Data* vs Classes:

![alt text][image4]

As seen above, the histogram distribution for all the above 3 are identical which is ideal to ensure the model does not train specifically for a single data set.

### Design, Train and Test a Model Architecture

#### 1. Preprocessing: Converting to grayscale and normalizing.

As a first step, I decided to convert the images to grayscale because it reduces the amount of features and thus reduces execution time. Additionally, several research papers have shown good results with grayscaling of the images. [Yann LeCun - Traffic Sign Recognition with Multi-Scale Convolutional Networks](http://yann.lecun.com/exdb/publis/pdf/sermanet-ijcnn-11.pdf)

Here are 12 examples of traffic sign image after grayscaling.

![alt text][image5]

As a last step, I normalized the image data with the method of (pixel-128)/ 128 so that the data has mean zero and equal variance.

#### 2. Model Architecture

The model architecture is based on the LeNet model architecture as shown:

![alt text][image13]
Source: Yan LeCun

After modifying, my final model consisted of the following layers:

| Layer         		|     Description	        					| 
|:---------------------:|:---------------------------------------------:| 
| Layer 1:	                                      | 
| Input         		| 32x32x1 gray scale image   							| 
| Convolution 5x5     	| 1x1 stride, valid padding, outputs 28x28x6 	|
| Activation 					| RELU											|
| Max pooling	      	| 2x2 stride,  outputs 14x14x6			|
| Layer 2:	                                      | 
| Convolution 5x5	    | 1x1 stride, valid padding, outputs 10x10x16		|
| Activation 					| RELU											|
| Max pooling	      	| 2x2 stride,  outputs 5x5x16			|
| Flatten      	| outputs 400			|
| Regularization					| Dropout								|
| Layer 3:	                                      | 
| Fully connected		| outputs 120        									|
| Activation 					| RELU											|
| Regularization					| Dropout								|
| Layer 4:	                                      | 
| Fully connected		| outputs 84        									|
| Activation 					| RELU											|
| Regularization					| Dropout								|
| Layer 5:	                                      | 
| Fully connected		| outputs 43        									|
 
#### 3. Model Training

To train the model, I used an Adam optimizer and the following hyperparamters:

* number of epochs: 150
* batch size: 128
* keep probalbility of the dropout layer: 0.5
* Variables were initialized using the truncated normal distribution with mu = 0.0 and sigma = 0.1
* learning rate: 0.001

My final model results were:

* training set accuracy: 99.6%
* validation set accuracy: 96.7% 
* model execution time: 568.30939 seconds
* test set accuracy: 95.2%

#### 4. Solution Approach

* The first architecture that was tried was the classic LeNet-5 and why it was chosen as the model already displayed promising results (>80%). This was also done by feeding in grayscale images to reduce features thus reducing training time.
* The initial architecture needed to be modified to achieve an accuracy of at least 93%.
* The initial architecture was adjusted by changing the number of outputs to 43 as it is the number of classes we have. After that a regularization method called dropout was applied in layers 2,3,4 to prevent overfitting. The extra added layer immediately boosted accuracy to ~93% with hyperparamters of 128 batch size, 100 epochs, and 0.001 learning rate.
* The model already achieved >93% and I assumed training it an extra 50 epochs might give it a little more boost, and so it did to ~95%.

The below plot displays the learning statistics for validatin set accuracy:

![alt text][image5]
 
### Test a Model on New Images

#### 1. Saving new images from google

Here are five German traffic signs that I found on the web:

![alt text][image6] 

The images might be hard to classify as they were not cropped appropriately but by forcefully resizing to 32x32 thus losing its 'accuracy'.

#### 2. Performance 

Here are the results of the prediction:

Predictions from model: [11, 1, 25, 34, 12, 17]
Image Labels: [11, 1, 25, 34, 12, 17]

The above labels correspond to the ClassId in [signnames.csv](https://github.com/ianleongg/Traffic-Sign-Classifier/blob/master/signnames.csv)

For ease of view after extracting ClassId:

| Image			        |     Prediction	        					| 
|:---------------------:|:---------------------------------------------:| 
| Right-of-way at the next intersection   		| Right-of-way at the next intersection									| 
| Speed limit (30km/h)   			| Speed limit (30km/h)									|
| Road work				| Road work									|
| Turn left ahead    		| Turn left ahead			 				|
| Priority road		| Priority road   							|
| No entry		| No entry   							|

The model was able to correctly guess 6 of the 6 traffic signs, which gives an accuracy of 100%. This compares favorably to the accuracy on the test set of 95.2%.

#### 3. Model Certainty - Softmax Probabilities

The following images shows the softmax probabilities of the 6 traffic sign images obtained from the web and evaluated using our trained model.

![alt text][image7]
![alt text][image8]
![alt text][image9]
![alt text][image10]
![alt text][image11]
![alt text][image12]

### (Extra) Visualizing the Neural Network (See Step 4 of the Ipython notebook for more details)
Example of what feature map outputs look like:
![alt text][image14]

### Additional Reading
#### Materials
* [Fast AI](https://www.fast.ai/)
* [A Guide To Deep Learning](http://yerevann.com/a-guide-to-deep-learning/)
* [Dealing with unbalanced data](https://medium.com/@vivek.yadav/dealing-with-unbalanced-data-generating-additional-data-by-jittering-the-original-image-7497fe2119c3#.obfuq3zde)
* [Improved Performance of Deep Learning On Traffic Sign Classification](https://medium.com/@vivek.yadav/improved-performance-of-deep-learning-neural-network-models-on-traffic-sign-classification-using-6355346da2dc#.tq0uk9oxy)

#### Bacth size discussion
* [How Large Should the Batch Size be](https://stats.stackexchange.com/questions/140811/how-large-should-the-batch-size-be-for-stochastic-gradient-descent)

#### Adam optimizer discussion
* [Optimizing Gradient Descent](https://ruder.io/optimizing-gradient-descent/index.html#adam)

#### Dropouts
* [Analysis of Dropout](https://pgaleone.eu/deep-learning/regularization/2017/01/10/anaysis-of-dropout/)
