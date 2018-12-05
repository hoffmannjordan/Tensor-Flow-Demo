# TensorFlow Demo for Mathematicians and Physicists
TensorFlow demo for Harvard Machine Learning Supergroup. The goal is to get you up and running with rather bare-bones code for three different tasks.

Code written by Jordan Hoffmann (mainly Demo 2 + 3, flat-folding code for Demo 1) and Shruti Mishra (Demo 1)

# Demo 1
Given a sheet that has been folded, can a computer tell how many times it has been folded? It is relatively straightforward for small crease numbers, but as the number increases, the specific locations of the creases can greatly impact the amount of mileage that is present. Below, we show an image that shows a few examples of creased sheets at different fold numbers.
![Demo_1](../master/ims/PredN.png)

# Demo 2
Here, I wanted to cook up a slightly more complicated example that uses a **side stream** in addition to the typical input. Therefore, in this
demo we will be doing a PDE related problem. We are solving:

<a href="https://www.codecogs.com/eqnedit.php?latex=u^{(2,0,0)}(t,x,y)-\nabla&space;_{\{x,y\}}^{}u(t,x,y)=0" target="_blank"><img src="https://latex.codecogs.com/gif.latex?u^{(2,0,0)}(t,x,y)-\nabla&space;_{\{x,y\}}^{}u(t,x,y)=0" title="u^{(2,0,0)}(t,x,y)-\nabla _{\{x,y\}}^{}u(t,x,y)=0" /></a>

Subject to:
<a href="https://www.codecogs.com/eqnedit.php?latex=u(0,x,y)=e^{Z&space;\left(-\left((x-X)^2&plus;(y-Y)^2\right)\right)}" target="_blank"><img src="https://latex.codecogs.com/gif.latex?u(0,x,y)=e^{Z&space;\left(-\left((x-X)^2&plus;(y-Y)^2\right)\right)}" title="u(0,x,y)=e^{Z \left(-\left((x-X)^2+(y-Y)^2\right)\right)}" /></a>
and:
<a href="https://www.codecogs.com/eqnedit.php?latex=u^{(1,0,0)}(0,x,y)=0" target="_blank"><img src="https://latex.codecogs.com/gif.latex?u^{(1,0,0)}(0,x,y)=0" title="u^{(1,0,0)}(0,x,y)=0" /></a>. We solve it on an irregular geometry. We then try to predict the total amplitude within a small region of the solution domain some time later. That is, we randomly set X,Y, and Z for each run. However, we store these values and pass them along as a side stream for the network.

Here is a plot of the solution over time, always with the same color bar:
![Grid](../master/ims/grid.png)

Here is an example of a solution at t=6.
![Setup](../master/ims/setup.png)

Here, the goal is to given the solution at t=6, the location of the original pulse, and it's amplitude, say something about the result at t=6.28. How different do they look? Below, we plot the solution at two times.
![Compare](../master/ims/diff.png)

Not totally the same! In this demo, we train a neural network to make predictions. Here, I tried to use a more typical coding style like that in a large project. 
To try to quantify some aspect, here we try to predict the summed amplitude in the lower right quadrant in the second image. Training the network on 5000 examples, a small sample of which are in data_small.zip, we get the results below:
![run0](../master/ims/res0.png)

# Demo 3
Demo problem recoloring an image given colored images. Use the left lava lamp from this video for training:
https://www.youtube.com/watch?v=rSzjFvMFQhg
Can we take an image like that on the left, separate the two lava lamps, and then accurately recolor one of them, fed a video of the other side? **Note:** I ended up taking the one on the right for training. 
![Setup](../master/ims/Lava_Lamp_Setup.png)
To do this, we try using a network that uses `conv2d` and `conv2d_transpose` layers. In the figure below, we show the input (gray scale), the result from early in training, somewhere near the middle of training, and the end of training. At the bottom, we show the target image. 
![Prediction over Time](../master/ims/PredT.png)

## Video
Click on the below image to be redirected to a youtube video.
[![Video](https://img.youtube.com/vi/9HE61S2OagU/0.jpg)](https://www.youtube.com/watch?v=9HE61S2OagU)

# Supplementary Information 
## Resources

https://developers.google.com/machine-learning/crash-course/first-steps-with-tensorflow/video-lecture

https://www.tensorflow.org/tutorials/

https://www.tensorflow.org/tutorials/keras/basic_classification
