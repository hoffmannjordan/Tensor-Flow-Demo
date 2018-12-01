# Tensor Flow Demo for Mathematicians and Physicist 
Tensor Flow demo for Harvard Machine Learning Supergroup. The goal is to get you up and running with rather bare-bones code for three different tasks.

Code written by Jordan Hoffmann and Shruti Mishra

# Demo 1

# Demo 2
Demo problem with solving:

<a href="https://www.codecogs.com/eqnedit.php?latex=u^{(2,0,0)}(t,x,y)-\nabla&space;_{\{x,y\}}^{}u(t,x,y)=0" target="_blank"><img src="https://latex.codecogs.com/gif.latex?u^{(2,0,0)}(t,x,y)-\nabla&space;_{\{x,y\}}^{}u(t,x,y)=0" title="u^{(2,0,0)}(t,x,y)-\nabla _{\{x,y\}}^{}u(t,x,y)=0" /></a>

Subject to:
<a href="https://www.codecogs.com/eqnedit.php?latex=u(0,x,y)=e^{Z&space;\left(-\left((x-X)^2&plus;(y-Y)^2\right)\right)}" target="_blank"><img src="https://latex.codecogs.com/gif.latex?u(0,x,y)=e^{Z&space;\left(-\left((x-X)^2&plus;(y-Y)^2\right)\right)}" title="u(0,x,y)=e^{Z \left(-\left((x-X)^2+(y-Y)^2\right)\right)}" /></a>
and:
<a href="https://www.codecogs.com/eqnedit.php?latex=u^{(1,0,0)}(0,x,y)=0" target="_blank"><img src="https://latex.codecogs.com/gif.latex?u^{(1,0,0)}(0,x,y)=0" title="u^{(1,0,0)}(0,x,y)=0" /></a>. We solve it on an irregular geometry. We then try to predict the total amplitude within a small region of the solution domain some time later.

Here is an example of a solution at t=6.
![Setup](../master/ims/setup.png)

Here, the goal is to given the solution at t=6, the location of the original pulse, and it's amplitude, say something about the result at t=6.28. How different do they look? Below, we plot the solution at two times.
![Compare](../master/ims/diff.png)

Not totally the same! In this demo, we train a neural network to make predictions. Here, I tried to use a more typical coding style like that in a large project. 

# Demo 3
Demo problem recoloring an image given colored images. Use the left lava lamp from this video for training:
https://www.youtube.com/watch?v=rSzjFvMFQhg
