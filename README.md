#### Magnetic resonance imaging (MRI) 

It is an advanced imaging technique that is used to observe a variety of diseases and parts of the body.
Neural networks can analyze these images individually (as a radiologist would) or combine them into a single 3D volume to make predictions.
At a high level, MRI works by measuring the radio waves emitting by atoms subjected to a magnetic field.

We want to be able to represent the MRI data in a form that we can input into the segmentation model. MRI images are not just a single 2D image like X-rays, they are a sequence of 3D volume seen in the axial view. Furthermore, an MRI example will be made up of multiple sequences, and this will consist of multiple 3D volumes. We'll look at how we can combine these multiple 3D volumes into one 3D volume. Let's pick a slice through the brain. A slice through the brain viewed on three different MRI sequences. The key idea that we will use to combine the information from different sequences is to treat them as different channels. We can say one of the channels is red, one is green, and one is a blue channel in the same way that we have three channels of an RGB image. The analogy of the three channels representing the RGB channels is mostly useful for us to visualize how we're going to combine the channels. 

![Channels](https://github.com/lopeselio/Brain-Tumor-Image-Segmentation-For-Magnetic-Resonance-Imaging/blob/master/Channels.PNG)

![Combined](https://github.com/lopeselio/Brain-Tumor-Image-Segmentation-For-Magnetic-Resonance-Imaging/blob/master/Combined.PNG)

The idea of using different channels also extends to when we may have four or five sequences, which can be represented with four or five channels. Once each sequence is represented with the different channel, what we do now is combine the sequences together to produce one image, which is the combination of all of the sequences. To us, this is now an RGB image. To the machine, these are channels stacked in the depth dimension. One challenge with combining these sequences is that they may not be aligned with each other. For instance, if a patient moves between the acquiring of each of these sequences, their head might be tilted in one sequence compared with the others. If the images are not aligned with each other when we combine them, the brain region at one location in the red channel does not correspond to the same location in the green or the blue channels. A preprocessing approach that's often used to fix this is called **image registration.** 
