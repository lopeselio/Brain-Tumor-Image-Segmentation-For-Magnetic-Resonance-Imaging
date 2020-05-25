#### Magnetic resonance imaging (MRI) 

It is an advanced imaging technique that is used to observe a variety of diseases and parts of the body.
Neural networks can analyze these images individually (as a radiologist would) or combine them into a single 3D volume to make predictions.
At a high level, MRI works by measuring the radio waves emitting by atoms subjected to a magnetic field.

We want to be able to represent the MRI data in a form that we can input into the segmentation model. MRI images are not just a single 2D image like X-rays, they are a sequence of 3D volume seen in the axial view. Furthermore, an MRI example will be made up of multiple sequences, and this will consist of multiple 3D volumes. We'll look at how we can combine these multiple 3D volumes into one 3D volume. Let's pick a slice through the brain. A slice through the brain viewed on three different MRI sequences. The key idea that we will use to combine the information from different sequences is to treat them as different channels. We can say one of the channels is red, one is green, and one is a blue channel in the same way that we have three channels of an RGB image. The analogy of the three channels representing the RGB channels is mostly useful for us to visualize how we're going to combine the channels. 

![Channels](https://github.com/lopeselio/Brain-Tumor-Image-Segmentation-For-Magnetic-Resonance-Imaging/blob/master/Channels.PNG) ![Combined](https://github.com/lopeselio/Brain-Tumor-Image-Segmentation-For-Magnetic-Resonance-Imaging/blob/master/Combined.PNG)

The idea of using different channels also extends to when we may have four or five sequences, which can be represented with four or five channels. Once each sequence is represented with the different channel, what we do now is combine the sequences together to produce one image, which is the combination of all of the sequences. To us, this is now an RGB image. To the machine, these are channels stacked in the depth dimension. One challenge with combining these sequences is that they may not be aligned with each other. For instance, if a patient moves between the acquiring of each of these sequences, their head might be tilted in one sequence compared with the others. If the images are not aligned with each other when we combine them, the brain region at one location in the red channel does not correspond to the same location in the green or the blue channels. A preprocessing approach that's often used to fix this is called **image registration.** 

### Segmentation 2-dim and 3-dim
Segmentation is the process of defining the boundaries of various tissues. In this case, we're trying to define the boundaries of tumors. We can also think of segmentation as the task of determining the class of every point in the 3D volume. These points in 2D space are called **pixels** and in 3D space are called **voxels**

#### 2-D segmentation and 3-D segmentation
![2d](https://github.com/lopeselio/Brain-Tumor-Image-Segmentation-For-Magnetic-Resonance-Imaging/blob/master/2Dapproach.PNG)     ![3D](https://github.com/lopeselio/Brain-Tumor-Image-Segmentation-For-Magnetic-Resonance-Imaging/blob/master/3Dapproach.PNG)

In the **2D approach**, we break up the 3D MRI volume we've built into many 2D slices. Each one of these slices is passed into a segmentation model which outputs the segmentation for that slice, one by one. The drawback with this 2D approach is that we might lose important 3D context when using this approach. For instance, if there is a tumor in one slice, there is likely to be a tumor in the slices right adjacent to it. Since we're passing in slices one at a time into the network, the network is not able to learn this useful context.

In the **3D approach** the size of the MRI volume makes it impossible to pass it in all at once into the model. It would simply take too much memory and computation.  The disadvantage with this 3D approach is that we might still lose important spatial cortex. For instance, if there is a tumor in one sub volume, there is likely to be a tumor in the sub volumes around it too. Since we're passing in sub volumes one at a time into the network, the network will not be able to learn this possibly useful context. The silver lining with the 3D approach is that that we're capturing some context in all of the width, height, and depth mentions. This covers a 3D approach to segmentation.

## Tumor images from MRI taken from random layers. The image is divided into 154 layers. There are four types of tumors
1. Normal
2. Endema
3. Enhancing 
4. Non-Enhancing
You can check them [here](https://github.com/lopeselio/Brain-Tumor-Image-Segmentation-For-Magnetic-Resonance-Imaging/blob/master/analysing%20the%203D%20MRI.ipynb)

![1](https://github.com/lopeselio/Brain-Tumor-Image-Segmentation-For-Magnetic-Resonance-Imaging/blob/master/layer%2050%204%20types.PNG)    ![2](https://github.com/lopeselio/Brain-Tumor-Image-Segmentation-For-Magnetic-Resonance-Imaging/blob/master/layer%2054.PNG)

## UNET Architecture 
![2](https://github.com/lopeselio/Brain-Tumor-Image-Segmentation-For-Magnetic-Resonance-Imaging/blob/master/Unet.PNG) ![3](https://github.com/lopeselio/Brain-Tumor-Image-Segmentation-For-Magnetic-Resonance-Imaging/blob/master/3dUNET.PNG)

One of the most popular architectures for segmentation has been the U-Net. The U-Net was first designed for biomedical image segmentation and demonstrated great results on the task of cell tracking. The cool thing about the U-Net, is that it can achieve relatively good results, even with hundreds of examples.

The U-Net consists of two paths: a contracting path, and an expanding path. The contracting path is a typical convolutional network like used in image classification. It consists of repeated application of convolution and pooling operations. The convolution operation here is called a down convolution. The key here, is that in the contracting path, our feature maps gets spatially smaller, which is why it's called a contraction. Then, there is the expanding path. The expanding path in some ways is doing the opposite of the contracting path. It's taking our small feature maps through a series of up-sampling and up-convolution steps to get back to the original size of the image. It also concatenates the up-sample representations at each step with the corresponding feature maps at the contraction pathway. Finally, at the last step, the architecture outputs the probability of tumor for every pixel in the image. The U-Net architecture can be trained on input output pairs of 2D slices in the 2D approach. We've covered the U-Net for the 2D approach. Let's see what we can do when we have 3D subvolumes in the 3D approach. We can feed any 3D subvolume into a segmentation architecture, if we can replace all of the 2D operations with their 3D counterparts. This is exactly what an extension to the U-Net called the 3D U-Net does. The 2D convolutions become 3D convolutions, and the 2D pooling layers become 3D pooling layers. It's okay if you haven't seen 3D convolutions before. All that's important to understand here is that the 3D U-Net allows us to pass in 3D subvolumes and get an output for every voxel in the volume specifying the probability of tumor. The 3D U-Net can be trained on subvolume input and outputs as part of the 3D approach.  3D U-Net allows us to pass in 3D subvolumes and get an output for every voxel in the volume specifying the probability of tumor. The 3D U-Net can be trained on subvolume input and outputs as part of the 3D approach.


### Data Augmentation and loss function 
One key difference with data augmentation during segmentation is we now have a segmentation output. So when we rotate an input image by 90 degrees to produce a transformed input. We also need to rotate the output segmentations by 90 degrees to get our transformed output segmentation. The second difference is that we now have 3D volumes instead of 2D images. So the transformations have to apply to the whole 3D volume. With this, we almost have all of the pieces necessary to train our brain tumor segmentation model. The final thing we need to look at is the loss function.

<img src="https://github.com/lopeselio/Brain-Tumor-Image-Segmentation-For-Magnetic-Resonance-Imaging/blob/master/dataaugmentation.PNG" height="500" width="500">
<img src="https://github.com/lopeselio/Brain-Tumor-Image-Segmentation-For-Magnetic-Resonance-Imaging/blob/master/groundtruth%26pred.PNG" height="500" width="500">

A simple 2D example here will allow us to get a quick intuition. Here P represents the output of the segmentation model on 9 pixels. Whereat each location, we have the predicted probability of tumor. G specifies the ground truth on each of these pixel locations. Three of the nine pixels are tumor represented as 1, and the remaining six are normal brain tissue represented as 0.
