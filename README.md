# Hippocampus Segmentation
This project is a binary segmentation problem for 3D MRI scans. The input data is for left and right hemispheres for 50 datapoints in total in NifTi file format. This project tries to workaround this data constraint to get the best possible performance from the ML model.

## Architecture
The architecture used in U-Net which is is a popular deep learning architecture known for image segmentation, recognized for its U-shaped design, skip connections, and effectiveness in various segmentation tasks.
### Approach 1 - 3D U-Net for 50 images
Since the data is 3 dimensional, it was ideal to use a 3D U-Net. However, due to data and computational resource constraints, a 3D model would introduce even more parameters, making it tougher to train. Thus, the results obtained from a 3D U-Net are quite unsatisfactory.

### Approach 2 - 48 channels for 50 images
The next approach was to use 2D U-Net itself but instead of using 1 channel ( grey-scale images ), I use 48 input channels, one for each layer in the z-axis to capture some relation between the 48 layers of . This gave substantially better results than the 3D U-Net but still ample scope for improvement.

### Approach 3 - 48 * 50 single channel images
Since we are not capturing the relational information between layers, we could sample randomly from the 48 layers in image giving us 2400 test slices. Then we use the same 2D U-Net to predict the outputs. There is some improvement but we could try a novel idea, I experimented with after discussing with colleagues.

### Approach 4 - 48 * 50 double output images
Right now, we're dealing with this as a regression problem and enforcing cutoffs. What we could do is make it a classification problem. We could output 2 channels instead of 1 and both with interchanged labels. Either we light up a pixel or we don't. In one channel we light up all background pixels and in the other we light up all foreground pixels by inverting their masks.
The final image is then chosen by comparing the magnitude of the pixel value in each mask, choosing the higher value state. This gives us good results with both BCELoss and Cross Entropy Loss.
<p align="center">
<img src="https://github.com/Anika-Roy/Hippocampus-Segmentation/assets/102136135/cfcc373f-34d4-45e7-95b5-e1eeb5b343c3" height=300 width=450>
</p>


These are all the approaches I've implemented so far, However, there are a few more strategies I would explore in the future

### Possibility 5 - Training on a larger dataset then finetuning
The network would need some more data for understanding what an MRI actually is, what regions are important, which aren't. Then we could further fine-tune it to our objective and our model. This is where we could use the secondary loss function.

### Possibility 6 - Reducing False positive error
I observed that the model often gave false positive results. What many researchers do, is first train a classifier than would determine whether the Hippocampus is present or not. Then we could train a model on those images that gave positive results to localise it.

## Experiment
### Understanding the Dataset and MRI scans

![image](https://github.com/Anika-Roy/Hippocampus-Segmentation/assets/102136135/24cfe8a0-931f-4c21-aad7-b146adc4fe21)


**About the datasets:**

(source : Continual Hippocampus Segmentation with Transformers)
1. Dryad: contains 50 cases of healthy patients
2. HarP: The Harmonized Hippocampal Protocol dataset, which we refer to as HarP, contains healthy subjects and patients with Alzheimer’s disease.

I will be using the Dryad dataset for this project.
<br><br>
**Data format**

Given the file format provided, it appears that we have two types of files for each case in the "Dryad" dataset:
1. **s01_L_gt.nii.gz**: This file likely contains the ground truth segmentation masks for the left hippocampus of a subject from the "Dryad" dataset.

2. **s01_L.nii.gz**: This file likely contains the MRI image data (e.g., T1-weighted images) for the left hippocampus of the same subject.

The filenames indicate that they correspond to the left hemisphere of the hippocampus ("L"). Since have similar files for the right hemisphere (e.g., "s01_R_gt.nii.gz" and "s01_R.nii.gz"), we will follow a similar process for those as well.
### Model Architechture: U-Net
<p align="center"><img title="U-Net architechture Diagram" alt="U-Net architechture Diagram" src="https://miro.medium.com/v2/resize:fit:1400/1*f7YOaE4TWubwaFF7Z1fzNw.png" height=500 width=800></p>

UNet, evolved from the traditional convolutional neural network, was first designed and applied in 2015 to process biomedical images. 

<i>The reason it is able to localise and distinguish borders of abnormality and disease is by doing classification on every pixel, so the input and output share the same size.</i>

The U-Net architecture is characterized by its U-shaped structure, which consists of a contracting path followed by an expansive path. Here's an overview of its key components:

1. Contracting Path (Encoder)
2. Bottleneck
3. Expansive Path (Decoder)
4. Output Layer

> Small sidenote: In the Downsampling network, simple CNN architectures are used and abstract representations of the input image are produced. In the Upsampling network, the abstract image representations are upsampled using various techniques to make their spatial dimensions equal to the input image.

(source: https://towardsdatascience.com/unet-line-by-line-explanation-9b191c76baf5, https://towardsdatascience.com/transposed-convolution-demystified-84ca81b4baba#:~:text=In%20the%20Downsampling%20network%2C%20simple,equal%20to%20the%20input%20image.)


### Losses
1. <i>Dice-Loss</i>

Dice Coefficient is similar to IoU Coefficient, both being positively correlated. I used this metric since it is very intuitive in image segmentation tasks. It is a region-based loss (as shown in the table below).  It can very well handle the class imbalance
in terms of pixel count for foreground and background

<p align="center">
  
| Type                           | Loss Function                    |
|--------------------------------|----------------------------------|
| Distribution-based Loss        | Binary Cross-Entropy             |
|                                | Weighted Cross-Entropy           |
|                                | Balanced Cross-Entropy           |
|                                | Focal Loss                       |
| Region-based Loss              | Dice Loss                        |
|                                | Sensitivity-Specificity Loss     |
|                                | Tversky Loss                     |
|                                | Focal Tversky Loss               |

</p>
The Dice coefficient ranges from 0 to 1, with 0 indicating no overlap between the sets and 1 indicating perfect overlap or similarity.

The smooth parameter is added to avoid division by zero errors when both intersection and union are zero. It's a small positive value that prevents instability in the loss calculation
<p align="center"><img src="https://cdn-images-1.medium.com/max/1600/0*HuENmnLgplFLg7Xv" height=300 width=550></p>

2. <i>Cross-Entropy Loss</i>

Cross-entropy can then be used to calculate the difference between the two probability distributions. One of the main advantages of cross-entropy loss is that it is easy to implement 
and optimize. Most neural network frameworks provide built-in functions for cross-entropy loss and its gradients. Cross-entropy loss also has a smooth and convex shape, which makes 
it easier for gradient-based optimization methods to find the global minimum.

(source: https://www.linkedin.com/advice/0/what-advantages-disadvantages-using-cross-entropy)

### Extending my pipeline to create a flip robust implementation
This will be done by:
1. Adding another loss function
2. Augmenting data by flipping the original dataset horizontally

<br>
<i>Why should we add another loss function?</i>

1. **Multi-Objective Optimization**: By adding a second loss function, we can optimize multiple objectives simultaneously. For example, in a semantic segmentation task, we may want to optimize for pixel-wise accuracy as well as object-level IoU (Intersection over Union).

2. **Balancing Trade-offs**: Different loss functions emphasize different aspects of the learning task. One loss function may encourage the model to focus on minimizing errors on one aspect, while another may emphasize a different aspect.

3. **Regularization**: Each loss function contributes to the overall loss, and this can act as a form of regularization discouraging the model from fitting the training data too closely.

4. **Handling Imbalanced Data**: In classification tasks with imbalanced datasets, we can use multiple loss functions to give different weights to classes based on their importance. For example, in a medical diagnosis task, false negatives (missed cases) might be more critical than false positives (false alarms), so we could assign higher weights to the former.

5. **Domain Adaptation**: In transfer learning or domain adaptation scenarios, we may have a pre-trained model on one domain and fine-tune it for another. In such cases, we can use a second loss function tailored to the target domain to guide the adaptation process effectively.

#### Finally we see, that the 2 channel implementation works quite well:) 
#### Hope you learnt something!
