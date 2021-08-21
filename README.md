# Nucleus-segmentation-using-U-Net

# Image Segmentation 
Image segmentation is a process of partitioning an image into multiple segments using computer vision techniques. This is different from image recognition, which assigns one or more lables to the entire image; and object detection which localizes  objects within an image by drawing a bounding box around them. Segmentation is providing pixel wise information of the contents of an image. 

Image segmentation can be further divided into two broad categories: semantic segmentation and instance segmentation. 
* ## Semantic segmentation
In semantic segmentation, each pixel is classified into a particular category. For example, if we have an image of a town, then it may contain building, people, road, vehicle. So, in semantic segmentation any pixel belonging to buildings is assigned to the same building classss. We can consider semantic segmentation as a classificationn on pixel level. 
<img src = "https://tariq-hasan.github.io/assets/images/semantic_segmentation.png">
* ## Instance segmentation 
Instance segmentation goes one step deeper and seperated disticnt objects belonging the same class. If there is two buildings each building would be assigned the building label, but with different color because they are different instances of the class. 

<img src = 'https://www.cogitotech.com/wp-content/uploads/2020/10/Instance-Segmentation-Deep-Learning.jpg.webp'>

# Application areas
Segmentation allows us to partitioning images into meaningful parts and understand the scene better. The application areas include  
* photo/video editing
* traffic controls systems
* biomedical image analysis
* automonomous vehicles etc. 

# Structure 
Deep learning approaches for segmentation have achieved high improvement compared to traditional segmentation techniques. The general structure of segmentation architecture includes encoder path and decoder path. In the encoder path, we narrow down the image and increase the dimensions with pooling and convolution operation. In the decoder path, the do upsampling and reaches out the output.  

# The U-Net Model 
The name derives from the network architecutre. The encoder path is made of many blocks which downsampling the images using pooling and the decoder mirrors those blocks in the opposite direction and upsampling the image to the original input size. Skip connections cut across the U to improve performace. 


In this work, the cell images are segmented using U-Net architecure. The following the architecture structure of the U-Net. 

<img src="https://lmb.informatik.uni-freiburg.de/people/ronneber/u-net/u-net-architecture.png">

# Dataset Description: 
You can download the dataset from the following link;
https://www.kaggle.com/c/data-science-bowl-2018/data 

This dataset contains a large number of segmented nuclei images. The images were acquired under a variety of conditions and vary in the cell type, magnification, and imaging modality (brightfield vs. fluorescence). The dataset is designed to challenge an algorithm's ability to generalize across these variations.

Each image is represented by an associated ImageId. Files belonging to an image are contained in a folder with this ImageId. Within this folder are two subfolders:

* images contains the image file.
* masks contains the segmented masks of each nucleus. This folder is only included in the training set. Each mask contains one nucleus. Masks are not allowed to overlap (no pixel belongs to two masks).
The second stage dataset will contain images from unseen experimental conditions. To deter hand labeling, it will also contain images that are ignored in scoring. The metric used to score this competition requires that your submissions are in run-length encoded format. Please see the evaluation page for details.

As with any human-annotated dataset, you may find various forms of errors in the data. You may manually correct errors you find in the training set. The dataset will not be updated/re-released unless it is determined that there are a large number of systematic errors. The masks of the stage 1 test set will be released with the release of the stage 2 test set.

# Visualization 
For visualizing the performance I used tensorboad. 
The command for running the tensorboard is the following 
!tensorboard --logdir=new_logs/ --host localhost --port 8088 

# Web Application 
This work is in progress 


# Project structure 
    - Images
        --Test 
        --Train 
    - logs 
    - data_loader.py 
    - main.py 
    - unet.py
    - nuclie_segmentation.h5
    - README.md 
    - static 
        -- css
        -- js 
        -- uploads 
    - templates
        -- index.html 
    - venv 
    - requirements.txt 


