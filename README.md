# COVID-19 Chest X-Ray Data Analysis

[![N|Solid](https://i.ibb.co/wQkHfSR/ezgif-com-crop.gif)](https://github.com/devindatt/covid19-chest_xrays-analysis/blob/master/ezgif.com-crop.gif)
#
https://github.com/devindatt/covid19-chest_xrays-analysis/blob/master/ezgif.com-crop.gif

## Disclaimer: 
##### *This analysis is only performed on a SMALL dataset (50 images) and shouldn't be considered true academic research of any kind to draw on any conclusions of its validity. Further research needs to be performed by trained medical practitioner on a much larger dataset (+10,000s) to see a true correlation.  Having said that there does seems to be an interesting relationship that could help medical staff for quick diagnosis.*
 
 #
 #
 #
 
 ## Objective
  - To apply Data Science processes to train a deep learning model using Keras and TensorFlow to see if we can "predict" COVID-19 from only analyzing chest X-rays of patients
  - To see if AI can be used to build a quick diagnosis tool for incoming patients



## Reasoning for this Analysis
- COVID-19 tests are currently hard to come by so we need to rely on other diagnosis measures.
- COVID-19 attacks the epithelial cells that line the respiratory tract (source here)
- Nearly all hospitals have X-ray imaging machines use X-rays to analyze the health of a patient’s lungs
- X-rays and CT scans are used to diagnose pneumonia, lung inflammation, abscesses, and/or enlarged lymph nodes.
- Since X-ray analysis requires a radiology expert to interpret scan results which are in short supply can we develop tools to shortcut the path to a reliable diagnosis for medical practitioners 

#
#

## Dataset Details

##### COVID19 Data:
- Created by [Dr. Joseph Cohen](https://josephpcohen.com/w/), a postdoctoral fellow of Sergio Bengio at the University of Montreal.
- This is is dataset that consists of scans of chest x-ray images [found here](https://github.com/ieee8023/covid-chestxray-dataset/tree/master/images) for cases of MERS, SARS, ARDS as well as COVID-19
- Select only the scans for COVID-19 cases, which only had 25 images
- Choosing only Posterior-Anterior view (back-to-front) scans, [assuming for now this is the best view](https://reference.medscape.com/features/slideshow/chest-x-ray) for a model to detect the presence of the virus

##### Non-COVID19 Data:
- To balance the dataset I choose non-convid19 cases from the Kaggle chest X-ray Pneumonia dataset
- Total images in curated 50 X-ray scans (25 COVID19, 25 Non-COVID19)
#
#
[![N|Solid](https://i.ibb.co/gzhfmn1/covid19-keras-dataset2.png)](https://i.ibb.co/gzhfmn1/covid19-keras-dataset2.png)


#
#


## Running Code
#
Assuming you download and keep the default settings, the dataset should be in the 'dataset' folder and the model file is in the main directory, the following command should allow you to run the training model as the only mandatory parameter is the location of the dataset folder.

```sh
$ python3 train_covid19.py --dataset dataset 
```
#
The above command will output the resulting train/validation plots in a file with a default name 'plot.png'. If you want to change this name you can simply use the 'plot' parameter and use any name, such as:
```sh
$ python3 train_covid19.py --dataset dataset --plot plot_filename.png 
```
#
#
You can also use short hand parameters and even use your own model instead of the default 'covid19' model:
```sh
$ python3 train_covid19.py -d dataset -m covid19.model -p plot_filename.png 
```


## Analysis Step:
Initial model settings, which you can change
- Learning rate = 0.001
- Trainig epochs = 25
- Batch size = 8

1) Gather the images in the dataset directory and initialize arrays one for list of image data and another for the image class (covid or non-covid)
2) Resize images to 224x224 pixels ignoring aspect ratio
3) Convert data and labels to NumPy arrays while normalizing pixel intensities to the range
4) Compute one-hot encoding on the labels to allow easier feature analysis
5) Split into 80% training and 20% testing datasets
6) Initialize the training data augmentation object generator
7) Perform Transfer Learning by loading in the VGG16 model layers but leave off the Fully Connected layer as this will need to be retrained
8) Create a new Fully Connected layer and connect to the base model
9) Start training new model but freeze the base parameters so it only trains the FC layers
10) Make some predications
11) Print out results
12) Save trained model

#
#
## Results:

[![N|Solid](https://i.ibb.co/55MpSWj/result-sshot1.png)](https://i.ibb.co/55MpSWj/result-sshot1.png)
#
- A - Shows the Training Loss is low (~28%) and high accuracy (95%)
- B - Shows the models Precision is High (83%), and model to the ground truth Recall (89%), F1 score (~90%)
- C - Shows on the validation testing High accuracy (90%), sensitivity (80%), and specificity (100%)
#
[![N|Solid](https://i.ibb.co/wCLLLJk/plot3.png)](https://i.ibb.co/wCLLLJk/plot3.png)

The green circle shows how the validation results are closely tracking the training results as increasing the epochs, which is what we want to see if there is something the model able to see some type of correlation.

My ethical conclusion here is,  you don’t need a degree in medicine to make an impact in the medical field — deep learning practitioners working closely with doctors and medical professionals can solve complex problems, save lives, and make the world a better place.


#
#
## Fraud associated with COVID19 Outbreak:
- Selling fake COVID19 test kits - namely by [selling fake COVID-19 test kits](https://abc7news.com/5995593/)
- Victims on social media falling for fake COVID-19 home testing kits  - [finding victims on social media platforms and chat applications](https://www.edgeprop.my/content/1658343/covid-19-home-testing-kits-are-fake-medical-authority)

#
#

