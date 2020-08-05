# AIR-Act2Act.pytorch
A pytorch implementation of AIR-Act2Act.

## Setting 
-   Python = 3.6.10     
-   Pytorch = 0.4.1    

## Source files
    .
    ├── data files/                 # Path to save the extracted data files
    ├── joint files/                # Path to move the joint files of AIR-Act2Act dataset
    ├── models/                     # Path to save the trained models
    │   └── k-means/                # K-means clustering models to label the user behavior classes
    ├── server/
    │   └── server.exe              # Connect to a simulated Pepper robot in Chreographe
    ├── utils/
    │   ├── AIR.py                  # Read AIR-Act2Act data files
    │   ├── draw.py                 # Draw 3D skeletons
    │   ├── kinect.py               # Run Kinect camera
    │   └── robot.py                # Convert between 3D joint position and robo joint angles
    ├── .gitignore
    ├── LICENSE.md
    ├── LICENSE_ko.md
    ├── README.md
    ├── constants.py                # Global constants
    ├── data.py                     # Get AIR-Act2Act dataset
    ├── demo.py                     # Demo with Kinect camera
    ├── gen_behavior.py             # Test behavior generation
    ├── k_clustering.py             # Label user behavior class using K-means clustering
    ├── model.py                    # Deep neural network
    ├── preprocess.py               # Generate training and test data
    ├── recog_subaction.py          # Train and test DNN for user behavior recognition
    └── requirements.txt

## Overall system
  
<img src="https://user-images.githubusercontent.com/13827622/89387344-50651480-d73d-11ea-98c5-d32dc093d07c.png" width="65%">

## User recognition

### LSTM-based model

<img src="https://user-images.githubusercontent.com/13827622/89415107-63400f00-d766-11ea-9008-6634fb496087.png" width="55%">

### How to train

1. Download the AIR-Act2Act dataset [here](http://nanum.etri.re.kr:8080/etriPortal/login?language=en).  
You need to join as a member to get to the download page.  
The data you need is the refined 3D skeleton files (.joint) of P001-P050.  
For more information on the AIR-Act2Act dataset, please visit [here](https://ai4robot.github.io/air-act2act-en/#). 
2. Put all ".joint" files in 'joint files/'.  
3. Run ```python preprocess.py``` to extract training and test data.   
4. Run ```python recog_subaction.py --mode train``` to train the model.  
All trained models are stored in 'models/lstm/'

### How to test

1. If you want to test with the data extracted from AIR-Act2Act,  
and run ```python recog_subaction.py --mode test```.  
You need to enter the model number and data number to test in command line.  
The recognition results will be printed as follows:  
```
true:
[4, 4, 4, 4, 4, 4, 5, 5, 5, 5, 5, 5, 5, 5, 5, 5, 5, 5, 5, 5, 5, 5, 5, 5, 6, 6, 6, 6, 6, 6, 6, 6, 6, 0, 0, 0, 0]
pred:
[4, 4, 4, 4, 4, 4, 5, 5, 5, 5, 5, 5, 5, 5, 5, 5, 5, 5, 5, 5, 5, 5, 5, 5, 6, 6, 6, 6, 6, 6, 6, 6, 6, 0, 0, 0, 0]
```
2. If you want to test with a Kinect camera, you need to install [Kinect for Windows SDK 2.0](https://www.microsoft.com/en-us/download/details.aspx?id=44561).  
(To be updated)

## Robot behavior generation

(To be updated)

## Contact
Please email wrko@etri.re.kr if you have any questions or comments.  

## LICENSE
This software is a part of AIR, and follows the [AIR License and Service Agreement](LICENSE.md).