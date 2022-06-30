# AIR-Act2Act.pytorch
A pytorch implementation of AIR-Act2Act.


## Overall system of the AIR-Act2Act
A Kinect sensor captures the user's 3D joint positions through skeletal tracking.  
Then, the user's behavior is recognized using a deep neural network (DNN).  
The robot's behavior that responds appropriately to the user's behavior   
is selected according to the predefined rules.  
Finally, the selected behavior is modified according to the user's posture.  

<img src="https://user-images.githubusercontent.com/13827622/89387344-50651480-d73d-11ea-98c5-d32dc093d07c.png" width="70%">


## Source files
    .
    ├── data files/                 # Path to save the extracted data files
    ├── joint files/                # Path to move the joint files of AIR-Act2Act dataset
    ├── robot/
    │   ├── server/
    │   │    └── server.exe         # Connect to a virtual Pepper robot in Chreographe
    │   ├── adapter.py              # Adapt robot behaviors to the user posture
    │   └── selecter.py             # Select robot behaviors according to the user behavior
    ├── user/
    │   ├── models/
    │   │    ├── k-means/           # K-means clustering models to label the user behavior classes
    │   │    └── lstm/              # DNN models to recognize the user behavior
    │   ├── classifier.py           # Train and test the DNN models for user behavior recognition
    │   ├── constants.py            # Global constants
    │   ├── data.py                 # Get AIR-Act2Act dataset
    │   ├── label.py                # Label user behavior classes using K-means clustering
    │   └── model.py                # Deep neural network
    ├── utils/
    │   ├── AIR.py                  # Read AIR-Act2Act data files
    │   ├── draw.py                 # Draw 3D skeletons
    │   ├── kcluster.py             # K-means clustering
    │   ├── kinect.py               # Run Kinect camera
    │   ├── openpose.py             # Detect 2D skeleton from RGB image
    │   └── robot.py                # Convert between 3D joint position and robot joint angles
    ├── .gitignore
    ├── LICENSE.md
    ├── LICENSE_ko.md
    ├── README.md
    ├── generate.py                 # Demo: generating robot behaviors 
    ├── setting.py                  # Global constants
    ├── preprocess.py               # Generate training and test data
    └── recognize.py                # Demo: recognizing user behaviors
    └── requirements.txt            # installed packages


## Installation 
The scripts are tested on Windows 10 and Anaconda Python 3.6.  
You also need to install the following modules.  

```
$ conda install pytorch==1.1.0 torchvision==0.3.0 cudatoolkit=9.0 -c pytorch  
$ pip install simplejson tqdm matplotlib seaborn argparse opencv-python  
$ pip install pandas==0.25.0  
$ pip install scikit-learn==0.22.1  
```
[CUDA Toolkit](https://developer.nvidia.com/cuda-downloads) 9.0 installation

    
## Prerequisites

### To train and test with the AIR-Act2Act dataset 
We already provide a trained model in 'models/lstm/A001A004A005A007A008/vector/2D/False/model_0045.pth'.  
But if you want to train the model by yourself, download the [AIR-Act2Act dataset](https://nanum.etri.re.kr/share/list?lang=En_us).  
You need to join as a member to get to the download page.  
The data all you need is the refined 3D skeleton files (.joint) of P001-P050.  
For more information on the AIR-Act2Act dataset, please visit [here](https://ai4robot.github.io/air-act2act-en/#). 

### To test with a Webcam
If you want to use a Webcam to test, you need to download [OpenPose v1.6.0](https://github.com/CMU-Perceptual-Computing-Lab/openpose/releases/tag/v1.6.0).   
Then, install the OpenPose using [CMake](https://cmake.org/download/) following the [instructions](https://velog.io/@oneul1213/OpenPose-%EA%B0%9C%EC%9A%94-%EB%B0%8F-%EC%84%A4%EC%B9%98%ED%95%98%EA%B8%B0).  
After setting ```openpose_path```, you can test the OpenPose by running ```python utils\openpose.py```.

### To test with a Kinect camera
If you want to use a Kinect camera to test, you need to install [Kinect for Windows SDK 2.0](https://www.microsoft.com/en-us/download/details.aspx?id=44561).    
Then, install the Pykinect2 and PyGame modules: ```pip install pykinect2 pygame```.  
You can test your Kinect sensor by running ```python utils\kinect.py```  
To solve ```AssertionError: 80```, replace the ```.py``` files in ```Lib\site-packages\pykinect2``` with [these files](https://github.com/Kinect/PyKinect2/tree/master/pykinect2).  
To solve ```ImportError("Wrong version")```, reinstall comtypes as ```pip install comtypes==1.1.4```  
  
### To test with a virtual Pepper robot
If you want to use a virtual Pepper robot, you need to download [Choregraphe](https://www.robotlab.com/choregraphe-download).  
We recommend to download **Choregraphe for PC** for **NAO V6**.  
After installing Choregraphe, launch it.  
Then, cilck [Edit] - [Preferences] - [Virtual Robot] and select Pepper for Robot model.  
You need to remember the port number which is written at the bottom.  


## User recognition

### LSTM-based model
To recognize the user's behavior, we use a long short-term memory (LSTM) based model.  
The LSTM is a popular model in sequential data understanding and makes a great success.  
The input is set to a sequence of feature vectors of user poses.  
The output is set to a one-hot vector of behavior class label.  
The user recognition is performed at 10 fps. 

<img src="https://user-images.githubusercontent.com/13827622/89415107-63400f00-d766-11ea-9008-6634fb496087.png" width="60%">  

### How to train with AIR-Act2Act data (optional)
1. Put all ".joint" files of the AIR-Act2Act dataset in ```joint files/```.   
1. Run ```python preprocess.py``` to extract training and test data.   
1. Run ```python user\label.py``` to label sub-action classes for all data.    
    1. If the K-means clustering models do not exist in ```user/models/k-means/```,  
    run ```python utils\kcluster.py``` after changing parameters.  
1. Change parameters in ```setting.py```.  
    ```
    INPUT_DATA_TYPE = '2D'  # {'2D', '3D'}
    NORM_METHOD = 'vector'  # {'vector', 'torso'}
    B_HANDS = False  # {True, False}
    ACTIONS = ['A001', 'A004', 'A005', 'A007', 'A008']
    ```
1. Run ```python user\classifier.py -m train``` to train the model.  
   (All trained models are stored in ```user/models/lstm/```)

### How to verify the trained model
1. Run ```python user\classifier.py -m verify```.  
1. Enter the model number to verify, into the command line.  
1. The average accuracy and loss of the model will be printed on the command line as follows:  
    ```  
    Test Loss: 0.03972, Test Acc: 0.98527  
    ```
1. The confusion matrix of the model will be displayed on a pop-up window.  
    <img src="https://user-images.githubusercontent.com/13827622/91275401-137bc300-e7bb-11ea-9029-6656fa4607c3.png" width="70%">  

### How to test with AIR-Act2Act data
1. Run ```python recognize.py -m data```.  
1. If you want to test a specific model, e.g. 'model_0010.pth',  
   run ```python recognize.py -m data -l 10```.  
1. Enter the data number to test, into the command line.  
1. The recognition results will be displayed on a pop-up window.  

### How to test with Webcam (in real-time)
1. Set ```INPUT_DATA_TYPE = '2D``` in ```setting.py```.  
1. Connect to the Webcam.
1. Run ```python recognize.py -m webcam```.  
1. If you want to test a specific model, e.g. 'model_0010.pth',  
   run ```python recognize.py -m webcam -l 10```.  
1. The captured video and the recognition results will be displayed on pop-up windows.  

### How to test with Kinect camera (in real-time)
1. Set ```INPUT_DATA_TYPE = '3D``` in ```setting.py```.  
1. Connect to the Kinect camera.
1. Run ```python recognize.py -m kinect```.  
1. If you want to test a specific model, e.g. 'model_0010.pth',  
   run ```python recognize.py -m kinect -l 10```.  
1. The captured video and the recognition results will be displayed on pop-up windows.  

## Robot behavior generation
The robot generates a behavior that suits the user's behavior according to predefined rules.  
Robot behavior adaptation has not yet been implemented.
It will be updated soon.

### How to test with AIR-Act2Act data
1. Run ```python generate.py -m data```.  
1. If you want to test a specific model, e.g. 'model_0010.pth',  
   run ```python generate.py -m data -l 10```   
1. Enter the data number to test into the command line.   
1. The input user behavior will be displayed on a pop-up window.  
   The selected robot behavior will be printed on the command line as follows:  
    ```  
    robot: stand  
    robot: stand  
    ...  
    robot: handshake  
    robot: handshake  
    ...  
    robot: avoid  
    ```  
    
### How to test with Webcam (in real-time)
1. Set ```INPUT_DATA_TYPE = '2D``` in ```setting.py```.  
1. Connect to the Webcam.
1. Run ```python generate.py -m webcam```.  
1. If you want to test a specific model, e.g. 'model_0010.pth',  
   run ```python generate.py -m webcam -l 10```   
1. The captured video and the recognition results will be displayed on pop-up windows.  
   The selected robot behavior will be printed on the command line.
    
### How to test with Kinect camera (in real-time)
1. Set ```INPUT_DATA_TYPE = '3D``` in ```setting.py```.  
1. Connect to the Kinect camera.
1. Run ```python generate.py -m kinect```.  
1. If you want to test a specific model, e.g. 'model_0010.pth',  
   run ```python generate.py -m kinect -l 10```   
1. The captured video and the recognition results will be displayed on pop-up windows.  
   The selected robot behavior will be printed on the command line.

### How to test with virtual Pepper robot
1. Launch Choregraphe and connect to a virtual Pepper robot.  
1. To send commands to Choregraphe, open command prompt and run server:  
   ```robot\server\server-1.0.exe -p {port number} -r pepper```  
1. Open an other command prompt, run ```python generate.py -m {mode}, -l {model number} --use_robot```.  
1. The captured video and the recognition results will be displayed on pop-up windows.  
   The selected robot behavior will be generated by a virtual Pepper in Choregraphe.  
      

## Contact
Please email wrko@etri.re.kr if you have any questions or comments.  


## Publication
Woo-Ri Ko, et al., "Adaptive Behavior Generation of Social Robots Based on User Behavior Recognition," 
Workshop on Social HRI of Human-Care Service Robots in RO-MAN 2020.  


## LICENSE
This software is a part of AIR, and follows the [AIR License and Service Agreement](LICENSE.md).
