# AIR-Act2Act.pytorch
A pytorch implementation of AIR-Act2Act.


## Overall system of the AIR-Act2Act
A Kinect sensor captures the user’s 3D joint positions through skeletal tracking.  
Then, the user's behavior is recognized using a deep neural network (DNN).  
The robot's behavior that responds appropriately to the user's behavior   
is selected according to the predefined rules.  
Finally, the selected behavior is modified according to the user's posture.  

<img src="https://user-images.githubusercontent.com/13827622/89387344-50651480-d73d-11ea-98c5-d32dc093d07c.png" width="70%">


## Source files
    .
    ├── data files/                 # Path to save the extracted data files
    ├── joint files/                # Path to move the joint files of AIR-Act2Act dataset
    ├── models/                     # Path to save the trained models
    │   └── k-means/                # K-means clustering models to label the user behavior classes
    ├── server/
    │   └── server.exe              # Connect to a virtual Pepper robot in Chreographe
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
    ├── gen_behavior.py             # Select and adapt robot behaviors
    ├── k_clustering.py             # Label user behavior class using K-means clustering
    ├── model.py                    # Deep neural network
    ├── preprocess.py               # Generate training and test data
    └── recog_subaction.py          # Train and test the DNN for user behavior recognition


## Installation 
The scripts are tested on Windows 10 and Anaconda Python 3.6.  
You also need to install the following modules.  

$ pip install simplejson tqdm matplotlib argparse pandas  
$ pip install scikit-learn==0.22.1  
$ conda install pytorch=0.4.1 cuda92 -c pytorch  
cuda 9.2 installation - [here](https://developer.nvidia.com/cuda-92-download-archive?target_os=Windows&target_arch=x86_64&target_version=10&target_type=exelocal)

    
## Prerequisites

### To train and test with the AIR-Act2Act dataset 
We already provide a trained model in 'models/lstm/vector/model_0013.pth'.  
But if you want to train the model by yourself, download the [AIR-Act2Act dataset](http://nanum.etri.re.kr:8080/etriPortal/login?language=en).  
You need to join as a member to get to the download page.  
The data all you need is the refined 3D skeleton files (.joint) of P001-P050.  
For more information on the AIR-Act2Act dataset, please visit [here](https://ai4robot.github.io/air-act2act-en/#). 

### To test with a Kinect camera
If you want to use a Kinect camera to test, you need to install [Kinect for Windows SDK 2.0](https://www.microsoft.com/en-us/download/details.aspx?id=44561).    
Then, install the Pykinect2 and PyGame modules: ```pip install pykinect2 pygame```.  
  
### To test with a virtual Pepper robot
If you want to use a virtual Pepper robot, you need to download [Choregraphe](https://www.robotlab.com/choregraphe-download).  
We recommend to download **Choregraphe for PC** for **NAO V6**.  
After installing Choregraphe, launch it.  
Then, cilck [Edit] - [Preferences] - [Virtual Robot] and select Pepper for Robot model.  
You need to remember the port number written at the bottom.  


## User recognition

### LSTM-based model
To recognize the user's behavior, we use a long short-term memory (LSTM) based model.  
The LSTM is a popular model in sequential data understanding and makes a great success.  
The input is set to a sequence of feature vectors of user poses.  
The output is set to a one-hot vector of behavior class label.  
The user recognition is performed at 10 fps. 

<img src="https://user-images.githubusercontent.com/13827622/89415107-63400f00-d766-11ea-9008-6634fb496087.png" width="60%">  

### How to train with AIR-Act2Act data
1. Put all ".joint" files of the AIR-Act2Act dataset in 'joint files/'.  
2. Run ```python preprocess.py``` to extract training and test data.   
3. Run ```python recog_subaction.py -m train``` to train the model.  
All trained models are stored in 'models/lstm/'

### How to test with AIR-Act2Act data
1. Run ```python recog_subaction.py -m test```.  
2. Enter the model number and data number to test, into the command line.  
3. The recognition results will be printed as follows:  
    ```  
    true:  
    [4, 4, 4, 4, 4, 4, 5, 5, 5, 5, 5, 5, 5, 5, 5, 5, 5, 5, 5, 5, 5, 5, 5, 5, 6, 6, 6, 6, 6, 6, 6, 6, 6, 0, 0, 0, 0]  
    pred:  
    [4, 4, 4, 4, 4, 4, 5, 5, 5, 5, 5, 5, 5, 5, 5, 5, 5, 5, 5, 5, 5, 5, 5, 5, 6, 6, 6, 6, 6, 6, 6, 6, 6, 0, 0, 0, 0]  
    ```  
    
### How to test with a Kinect camera
1. Run ```python demo.py -m recognize```.  
2. The captured video will be displayed on a pop-up window.  
The recognized user behavior will be printed on the command line as follows:  
    ```  
    stand  
    stand  
    ...  
    raise right hand  
    raise right hand  
    ...  
    threaten to hit with right hand  
    ...  
    ```


## Robot behavior generation

The robot generates a behavior that suits the user's behavior according to predefined rules.  
Robot behavior adaptation has not yet been implemented.
It will be updated soon.

### How to test with AIR-Act2Act data
1. Run ```python gen_behavior.py -m test```.  
2. Select the trained model as ```python gen_behavior.py -m test -l 10```.  
3. Enter the data number to test into the command line.   
4. The input user behavior will be displayed on a pop-up window.  
The selected robot behavior will be printed on the command line as follows:  
    ```  
    stand  
    stand  
    ...  
    handshake  
    handshake  
    ...  
    avoid  
    ```  

### How to test with AIR-Act2Act data and virtual Pepper robot
1. Launch Choregraphe and connect to a virtual Pepper robot.  
2. To send commands to Choregraphe, open command prompt and run server:  
```server\server.exe -p {port number}```  
3. Open an other command prompt, run ```python gen_behavior.py -m test_pepper```.  
4. The input user behavior will be displayed on a pop-up window.  
The selected robot behavior will be generated by a virtual Pepper in Choregraphe.  

### How to test with Kinect camera and virtual Pepper robot
1. Launch Choregraphe and connect to a virtual Pepper robot.  
2. Run server: ```server\server.exe -p {port number}```  
3. Run Act2Act: ```python demo.py -m generate```  
4. The captured video will be displayed on a pop-up window.  
The selected robot behavior will be generated by a virtual Pepper in Choregraphe. 
      

## Contact
Please email wrko@etri.re.kr if you have any questions or comments.  


## Publication
Woo-Ri Ko, et al., "Adaptive Behavior Generation of Social Robots Based on User Behavior Recognition," 
Workshop on Social HRI of Human-Care Service Robots in RO-MAN 2020, *submitted*.  


## LICENSE
This software is a part of AIR, and follows the [AIR License and Service Agreement](LICENSE.md).
