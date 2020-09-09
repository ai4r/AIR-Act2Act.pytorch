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
    │   ├── k_clustering.py         # Label user behavior classes using K-means clustering
    │   └── model.py                # Deep neural network
    ├── utils/
    │   ├── AIR.py                  # Read AIR-Act2Act data files
    │   ├── draw.py                 # Draw 3D skeletons
    │   ├── kinect.py               # Run Kinect camera
    │   └── robot.py                # Convert between 3D joint position and robot joint angles
    ├── .gitignore
    ├── LICENSE.md
    ├── LICENSE_ko.md
    ├── README.md
    ├── generate.py                 # Demo: generating robot behaviors 
    ├── setting.py                  # Global constants
    ├── preprocess.py               # Generate training and test data
    └── recognize.py                # Demo: recognizing user behaviors


## Installation 
The scripts are tested on Windows 10 and Anaconda Python 3.6.  
You also need to install the following modules.  

$ pip install simplejson tqdm matplotlib argparse  
$ pip install scikit-learn==0.22.1  
$ pip install pandas==0.25.0  
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

### How to train with AIR-Act2Act data (optional)
1. Put all ".joint" files of the AIR-Act2Act dataset in 'joint files/'.  
2. Run ```python preprocess.py``` to extract training and test data.   
3. Run ```python user\classifier.py -m train``` to train the model.  
   All trained models are stored in 'models/lstm/'

### How to verify a trained model
1. Run ```python user\classifier.py -m verify```.  
2. Enter the model number to verify, into the command line.  
3. The average accuracy and loss of the model will be printed on the command line as follows:  
    ```  
    Validation Loss: 0.03972, Validation Acc: 0.98527  
    ```
4. The confusion matrix of the model will be displayed on a pop-up window.  
    <img src="https://user-images.githubusercontent.com/13827622/91275401-137bc300-e7bb-11ea-9029-6656fa4607c3.png" width="70%">  

### How to test a trained model with AIR-Act2Act data
1. Run ```python recognize.py -m data```.  
2. If you want to test a specific model, e.g. 'model_0010.pth',  
   run ```python recognize.py -m data -l 10```.  
2. Enter the data number to test, into the command line.  
3. The recognition results will be displayed on a pop-up window.  

### How to test a trained model with Kinect camera
1. Connect to the Kinect camera.
2. Run ```python recognize.py -m kinect```.  
3. If you want to test a specific model, e.g. 'model_0010.pth',  
   run ```python recognize.py -m kinect -l 10```.  
4. The captured video and the recognition results will be displayed on pop-up windows.  


## Robot behavior generation
The robot generates a behavior that suits the user's behavior according to predefined rules.  
Robot behavior adaptation has not yet been implemented.
It will be updated soon.

### How to test with AIR-Act2Act data
1. Run ```python generate.py -m data```.  
2. If you want to test a specific model, e.g. 'model_0010.pth',  
   run ```python generate.py -m data -l 10```   
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
    
### How to test with Kinect camera
1. Connect to the Kinect camera.
2. Run ```python generate.py -m kinect```.  
3. If you want to test a specific model, e.g. 'model_0010.pth',  
   run ```python generate.py -m kinect -l 10```   
4. The captured video and the recognition results will be displayed on pop-up windows.  
   The selected robot behavior will be printed on the command line.

### How to test with virtual Pepper robot
1. Launch Choregraphe and connect to a virtual Pepper robot.  
2. To send commands to Choregraphe, open command prompt and run server:  
   ```server\server.exe -p {port number}```  
3. Open an other command prompt, run ```python generate.py -m {mode}, -l {model number} --use_robot```.  
4. The captured video and the recognition results will be displayed on pop-up windows.  
   The selected robot behavior will be generated by a virtual Pepper in Choregraphe.  
      

## Contact
Please email wrko@etri.re.kr if you have any questions or comments.  


## Publication
Woo-Ri Ko, et al., "Adaptive Behavior Generation of Social Robots Based on User Behavior Recognition," 
Workshop on Social HRI of Human-Care Service Robots in RO-MAN 2020.  


## LICENSE
This software is a part of AIR, and follows the [AIR License and Service Agreement](LICENSE.md).
