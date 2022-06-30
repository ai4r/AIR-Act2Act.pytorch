# From Python
# It requires OpenCV installed for Python
import sys
import os
import cv2
from sys import platform

from setting import W_VIDEO, H_VIDEO

openpose_path = "C:/Users/wrko/Desktop/Code/openpose"
try:
    # Windows Import
    if platform == "win32":
        # Change these variables to point to the correct folder (Release/x64 etc.)
        sys.path.append(openpose_path + '/build/python/openpose/Release')
        os.environ['PATH'] += ';' + openpose_path + '/build/x64/Release;'
        os.environ['PATH'] += ';' + openpose_path + '/build/bin;'
        import pyopenpose as op
    else:
        # Change these variables to point to the correct folder (Release/x64 etc.)
        sys.path.append(openpose_path + '/python')
        from openpose import pyopenpose as op
except ImportError as e:
    print('Error: OpenPose library could not be found. '
          'Did you enable `BUILD_PYTHON` in CMake and have this Python script in the right folder?')
    raise e

# Custom Params (refer to include/openpose/flags.hpp for more parameters)
params = dict()
params["model_folder"] = openpose_path + "/models/"
params["render_pose"] = 1
params["net_resolution"] = "-1x368"
params["disable_blending"] = False

# Starting OpenPose
opWrapper = op.WrapperPython()
opWrapper.configure(params)
opWrapper.start()


def pose_keypoints(image):
    # Process Image
    datum = op.Datum()
    datum.cvInputData = image
    opWrapper.emplaceAndPop([datum])

    # Return Results
    return datum.poseKeypoints, datum.cvOutputData


def main():
    cap = cv2.VideoCapture(0)
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, W_VIDEO)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, H_VIDEO)

    while True:
        # 2d skeleton from video
        ret, frame = cap.read()
        key_points, output_data = pose_keypoints(frame)

        output_data = cv2.flip(output_data, 3)
        cv2.imshow(f'{W_VIDEO}x{H_VIDEO}', output_data)
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break


if __name__ == '__main__':
    main()
