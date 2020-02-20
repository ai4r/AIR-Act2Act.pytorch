import os
import simplejson as json
import numpy as np


def read_joint(path):
    if not os.path.isfile(path):
        return None
    try:
        with open(path) as fp:
            data = fp.read()
            body_info = json.loads(data)
    except:
        print('Error occurred: ' + path)

    return body_info


def vectorize(joint):
    return np.array([joint['x'], joint['y'], joint['z']]).astype('float32')


def move_camera_to_front(body_info, body_id):
    for f in range(len(body_info)):
        if body_info[f][body_id] is None:
            continue

        # joints of the trunk
        reference_body = body_info[f][body_id]["joints"]
        r_4_kinect = vectorize(reference_body[4])  # shoulderLeft
        r_8_kinect = vectorize(reference_body[8])  # shoulderRight
        r_20_kinect = (r_4_kinect + r_8_kinect) / 2  # spineShoulder
        r_0_kinect = vectorize(reference_body[0])  # torso
        dist_to_camera = np.linalg.norm(r_20_kinect)

        # find the front direction vector
        front_vector = np.cross(r_8_kinect - r_4_kinect, r_0_kinect - r_4_kinect)
        norm = np.linalg.norm(front_vector, axis=0, ord=2)
        norm = np.finfo(front_vector.dtype).eps if norm == 0 else norm
        normalized_front_vector = front_vector / norm * dist_to_camera
        cam_pos = r_20_kinect + normalized_front_vector  # to
        cam_dir = -normalized_front_vector
        if any(x != 0 for x in cam_dir):
            start_frame = f
            break

    # rotation factors
    eye = cam_pos
    at = cam_dir
    up = r_20_kinect - r_0_kinect

    norm_at = np.linalg.norm(at, axis=0, ord=2)
    norm_up = np.linalg.norm(up, axis=0, ord=2)
    norm_at = np.finfo(at.dtype).eps if norm_at == 0 else norm_at
    norm_up = np.finfo(up.dtype).eps if norm_up == 0 else norm_up
    z_c = at / norm_at
    y_c = up / norm_up
    x_c = np.cross(y_c, z_c)

    for f in range(start_frame, len(body_info)):
        body = body_info[f][body_id]["joints"]
        # for all the 25 joints within each skeleton
        for j in range(len(body)):
            joint = body[j]

            x = joint['x'] * x_c[0] + joint['y'] * x_c[1] + joint['z'] * x_c[2] - np.dot(eye, x_c)
            y = joint['x'] * y_c[0] + joint['y'] * y_c[1] + joint['z'] * y_c[2] - np.dot(eye, y_c)
            z = joint['x'] * z_c[0] + joint['y'] * z_c[1] + joint['z'] * z_c[2] - np.dot(eye, z_c)

            joint['x'] = x
            joint['y'] = y
            joint['z'] = z

    return body_info


# move origin to torso and
# normalize to the distance between torso and spineShoulder
def norm_to_torso(body):
    torso, spineShoulder, head, shoulderLeft, elbowLeft, wristLeft, shoulderRight, elbowRight, wristRight = \
        get_upper_body_joints(body)

    features = list()
    features.extend(norm_to_distance(torso, spineShoulder, spineShoulder))
    features.extend(norm_to_distance(torso, spineShoulder, head))
    features.extend(norm_to_distance(torso, spineShoulder, shoulderLeft))
    features.extend(norm_to_distance(torso, spineShoulder, elbowLeft))
    features.extend(norm_to_distance(torso, spineShoulder, wristLeft))
    features.extend(norm_to_distance(torso, spineShoulder, shoulderRight))
    features.extend(norm_to_distance(torso, spineShoulder, elbowRight))
    features.extend(norm_to_distance(torso, spineShoulder, wristRight))
    return features


def get_upper_body_joints(body):
    shoulderRight = vectorize(body[8])
    shoulderLeft = vectorize(body[4])
    elbowRight = vectorize(body[9])
    elbowLeft = vectorize(body[5])
    wristRight = vectorize(body[10])
    wristLeft = vectorize(body[6])

    torso = vectorize(body[0])
    spineShoulder = vectorize(body[20])
    head = vectorize(body[3])

    return torso, spineShoulder, head, shoulderLeft, elbowLeft, wristLeft, shoulderRight, elbowRight, wristRight


def norm_to_distance(origin, basis, joint):
    norm = np.linalg.norm(basis - origin)
    norm = np.finfo(basis.dtype).eps if norm == 0 else norm
    result = (joint - origin) / norm if any(joint) else joint
    return result


def denorm_from_torso(features):
    # pelvis
    pelvis = np.array([0, 0, 0])

    # other joints (spineShoulder, head, shoulderLeft, elbowLeft, wristLeft, shoulderRight, elbowRight, wristRight)
    spine_len = 3.
    features = np.array(features) * spine_len

    return np.vstack((pelvis, np.split(features, 8)))
