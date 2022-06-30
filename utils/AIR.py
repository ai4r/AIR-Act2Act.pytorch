import os
import simplejson as json
import numpy as np

# Kinect v.2 camera coefficients
KINECT_CO = dict()
KINECT_CO['cfx'], KINECT_CO['cfy'] = 1144.361, 1147.337  # (mm)
KINECT_CO['cu0'], KINECT_CO['cv0'] = 966.359, 548.038
KINECT_CO['dfx'], KINECT_CO['dfy'] = 388.198, 389.033  # (mm)
KINECT_CO['du0'], KINECT_CO['dv0'] = 253.270, 213.934
KINECT_CO['k1'], KINECT_CO['k2'], KINECT_CO['k3'] = 0.108, -0.125, 0.062
KINECT_CO['p1'], KINECT_CO['p2'] = -0.001, -0.003
co = KINECT_CO


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


def vectorize(joint, type):
    if type == '2D':
        return vectorize2D(joint)
    if type == '3D':
        return vectorize3D(joint)


def vectorize2D(joint):
    return np.array([joint['colorX'], joint['colorY']])


def vectorize3D(joint):
    return np.array([joint['x'], joint['y'], joint['z']]).astype('float32')


def move_camera_to_front(body_info, body_id):
    for f in range(len(body_info)):
        if body_info[f][body_id] is None:
            continue

        # joints of the trunk
        reference_body = body_info[f][body_id]["joints"]
        r_4_kinect = vectorize3D(reference_body[4])  # shoulderLeft
        r_8_kinect = vectorize3D(reference_body[8])  # shoulderRight
        r_20_kinect = (r_4_kinect + r_8_kinect) / 2  # spineShoulder
        r_0_kinect = vectorize3D(reference_body[0])  # torso
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

    # 3d-to-2d projection coefficients
    for f in range(start_frame):
        body = body_info[f][body_id]["joints"]
        for j in range(len(body)):
            body[j]['colorX'] = 0.
            body[j]['colorY'] = 0.

    for f in range(start_frame, len(body_info)):
        body = body_info[f][body_id]["joints"]
        for j in range(len(body)):
            joint = body[j]

            x = joint['x'] * x_c[0] + joint['y'] * x_c[1] + joint['z'] * x_c[2] - np.dot(eye, x_c)
            y = joint['x'] * y_c[0] + joint['y'] * y_c[1] + joint['z'] * y_c[2] - np.dot(eye, y_c)
            z = joint['x'] * z_c[0] + joint['y'] * z_c[1] + joint['z'] * z_c[2] - np.dot(eye, z_c)

            joint['x'] = x
            joint['y'] = y
            joint['z'] = z

            # project 3d point cloud to 2d depth map
            r2 = pow(x, 2) + pow(y, 2)
            x_corrected = x * (1 + co['k1'] * r2 + co['k2'] * pow(r2, 2) + co['k3'] * pow(r2, 3)) + \
                          2 * co['p1'] * x * y + co['p2'] * (r2 + 2 * pow(x, 2))
            y_corrected = y * (1 + co['k1'] * r2 + co['k2'] * pow(r2, 2) + co['k3'] * pow(r2, 3)) + \
                          co['p1'] * (r2 + 2 * pow(y, 2)) + 2 * co['p2'] * x * y

            joint['depthX'] = co['du0'] + co['dfx'] * x_corrected / z
            joint['depthY'] = co['dv0'] - co['dfy'] * y_corrected / z

            joint['colorX'] = co['cu0'] + co['cfx'] * x_corrected / z
            joint['colorY'] = co['cv0'] - co['cfy'] * y_corrected / z

    return body_info


def norm_features(body, method='vector', type='3D', b_hand=True):
    if method != 'torso' and method != 'vector':
        raise Exception(f'Wrong normalization method: {method}')
    if type != '2D' and type != '3D':
        raise Exception(f'Wrong input data type: {type}')
    if not isinstance(b_hand, bool):
        raise Exception(f'Wrong hand data type: {b_hand}')

    if method == 'torso':
        return norm_to_torso(body, type, b_hand)
    if method == 'vector':
        return norm_to_vector(body, type, b_hand)


# move origin to torso and
# normalize to the distance between torso and spineShoulder
def norm_to_torso(body, type, b_hand):
    torso, spineShoulder, head, shoulderLeft, elbowLeft, wristLeft, handLeft, shoulderRight, elbowRight, wristRight, handRight = \
        get_upper_body_joints(body, type)

    features = list()
    features.extend(norm_to_distance(torso, spineShoulder, spineShoulder))
    features.extend(norm_to_distance(torso, spineShoulder, head))
    features.extend(norm_to_distance(torso, spineShoulder, shoulderLeft))
    features.extend(norm_to_distance(torso, spineShoulder, elbowLeft))
    features.extend(norm_to_distance(torso, spineShoulder, wristLeft))
    if b_hand:
        features.extend(norm_to_distance(torso, spineShoulder, handLeft))
    features.extend(norm_to_distance(torso, spineShoulder, shoulderRight))
    features.extend(norm_to_distance(torso, spineShoulder, elbowRight))
    features.extend(norm_to_distance(torso, spineShoulder, wristRight))
    if b_hand:
        features.extend(norm_to_distance(torso, spineShoulder, handRight))
    return features


def get_upper_body_joints(body, type):
    shoulderRight = vectorize(body[8], type)
    shoulderLeft = vectorize(body[4], type)
    elbowRight = vectorize(body[9], type)
    elbowLeft = vectorize(body[5], type)
    wristRight = vectorize(body[10], type)
    wristLeft = vectorize(body[6], type)
    handRight = vectorize(body[23], type)
    handLeft = vectorize(body[21], type)

    torso = vectorize(body[0], type)
    spineShoulder = vectorize(body[20], type)
    head = vectorize(body[3], type)

    return torso, spineShoulder, head, shoulderLeft, elbowLeft, wristLeft, handLeft, shoulderRight, elbowRight, wristRight, handRight


def norm_to_distance(origin, basis, joint):
    norm = np.linalg.norm(basis - origin)
    norm = np.finfo(basis.dtype).eps if norm == 0 else norm
    result = (joint - origin) / norm if any(joint) else joint
    return result


def norm_to_vector(body, type, b_hand):
    torso, spineShoulder, head, shoulderLeft, elbowLeft, wristLeft, handLeft, shoulderRight, elbowRight, wristRight, handRight = \
        get_upper_body_joints(body, type)

    features = list()
    features.extend(norm_to_distance(torso, spineShoulder, spineShoulder))
    features.extend(norm_to_distance(spineShoulder, head, head))
    features.extend(norm_to_distance(spineShoulder, shoulderLeft, shoulderLeft))
    features.extend(norm_to_distance(shoulderLeft, elbowLeft, elbowLeft))
    features.extend(norm_to_distance(elbowLeft, wristLeft, wristLeft))
    if b_hand:
        features.extend(norm_to_distance(wristLeft, handLeft, handLeft))
    features.extend(norm_to_distance(spineShoulder, shoulderRight, shoulderRight))
    features.extend(norm_to_distance(shoulderRight, elbowRight, elbowRight))
    features.extend(norm_to_distance(elbowRight, wristRight, wristRight))
    if b_hand:
        features.extend(norm_to_distance(wristRight, handRight, handRight))
    return features


def denorm_features(features, method='vector', type='3D', b_hand=True):
    if method != 'torso' and method != 'vector':
        raise Exception(f'Wrong normalization method: {method}')
    if type != '2D' and type != '3D':
        raise Exception(f'Wrong input data type: {type}')
    if not isinstance(b_hand, bool):
        raise Exception(f'Wrong hand data type: {b_hand}')

    if method == 'torso':
        return denorm_from_torso(features, type, b_hand)
    if method == 'vector':
        return denorm_from_vector(features, type, b_hand)


def denorm_from_torso(features, type, b_hand):
    # pelvis
    pelvis = np.array([0, 0, 0]) if type == '2D' else np.array([0, 0])

    # other joints (spineShoulder, head, shoulderLeft, elbowLeft, wristLeft, handLeft, shoulderRight, elbowRight, wristRight, handRight)
    spine_len = 3.
    features = np.array(features) * spine_len

    if b_hand:
        joints = np.vstack((pelvis, np.split(features, 10)))
    else:
        joints = np.vstack((pelvis, np.split(features, 8)))
        joints = np.insert(joints, 8, joints[7], axis=0)
        joints = np.insert(joints, 4, joints[3], axis=0)

    return joints


def denorm_from_vector(features, type, b_hand):
    pelvis = np.array([0, 0, 0]) if type == '3D' else np.array([0, 0])
    if b_hand:
        v_spineShoulder, v_head, \
        v_shoulderLeft, v_elbowLeft, v_wristLeft, v_handLeft, \
        v_shoulderRight, v_elbowRight, v_wristRight, v_handRight \
            = np.split(np.array(features), 10)
    else:
        v_spineShoulder, v_head, \
        v_shoulderLeft, v_elbowLeft, v_wristLeft, \
        v_shoulderRight, v_elbowRight, v_wristRight \
            = np.split(np.array(features), 8)

    spine_len = 3.
    spineShoulder = v_spineShoulder * spine_len
    head = spineShoulder + v_head * spine_len / 2.
    shoulderLeft = spineShoulder + v_shoulderLeft * spine_len / 3.
    elbowLeft = shoulderLeft + v_elbowLeft * spine_len / 2.
    wristLeft = elbowLeft + v_wristLeft * spine_len / 2.
    shoulderRight = spineShoulder + v_shoulderRight * spine_len / 3.
    elbowRight = shoulderRight + v_elbowRight * spine_len / 2.
    wristRight = elbowRight + v_wristRight * spine_len / 2.

    if not b_hand:
        return np.vstack((pelvis, spineShoulder, head,
                          shoulderLeft, elbowLeft, wristLeft, wristLeft,
                          shoulderRight, elbowRight, wristRight, wristRight))
    else:
        handLeft = wristLeft + v_handLeft * spine_len / 4.
        handRight = wristRight + v_handRight * spine_len / 4.
        return np.vstack((pelvis, spineShoulder, head,
                          shoulderLeft, elbowLeft, wristLeft, handLeft,
                          shoulderRight, elbowRight, wristRight, handRight))
