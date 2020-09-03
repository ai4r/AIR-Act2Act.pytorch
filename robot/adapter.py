from math import atan2, degrees

POSES = dict()
POSES['stand'] = \
    [-0.0357290804386, .0772484987974,
     1.7656172514, 0.104678653181, -1.71877264977, -0.113438166678,
     1.74494993687, -0.104725584388, 1.69289374352, 0.102678008378]
POSES['bow'] = \
    [-0.826445043087, 0.0681400820613,
     1.33279013634, 0.0928900837898, -1.71842157841, -0.112014353275,
     1.33279013634, -0.0903749540448, 1.69248425961, 0.0986629277468]
POSES['handshake'] = \
    [-0.0357313230634, 0.0772484987974,
     1.76606106758, 0.104689441621, -1.71871554852, -0.113334469497,
     0.0, -0.0917486473918, 1.68654882908, 0.]
POSES['hug'] = \
    [-0.397290990651, 0.0772485136986,
     0.210701853037, 0.106823422015, -1.36332726479, -0.379669308662,
     0.210701853037, -0.106823422015, 1.36332726479, 0.379669308662]
POSES['avoid'] = \
    [0.3069524228518884, 0.1870072584919955,
     -0.09544009163085918, -0.07582597981979501, -1.342600244783586, -1.413778030964168,
     0.33340909260840057, -0.13373375503589732, 1.1374428138295134, 1.4061889935828833]


def adapt_behavior(behavior, pose=None):
    if pose is None:
        return POSES[behavior]
    if behavior == 'stand' or behavior == 'bow':
        return POSES[behavior]

    if behavior == 'hug':
        return hug(pose)
    if behavior == 'handshake':
        return handshake(pose)
    if behavior == 'avoid':
        return avoid(pose)

    raise Exception(f"Wrong behavior name: {behavior}")


# adapt hug behavior (assumption: head angle of robot is zero)
def hug(pose):
    shoulder_pos_y = pose[20]['y']
    shoulder_pos_z = pose[20]['z']
    angle_to_shoulder = degrees(atan2(shoulder_pos_y, shoulder_pos_z)) + 90
    # print(angle_to_shoulder)

    if angle_to_shoulder > 0:
        diff = [a - b for a, b in zip(POSES['hug'], POSES['stand'])]
        adapted_pose = [a + b * angle_to_shoulder / 90 for a, b in zip(POSES['stand'], diff)]
    else:
        adapted_pose = POSES['stand']
    return adapted_pose


def handshake(pose):
    right_hand_pos_x = pose[10]['x']
    right_hand_pos_y = pose[10]['y']
    right_hand_pos_z = pose[10]['z']

    angle_to_right_hand = degrees(atan2(right_hand_pos_y, right_hand_pos_z)) + 90
    if angle_to_right_hand > 0:
        diff = [a - b for a, b in zip(POSES['handshake'], POSES['stand'])]
        adapted_pose = [a + b * angle_to_right_hand / 90 for a, b in zip(POSES['stand'], diff)]
    else:
        adapted_pose = POSES['stand']

    if right_hand_pos_x < 0:
        angle_x_to_right_hand = atan2(-right_hand_pos_z, -right_hand_pos_x + 0.1)
        angle_x_to_right_hand /= 2.
        adapted_pose[7] = angle_x_to_right_hand

    return adapted_pose


def avoid(pose):
    if pose[6]['y'] > pose[10]['y']:
        hand_pos_y = pose[6]['y']
        hand_pos_z = pose[6]['z']
    else:
        hand_pos_y = pose[10]['y']
        hand_pos_z = pose[10]['z']

    angle_to_hand = degrees(atan2(hand_pos_y, hand_pos_z)) + 90
    if angle_to_hand > 0:
        diff = [a - b for a, b in zip(POSES['avoid'], POSES['stand'])]
        adapted_pose = [a + b * angle_to_hand / 90 for a, b in zip(POSES['stand'], diff)]
    else:
        adapted_pose = POSES['stand']
    return adapted_pose
