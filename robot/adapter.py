from math import atan2, degrees, radians, sqrt, cos, sin
from setting import INPUT_DATA_TYPE, W_VIDEO, H_VIDEO

POSES = dict()
POSES['stand'] = \
    [-0.0357290804386, -.2272484987974,
     1.7656172514, 0.104678653181, 0., -0.113438166678, -1.71877264977,
     1.7656172514, -0.104678653181, 0., 0.113438166678, 1.71877264977]
POSES['bow'] = \
    [-0.826445043087, 0.0681400820613,
     1.33279013634, 0.0903749540448, 0., -0.0986629277468, -1.69248425961,
     1.33279013634, -0.0903749540448, 0., 0.0986629277468, 1.69248425961]
POSES['approach'] = \
    [-0.0357290804386, -0.2272484987974,
     1.7656172514, 0.104678653181, 0., -0.113438166678, -1.71877264977,
     1.7656172514, -0.104678653181, 0., 0.113438166678, 1.71877264977]
POSES['handshake'] = \
    [-0.0357313230634, -0.2272484987974,
     1.7656172514, 0.104678653181, 0., -0.113438166678, -1.71877264977,
     0.1765617251, -0.0917486473918, 0., 0., 1.68654882908]
POSES['hug'] = \
    [-0.1357290804386, 0.0772485136986,
     0.0765617251, 0.106823422015, 0., -0.379669308662, -1.36332726479,
     0.0765617251, -0.106823422015, 0., 0.379669308662, 1.36332726479]
POSES['high-five'] = \
    [-0.0357313230634, -0.2272484987974,
     1.7656172514, 0.104678653181, 0., -0.113438166678, -1.71877264977,
     0.0765617251, -0.0917486473918, 1.0656172514, 0.73279013634, -1.1656172514]
POSES['avoid'] = \
    [0.1069524228518884, 0.1870072584919955,
     -0.09544009163085918, -0.07582597981979501, -1.342600244783586, -1.413778030964168, 0.,
     0.33340909260840057, -0.13373375503589732, 1.1374428138295134, 1.4061889935828833, 0.]


def adapt_behavior(behavior, pose=None):
    # behaviors no need to be adapted
    if pose is None:
        return POSES[behavior]
    if behavior == 'stand' or behavior == 'bow':
        return POSES[behavior]

    # behaviors should be adapted
    # if behavior == 'approach':
    #     return approach(pose)
    # if behavior == 'hug':
    #     return hug(pose)
    if behavior == 'handshake':
        return handshake(pose)
    if behavior == 'avoid':
        return avoid(pose)
    if behavior == 'high-five':
        return highfive(pose)

    raise Exception(f"Wrong behavior name: {behavior}")


# adapt approach behavior
def approach(pose):
    shoulder_pos_x = pose[20]['x']
    shoulder_pos_z = pose[20]['z']

    theta = atan2(shoulder_pos_x, shoulder_pos_z)

    dist_to_user = sqrt(shoulder_pos_x ** 2 + shoulder_pos_z ** 2)
    dist_to_move = max(0.0, dist_to_user - 1.0)

    x_to_move = dist_to_move * cos(theta)
    y_to_move = dist_to_move * sin(theta)

    adapted_pose = list(POSES['approach'])
    # adapted_pose.extend([x_to_move, y_to_move, theta])
    adapted_pose.extend([1.0, 0, 0])  # should be edited
    return adapted_pose


# adapt hug behavior
# assumption: head angle of robot is zero
def hug(pose):
    shoulder_pos_y = pose[20]['y']
    shoulder_pos_z = pose[20]['z']

    angle_to_shoulder = degrees(atan2(shoulder_pos_y, shoulder_pos_z)) + 90
    if angle_to_shoulder > 0.:
        diff = [a - b for a, b in zip(POSES['hug'], POSES['stand'])]
        adapted_pose = [a + b * angle_to_shoulder / 90. for a, b in zip(POSES['stand'], diff)]
    else:
        adapted_pose = POSES['stand']
    return adapted_pose


# adapt handshake behavior
def handshake(pose):
    if INPUT_DATA_TYPE == '3D':
        return handshake_3D(pose)
    if INPUT_DATA_TYPE == '2D':
        return handshake_2D(pose)


def handshake_3D(pose):
    right_hand_pos_x = pose[10]['x']
    right_hand_pos_y = pose[10]['y']
    right_hand_pos_z = pose[10]['z']

    angle_y_to_right_hand = degrees(atan2(right_hand_pos_y, right_hand_pos_z)) + 90
    if angle_y_to_right_hand > 0:
        diff = [a - b for a, b in zip(POSES['handshake'], POSES['stand'])]
        adapted_pose = [a + b * angle_y_to_right_hand / 90. for a, b in zip(POSES['stand'], diff)]
    else:
        adapted_pose = POSES['stand']

    shoulder_length = 0.1
    if right_hand_pos_x < -shoulder_length:
        angle_x_to_right_hand = atan2(-right_hand_pos_x - shoulder_length, right_hand_pos_z)
        adapted_pose[8] = -angle_x_to_right_hand  # RShoulderRoll
        adapted_pose[10] = POSES['handshake'][10]  # RElbowRoll
    else:
        angle_x_to_right_hand = atan2(right_hand_pos_x + shoulder_length, right_hand_pos_z)
        adapted_pose[8] = POSES['handshake'][8]  # RShoulderRoll
        adapted_pose[10] = angle_x_to_right_hand  # RElbowRoll
    return adapted_pose


def handshake_2D(pose):
    right_hand_pos_x = pose[10]['colorX'] + W_VIDEO * .2
    right_hand_pos_y = pose[10]['colorY']

    angle_y_to_right_hand = (H_VIDEO - right_hand_pos_y) / H_VIDEO * 120 + 60  # 180 to 45
    angle_y_to_right_hand = min(max(angle_y_to_right_hand, 70), 160)

    diff = [a - b for a, b in zip(POSES['handshake'], POSES['stand'])]
    adapted_pose = [a + b * angle_y_to_right_hand / 90. for a, b in zip(POSES['stand'], diff)]

    angle_x_to_right_hand = right_hand_pos_x / W_VIDEO * 90 - 45  # -45 to 45
    # print(angle_x_to_right_hand)
    if angle_x_to_right_hand < 0:
        adapted_pose[8] = radians(angle_x_to_right_hand) * 1.5  # RShoulderRoll
        adapted_pose[10] = POSES['handshake'][10]  # RElbowRoll
    else:
        adapted_pose[8] = POSES['handshake'][8]  # RShoulderRoll
        adapted_pose[10] = radians(angle_x_to_right_hand) * 1.5  # RElbowRoll

    return adapted_pose


# adapt high-five behavior
def highfive(pose):
    if INPUT_DATA_TYPE == '3D':
        return highfive_3D(pose)
    if INPUT_DATA_TYPE == '2D':
        return highfive_2D(pose)


def highfive_3D(pose):
    adapted_pose = list(POSES['high-five'])

    right_hand_pos_x = pose[10]['colorX']
    right_hand_pos_y = pose[10]['colorY']

    angle_y_to_right_hand = (H_VIDEO - right_hand_pos_y) / H_VIDEO * 180  # 180 to 0
    angle_y_to_right_hand = min(max(angle_y_to_right_hand, 45), 120)
    # print(angle_y_to_right_hand)
    diff_y = POSES['high-five'][7] - POSES['stand'][7]
    adapted_pose[7] = POSES['stand'][7] + diff_y * angle_y_to_right_hand / 90.

    angle_x_to_right_hand = right_hand_pos_x / W_VIDEO * 90 - 45  # -45 to 45
    if angle_x_to_right_hand < 0:
        adapted_pose[8] = radians(angle_x_to_right_hand) * 2  # RShoulderRoll
        adapted_pose[9] = POSES['high-five'][9]  # RElbowYaw
        adapted_pose[10] = POSES['high-five'][10]  # RElbowRoll
        adapted_pose[11] = POSES['high-five'][11] - radians(angle_x_to_right_hand) * .5  # RWristYaw
    else:
        adapted_pose[8] = POSES['high-five'][8]  # RShoulderRoll
        adapted_pose[9] = POSES['high-five'][9] - radians(angle_x_to_right_hand)  # RElbowYaw
        adapted_pose[10] = POSES['high-five'][10]  # RElbowRoll
        adapted_pose[11] = POSES['high-five'][11] + radians(angle_x_to_right_hand) * .5  # RWristYaw

    return adapted_pose


def highfive_2D(pose):
    adapted_pose = list(POSES['high-five'])

    right_hand_pos_x = pose[10]['colorX'] + W_VIDEO * .05
    right_hand_pos_y = pose[10]['colorY']

    angle_y_to_right_hand = (H_VIDEO - right_hand_pos_y) / H_VIDEO * 115 + 45  # 180 to 45
    angle_y_to_right_hand = min(max(angle_y_to_right_hand, 90), 155)
    # print(angle_y_to_right_hand)
    diff_y = POSES['high-five'][7] - POSES['stand'][7]
    adapted_pose[7] = POSES['stand'][7] + diff_y * angle_y_to_right_hand / 90.

    angle_x_to_right_hand = right_hand_pos_x / W_VIDEO * 120 - 60  # -60 to 60
    angle_x_to_right_hand = min(max(angle_x_to_right_hand, -30), 15)

    if angle_x_to_right_hand < 0:
        adapted_pose[8] = radians(angle_x_to_right_hand) * 2  # RShoulderRoll
        adapted_pose[9] = POSES['high-five'][9]  # RElbowYaw
        adapted_pose[10] = POSES['high-five'][10]  # RElbowRoll
        adapted_pose[11] = POSES['high-five'][11] - radians(angle_x_to_right_hand) * .5  # RWristYaw
    else:
        adapted_pose[8] = POSES['high-five'][8]  # RShoulderRoll
        adapted_pose[9] = POSES['high-five'][9] - radians(angle_x_to_right_hand)  # RElbowYaw
        adapted_pose[10] = POSES['high-five'][10]  # RElbowRoll
        adapted_pose[11] = POSES['high-five'][11] + radians(angle_x_to_right_hand) * .5  # RWristYaw

    return adapted_pose


# adapt avoid behavior
def avoid(pose):
    if INPUT_DATA_TYPE == '3D':
        return avoid_3D(pose)
    if INPUT_DATA_TYPE == '2D':
        return avoid_2D(pose)


def avoid_3D(pose):
    if pose[6]['y'] > pose[10]['y']:
        hand_pos_y = pose[6]['y']
        hand_pos_z = pose[6]['z']
    else:
        hand_pos_y = pose[10]['y']
        hand_pos_z = pose[10]['z']

    angle_to_hand = degrees(atan2(hand_pos_y, hand_pos_z)) + 90
    if angle_to_hand > 0.:
        diff = [a - b for a, b in zip(POSES['avoid'], POSES['stand'])]
        adapted_pose = [a + b * angle_to_hand / 90. for a, b in zip(POSES['stand'], diff)]
    else:
        adapted_pose = POSES['stand']
    return adapted_pose


def avoid_2D(pose):
    adapted_pose = list(POSES['avoid'])

    if pose[6]['colorY'] > pose[10]['colorY']:
        hand_pos_y = pose[10]['colorY']
    else:
        hand_pos_y = pose[6]['colorY']

    angle_y_to_hand = (H_VIDEO - hand_pos_y) / H_VIDEO * 180  # 180 to 0
    angle_y_to_hand = min(max(angle_y_to_hand, 45), 120)
    diff_y = POSES['avoid'][0] - POSES['stand'][0]
    adapted_pose[0] = POSES['stand'][0] + diff_y * angle_y_to_hand / 90.

    return adapted_pose


