# for common use

KINECT_FRAME_RATE = 30  # frame rate of kinect camera
TARGET_FRAME_RATE = 10  # frame rate of extracted data

SUBACTION_NAMES = dict()
SUBACTION_NAMES[0] = "stand"
SUBACTION_NAMES[1] = "open the door"
SUBACTION_NAMES[2] = "hand on wall"
SUBACTION_NAMES[3] = "not shown"
SUBACTION_NAMES[4] = "raise right hand"
SUBACTION_NAMES[5] = "wave right hand"
SUBACTION_NAMES[6] = "lower right hand"
SUBACTION_NAMES[7] = "cry with right hand"
SUBACTION_NAMES[8] = "raise both hands"
SUBACTION_NAMES[9] = "cry with both hands"
SUBACTION_NAMES[10] = "lower left hand or both hands"
SUBACTION_NAMES[11] = "raise or cry with left hand"
SUBACTION_NAMES[12] = "threaten to hit with right hand"
SUBACTION_NAMES[13] = "raise or threaten to hit with left hand"

sub_action_mapping_1 = {0: 1, 1: 0, 2: 2, 3: 3}
sub_action_mapping_2 = {0: 0, 1: 8, 2: 4, 3: 7, 4: 11, 5: 5, 6: 9, 7: 6, 8: 10}
sub_action_mapping_3 = {0: 0, 1: 5, 2: 6, 3: 12, 4: 10, 5: 4, 6: 13}


def gen_sequence(data, length):
    for start_idx in range(len(data) - length + 1):
        yield list(data[start_idx:start_idx + length])


POSE_IDS = {'stand': 0, 'bow': 1, 'handshake': 2, 'hug': 3, 'avoid': 4}
POSES = dict()
POSES[0] = [-0.0357290804386, .0772484987974,
            1.7656172514, 0.104678653181, -1.71877264977, -0.113438166678,
            1.74494993687, -0.104725584388, 1.69289374352, 0.102678008378]
POSES[1] = [-0.826445043087, 0.0681400820613,
            1.33279013634, 0.0928900837898, -1.71842157841, -0.112014353275,
            1.33279013634, -0.0903749540448, 1.69248425961, 0.0986629277468]
POSES[2] = [-0.0357313230634, 0.0772484987974,
            1.76606106758, 0.104689441621, -1.71871554852, -0.113334469497,
            0.669365823269, -0.0917486473918, 1.68654882908, 0.]
POSES[3] = [-0.397290990651, 0.0772485136986,
            0.210701853037, 0.106823422015, -1.36332726479, -0.379669308662,
            0.210701853037, -0.106823422015, 1.36332726479, 0.379669308662]
POSES[4] = [0.3069524228518884, 0.1870072584919955,
            -0.09544009163085918, -0.07582597981979501, -1.342600244783586, -1.413778030964168,
            0.33340909260840057, -0.13373375503589732, 1.1374428138295134, 1.4061889935828833]
