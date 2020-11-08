# for common use

KINECT_FRAME_RATE = 30  # frame rate of kinect camera
TARGET_FRAME_RATE = 10  # frame rate of extracted data

SUBACTION_NAMES = dict()
SUBACTION_NAMES[0] = "stand"
SUBACTION_NAMES[1] = "open the door"
SUBACTION_NAMES[2] = "hand on wall"
SUBACTION_NAMES[3] = "not shown"
SUBACTION_NAMES[4] = "call with right hand"
SUBACTION_NAMES[5] = "call with left hand"
SUBACTION_NAMES[6] = "call with both hand"
SUBACTION_NAMES[7] = "lower hands"
SUBACTION_NAMES[8] = "raise right hand"
SUBACTION_NAMES[9] = "wave right hand"
SUBACTION_NAMES[10] = "raise both hands"
SUBACTION_NAMES[11] = "cry with right hand"
SUBACTION_NAMES[12] = "cry with left hand"
SUBACTION_NAMES[13] = "cry with both hands"
SUBACTION_NAMES[14] = "threaten to hit with right hand"
SUBACTION_NAMES[15] = "threaten to hit with left hand"

sub_action_mapping_1 = {0: 1, 1: 0, 2: 3, 3: 2}
sub_action_mapping_2 = {0: 0, 1: 8, 2: 7, 3: 7, 4: 4, 5: 9, 6: 6, 7: 5}
sub_action_mapping_3 = {0: 0, 1: 7, 2: 8, 3: 12, 4: 7, 5: 11, 6: 10, 7: 13}
sub_action_mapping_4 = {0: 7, 1: 14, 2: 0, 3: 9, 4: 8, 5: 7, 6: 15}
