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