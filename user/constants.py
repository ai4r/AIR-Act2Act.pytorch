# for common use
from setting import ACTIONS

KINECT_FRAME_RATE = 30  # frame rate of kinect camera
TARGET_FRAME_RATE = 10  # frame rate of extracted data

ALL_SUBACTION_NAMES = dict()
ALL_SUBACTION_NAMES[0] = "stand"
ALL_SUBACTION_NAMES[1] = "open the door"
ALL_SUBACTION_NAMES[2] = "hand on wall"
ALL_SUBACTION_NAMES[3] = "not shown"
ALL_SUBACTION_NAMES[4] = "call with right hand"
ALL_SUBACTION_NAMES[5] = "call with left hand"
ALL_SUBACTION_NAMES[6] = "call with both hands"
ALL_SUBACTION_NAMES[7] = "lower hands"
ALL_SUBACTION_NAMES[8] = "raise right hand"
ALL_SUBACTION_NAMES[9] = "wave right hand"
ALL_SUBACTION_NAMES[10] = "raise both hands"
ALL_SUBACTION_NAMES[11] = "cry with right hand"
ALL_SUBACTION_NAMES[12] = "cry with left hand"
ALL_SUBACTION_NAMES[13] = "cry with both hands"
ALL_SUBACTION_NAMES[14] = "high-five with right hand"
ALL_SUBACTION_NAMES[15] = "threaten to hit with right hand"
ALL_SUBACTION_NAMES[16] = "threaten to hit with left hand"

sub_action_mapping_1 = {0: 1, 1: 0, 2: 3, 3: 2}
sub_action_mapping_2 = {0: 0, 1: 8, 2: 7, 3: 7, 4: 4, 5: 9, 6: 6, 7: 5}
sub_action_mapping_3 = {0: 0, 1: 7, 2: 8, 3: 12, 4: 7, 5: 11, 6: 10, 7: 13}
sub_action_mapping_4 = {0: 7, 1: 15, 2: 0, 3: 15, 4: 8, 5: 7, 6: 16}   # org: {3: 9}
sub_action_mapping_5 = {0: 0, 1: 14, 2: 7, 3: 8}
sub_action_mapping_6 = {0: 8, 1: 7, 2: 0, 3: 9}

SUBACTION_NAMES = list()
SUBACTION_NAMES.append(ALL_SUBACTION_NAMES[0])
if "A001" in ACTIONS:
    SUBACTION_NAMES.append(ALL_SUBACTION_NAMES[1])
    SUBACTION_NAMES.append(ALL_SUBACTION_NAMES[2])
    SUBACTION_NAMES.append(ALL_SUBACTION_NAMES[3])
if "A003" in ACTIONS:
    SUBACTION_NAMES.append(ALL_SUBACTION_NAMES[4])
    SUBACTION_NAMES.append(ALL_SUBACTION_NAMES[5])
    SUBACTION_NAMES.append(ALL_SUBACTION_NAMES[6])
SUBACTION_NAMES.append(ALL_SUBACTION_NAMES[7])
SUBACTION_NAMES.append(ALL_SUBACTION_NAMES[8])
if "A005" in ACTIONS:
    SUBACTION_NAMES.append(ALL_SUBACTION_NAMES[9])
if "A006" in ACTIONS:
    SUBACTION_NAMES.append(ALL_SUBACTION_NAMES[10])
    SUBACTION_NAMES.append(ALL_SUBACTION_NAMES[11])
    SUBACTION_NAMES.append(ALL_SUBACTION_NAMES[12])
    SUBACTION_NAMES.append(ALL_SUBACTION_NAMES[13])
if "A007" in ACTIONS:
    SUBACTION_NAMES.append(ALL_SUBACTION_NAMES[14])
if "A008" in ACTIONS:
    SUBACTION_NAMES.append(ALL_SUBACTION_NAMES[15])
    SUBACTION_NAMES.append(ALL_SUBACTION_NAMES[16])
