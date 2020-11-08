import os
import sys

sys.path.append(os.path.join(os.path.dirname(__file__), '..'))
from user.constants import SUBACTION_NAMES

ROBOT_BEHAVIORS = dict()
ROBOT_BEHAVIORS[0] = 'stand'
ROBOT_BEHAVIORS[1] = 'bow'
ROBOT_BEHAVIORS[2] = 'approach'
ROBOT_BEHAVIORS[3] = 'handshake'
ROBOT_BEHAVIORS[4] = 'hug'
ROBOT_BEHAVIORS[5] = 'avoid'


def select_behavior(user_behaviors):
    if all(SUBACTION_NAMES[behavior] == "stand" for behavior in user_behaviors) or \
            all("lower" in SUBACTION_NAMES[behavior] for behavior in user_behaviors):
        return 'stand'

    if all(SUBACTION_NAMES[behavior] == "not shown" for behavior in user_behaviors[:-1]):
        if SUBACTION_NAMES[user_behaviors[-1]] == "stand" or SUBACTION_NAMES[user_behaviors[-1]] == "open the door":
            return 'bow'

    if all("call" in SUBACTION_NAMES[behavior] for behavior in user_behaviors):
        return 'approach'

    if all(SUBACTION_NAMES[behavior] == "raise right hand" for behavior in user_behaviors):
        return 'handshake'

    if all("cry" in SUBACTION_NAMES[behavior] for behavior in user_behaviors):
        return 'hug'

    if all("threaten" in SUBACTION_NAMES[behavior] for behavior in user_behaviors):
        return 'avoid'
