# for common use

KINECT_FRAME_RATE = 30  # frame rate of kinect camera
TARGET_FRAME_RATE = 10  # frame rate of extracted data

SUBACTION_NAMES = dict()
SUBACTION_NAMES[0] = "가만히 있기"
SUBACTION_NAMES[1] = "문 열기"
SUBACTION_NAMES[2] = "손으로 벽 집기"
SUBACTION_NAMES[3] = "아무도 없음"
SUBACTION_NAMES[4] = "오른손 올리기"
SUBACTION_NAMES[5] = "오른손 흔들기"
SUBACTION_NAMES[6] = "오른손 내리기"
SUBACTION_NAMES[7] = "오른손 울기"
SUBACTION_NAMES[8] = "두손 올리기"
SUBACTION_NAMES[9] = "두손 울기"
SUBACTION_NAMES[10] = "두손/왼손 내리기"
SUBACTION_NAMES[11] = "왼손 올리기/울기"
SUBACTION_NAMES[12] = "오른손 때리려하기"
SUBACTION_NAMES[13] = "왼손 올리기/때리려하기"


def gen_sequence(data, length):
    for start_idx in range(len(data) - length + 1):
        yield list(data[start_idx:start_idx + length])
