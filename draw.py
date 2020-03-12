import os
import glob
import pickle
import numpy as np
import matplotlib.animation as animation
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D

from subaction import make_dataframe, SEQ_LENGTH, N_CLUSTERS

'''
A001 : 문열고 들어오기
k=3
0: 들어와서 정지까지의 행동
1: 사람이 없는 frame
2: 문을열고 들어오는 행동

A002 : 멀리서 쳐다보지 않기
k=3
0: 팔 구부리기
1: 측면 가만히 있기
2: 정면 가만히 있기

A003 : 이리 오라고 손짓하기 (음.. 1과 2 차이가뭐지?)
k=3
0: 가만히있기
1: 팔올리고내리기
2: 팔올리고내리기

A004 : 가까이에서 쳐다보기
k=4
0: 가만히있기
1: 가만히있기
2: 가만히있기
3: 가만히있기

A005 : 한 손을 앞으로 내밀기
k=3
0: 가만히있기
1: 손흔들다가 내리기
2: 손내밀기

A006 : 손으로 얼굴을 가리기
k=4
0: 두 손으로 울기
1: 내리기 & 가만히 있기 
2: 왼손으로 울기
3: 오른손으로 울기

A007 : 하이파이브 하기
k=4
0: 손내밀기
1: 손내리기
2: 손내밀고 가만히 있기
3: 가만히있기

A008 : 떄리려 손들기
k=4
0: 왼손들기
1: ?
2: 가만히 있기
3: 오른손 들기

A009 : 저리 가라며 손을 휘젓기
k=5
0: 오른손올리기
1: 가만히 있기
2: 왼손내리기
3: 왼손올리기
4: 오른손내리기

A010 : 뒤돌아 나가기
k=3
0: 정면 동작
1: 측면 가만히
2: 동작?

'''


# move origin to torso and
# normalize to the distance between torso and spineShoulder
def norm_to_torso(body):
    torso, spineShoulder, head, shoulderLeft, elbowLeft, wristLeft, shoulderRight, elbowRight, wristRight = \
        get_upper_body_joints(body)

    def norm_to_distance(origin, basis, joint):
        norm = np.linalg.norm(basis - origin)
        norm = np.finfo(basis.dtype).eps if norm == 0 else norm
        result = (joint - origin) / norm if any(joint) else joint
        return result

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
    def vectorize(joint):
        return np.array([joint['x'], joint['y'], joint['z']]).astype('float32')

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


def denorm_from_torso(features):
    # pelvis
    pelvis = np.array([0, 0, 0])

    # other joints (spineShoulder, head, shoulderLeft, elbowLeft, wristLeft, shoulderRight, elbowRight, wristRight)
    spine_len = 3.
    features = np.array(features) * spine_len

    return np.vstack((pelvis, np.split(features, 8)))


def draw(features, save_path=None, b_show=False):
    fig = plt.figure()
    axes = [fig.add_subplot(1, len(features), idx + 1, projection='3d') for idx in range(len(features))]
    anim = animation.FuncAnimation(fig, animate_3d, interval=100, blit=True, fargs=(features, axes),
                                   frames=len(features[0]), repeat=True)
    writer = animation.writers['ffmpeg'](fps=10)
    anim.save(save_path, writer=writer, dpi=250) if save_path else None
    plt.show() if b_show else None
    plt.close()


def animate_3d(f, features, axes):
    ret_artists = list()
    for idx in range(len(features)):
        init_axis(axes[idx])
        cur_features = features[idx][f] if f < len(features[idx]) else features[idx][-1]
        cur_features = cur_features[1:]
        cur_features = denorm_from_torso(cur_features)
        pelvis, neck, head, lshoulder, lelbow, lwrist, rshoulder, relbow, rwrist = cur_features
        ret_artists.extend(draw_parts(axes[idx], [pelvis, neck, head]))
        ret_artists.extend(draw_parts(axes[idx], [neck, lshoulder, lelbow, lwrist]))
        ret_artists.extend(draw_parts(axes[idx], [neck, rshoulder, relbow, rwrist]))
        # ret_artists.extend([axes[idx].text(0, 0, 0, '{0}/{1}'.format(f + 1, len(features[idx])))])

        try:
            if f >= SEQ_LENGTH - 1:
                sequence = features[idx][f-SEQ_LENGTH+1:f+1]
                df = make_dataframe([sequence], SEQ_LENGTH)
                sub_action = km_model.predict(df)
                ret_artists.append(axes[idx].text(0, 0, 0, F"{sub_action}\n{f+1}/{len(features[idx])}"))
        except Exception as e:
            print(e)
            break

    return ret_artists


def init_axis(ax):
    ax.clear()

    ax.set_xlabel('x')
    ax.set_ylabel('y')
    ax.set_zlabel('z')

    max = 2.
    ax.set_xlim3d(-max, max)
    ax.set_ylim3d(0, 2 * max)
    ax.set_zlim3d(-max, max)

    ax.view_init(elev=-80, azim=90)
    ax._axis3don = False


def draw_parts(ax, joints):
    def add_points(points):
        xs, ys, zs = list(), list(), list()
        for point in points:
            xs.append(point[0])
            ys.append(point[1])
            zs.append(point[2])
        return xs, ys, zs

    xs, ys, zs = add_points(joints)
    ret = ax.plot(xs, ys, zs, color='b')
    return ret


def main():
    # 보고싶은 액션 입력
    action = "A006"

    index = int(action[-2:]) - 1
    global km_model
    km_model = pickle.load(open(f"./models/k-means/{action}_full_{N_CLUSTERS[index]}_cluster.pkl", "rb"))

    # show all test data
    data_files = glob.glob(os.path.normpath(os.path.join('./data files/valid data', F"*{action}*.npz")))
    data_files.sort()
    n_data = len(data_files)

    print('There are %d data.' % n_data)
    for data_idx in range(n_data):
        print('%d: %s' % (data_idx, os.path.basename(data_files[data_idx])))

    # select data name to draw
    while True:
        var = int(input("Input data number to display: "))
        data_file = data_files[var]

        with np.load(data_file, allow_pickle=True) as data:
            human_data = [norm_to_torso(human) for human in data['human_info']]
            third_data = data['third_info']

            sampled_human_seq = human_data[::3]
            sampled_third_seq = third_data[::3]
            sequence = np.concatenate((sampled_third_seq, sampled_human_seq), axis=1)

            draw([sequence], save_path=None, b_show=True)


if __name__ == "__main__":
    main()
