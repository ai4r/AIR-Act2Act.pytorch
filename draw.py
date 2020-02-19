import os
import glob
import numpy as np
import matplotlib.animation as animation
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D

from utils.AIR import norm_to_torso


def main():
    # show all test data
    data_files = glob.glob(os.path.normpath(os.path.join('./data files/valid data', "*A005*.npz")))
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
            human_data = [denorm_from_torso(norm_to_torso(human)) for human in data['human_info']]
            sampled_human_seq = human_data[::3]
            draw([sampled_human_seq], save_path=None, b_show=True)


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


def animate_3d(f, features, axes):
    ret_artists = list()
    for idx in range(len(features)):
        init_axis(axes[idx])
        cur_features = features[idx][f] if f < len(features[idx]) else features[idx][-1]
        pelvis, neck, head, lshoulder, lelbow, lwrist, rshoulder, relbow, rwrist = cur_features
        ret_artists.extend(draw_parts(axes[idx], [pelvis, neck, head]))
        ret_artists.extend(draw_parts(axes[idx], [neck, lshoulder, lelbow, lwrist]))
        ret_artists.extend(draw_parts(axes[idx], [neck, rshoulder, relbow, rwrist]))
        ret_artists.extend([axes[idx].text(0, 0, 0, '{0}/{1}'.format(f + 1, len(features[idx])))])
    return ret_artists


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


if __name__ == "__main__":
    main()
