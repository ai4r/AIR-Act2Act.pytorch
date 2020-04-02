import matplotlib.animation as animation
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D


def draw(features, results, save_path=None, b_show=False):
    fig = plt.figure()
    axes = [fig.add_subplot(1, len(features), idx + 1, projection='3d') for idx in range(len(features))]
    anim = animation.FuncAnimation(fig, animate_3d, interval=100, blit=True, fargs=(features, results, axes),
                                   frames=len(features[0]), repeat=True)

    writer = animation.writers['ffmpeg'](fps=10)
    anim.save(save_path, writer=writer, dpi=250) if save_path else None

    figManager = plt.get_current_fig_manager()
    figManager.window.showMaximized()
    plt.show() if b_show else None
    plt.close()


def animate_3d(f, features, results, axes):
    ret_artists = list()
    for idx in range(len(features)):
        init_axis(axes[idx])
        pelvis, neck, head, lshoulder, lelbow, lwrist, rshoulder, relbow, rwrist = features[idx][f]
        ret_artists.extend(draw_parts(axes[idx], [pelvis, neck, head]))
        ret_artists.extend(draw_parts(axes[idx], [neck, lshoulder, lelbow, lwrist]))
        ret_artists.extend(draw_parts(axes[idx], [neck, rshoulder, relbow, rwrist]))

        if results is not None:
            ret_artists.append(axes[idx].text(0, 0, 0, F"{results[idx][f]}\n{f+1}/{len(features[idx])}",
                                              fontsize=40))

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
