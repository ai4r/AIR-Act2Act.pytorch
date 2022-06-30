import time
import matplotlib.animation as animation
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from textwrap import wrap
from setting import INPUT_DATA_TYPE

# plt.rc('font', family='NanumGothic')


class Artist:
    def __init__(self, n_plot=1):
        self.n_plot = n_plot
        self.last_draw_time = time.time()

        fig = plt.figure()
        if INPUT_DATA_TYPE == '3D':
            self.axes = [fig.add_subplot(1, self.n_plot, idx + 1, projection='3d') for idx in range(self.n_plot)]
        if INPUT_DATA_TYPE == '2D':
            self.axes = [fig.add_subplot(1, self.n_plot, idx + 1) for idx in range(self.n_plot)]
        for idx in range(self.n_plot):
            self.init_axis(self.axes[idx])

        # self.figManager = plt.get_current_fig_manager()
        # fig.canvas.manager.window.activateWindow()
        # self.figManager.window.raise_()
        plt.show(block=False)

    def __del__(self):
        plt.close()

    def update(self, features, results, frame_info, fps=1000):
        wait_time = 1. / fps
        while True:
            if time.time() - self.last_draw_time > wait_time:
                self.last_draw_time = time.time()

                ret_artists = list()
                for idx in range(self.n_plot):
                    self.init_axis(self.axes[idx])
                    pelvis, neck, head, lshoulder, lelbow, lwrist, lhand, rshoulder, relbow, rwrist, rhand = features[idx]
                    ret_artists.extend(self.draw_parts(self.axes[idx], [pelvis, neck, head]))
                    ret_artists.extend(self.draw_parts(self.axes[idx], [neck, lshoulder, lelbow, lwrist, lhand]))
                    ret_artists.extend(self.draw_parts(self.axes[idx], [neck, rshoulder, relbow, rwrist, rhand]))

                    result = "\n".join(wrap(results[idx], 15)) if results[idx] is not None else ''
                    if INPUT_DATA_TYPE == '2D':
                        ret_artists.append(self.axes[idx].text(0, 0, f"{result}\n{frame_info[idx]}", fontsize=20))
                    if INPUT_DATA_TYPE == '3D':
                        ret_artists.append(self.axes[idx].text(0, 0, 0, f"{result}\n{frame_info[idx]}", fontsize=20))

                plt.show(block=False)
                plt.pause(0.001)
                return

    def init_axis(self, ax):
        ax.clear()
        ax.set_xlabel('x')
        ax.set_ylabel('y')
        max = 2.
        if INPUT_DATA_TYPE == '3D':
            ax.set_zlabel('z')
            ax.set_xlim3d(-max * 0.7, max * 0.7)
            ax.set_ylim3d(0, 2 * max)
            ax.set_zlim3d(-max, max)
            ax.view_init(elev=-80, azim=90)
        if INPUT_DATA_TYPE == '2D':
            ax.set_xlim(max * 2.0, -max * 2.0)
            ax.set_ylim(max * 0.5, -max * 3.5)
        ax._axis3don = False

    def draw_parts(self, ax, joints):
        def add_points(points):
            xs, ys, zs = list(), list(), list()
            for point in points:
                xs.append(point[0])
                ys.append(point[1])
                if INPUT_DATA_TYPE == '3D':
                    zs.append(point[2])
            return xs, ys, zs

        xs, ys, zs = add_points(joints)
        if INPUT_DATA_TYPE == '3D':
            ret = ax.plot(xs, ys, zs, color='b')
        if INPUT_DATA_TYPE == '2D':
            ret = ax.plot(xs, ys, color='b')
        return ret
