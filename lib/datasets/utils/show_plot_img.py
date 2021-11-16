from __future__ import print_function, division
import numpy as np
import matplotlib

matplotlib.use('Agg')
import matplotlib.pyplot as plt

plt.switch_backend('agg')
import pdb


class MultiImgPloter:
    def __init__(self, **args):
        self.arg_num = len(args)
        self.arg_key_list = list(args.keys())
        self.plot_img = self._integrate_img(args)

        self.savefig_path = ''

    def _integrate_img(self, args):
        if self.arg_num == 1:
            return args['img1']
        elif self.arg_num == 2:
            f, ax = plt.subplots(1, 2, figsize=(14, 6))
            # f.tight_layout()

            f.subplots_adjust(hspace=.01, wspace=.01, top=0.98, bottom=0.02, left=0.02, right=0.98)
            ax = ax.ravel()

            for i in range(self.arg_num):
                ax[i].imshow(args[self.arg_key_list[i]])
                ax[i].axis('off')
            f.canvas.draw()
            width, height = f.get_size_inches() * f.get_dpi()
            mplimage = np.frombuffer(f.canvas.tostring_rgb(), dtype='uint8').reshape(int(height), int(width), 3)
            plt.cla()
            plt.close(f)
            return mplimage
        elif self.arg_num == 3:
            f, ax = plt.subplots(1, 3, figsize=(9, 9))
            f.tight_layout()
            f.subplots_adjust(hspace=.01, wspace=.01, top=0.98, bottom=0.02, left=0.02, right=0.98)
            ax = ax.ravel()
            # pdb.set_trace()

            ax[0].imshow(args['img1'])
            ax[0].axis('off')

            ax[1].imshow(args['img2'])
            ax[1].axis('off')

            ax[2].imshow(args['img3'])
            ax[2].axis('off')

            # for i in range(self.arg_num):
            #     ax[i].imshow(args[self.arg_key_list[i]])
            #     ax[i].axis('off')
            f.canvas.draw()
            width, height = f.get_size_inches() * f.get_dpi()
            mplimage = np.frombuffer(f.canvas.tostring_rgb(), dtype='uint8').reshape(int(height), int(width), 3)
            plt.cla()
            plt.close(f)
            return mplimage
        elif self.arg_num == 4:
            f, ax = plt.subplots(2, 2, figsize=(13, 8))
            # f.tight_layout()

            f.subplots_adjust(hspace=.01, wspace=.01, top=0.98, bottom=0.02, left=0.02, right=0.98)
            ax = ax.ravel()

            for i in range(self.arg_num):
                ax[i].imshow(args[self.arg_key_list[i]])
                ax[i].axis('off')
            f.canvas.draw()
            width, height = f.get_size_inches() * f.get_dpi()
            mplimage = np.frombuffer(f.canvas.tostring_rgb(), dtype='uint8').reshape(int(height), int(width), 3)
            plt.cla()
            plt.close(f)
            return mplimage

    def get_mplimage(self):
        return self.plot_img


if __name__ == '__main__':
    import cv2

    img1 = np.ones((500, 500, 3), dtype=np.uint8)
    img2 = np.ones((500, 500), dtype=np.uint8)
    img3 = np.ones((500, 500), dtype=np.uint8)
    img = MultiImgPloter(img1=img1, img2=img2, img3=img3).get_mplimage()

