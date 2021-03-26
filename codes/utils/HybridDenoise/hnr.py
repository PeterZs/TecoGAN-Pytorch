import os.path
import argparse

import cv2
import imageio
try:
    import bm3d
except:
    print("\033[91mNo BM3D module installed.\033[00m")
import numpy as np

from skimage import img_as_ubyte, img_as_float
from skimage.restoration import denoise_wavelet
from skimage.color import rgb2yuv, yuv2rgb
from skimage.filters.rank import median
from skimage.morphology import disk

import utils

kernel = np.array([[1, 1,  1, 1, 1],
        [1, 5,  5, 5, 1],
        [1, 5, 44, 5, 1],
        [1, 5,  5, 5, 1],
        [1, 1,  1, 1, 1]]) / 100.0


class HybridNoiseReduction:

    @staticmethod
    def yuv_split(rgb_img):
        yuv = rgb2yuv(rgb_img)
        y, u, v = cv2.split(yuv)
        return y, u, v

    @staticmethod
    def yuv_combine(y, u, v):
        yuv = cv2.merge((y, u, v))
        rgb = yuv2rgb(yuv)
        return rgb

    @staticmethod
    def freq_domain_denoise(noisy_y, threshold_sigma=0.05):
        denoised_y = denoise_wavelet(noisy_y, sigma=threshold_sigma, multichannel=False)
        return denoised_y

    @staticmethod
    def freq_domain_denoise_smooth(noisy_y, threshold_sigma=0.05):
        # denoised_y = denoise_wavelet(noisy_y, sigma=threshold_sigma, multichannel=False)
        # denoised_y = cv2.filter2D(denoised_y, -1, kernel=kernel)
        blocks = utils.blockshaped(noisy_y, nrows=64, ncols=64)
        # blocks = view_as_blocks(noisy_y, block_shape=(16, 16))

        denoised_block = np.zeros_like(blocks)
        for i in range(blocks.shape[0]):
            curr_block = denoise_wavelet(blocks[i], sigma=threshold_sigma)
            denoised_block[i] = curr_block
        denoised_y = utils.unblockshaped(denoised_block, noisy_y.shape[0], noisy_y.shape[1])
        denoised_y = cv2.resize(denoised_y, None, fx=1.5, fy=1.5, interpolation=cv2.INTER_LINEAR)
        denoised_y = cv2.resize(denoised_y, None, fx=1/1.5, fy=1/1.5, interpolation=cv2.INTER_LINEAR)
        return denoised_y

    @staticmethod
    def median_denoise(noisy_y, radius=5):
        noisy_y = img_as_ubyte(noisy_y)
        denoised_y = median(noisy_y, disk(radius))
        thres_idx = np.abs(denoised_y - noisy_y) > 255
        denoised_y[thres_idx] = noisy_y[thres_idx]
        denoised_y = img_as_float(denoised_y)
        return denoised_y

    @staticmethod
    def gaussian_denoise(noisy_y, radius=5):
        denoised_y = cv2.GaussianBlur(noisy_y, (radius, radius), sigmaX=9.0, sigmaY=0)
        return denoised_y

    @staticmethod
    def nlm_denoise(noisy_y, h_size=5):
        noisy_y = img_as_ubyte(noisy_y)
        y_denoised = cv2.fastNlMeansDenoising(noisy_y, h=h_size, templateWindowSize=7, searchWindowSize=21)
        denoised_y = img_as_float(y_denoised)
        return denoised_y

    @staticmethod
    def pyr_denoise(lap_pyr_y, alpha=0.2):
        lap_pyr_y[-1] = alpha*lap_pyr_y[-1]
        lap_pyr_y[-2] = alpha*lap_pyr_y[-2]
        return lap_pyr_y

    @staticmethod
    def lap_pyr(img, num_level=5):

        # Gaussian Pyramid
        layer = img.copy()
        gaussian_pyramid = [layer]
        for i in range(num_level + 1):
            layer = cv2.pyrDown(layer)
            gaussian_pyramid.append(layer)

        # Laplacian Pyramid
        layer = gaussian_pyramid[num_level]
        laplacian_pyramid = [layer]
        for i in range(num_level, 0, -1):
            size = (gaussian_pyramid[i - 1].shape[1], gaussian_pyramid[i - 1].shape[0])
            gaussian_expanded = cv2.pyrUp(gaussian_pyramid[i], dstsize=size)
            laplacian = cv2.subtract(gaussian_pyramid[i - 1], gaussian_expanded)
            laplacian_pyramid.append(laplacian)

        return laplacian_pyramid

    @staticmethod
    def lap_pyr_rec(laplacian_pyramid, num_level=5):
        reconstructed_image = laplacian_pyramid[0]
        for i in range(1, num_level + 1):
            size = (laplacian_pyramid[i].shape[1], laplacian_pyramid[i].shape[0])
            reconstructed_image = cv2.pyrUp(reconstructed_image, dstsize=size)
            reconstructed_image = cv2.add(reconstructed_image, laplacian_pyramid[i])
        return reconstructed_image

    @staticmethod
    def get_blending_mask(laplacian_pyramid, blend_freq_layer, morph_kernel_size=1):
        kernel = np.ones((morph_kernel_size, morph_kernel_size))
        h, w = laplacian_pyramid[-1].shape[0], laplacian_pyramid[-1].shape[1]
        ret, th = cv2.threshold(img_as_ubyte(np.abs(laplacian_pyramid[blend_freq_layer])), 0, 1,
                                cv2.THRESH_BINARY + cv2.THRESH_OTSU)
        th = np.float32(th)
        opening = cv2.morphologyEx(th, cv2.MORPH_OPEN, kernel)
        opening = cv2.morphologyEx(opening, cv2.MORPH_DILATE, kernel)
        mask = cv2.resize(opening, dsize=(w, h))
        mask = cv2.GaussianBlur(mask, (15, 15), 0)
        return mask

    def mask_blend(self, y, y_denoise, mask, no_blend):
        if no_blend:
            return y_denoise
        else:
            return y_denoise * (1 - mask) + y * mask

    def alpha_blend(self, y, y_denoise, alpha=0.5):
            return y_denoise * alpha + y * (1 - alpha)

    def run(self, rgb_img, nr_method, nr_parm, no_blend, mask_level, blend_morph_kernel, 
                blend_alpha=0.5):
        y, u, v = self.yuv_split(rgb_img)
        laplacian_pyramid = self.lap_pyr(y)

        # rec = self.lap_pyr_rec(laplacian_pyramid)
        # imageio.imwrite('test.jpg', img_as_ubyte(rec))
        # imageio.imwrite('test_lap.jpg', img_as_float(laplacian_pyramid[-1]))
        #
        # laplacian_pyramid[-1] = np.zeros_like(laplacian_pyramid[-1])
        # rec = self.lap_pyr_rec(laplacian_pyramid)
        # imageio.imwrite('test_zero.jpg', img_as_ubyte(rec))

        if nr_method == 'freq':
            y_denoised = self.freq_domain_denoise(y, threshold_sigma=nr_parm)
        if nr_method == 'freq_smooth':
            y_denoised = self.freq_domain_denoise_smooth(y, threshold_sigma=nr_parm)
        elif nr_method == "median":
            y_denoised = self.median_denoise(y, radius=int(nr_parm))
        elif nr_method == "gaussian":
            y_denoised = self.gaussian_denoise(y, radius=int(nr_parm))
        elif nr_method == 'tvl1':
            y_denoised = cv2.denoise_TVL1(y, y)
        elif nr_method == 'nlm':
            y_denoised = self.nlm_denoise(y, int(nr_parm))
        elif nr_method == 'bm3d':
            y_denoised = bm3d.bm3d(y, nr_parm)
        elif nr_method == 'lap_pyr':
            denoised_pyr = self.pyr_denoise(laplacian_pyramid, nr_parm)
            y_denoised = self.lap_pyr_rec(denoised_pyr)
        elif nr_method == 'bilateral':
            y_denoised = img_as_float(cv2.bilateralFilter(img_as_ubyte(y), int(nr_parm), 20, 5))
        else:
            y_denoised = self.nlm_denoise(y, int(nr_parm))

        mask = self.get_blending_mask(laplacian_pyramid, blend_freq_layer=mask_level, morph_kernel_size=blend_morph_kernel)

        y_final = self.alpha_blend(y, y_denoised, blend_alpha)
        # y_final = self.mask_blend(y, y_final, mask, no_blend)


        rgb_final = self.yuv_combine(y_final, u, v)

        return np.clip(rgb_final, a_min=0, a_max=1), np.clip(mask, a_min=0, a_max=1)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Process some integers.')
    parser.add_argument('--input_fn', required=True, type=str, help='input filename')
    parser.add_argument('--nr_method', default='freq', type=str, help='denoising method')
    parser.add_argument('--nr_parm', default=0.05, type=float, help='denoising parm')
    parser.add_argument('-no_blend', action='store_true', help='do not apply blending mask')
    parser.add_argument('--mask_level', default=-2, type=int, help='laplacian pyramid blending level')
    parser.add_argument('--blend_morph_kernel', default=1, type=int, help='kernel size for morphlogical operation default 1 is none')

    args = parser.parse_args()

    input_fn_base = os.path.splitext(os.path.basename(args.input_fn))[0]

    rgb_noisy = imageio.imread(args.input_fn)
    hnr = HybridNoiseReduction()
    rgb_denoise, mask = hnr.run(img_as_float(rgb_noisy),
                                nr_method=args.nr_method,
                                nr_parm=args.nr_parm,
                                no_blend=args.no_blend,
                                mask_level=args.mask_level,
                                blend_morph_kernel=args.blend_morph_kernel)

    mask_fn = 'outputs/{}_mask_{}_parm_{}_level_{}_morph_{}.png'.format(input_fn_base, args.nr_method,  str(args.nr_parm), str(args.mask_level), str(args.blend_morph_kernel))
    out_fn = 'outputs/{}_output_{}_parm_{}_level_{}_morph_{}.jpg'.format(input_fn_base, args.nr_method, str(args.nr_parm), str(args.mask_level), str(args.blend_morph_kernel))

    imageio.imwrite(mask_fn, img_as_ubyte(mask))
    imageio.imwrite(out_fn, img_as_ubyte(rgb_denoise))
