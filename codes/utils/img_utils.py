#! /usr/bin/env python
# -*- coding: utf-8 -*-

"""
Image deformation using moving least squares.

    * Affine deformation
    * Affine inverse deformation
    * Similarity deformation
    * Similarity inverse deformation
    * Rigid deformation
    * Rigid inverse deformation (* This algorithm is approximate, because the inverse formula 
                                   of the rigid deformation is not easy to infer)

For more details please reference the documentation: 

    Moving-Least-Squares/doc/Image Deformation.pdf

or the original paper:

    Image deformation using moving least squares
    Schaefer, Mcphail, Warren.

Note:
    In the original paper, the author missed the weight w_j in formular (5).
    In addition, all the formulars in section 2.1 miss the w_j.
    And I have corrected this point in my documentation.

@author: Jarvis ZHANG
@date: 2017/8/8
@editor: VS Code
"""
import numpy as np
from skimage.transform import rescale
import scipy.io
import sys
from PIL import Image
from scipy.signal import convolve2d
from skimage.draw import circle
from scipy.ndimage import rotate
import colour_demosaicing
import glob
np.seterr(divide='ignore', invalid='ignore')
# from easydict import EasyDict as edict
# CONFIG = edict(yaml.load(open('blur_config.yaml', 'r')))
import random
import cv2

from skimage import img_as_ubyte, img_as_float, draw
from skimage.restoration import denoise_wavelet
from skimage.color import rgb2yuv, yuv2rgb
from skimage.filters.rank import median
from skimage.morphology import disk
from .HybridDenoise.hybrid_denoise import hybrid_denoise

VIVO_NOISE = {
    '100iso': [[0.10302168, 0.22796172],
               [0.1037082, 0.20281659],
               [0.09871322, 0.34510456]],
    '500iso': [[0.51525423, 1.04507114],
               [0.51331967, 1.17793258],
               [0.48696414, 1.22589979]],
    '800iso': [[0.80884628, 2.43097323],
               [0.81739142, 2.38013651],
               [0.76519675, 2.49142857]],
    '1600iso': [[1.59314132, 7.88423861],
                [1.59319833, 8.21943797],
                [1.50654706, 8.25159436]],
    '3200iso': [[3.18628265, 7.88423861],
                [3.18639667, 8.21943797],
                [3.01309412, 8.25159436]]
}

MI_NOTE10_NOISE = {
    '100iso': [[0.0641, 0.2927], [0.0608, 0.2816], [0.0623, 0.3453]],
    '300iso': [[0.1927, 0.4707], [0.1759, 0.424], [0.1759, 0.424]],
    '800iso': [[0.5353, 1.1777], [0.4554, 1.2451], [0.5192, 0.9204]],
    '1600iso': [[1.0976, 3.1551], [0.863, 3.9203], [1.0158, 3.0116]],
    '3200iso': [[2.1691, 10.7588], [1.7578, 11.4467], [1.9651, 11.6351]],
    '4800iso': [[3.2640, 23.4308], [2.5920, 23.9907], [2.9760, 25.9715]],
    '12000iso': [[8.1600, 143.1812], [6.4800, 142.5315], [7.4500, 161.4467]],
    'r': {'s': 0.068, 'r0': 0.0099, 'r1': 0.6212},
    'b': {'s': 0.054, 'r0': 0.0098, 'r1': 1.4115},
    'g': {'s': 0.062, 'r0': 0.0112, 'r1': 0.1667},
}


def noise_meta_func(iso_, s, r0, r1):
    return iso_ / 100.0 * s, (iso_ / 100.0) ** 2 * r0 + r1

def apply_sharpen(img, sr=None):
    img_float = img.astype(np.float32) / 255.0
    img_blur = cv2.GaussianBlur(img_float, (5, 5), sigmaX=1.0, sigmaY=1.0)
    details = img_float - img_blur
    sharpen_rate = random.uniform(0.3, 1.1) if sr is None else sr
    img_sharpen = np.clip(img_float + sharpen_rate * details, 0.0, 1.0)
    img = (img_sharpen * 255.0).astype(np.uint8)

    return img

def apply_haze(img):
    a = random.uniform(0.5, 0.8)
    h, w = img.shape[:2]
    if random.random() > 0.5:
        # Base light mask
        light_mask = np.zeros(img.shape, dtype=np.float32)

        # Calculate end points
        left_ = random.randint(h // 3, h)
        right_ = random.randint(h // 3, h)
        k = (right_ - left_) / float(w - 1)
        b = left_
        end_points = np.floor(k * np.asarray(list(range(0, w))) + b).astype(np.int)
        end_points = np.clip(end_points, 0, h - 1)

        for idx, ep in enumerate(end_points):
            v_slice = ((np.linspace(255, 0, ep + 1)) / 255.0) ** 0.7 * 255.0
            light_mask[0:ep + 1, idx, :] = v_slice.reshape((ep + 1, 1))

        all_cover_flag = False
    else:
        light_mask = np.ones(img.shape, dtype=np.float32) * 255.0
        all_cover_flag = True

    out_img = np.clip(img * a + light_mask * (1 - a), 0, 255).astype(np.uint8)

    return out_img, all_cover_flag

def apply_sharpen(img):
    img_float = img.astype(np.float32) / 255.0
    img_blur = cv2.GaussianBlur(img_float, (5, 5), sigmaX=1.0, sigmaY=1.0)
    details = img_float - img_blur
    sharpen_rate = random.uniform(0.3, 1.0)
    img_sharpen = np.clip(img_float + sharpen_rate * details, 0.0, 1.0)
    img = (img_sharpen * 255.0).astype(np.uint8)

    return img


def mls_affine_deformation_1pt(p, q, v, alpha=1):
    ''' Calculate the affine deformation of one point.
    This function is used to test the algorithm.
    '''
    ctrls = p.shape[0]
    np.seterr(divide='ignore')
    w = 1.0 / np.sum((p - v) ** 2, axis=1) ** alpha
    w[w == np.inf] = 2 ** 31 - 1
    pstar = np.sum(p.T * w, axis=1) / np.sum(w)
    qstar = np.sum(q.T * w, axis=1) / np.sum(w)
    phat = p - pstar
    qhat = q - qstar
    reshaped_phat1 = phat.reshape(ctrls, 2, 1)
    reshaped_phat2 = phat.reshape(ctrls, 1, 2)
    reshaped_w = w.reshape(ctrls, 1, 1)
    pTwp = np.sum(reshaped_phat1 * reshaped_w * reshaped_phat2, axis=0)
    try:
        inv_pTwp = np.linalg.inv(pTwp)
    except np.linalg.linalg.LinAlgError:
        if np.linalg.det(pTwp) < 1e-8:
            new_v = v + qstar - pstar
            return new_v
        else:
            raise
    mul_left = v - pstar
    mul_right = np.sum(reshaped_phat1 * reshaped_w * qhat[:, np.newaxis, :], axis=0)
    new_v = np.dot(np.dot(mul_left, inv_pTwp), mul_right) + qstar
    return new_v


def mls_affine_deformation(image, p, q, alpha=1.0, density=1.0):
    ''' Affine deformation
    ### Params:
        * image - ndarray: original image
        * p - ndarray: an array with size [n, 2], original control points
        * q - ndarray: an array with size [n, 2], final control points
        * alpha - float: parameter used by weights
        * density - float: density of the grids
    ### Return:
        A deformed image.
    '''
    height = image.shape[0]
    width = image.shape[1]
    # Change (x, y) to (row, col)
    q = q[:, [1, 0]]
    p = p[:, [1, 0]]

    # Make grids on the original image
    gridX = np.linspace(0, width, num=int(width * density), endpoint=False)
    gridY = np.linspace(0, height, num=int(height * density), endpoint=False)
    vy, vx = np.meshgrid(gridX, gridY)
    grow = vx.shape[0]  # grid rows
    gcol = vx.shape[1]  # grid cols
    ctrls = p.shape[0]  # control points

    # Precompute
    reshaped_p = p.reshape(ctrls, 2, 1, 1)  # [ctrls, 2, 1, 1]
    reshaped_v = np.vstack((vx.reshape(1, grow, gcol), vy.reshape(1, grow, gcol)))  # [2, grow, gcol]

    w = 1.0 / np.sum((reshaped_p - reshaped_v) ** 2, axis=1) ** alpha  # [ctrls, grow, gcol]
    w[w == np.inf] = 2 ** 31 - 1
    pstar = np.sum(w * reshaped_p.transpose(1, 0, 2, 3), axis=1) / np.sum(w, axis=0)  # [2, grow, gcol]
    phat = reshaped_p - pstar  # [ctrls, 2, grow, gcol]
    reshaped_phat1 = phat.reshape(ctrls, 2, 1, grow, gcol)  # [ctrls, 2, 1, grow, gcol]
    reshaped_phat2 = phat.reshape(ctrls, 1, 2, grow, gcol)  # [ctrls, 1, 2, grow, gcol]
    reshaped_w = w.reshape(ctrls, 1, 1, grow, gcol)  # [ctrls, 1, 1, grow, gcol]
    pTwp = np.sum(reshaped_phat1 * reshaped_w * reshaped_phat2, axis=0)  # [2, 2, grow, gcol]
    try:
        inv_pTwp = np.linalg.inv(pTwp.transpose(2, 3, 0, 1))  # [grow, gcol, 2, 2]
        flag = False
    except np.linalg.linalg.LinAlgError:
        flag = True
        det = np.linalg.det(pTwp.transpose(2, 3, 0, 1))  # [grow, gcol]
        det[det < 1e-8] = np.inf
        reshaped_det = det.reshape(1, 1, grow, gcol)  # [1, 1, grow, gcol]
        adjoint = pTwp[[[1, 0], [1, 0]], [[1, 1], [0, 0]], :, :]  # [2, 2, grow, gcol]
        adjoint[[0, 1], [1, 0], :, :] = -adjoint[[0, 1], [1, 0], :, :]  # [2, 2, grow, gcol]
        inv_pTwp = (adjoint / reshaped_det).transpose(2, 3, 0, 1)  # [grow, gcol, 2, 2]
    mul_left = reshaped_v - pstar  # [2, grow, gcol]
    reshaped_mul_left = mul_left.reshape(1, 2, grow, gcol).transpose(2, 3, 0, 1)  # [grow, gcol, 1, 2]
    mul_right = reshaped_w * reshaped_phat1  # [ctrls, 2, 1, grow, gcol]
    reshaped_mul_right = mul_right.transpose(0, 3, 4, 1, 2)  # [ctrls, grow, gcol, 2, 1]
    A = np.matmul(np.matmul(reshaped_mul_left, inv_pTwp), reshaped_mul_right)  # [ctrls, grow, gcol, 1, 1]
    reshaped_A = A.reshape(ctrls, 1, grow, gcol)  # [ctrls, 1, grow, gcol]

    # Calculate q
    reshaped_q = q.reshape((ctrls, 2, 1, 1))  # [ctrls, 2, 1, 1]
    qstar = np.sum(w * reshaped_q.transpose(1, 0, 2, 3), axis=1) / np.sum(w, axis=0)  # [2, grow, gcol]
    qhat = reshaped_q - qstar  # [ctrls, 2, grow, gcol]

    # Get final image transfomer -- 3-D array
    transformers = np.sum(reshaped_A * qhat, axis=0) + qstar  # [2, grow, gcol]

    # Correct the points where pTwp is singular
    if flag:
        blidx = det == np.inf  # bool index
        transformers[0][blidx] = vx[blidx] + qstar[0][blidx] - pstar[0][blidx]
        transformers[1][blidx] = vy[blidx] + qstar[1][blidx] - pstar[1][blidx]

    # Removed the points outside the border
    transformers[transformers < 0] = 0
    transformers[0][transformers[0] > height - 1] = 0
    transformers[1][transformers[1] > width - 1] = 0

    # Mapping original image
    transformed_image = np.ones_like(image) * 255
    new_gridY, new_gridX = np.meshgrid((np.arange(gcol) / density).astype(np.int16),
                                       (np.arange(grow) / density).astype(np.int16))
    transformed_image[tuple(transformers.astype(np.int16))] = image[new_gridX, new_gridY]  # [grow, gcol]

    transformers = transformers * 2 / (float)(image.shape[0] - 1) - 1

    return transformed_image, transformers


def mls_affine_deformation_inv(image, p, q, alpha=1.0, density=1.0):
    ''' Affine inverse deformation
    ### Params:
        * image - ndarray: original image
        * p - ndarray: an array with size [n, 2], original control points
        * q - ndarray: an array with size [n, 2], final control points
        * alpha - float: parameter used by weights
        * density - float: density of the grids
    ### Return:
        A deformed image.
    '''
    height = image.shape[0]
    width = image.shape[1]
    # Change (x, y) to (row, col)
    q = q[:, [1, 0]]
    p = p[:, [1, 0]]

    # Make grids on the original image
    gridX = np.linspace(0, width, num=int(width * density), endpoint=False)
    gridY = np.linspace(0, height, num=int(height * density), endpoint=False)
    vy, vx = np.meshgrid(gridX, gridY)
    grow = vx.shape[0]  # grid rows
    gcol = vx.shape[1]  # grid cols
    ctrls = p.shape[0]  # control points

    # Compute
    reshaped_p = p.reshape(ctrls, 2, 1, 1)  # [ctrls, 2, 1, 1]
    reshaped_q = q.reshape((ctrls, 2, 1, 1))  # [ctrls, 2, 1, 1]
    reshaped_v = np.vstack((vx.reshape(1, grow, gcol), vy.reshape(1, grow, gcol)))  # [2, grow, gcol]

    w = 1.0 / np.sum((reshaped_p - reshaped_v) ** 2, axis=1) ** alpha  # [ctrls, grow, gcol]
    w[w == np.inf] = 2 ** 31 - 1
    pstar = np.sum(w * reshaped_p.transpose(1, 0, 2, 3), axis=1) / np.sum(w, axis=0)  # [2, grow, gcol]
    phat = reshaped_p - pstar  # [ctrls, 2, grow, gcol]
    qstar = np.sum(w * reshaped_q.transpose(1, 0, 2, 3), axis=1) / np.sum(w, axis=0)  # [2, grow, gcol]
    qhat = reshaped_q - qstar  # [ctrls, 2, grow, gcol]

    reshaped_phat = phat.reshape(ctrls, 2, 1, grow, gcol)  # [ctrls, 2, 1, grow, gcol]
    reshaped_phat2 = phat.reshape(ctrls, 1, 2, grow, gcol)  # [ctrls, 2, 1, grow, gcol]
    reshaped_qhat = qhat.reshape(ctrls, 1, 2, grow, gcol)  # [ctrls, 1, 2, grow, gcol]
    reshaped_w = w.reshape(ctrls, 1, 1, grow, gcol)  # [ctrls, 1, 1, grow, gcol]
    pTwq = np.sum(reshaped_phat * reshaped_w * reshaped_qhat, axis=0)  # [2, 2, grow, gcol]
    try:
        inv_pTwq = np.linalg.inv(pTwq.transpose(2, 3, 0, 1))  # [grow, gcol, 2, 2]
        flag = False
    except np.linalg.linalg.LinAlgError:
        flag = True
        det = np.linalg.det(pTwq.transpose(2, 3, 0, 1))  # [grow, gcol]
        det[det < 1e-8] = np.inf
        reshaped_det = det.reshape(1, 1, grow, gcol)  # [1, 1, grow, gcol]
        adjoint = pTwq[[[1, 0], [1, 0]], [[1, 1], [0, 0]], :, :]  # [2, 2, grow, gcol]
        adjoint[[0, 1], [1, 0], :, :] = -adjoint[[0, 1], [1, 0], :, :]  # [2, 2, grow, gcol]
        inv_pTwq = (adjoint / reshaped_det).transpose(2, 3, 0, 1)  # [grow, gcol, 2, 2]
    mul_left = reshaped_v - qstar  # [2, grow, gcol]
    reshaped_mul_left = mul_left.reshape(1, 2, grow, gcol).transpose(2, 3, 0, 1)  # [grow, gcol, 1, 2]
    mul_right = np.sum(reshaped_phat * reshaped_w * reshaped_phat2, axis=0)  # [2, 2, grow, gcol]
    reshaped_mul_right = mul_right.transpose(2, 3, 0, 1)  # [grow, gcol, 2, 2]
    temp = np.matmul(np.matmul(reshaped_mul_left, inv_pTwq), reshaped_mul_right)  # [grow, gcol, 1, 2]
    reshaped_temp = temp.reshape(grow, gcol, 2).transpose(2, 0, 1)  # [2, grow, gcol]

    # Get final image transfomer -- 3-D array
    transformers = reshaped_temp + pstar  # [2, grow, gcol]

    # Correct the points where pTwp is singular
    if flag:
        blidx = det == np.inf  # bool index
        transformers[0][blidx] = vx[blidx] + qstar[0][blidx] - pstar[0][blidx]
        transformers[1][blidx] = vy[blidx] + qstar[1][blidx] - pstar[1][blidx]

    # Removed the points outside the border
    transformers[transformers < 0] = 0
    transformers[0][transformers[0] > height - 1] = 0
    transformers[1][transformers[1] > width - 1] = 0

    # Mapping original image
    transformed_image = image[tuple(transformers.astype(np.int16))]  # [grow, gcol]

    # Rescale image
    transformed_image = rescale(transformed_image, scale=1.0 / density, mode='reflect')

    return transformed_image


def mls_similarity_deformation(image, p, q, alpha=1.0, density=1.0):
    ''' Similarity deformation
    ### Params:
        * image - ndarray: original image
        * p - ndarray: an array with size [n, 2], original control points
        * q - ndarray: an array with size [n, 2], final control points
        * alpha - float: parameter used by weights
        * density - float: density of the grids
    ### Return:
        A deformed image.
    '''
    height = image.shape[0]
    width = image.shape[1]
    # Change (x, y) to (row, col)
    q = q[:, [1, 0]]
    p = p[:, [1, 0]]

    # Make grids on the original image
    gridX = np.linspace(0, width, num=int(width * density), endpoint=False)
    gridY = np.linspace(0, height, num=int(height * density), endpoint=False)
    vy, vx = np.meshgrid(gridX, gridY)
    grow = vx.shape[0]  # grid rows
    gcol = vx.shape[1]  # grid cols
    ctrls = p.shape[0]  # control points

    # Compute
    reshaped_p = p.reshape(ctrls, 2, 1, 1)  # [ctrls, 2, 1, 1]
    reshaped_v = np.vstack((vx.reshape(1, grow, gcol), vy.reshape(1, grow, gcol)))  # [2, grow, gcol]

    w = 1.0 / np.sum((reshaped_p - reshaped_v) ** 2, axis=1) ** alpha  # [ctrls, grow, gcol]
    sum_w = np.sum(w, axis=0)  # [grow, gcol]
    pstar = np.sum(w * reshaped_p.transpose(1, 0, 2, 3), axis=1) / sum_w  # [2, grow, gcol]
    phat = reshaped_p - pstar  # [ctrls, 2, grow, gcol]
    reshaped_phat1 = phat.reshape(ctrls, 1, 2, grow, gcol)  # [ctrls, 1, 2, grow, gcol]
    reshaped_phat2 = phat.reshape(ctrls, 2, 1, grow, gcol)  # [ctrls, 2, 1, grow, gcol]
    reshaped_w = w.reshape(ctrls, 1, 1, grow, gcol)  # [ctrls, 1, 1, grow, gcol]
    mu = np.sum(np.matmul(reshaped_w.transpose(0, 3, 4, 1, 2) *
                          reshaped_phat1.transpose(0, 3, 4, 1, 2),
                          reshaped_phat2.transpose(0, 3, 4, 1, 2)), axis=0)  # [grow, gcol, 1, 1]
    reshaped_mu = mu.reshape(1, grow, gcol)  # [1, grow, gcol]
    neg_phat_verti = phat[:, [1, 0], ...]  # [ctrls, 2, grow, gcol]
    neg_phat_verti[:, 1, ...] = -neg_phat_verti[:, 1, ...]
    reshaped_neg_phat_verti = neg_phat_verti.reshape(ctrls, 1, 2, grow, gcol)  # [ctrls, 1, 2, grow, gcol]
    mul_left = np.concatenate((reshaped_phat1, reshaped_neg_phat_verti), axis=1)  # [ctrls, 2, 2, grow, gcol]
    vpstar = reshaped_v - pstar  # [2, grow, gcol]
    reshaped_vpstar = vpstar.reshape(2, 1, grow, gcol)  # [2, 1, grow, gcol]
    neg_vpstar_verti = vpstar[[1, 0], ...]  # [2, grow, gcol]
    neg_vpstar_verti[1, ...] = -neg_vpstar_verti[1, ...]
    reshaped_neg_vpstar_verti = neg_vpstar_verti.reshape(2, 1, grow, gcol)  # [2, 1, grow, gcol]
    mul_right = np.concatenate((reshaped_vpstar, reshaped_neg_vpstar_verti), axis=1)  # [2, 2, grow, gcol]
    reshaped_mul_right = mul_right.reshape(1, 2, 2, grow, gcol)  # [1, 2, 2, grow, gcol]
    A = np.matmul((reshaped_w * mul_left).transpose(0, 3, 4, 1, 2),
                  reshaped_mul_right.transpose(0, 3, 4, 1, 2))  # [ctrls, grow, gcol, 2, 2]

    # Calculate q
    reshaped_q = q.reshape((ctrls, 2, 1, 1))  # [ctrls, 2, 1, 1]
    qstar = np.sum(w * reshaped_q.transpose(1, 0, 2, 3), axis=1) / np.sum(w, axis=0)  # [2, grow, gcol]
    qhat = reshaped_q - qstar  # [ctrls, 2, grow, gcol]
    reshaped_qhat = qhat.reshape(ctrls, 1, 2, grow, gcol).transpose(0, 3, 4, 1, 2)  # [ctrls, grow, gcol, 1, 2]

    # Get final image transfomer -- 3-D array
    temp = np.sum(np.matmul(reshaped_qhat, A), axis=0).transpose(2, 3, 0, 1)  # [1, 2, grow, gcol]
    reshaped_temp = temp.reshape(2, grow, gcol)  # [2, grow, gcol]
    transformers = reshaped_temp / reshaped_mu + qstar  # [2, grow, gcol]

    # Removed the points outside the border
    transformers[transformers < 0] = 0
    transformers[0][transformers[0] > height - 1] = 0
    transformers[1][transformers[1] > width - 1] = 0

    # Mapping original image
    transformed_image = np.ones_like(image) * 255
    new_gridY, new_gridX = np.meshgrid((np.arange(gcol) / density).astype(np.int16),
                                       (np.arange(grow) / density).astype(np.int16))
    transformed_image[tuple(transformers.astype(np.int16))] = image[new_gridX, new_gridY]  # [grow, gcol]

    return transformed_image


def mls_similarity_deformation_inv(image, p, q, alpha=1.0, density=1.0):
    ''' Similarity inverse deformation
    ### Params:
        * image - ndarray: original image
        * p - ndarray: an array with size [n, 2], original control points
        * q - ndarray: an array with size [n, 2], final control points
        * alpha - float: parameter used by weights
        * density - float: density of the grids
    ### Return:
        A deformed image.
    '''
    height = image.shape[0]
    width = image.shape[1]
    # Change (x, y) to (row, col)
    q = q[:, [1, 0]]
    p = p[:, [1, 0]]

    # Make grids on the original image
    gridX = np.linspace(0, width, num=int(width * density), endpoint=False)
    gridY = np.linspace(0, height, num=int(height * density), endpoint=False)
    vy, vx = np.meshgrid(gridX, gridY)
    grow = vx.shape[0]  # grid rows
    gcol = vx.shape[1]  # grid cols
    ctrls = p.shape[0]  # control points

    # Compute
    reshaped_p = p.reshape(ctrls, 2, 1, 1)  # [ctrls, 2, 1, 1]
    reshaped_q = q.reshape((ctrls, 2, 1, 1))  # [ctrls, 2, 1, 1]
    reshaped_v = np.vstack((vx.reshape(1, grow, gcol), vy.reshape(1, grow, gcol)))  # [2, grow, gcol]

    w = 1.0 / np.sum((reshaped_p - reshaped_v) ** 2, axis=1) ** alpha  # [ctrls, grow, gcol]
    w[w == np.inf] = 2 ** 31 - 1
    pstar = np.sum(w * reshaped_p.transpose(1, 0, 2, 3), axis=1) / np.sum(w, axis=0)  # [2, grow, gcol]
    phat = reshaped_p - pstar  # [ctrls, 2, grow, gcol]
    qstar = np.sum(w * reshaped_q.transpose(1, 0, 2, 3), axis=1) / np.sum(w, axis=0)  # [2, grow, gcol]
    qhat = reshaped_q - qstar  # [ctrls, 2, grow, gcol]
    reshaped_phat1 = phat.reshape(ctrls, 1, 2, grow, gcol)  # [ctrls, 1, 2, grow, gcol]
    reshaped_phat2 = phat.reshape(ctrls, 2, 1, grow, gcol)  # [ctrls, 2, 1, grow, gcol]
    reshaped_qhat = qhat.reshape(ctrls, 1, 2, grow, gcol)  # [ctrls, 1, 2, grow, gcol]
    reshaped_w = w.reshape(ctrls, 1, 1, grow, gcol)  # [ctrls, 1, 1, grow, gcol]

    mu = np.sum(np.matmul(reshaped_w.transpose(0, 3, 4, 1, 2) *
                          reshaped_phat1.transpose(0, 3, 4, 1, 2),
                          reshaped_phat2.transpose(0, 3, 4, 1, 2)), axis=0)  # [grow, gcol, 1, 1]
    reshaped_mu = mu.reshape(1, grow, gcol)  # [1, grow, gcol]
    neg_phat_verti = phat[:, [1, 0], ...]  # [ctrls, 2, grow, gcol]
    neg_phat_verti[:, 1, ...] = -neg_phat_verti[:, 1, ...]
    reshaped_neg_phat_verti = neg_phat_verti.reshape(ctrls, 1, 2, grow, gcol)  # [ctrls, 1, 2, grow, gcol]
    mul_right = np.concatenate((reshaped_phat1, reshaped_neg_phat_verti), axis=1)  # [ctrls, 2, 2, grow, gcol]
    mul_left = reshaped_qhat * reshaped_w  # [ctrls, 1, 2, grow, gcol]
    Delta = np.sum(np.matmul(mul_left.transpose(0, 3, 4, 1, 2),
                             mul_right.transpose(0, 3, 4, 1, 2)),
                   axis=0).transpose(0, 1, 3, 2)  # [grow, gcol, 2, 1]
    Delta_verti = Delta[..., [1, 0], :]  # [grow, gcol, 2, 1]
    Delta_verti[..., 0, :] = -Delta_verti[..., 0, :]
    B = np.concatenate((Delta, Delta_verti), axis=3)  # [grow, gcol, 2, 2]
    try:
        inv_B = np.linalg.inv(B)  # [grow, gcol, 2, 2]
        flag = False
    except np.linalg.linalg.LinAlgError:
        flag = True
        det = np.linalg.det(B)  # [grow, gcol]
        det[det < 1e-8] = np.inf
        reshaped_det = det.reshape(grow, gcol, 1, 1)  # [grow, gcol, 1, 1]
        adjoint = B[:, :, [[1, 0], [1, 0]], [[1, 1], [0, 0]]]  # [grow, gcol, 2, 2]
        adjoint[:, :, [0, 1], [1, 0]] = -adjoint[:, :, [0, 1], [1, 0]]  # [grow, gcol, 2, 2]
        inv_B = (adjoint / reshaped_det).transpose(2, 3, 0, 1)  # [2, 2, grow, gcol]

    v_minus_qstar_mul_mu = (reshaped_v - qstar) * reshaped_mu  # [2, grow, gcol]

    # Get final image transfomer -- 3-D array
    reshaped_v_minus_qstar_mul_mu = v_minus_qstar_mul_mu.reshape(1, 2, grow, gcol)  # [1, 2, grow, gcol]
    transformers = np.matmul(reshaped_v_minus_qstar_mul_mu.transpose(2, 3, 0, 1),
                             inv_B).reshape(grow, gcol, 2).transpose(2, 0, 1) + pstar  # [2, grow, gcol]

    # Correct the points where pTwp is singular
    if flag:
        blidx = det == np.inf  # bool index
        transformers[0][blidx] = vx[blidx] + qstar[0][blidx] - pstar[0][blidx]
        transformers[1][blidx] = vy[blidx] + qstar[1][blidx] - pstar[1][blidx]

    # Removed the points outside the border
    transformers[transformers < 0] = 0
    transformers[0][transformers[0] > height - 1] = 0
    transformers[1][transformers[1] > width - 1] = 0

    # Mapping original image
    transformed_image = image[tuple(transformers.astype(np.int16))]  # [grow, gcol]

    # Rescale image
    transformed_image = rescale(transformed_image, scale=1.0 / density, mode='reflect')

    return transformed_image


def mls_rigid_deformation(image, p, q, alpha=1.0, density=1.0):
    ''' Rigid deformation
    ### Params:
        * image - ndarray: original image
        * p - ndarray: an array with size [n, 2], original control points
        * q - ndarray: an array with size [n, 2], final control points
        * alpha - float: parameter used by weights
        * density - float: density of the grids
    ### Return:
        A deformed image.
    '''
    height = image.shape[0]
    width = image.shape[1]
    # Change (x, y) to (row, col)
    q = q[:, [1, 0]]
    p = p[:, [1, 0]]

    # Make grids on the original image
    gridX = np.linspace(0, width, num=int(width * density), endpoint=False)
    gridY = np.linspace(0, height, num=int(height * density), endpoint=False)
    vy, vx = np.meshgrid(gridX, gridY)
    grow = vx.shape[0]  # grid rows
    gcol = vx.shape[1]  # grid cols
    ctrls = p.shape[0]  # control points

    # Compute
    reshaped_p = p.reshape(ctrls, 2, 1, 1)  # [ctrls, 2, 1, 1]
    reshaped_v = np.vstack((vx.reshape(1, grow, gcol), vy.reshape(1, grow, gcol)))  # [2, grow, gcol]

    w = 1.0 / np.sum((reshaped_p - reshaped_v) ** 2, axis=1) ** alpha  # [ctrls, grow, gcol]
    sum_w = np.sum(w, axis=0)  # [grow, gcol]
    pstar = np.sum(w * reshaped_p.transpose(1, 0, 2, 3), axis=1) / sum_w  # [2, grow, gcol]
    phat = reshaped_p - pstar  # [ctrls, 2, grow, gcol]
    reshaped_phat = phat.reshape(ctrls, 1, 2, grow, gcol)  # [ctrls, 1, 2, grow, gcol]
    reshaped_w = w.reshape(ctrls, 1, 1, grow, gcol)  # [ctrls, 1, 1, grow, gcol]
    neg_phat_verti = phat[:, [1, 0], ...]  # [ctrls, 2, grow, gcol]
    neg_phat_verti[:, 1, ...] = -neg_phat_verti[:, 1, ...]
    reshaped_neg_phat_verti = neg_phat_verti.reshape(ctrls, 1, 2, grow, gcol)  # [ctrls, 1, 2, grow, gcol]
    mul_left = np.concatenate((reshaped_phat, reshaped_neg_phat_verti), axis=1)  # [ctrls, 2, 2, grow, gcol]
    vpstar = reshaped_v - pstar  # [2, grow, gcol]
    reshaped_vpstar = vpstar.reshape(2, 1, grow, gcol)  # [2, 1, grow, gcol]
    neg_vpstar_verti = vpstar[[1, 0], ...]  # [2, grow, gcol]
    neg_vpstar_verti[1, ...] = -neg_vpstar_verti[1, ...]
    reshaped_neg_vpstar_verti = neg_vpstar_verti.reshape(2, 1, grow, gcol)  # [2, 1, grow, gcol]
    mul_right = np.concatenate((reshaped_vpstar, reshaped_neg_vpstar_verti), axis=1)  # [2, 2, grow, gcol]
    reshaped_mul_right = mul_right.reshape(1, 2, 2, grow, gcol)  # [1, 2, 2, grow, gcol]
    A = np.matmul((reshaped_w * mul_left).transpose(0, 3, 4, 1, 2),
                  reshaped_mul_right.transpose(0, 3, 4, 1, 2))  # [ctrls, grow, gcol, 2, 2]

    # Calculate q
    reshaped_q = q.reshape((ctrls, 2, 1, 1))  # [ctrls, 2, 1, 1]
    qstar = np.sum(w * reshaped_q.transpose(1, 0, 2, 3), axis=1) / np.sum(w, axis=0)  # [2, grow, gcol]
    qhat = reshaped_q - qstar  # [2, grow, gcol]
    reshaped_qhat = qhat.reshape(ctrls, 1, 2, grow, gcol).transpose(0, 3, 4, 1, 2)  # [ctrls, grow, gcol, 1, 2]

    # Get final image transfomer -- 3-D array
    temp = np.sum(np.matmul(reshaped_qhat, A), axis=0).transpose(2, 3, 0, 1)  # [1, 2, grow, gcol]
    reshaped_temp = temp.reshape(2, grow, gcol)  # [2, grow, gcol]
    norm_reshaped_temp = np.linalg.norm(reshaped_temp, axis=0, keepdims=True)  # [1, grow, gcol]
    norm_vpstar = np.linalg.norm(vpstar, axis=0, keepdims=True)  # [1, grow, gcol]
    transformers = reshaped_temp / norm_reshaped_temp * norm_vpstar + qstar  # [2, grow, gcol]

    # Removed the points outside the border
    transformers[transformers < 0] = 0
    transformers[0][transformers[0] > height - 1] = 0
    transformers[1][transformers[1] > width - 1] = 0

    # Mapping original image
    transformed_image = np.ones_like(image) * 255
    new_gridY, new_gridX = np.meshgrid((np.arange(gcol) / density).astype(np.int16),
                                       (np.arange(grow) / density).astype(np.int16))
    transformed_image[tuple(transformers.astype(np.int16))] = image[new_gridX, new_gridY]  # [grow, gcol]

    transformers = transformers * 2 / (float)(image.shape[0] - 1) - 1
    print(type(transformers))
    # transformers = np.transpose(transformers, ())
    return transformed_image, transformers


def mls_rigid_deformation_inv(image, p, q, alpha=1.0, density=1.0):
    ''' Rigid inverse deformation
    ### Params:
        * image - ndarray: original image
        * p - ndarray: an array with size [n, 2], original control points
        * q - ndarray: an array with size [n, 2], final control points
        * alpha - float: parameter used by weights
        * density - float: density of the grids
    ### Return:
        A deformed image.
    '''
    height = image.shape[0]
    width = image.shape[1]
    # Change (x, y) to (row, col)
    q = q[:, [1, 0]]
    p = p[:, [1, 0]]

    # Make grids on the original image
    gridX = np.linspace(0, width, num=int(width * density), endpoint=False)
    gridY = np.linspace(0, height, num=int(height * density), endpoint=False)
    vy, vx = np.meshgrid(gridX, gridY)
    grow = vx.shape[0]  # grid rows
    gcol = vx.shape[1]  # grid cols
    ctrls = p.shape[0]  # control points

    # Compute
    reshaped_p = p.reshape(ctrls, 2, 1, 1)  # [ctrls, 2, 1, 1]
    reshaped_q = q.reshape((ctrls, 2, 1, 1))  # [ctrls, 2, 1, 1]
    reshaped_v = np.vstack((vx.reshape(1, grow, gcol), vy.reshape(1, grow, gcol)))  # [2, grow, gcol]

    w = 1.0 / np.sum((reshaped_p - reshaped_v) ** 2, axis=1) ** alpha  # [ctrls, grow, gcol]
    w[w == np.inf] = 2 ** 31 - 1
    pstar = np.sum(w * reshaped_p.transpose(1, 0, 2, 3), axis=1) / np.sum(w, axis=0)  # [2, grow, gcol]
    phat = reshaped_p - pstar  # [ctrls, 2, grow, gcol]
    qstar = np.sum(w * reshaped_q.transpose(1, 0, 2, 3), axis=1) / np.sum(w, axis=0)  # [2, grow, gcol]
    qhat = reshaped_q - qstar  # [ctrls, 2, grow, gcol]
    reshaped_phat1 = phat.reshape(ctrls, 1, 2, grow, gcol)  # [ctrls, 1, 2, grow, gcol]
    reshaped_phat2 = phat.reshape(ctrls, 2, 1, grow, gcol)  # [ctrls, 2, 1, grow, gcol]
    reshaped_qhat = qhat.reshape(ctrls, 1, 2, grow, gcol)  # [ctrls, 1, 2, grow, gcol]
    reshaped_w = w.reshape(ctrls, 1, 1, grow, gcol)  # [ctrls, 1, 1, grow, gcol]

    mu = np.sum(np.matmul(reshaped_w.transpose(0, 3, 4, 1, 2) *
                          reshaped_phat1.transpose(0, 3, 4, 1, 2),
                          reshaped_phat2.transpose(0, 3, 4, 1, 2)), axis=0)  # [grow, gcol, 1, 1]
    reshaped_mu = mu.reshape(1, grow, gcol)  # [1, grow, gcol]
    neg_phat_verti = phat[:, [1, 0], ...]  # [ctrls, 2, grow, gcol]
    neg_phat_verti[:, 1, ...] = -neg_phat_verti[:, 1, ...]
    reshaped_neg_phat_verti = neg_phat_verti.reshape(ctrls, 1, 2, grow, gcol)  # [ctrls, 1, 2, grow, gcol]
    mul_right = np.concatenate((reshaped_phat1, reshaped_neg_phat_verti), axis=1)  # [ctrls, 2, 2, grow, gcol]
    mul_left = reshaped_qhat * reshaped_w  # [ctrls, 1, 2, grow, gcol]
    Delta = np.sum(np.matmul(mul_left.transpose(0, 3, 4, 1, 2),
                             mul_right.transpose(0, 3, 4, 1, 2)),
                   axis=0).transpose(0, 1, 3, 2)  # [grow, gcol, 2, 1]
    Delta_verti = Delta[..., [1, 0], :]  # [grow, gcol, 2, 1]
    Delta_verti[..., 0, :] = -Delta_verti[..., 0, :]
    B = np.concatenate((Delta, Delta_verti), axis=3)  # [grow, gcol, 2, 2]
    try:
        inv_B = np.linalg.inv(B)  # [grow, gcol, 2, 2]
        flag = False
    except np.linalg.linalg.LinAlgError:
        flag = True
        det = np.linalg.det(B)  # [grow, gcol]
        det[det < 1e-8] = np.inf
        reshaped_det = det.reshape(grow, gcol, 1, 1)  # [grow, gcol, 1, 1]
        adjoint = B[:, :, [[1, 0], [1, 0]], [[1, 1], [0, 0]]]  # [grow, gcol, 2, 2]
        adjoint[:, :, [0, 1], [1, 0]] = -adjoint[:, :, [0, 1], [1, 0]]  # [grow, gcol, 2, 2]
        inv_B = (adjoint / reshaped_det).transpose(2, 3, 0, 1)  # [2, 2, grow, gcol]

    vqstar = reshaped_v - qstar  # [2, grow, gcol]
    reshaped_vqstar = vqstar.reshape(1, 2, grow, gcol)  # [1, 2, grow, gcol]

    # Get final image transfomer -- 3-D array
    temp = np.matmul(reshaped_vqstar.transpose(2, 3, 0, 1),
                     inv_B).reshape(grow, gcol, 2).transpose(2, 0, 1)  # [2, grow, gcol]
    norm_temp = np.linalg.norm(temp, axis=0, keepdims=True)  # [1, grow, gcol]
    norm_vqstar = np.linalg.norm(vqstar, axis=0, keepdims=True)  # [1, grow, gcol]
    transformers = temp / norm_temp * norm_vqstar + pstar  # [2, grow, gcol]

    # Correct the points where pTwp is singular
    if flag:
        blidx = det == np.inf  # bool index
        transformers[0][blidx] = vx[blidx] + qstar[0][blidx] - pstar[0][blidx]
        transformers[1][blidx] = vy[blidx] + qstar[1][blidx] - pstar[1][blidx]

    # Removed the points outside the border
    transformers[transformers < 0] = 0
    transformers[0][transformers[0] > height - 1] = 0
    transformers[1][transformers[1] > width - 1] = 0
    # Mapping original image
    transformed_image = image[tuple(transformers.astype(np.int16))]  # [grow, gcol]
    # Rescale image
    # transformed_image = rescale(transformed_image, scale=1.0 / density, mode='reflect')
    transformers = transformers * 2 / (float)(image.shape[0] - 1) - 1

    return transformed_image, transformers


def debug_point(landmark, img):
    import cv2
    for point in landmark:
        cv2.circle(img, (point[0], point[1]), 1, (255, 255, 255), 1)
    return img


'''
0:清晰　
1:模糊
2:噪声
3:模糊＋噪声
4:超分
5:模糊＋超分
6:噪声＋超分
7:模糊＋超分＋噪声
'''


def op4_DefocusBlur_random(img):
    kernelidx = np.random.randint(0, len(defocusKernelDims))
    kerneldim = defocusKernelDims[kernelidx]
    return DefocusBlur(img, kerneldim)


def DefocusBlur(img, dim):
    imgarray = np.array(img, dtype="float32")
    kernel = DiskKernel(dim)
    convolved = convolve2d(imgarray, kernel, mode='same', fillvalue=255.0).astype("uint8")
    img = Image.fromarray(convolved)
    return img


def DiskKernel(dim):
    kernelwidth = dim
    kernel = np.zeros((kernelwidth, kernelwidth), dtype=np.float32)
    circleCenterCoord = dim // 2
    circleRadius = circleCenterCoord + 1

    rr, cc = draw.circle(circleCenterCoord, circleCenterCoord, circleRadius)
    kernel[rr, cc] = 1

    if (dim == 3 or dim == 5):
        kernel = Adjust(kernel, dim)

    normalizationFactor = np.count_nonzero(kernel)
    kernel = kernel / normalizationFactor
    return kernel


def Adjust(kernel, kernelwidth):
    kernel[0, 0] = 0
    kernel[0, kernelwidth - 1] = 0
    kernel[kernelwidth - 1, 0] = 0
    kernel[kernelwidth - 1, kernelwidth - 1] = 0
    return kernel


class Distortion:

    def __init__(self, distortion):
        self.distortion_strength = distortion.strength
        self.label = [0, 0, 0]
        self.distortion = list()
        self.down = 1

    def op1_gaussian_blur(self, img, bias=0):

        left = self.distortion_strength['Blur'][0]
        right = self.distortion_strength['Blur'][1]
        sigma = random.randint(left, right) + bias
        norm_sigma = float(sigma - left) / float(right - left)
        self.distortion.append(norm_sigma)
        if sigma >= 1 and sigma <= 6:
            sizeG = sigma * 12 + 1
            self.label[2] = 1
            return cv2.GaussianBlur(img, (sizeG, sizeG), sigma)
        return img

    def op2_down(self, img):
        left = self.distortion_strength.Down[0]
        right = self.distortion_strength.Down[1]
        op = random.randint(left, right)
        norm_op = float(op - left) / float(right - left)
        self.distortion.append(norm_op)

        scaleimg = 2 ** op
        self.down = scaleimg
        if scaleimg > 1:
            self.label[0] = 1
        newh = img.shape[0] // scaleimg
        neww = img.shape[1] // scaleimg
        return cv2.resize(img, (neww, newh), interpolation=cv2.INTER_AREA)

    def op2_up(self, img, image_size):
        return cv2.resize(img, image_size, interpolation=cv2.INTER_LINEAR)

    def op3_gaussian_noise(self, img):
        left = self.distortion_strength.Noise[0]
        right = self.distortion_strength.Noise[1]
        row, col, ch = img.shape
        mean = 0
        # var = 0.1
        # sigma = var ** 0.5
        var = random.randint(left, right)
        norm_var = float(var - left) / float(right - left)
        self.distortion.append(norm_var)
        if var == 0:
            return img
        sigma = var
        gauss = np.random.normal(mean, sigma, (row, col, ch))
        gauss = gauss.reshape(row, col, ch)
        noisy = img + gauss
        noisy[noisy < 0] = 0
        noisy[noisy > 255] = 255
        self.label[1] = 1
        return noisy.astype(np.uint8)

    def op4_DefocusBlur_random(self, img):
        kernelidx = np.random.randint(0, len(defocusKernelDims))
        kerneldim = defocusKernelDims[kernelidx]
        return DefocusBlur(img, kerneldim)

    # define constant distortion
    def op1_gaussian_blur_without_rand(self, img):
        sigma = self.distortion_strength['Blur'][1]
        self.distortion.append(sigma / 4.0)
        self.distortion.append(sigma)
        if sigma >= 1 and sigma <= 6:
            sizeG = sigma * 12 + 1
            self.label[2] = 1
            return cv2.GaussianBlur(img, (sizeG, sizeG), sigma)
        else:
            return img

    def op2_down_without_rand(self, img):
        op = self.distortion_strength['Down'][1]
        scaleimg = 2 ** op
        self.distortion.append(scaleimg / 2.0)
        newh = img.shape[0] // scaleimg
        neww = img.shape[1] // scaleimg
        return cv2.resize(img, (neww, newh), interpolation=cv2.INTER_AREA)

    def op3_gaussian_noise_without_rand(self, img):
        row, col, ch = img.shape
        mean = 0
        var = self.distortion_strength['Noise'][1]
        self.distortion.append(var / 4.0)
        sigma = var
        # Fix time seed
        np.random.seed(5)
        gauss = np.random.normal(mean, sigma, (row, col, ch))
        gauss = gauss.reshape(row, col, ch)
        noisy = img + gauss
        noisy[noisy < 0] = 0
        noisy[noisy > 255] = 255
        self.label[1] = 1
        return noisy.astype(np.uint8)

    def Distort_random(self, image, Size):
        return self.op2_up(self.op3_gaussian_noise(
            self.op2_down(self.op1_gaussian_blur(image))), Size).astype(np.uint8), self.down

    def Distort_constant_noise(self, image, Size):
        return self.op2_up(self.op3_gaussian_noise_without_rand(self.op2_down_without_rand(image))
                           , Size).astype(np.uint8), self.down

        # constant

    def Distort_constant_blur(self, image, Size):
        return self.op2_up(self.op2_down_without_rand(self.op1_gaussian_blur_without_rand(image))
                           , Size).astype(np.uint8), self.down

    def Distort_constant_down(self, image, Size):
        return self.op2_up(self.op2_down_without_rand(image)
                           , Size).astype(np.uint8), self.down

    # constant
    def Distort_constant(self, image, Size):
        return self.op2_up(
            self.op3_gaussian_noise_without_rand(self.op2_down_without_rand(self.op1_gaussian_blur_without_rand(image)))
            , Size).astype(np.uint8), self.down


class Distortion_v2:
    """
        New version of distortion
    """

    def __init__(self, distortion, ratio=1.):
        self.distortion_strength = distortion.strength
        self.raw_input_flag = distortion.get("flags", {}).get("raw_input_flag", False)
        self.down = 1
        self.kernel_files = glob.glob("codes/utils/kernels/*.mat")
        self.kernel_changes = list(np.linspace(1.0, 2.0, 5))
        self.kernel_angles = list(np.linspace(0, 360, 9))
        # scale ratio of degree
        self.ratio = ratio
        # distortion record list
        self.distortion_record = {}
        self.only_probability = False
        self.Max_Blurness = 16.0

    def _gama_trans(self, img, gama=2.2):
        img = np.power(img / 255., gama)
        return (img * 255.).astype(np.uint8)

    def Distort_random(self, img, img_size):
        self.o_size = img_size
        global_noise_flag = random.random()
        motion_blur_flag = random.random()
        defocus_flag = random.random()

        x = img
        if defocus_flag > 0.5:
            x = self._op_defocus(x)
        if motion_blur_flag > 0.5:
            x = self._op_motion(x)

        if global_noise_flag > 0.5:
            x = self._op_gaussian_noise(x)
        else:
            x = self._op_ychannel(x)

        # Downsize and upsize
        x, down_flag = self._op_down(x)

        # JEPG compression
        x = self._op_jpeg(x)

        if down_flag:
            x = self._op_up(x)

        return x, self.down

    def Distort_random_v2(self, img, img_size):
        self.o_size = img_size
        x = img
        # defocus blur
        x = self._op_defocus(x)
        # Downsize
        x, down_flag = self._op_down(x)
        # Noise
        x = self._op_gaussian_noise(x)

        if down_flag:
            x = self._op_up(x)

        return x, self.down

    def Distort_random_v3(self, img, img_size):
        self.o_size = img_size
        x = img
        if hasattr(self.distortion_strength, 'noDistortion'):
            small_num = self.distortion_strength.noDistortion.left
            large_num = self.distortion_strength.noDistortion.right
            chanceOfNoDistortion = random.randint(small_num, large_num)
            if chanceOfNoDistortion == small_num:
                return x, 1

        if hasattr(self.distortion_strength, 'Motion') and hasattr(self.distortion_strength, 'Defocus'):
            # defocus blur
            if hasattr(self.distortion_strength, 'high') and self.distortion_strength.high:
                chance = random.randint(0, 5)
                # 1/5 chance for motion blur
                chance = random.randint(0, 5)
                if chance == 0:
                    x = self._op_motion(x)
                # 1/5 chance for added
                elif chance == 1:
                    x = self._op_motion(x)
                    x = self._op_defocus(x)
                # 3/5 chance
                elif chance == 2 or chance == 3 or chance == 4:
                    pass
                # 1/5 chance for defocus blur
                else:
                    x = self._op_defocus(x)

            elif hasattr(self.distortion_strength, 'low') and self.distortion_strength.low:
                chance = random.randint(0, 3)
                if chance == 0:
                    x = self._op_motion(x)
                elif chance == 1:
                    x = self._op_motion(x)
                    x = self._op_defocus(x)
                else:
                    x = self._op_defocus(x)
            else:
                chance = random.randint(0, 5)
                # 1/5 chance for motion blur
                if chance == 0:
                    x = self._op_motion(x)
                # 1/5 chance for added
                elif chance == 1:
                    x = self._op_motion(x)
                    x = self._op_defocus(x)
                # 1/5 chance
                elif chance == 2:
                    pass
                # 2/5 chance for defocus blur
                else:
                    x = self._op_defocus(x)

        # Downsize
        x, down_flag = self._op_down(x)
        prob = random.randint(0, 1)
        if prob == 0:
            # Noise
            x = self._op_ychannel(x)
        else:
            x = self._op_gaussian_noise(x)
        # JPEG compress
        x = self._op_jpeg(x)

        # Beautyfy
        if hasattr(self.distortion_strength, 'Beautify'):
            x = self._op_beautyfy(x)

        if down_flag:
            x = self._op_up(x)

        return x, self.down

    # update dis
    def Distort_random_v4(self, img, img_size, noise_model="MI_NOTE10_NOISE"):
        shape = img.shape[:2]

        self._op_random_awb()

        if self.raw_input_flag:
            raw_ = self._op_resample(img)
            gt_raw = self._op_fold_raw(raw_)
        else:
            gt_raw = None

        self.o_size = img_size
        x = img.copy()
        # single disort
        flag_single = False
        sharpen_flag = True

        chance = random.randint(0, 5)
        print("chance",chance)

        print(random.random())

        # 1/6 chance for motion blur
        if chance == 0 or chance == 3:
            x = self._op_blur_motion(x) if random.random() > 0.7 else self._op_motion(x)
            if random.random() > 0.7:
                flag_single = True
        # 1/6 chance for added
        elif chance == 1:
            x = self._op_motion(x) if random.random() > 0.5 else self._op_blur_motion(x)
            x = self._op_defocus(x)
            if random.random() > 0.7:
                flag_single = True
        # 1/6 chance
        elif chance == 2:
            sharpen_flag = False
        # 3/6 chance for defocus blur
        else:
            x = self._op_defocus(x)

        # Downsize
        x, down_flag = self._op_down(x)
        prob = random.randint(0, 2)
        if prob == 0 and not self.raw_input_flag:
            # Noise
            if random.random() > 0.5:
                x = self._op_ychannel(x)
            else:
                x = self._op_gaussian_noise(x)
        else:
            # Adding noise, not considering post-processing
            # x = self._op_noise_raw(x, noise_model=noise_model)
            x = self._op_noise_and_sharpen(x, sharpen_flag=sharpen_flag)

        if not self.raw_input_flag:
            if flag_single:
                x = self._op_up(x)
                return None, x, self.down
            # JPEG compress
            x = self._op_jpeg(x)
            # Beautyfy
            if hasattr(self.distortion_strength, 'Beautify'):
                x = self._op_beautyfy(x)
            if down_flag:
                x = self._op_up(x)
        else:
            x = self._op_fold_raw(x)
            if down_flag:
                x = self._op_up(x)

        return gt_raw, x, self.down


    def Distort_random_v5(self, img, img_size, seed_, noise_model="MI_NOTE10_NOISE"):
        random.seed(seed_)
        shape = img.shape[:2]

        self._op_random_awb()

        if self.raw_input_flag:
            raw_ = self._op_resample(img)
            gt_raw = self._op_fold_raw(raw_)
        else:
            gt_raw = None

        self.o_size = img_size
        x = img.copy()
        # single disort
        flag_single = False
        sharpen_flag = True

        chance = random.randint(0, 5)

        # 1/6 chance for motion blur
        if chance == 0 or chance == 3:
            x = self._op_blur_motion(x) if random.random() > 0.7 else self._op_motion(x)
            if random.random() > 0.7:
                flag_single = True
        # 1/6 chance for added
        elif chance == 1:
            x = self._op_motion(x) if random.random() > 0.5 else self._op_blur_motion(x)
            x = self._op_defocus(x)
            if random.random() > 0.7:
                flag_single = True
        # 1/6 chance
        elif chance == 2:
            sharpen_flag = False
        # 3/6 chance for defocus blur
        else:
            x = self._op_defocus(x)

        # Downsize
        x, down_flag = self._op_down(x)
        prob = random.randint(0, 2)
        if prob == 0 and not self.raw_input_flag:
            # Noise
            if random.random() > 0.5:
                x = self._op_ychannel(x)
            else:
                x = self._op_gaussian_noise(x)
        else:
            # Adding noise, not considering post-processing
            # x = self._op_noise_raw(x, noise_model=noise_model)
            x = self._op_noise_and_sharpen(x, sharpen_flag=sharpen_flag)

        if not self.raw_input_flag:
            if flag_single:
                x = self._op_up(x)
                return None, x, self.down
            # JPEG compress
            x = self._op_jpeg(x)
            # Beautyfy
            if hasattr(self.distortion_strength, 'Beautify'):
                x = self._op_beautyfy(x)
            if down_flag:
                x = self._op_up(x)
        else:
            x = self._op_fold_raw(x)
            if down_flag:
                x = self._op_up(x)

        return gt_raw, x, self.down


    def _op_down(self, img):
        """Randomly downsample input image"""
        left = self.distortion_strength.Down[0]
        right = self.distortion_strength.Down[1]
        op = random.randint(left, right)
        norm_op = op / self.Max_Blurness if op != 0 else 0
        _function_name = sys._getframe().f_code.co_name
        self.distortion_record[_function_name] = norm_op

        if op == 0:
            down_flag = False
        else:
            down_flag = True

        if self.only_probability or not down_flag:
            return img, False

        scaleimg = op
        self.down = scaleimg
        # if scaleimg > 1:
        # self.label[0] = 1
        newh = img.shape[0] // scaleimg
        neww = img.shape[1] // scaleimg
        # Make sure it is odd
        newh += newh % 2
        neww += neww % 2
        return cv2.resize(img, (neww, newh), interpolation=cv2.INTER_AREA), down_flag

    def _op_up(self, img):
        """Resize up the image, if necessary. Work with `_op_down`"""
        if self.only_probability:
            return img
        if self.raw_input_flag:
            img = np.transpose(img, (1, 2, 0))
            img = cv2.resize(img, (self.o_size[0], self.o_size[1]), 
                    interpolation=cv2.INTER_LINEAR)
            img = np.transpose(img, (2, 0, 1))

            return img

        return cv2.resize(img, (self.o_size[0], self.o_size[1]),
                          interpolation=cv2.INTER_LINEAR)

    def lap_pyr(self, img, num_level=5):

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

    def lap_pyr_rec(self, laplacian_pyramid, var=5, num_level=5):
        reconstructed_image = laplacian_pyramid[0]

        _var_list = [var + x for x in range(num_level)]
        for i in range(1, num_level + 1):
            size = (laplacian_pyramid[i].shape[1], laplacian_pyramid[i].shape[0])
            reconstructed_image = cv2.pyrUp(reconstructed_image, dstsize=size)
            h, w, c = reconstructed_image.shape
            gaussian_noise = np.random.normal(0, _var_list[i - 1], (h, w, c))
            reconstructed_image = cv2.add(reconstructed_image, laplacian_pyramid[i])

            reconstructed_image = np.clip(reconstructed_image + gaussian_noise, 0.0, 255.0).astype(np.uint8)
        return reconstructed_image

    def _op_fold_raw(self, x):
        shape = x.shape[:2]
        raw = np.zeros((4, shape[0] // 2, shape[1] // 2), dtype=np.uint16)
        raw[0] = x[::2, ::2]
        raw[1] = x[1::2, ::2]
        raw[2] = x[::2, 1::2]
        raw[3] = x[1::2, 1::2]

        return raw

    def _op_random_awb(self):
        self.awb_r, self.awb_b = 0.5 * random.random() + 1.9, 0.4 * random.random() + 1.5

    def _op_resample(self, x):
        """x is uint8"""
        raw = np.zeros(x.shape[:2], dtype=np.float32)

        x = np.power(x / 255.0, 2.2)  # Inverse gamma

        # Resampling
        raw[::2, ::2] = x[::2, ::2, 0]  # R
        raw[1::2, ::2] = x[1::2, ::2, 1]  # G
        raw[::2, 1::2] = x[::2, 1::2, 1]  # G
        raw[1::2, 1::2] = x[1::2, 1::2, 2]  # B

        # Inverse AWB, randomly choosing AWB parameters
        #   from red [1.9, 2.4], blur [1.5, 1.9]
        raw[::2, ::2] /= self.awb_r  # awb_r
        raw[1::2, 1::2] /= self.awb_b  # awb_b
        raw = np.clip((raw * 1023), 0, 1023).astype(np.uint16)

        return raw

    def _op_noise_raw(self, x, noise_model="MI_NOTE10_NOISE"):
        """
            input x: RGB, uint8
        """
        if noise_model == "MI_NOTE10_NOISE":
            n_p_collect = MI_NOTE10_NOISE
        elif noise_model == "VIVO_NOISE":
            n_p_collect = VIVO_NOISE

        iso_range = self.distortion_strength.get("iso_range", ["1600iso", "3200iso"])
        iso_top = int(iso_range[1].rstrip("iso"))
        iso_down = int(iso_range[0].rstrip("iso"))
        iso = random.choice(iso_range)
        # if random.random() > 0.8:
        # # Has 30 percent chance to directly use iso values here
        # n_p = n_p_collect[iso]
        # else:
        # Otherwise interp according to the meta function
        """newest strategy is to directly interpolate the the two isos"""
        iso_ = (iso_top - iso_down) * random.random() + iso_down
        r_meta, b_meta, g_meta = \
            n_p_collect['r'], n_p_collect['b'], n_p_collect['g']
        n_p = []
        n_p.append(noise_meta_func(iso_, r_meta['s'], r_meta['r0'], r_meta['r1']))
        n_p.append(noise_meta_func(iso_, b_meta['s'], b_meta['r0'], b_meta['r1']))
        n_p.append(noise_meta_func(iso_, g_meta['s'], g_meta['r0'], g_meta['r1']))

        raw = self._op_resample(x)

        # Assume now already subtracted black level

        # Possion, different possion for different color channel
        r = raw[::2, ::2]
        g1 = raw[1::2, ::2]  # two g is identical till this step
        g2 = raw[::2, 1::2]  # two g is identical till this step
        b = raw[1::2, 1::2]
        gamma_r, beta_r = n_p[0][0], n_p[0][1]
        gamma_g, beta_g = n_p[2][0], n_p[2][1]
        gamma_b, beta_b = n_p[1][0], n_p[1][1]

        noise_r = np.sqrt(gamma_r * r + beta_r) * np.random.normal(0, 1, r.shape)
        noise_g1 = np.sqrt(gamma_g * g1 + beta_g) * np.random.normal(0, 1, g1.shape)
        noise_g2 = np.sqrt(gamma_g * g2 + beta_g) * np.random.normal(0, 1, g2.shape)
        noise_b = np.sqrt(gamma_b * b + beta_b) * np.random.normal(0, 1, b.shape)

        raw = raw.astype(np.float64)
        raw[::2, ::2] += noise_r  # R
        raw[1::2, ::2] += noise_g1  # G
        raw[::2, 1::2] += noise_g2  # G
        raw[1::2, 1::2] += noise_b  # B

        if self.raw_input_flag:
            raw = np.clip(raw, 0, 1023).astype(np.uint16)
            return raw
        # AWB
        raw[::2, ::2] *= self.awb_r  # awb_r
        raw[1::2, 1::2] *= self.awb_b  # awb_b
        raw = np.clip(raw, 0, 1023).astype(np.uint16)


        demosaicked_rgb = colour_demosaicing.demosaicing_CFA_Bayer_Menon2007(raw, 'RGGB')
        demosaicked_rgb = np.clip(demosaicked_rgb / 1023, 0, 1)
        x = np.power(demosaicked_rgb, 1 / 2.2)
        x = (x * 255.0).astype(np.uint8)

        return x

    def _op_median_denoise(self, x, radius=5):
        x = x[..., ::-1]
        yuv = rgb2yuv(x)
        noisy_y, u, v = cv2.split(yuv)
        noisy_y = img_as_ubyte(noisy_y)
        # x = img_as_ubyte(x)
        denoised_y = median(noisy_y, disk(radius))
        thres_idx = np.abs(denoised_y - noisy_y) > 255
        denoised_y[thres_idx] = noisy_y[thres_idx]
        denoised_y = img_as_float(denoised_y)
    
        yuv =cv2.merge((denoised_y, u, v))
        rgb = yuv2rgb(yuv)
        x = (np.clip(rgb[..., ::-1], 0, 1) * 255.).astype(np.uint8)

        return x

    def _op_nlm_denoise(self, img, h=None, tw=None, sw=None):
        # image_ycbcr = cv2.cvtColor(img, cv2.COLOR_RGB2YCR_CB)
        # y_channel = image_ycbcr[:, :, 0]

        # Op non local mean parameters
        h_ = random.choice([5, 7, 9]) if h is None else h
        sw_ = random.choice([19, 21, 23]) if tw is None else tw
        tw_ = random.choice([5, 7, 9]) if sw is None else sw

        denoised = cv2.fastNlMeansDenoisingColored(img, h=h_, hColor=h_,
                                             templateWindowSize=tw_, 
                                             searchWindowSize=sw_)
        # image_ycbcr[:, :, 0] = y_channel
        # noisy = image_ycbcr
        # noisy = cv2.cvtColor(noisy, cv2.COLOR_YCR_CB2RGB)
        # noisy = np.clip(noisy, 0, 255).astype(np.uint8)

        return denoised

    def _op_noise_and_sharpen(self, img, sharpen_flag=False):
        """
            Standard NLM hyper-parameter
            Adding noise to RAW
        """

        img = self._op_noise_raw(img)  # Adding noise to raw
        if self.raw_input_flag:
            """Directly return raw data"""
            return img

        # Y Channel Denoising
        if random.random() > 0.2:
            r_ = random.random()
            if 0.8 < r_:
                image_ycbcr = cv2.cvtColor(img, cv2.COLOR_RGB2YCR_CB)
                y_channel = image_ycbcr[:, :, 0]

                # Op non local mean parameters
                h_ = random.choice([5, 7, 9])
                sw_ = random.choice([19, 21, 23])
                tw_ = random.choice([5, 7, 9])

                y_channel = cv2.fastNlMeansDenoising(y_channel, h=h_, 
                        templateWindowSize=tw_, searchWindowSize=sw_)
                image_ycbcr[:, :, 0] = y_channel
                noisy = image_ycbcr
                noisy = cv2.cvtColor(noisy, cv2.COLOR_YCR_CB2RGB)
                noisy = np.clip(noisy, 0, 255).astype(np.uint8)
            elif 0.6 < r_ <= 0.8:
                noisy = self._op_hnr_denoise(img, method="lap_pyr_rec")
            elif 0.4 < r_ <= 0.6:
                noisy = self._op_hnr_denoise(img, method="median")
            elif 0.2 < r_ <= 0.4:
                noisy = self._op_hnr_denoise(img, method="freq")
            else:
                noisy = self._op_hnr_denoise(img, method="bilateral")

            # Sharpen the image after denoising
            if random.random() > 0.5 and sharpen_flag:
                noisy = apply_sharpen(noisy)
        else:
            noisy = img

        return noisy

    def _op_hnr_denoise(self, x, method="lap_pyr_rec"):
        x = hybrid_denoise(x, nr_method=method, blend_alpha=0.8)
        return x

    def _op_gaussian_noise(self, img, strength=2):
        """The 3 channel color noise"""

        left = self.distortion_strength.Noise[0]
        right = self.distortion_strength.Noise[1] * strength * self.ratio
        right = np.ceil(right)

        row, col, ch = img.shape
        mean = 0
        scale = random.randint(2, 5)
        var = random.randint(left, right)

        _function_name = sys._getframe().f_code.co_name
        self.distortion_record[_function_name] = float(var - left) / float(right - left)
        if self.only_probability:
            return img
        if var == 0:
            return img

        pro = random.randint(0, 1)
        if pro == 0:
            laplacian_pyramid = self.lap_pyr(img)
            noisy = self.lap_pyr_rec(laplacian_pyramid, var=var)
        else:
            gauss = np.random.normal(mean, var, (row, col, ch))
            # gauss = gauss.reshape(int(row / scale), int(col / scale), ch)
            # gauss = cv2.resize(gauss, (col, row), interpolation=cv2.INTER_LINEAR)

            # Addition
            noisy = img + gauss
            noisy[noisy < 0] = 0
            noisy[noisy > 255] = 255
            noisy = noisy.astype(np.uint8)

        # if hasattr(self.distortion_strength, 'non_local_mean') and self.distortion_strength.non_local_mean:
        #     pro = random.randint(0,1)
        #     if pro == 0:
        #         noisy = self._op_non_local_mean(noisy)
        if hasattr(self.distortion_strength, 'freq_denoise') and self.distortion_strength.freq_denoise:
            pro = random.randint(0, 1)
            if pro == 0:
                noisy = self._op_frequency_denoise(noisy, var)

        return noisy

    def _op_defocus(self, img, TYPE='uniform', radius=15):
        """ Defocus _opteration"""
        _function_name = sys._getframe().f_code.co_name
        left = self.distortion_strength.Defocus[0]
        right = self.distortion_strength.Defocus[1] * self.ratio
        right = np.ceil(right)

        defocusKernelDims = [i for i in range(left, int(right), 2)]
        TYPES = ['uniform', 'circle']
        TYPE = TYPES[np.random.randint(0, 2)]

        # (3, 9)
        radius = np.random.randint(np.ceil(left / 2), np.ceil(right / 2))
        self.distortion_record[_function_name] = (2 * radius + 1) / self.Max_Blurness
        if self.only_probability:
            return img
        sizeX, sizeY, channel_num = img.shape
        x, y = np.mgrid[-(radius + 1):(radius + 1), -(radius + 1):(radius + 1)]
        # construct uniform disk kernel
        if TYPE == 'uniform':
            disk = (np.sqrt(x ** 2 + y ** 2) < radius).astype(float)
            disk /= disk.sum()
        elif TYPE == 'circle':
            # circle disk kernel
            kernelidx = np.random.randint(0, len(defocusKernelDims))
            kerneldim = defocusKernelDims[kernelidx]
            disk = DiskKernel(dim=kerneldim)
        else:
            raise NotImplementedError
        # gama transfer
        img = self._gama_trans(img, gama=2.2)
        smoothed = cv2.filter2D(img, -1, disk)
        smoothed = self._gama_trans(smoothed, gama=1 / 2.2)
        return smoothed.astype(np.uint8)

    def _op_motion(self, img):
        """ Motion blur: all random variables generated inside the function"""
        _function_name = sys._getframe().f_code.co_name

        left = self.distortion_strength.Motion[0]
        right = np.ceil(self.distortion_strength.Motion[1] * self.ratio)
        # image = np.array(img)
        image = self._gama_trans(img, gama=2.2)
        degree = random.randint(left, int(right))
        self.distortion_record[_function_name] = degree / self.Max_Blurness
        if self.only_probability:
            return img

        angle = random.randint(0, 360)

        M = cv2.getRotationMatrix2D((degree / 2, degree / 2), angle, 1)

        motion_blur_kernel = np.diag(np.ones(degree))
        motion_blur_kernel = cv2.warpAffine(motion_blur_kernel, M, (degree, degree))

        # if motion_blur_kernel[0][0] == 0.0:
        # return img

        motion_blur_kernel = motion_blur_kernel / degree
        blurred = cv2.filter2D(image, -1, motion_blur_kernel)
        # convert to uint8
        cv2.normalize(blurred, blurred, 0, 255, cv2.NORM_MINMAX)
        blurred = np.array(blurred, dtype=np.uint8)
        # gama
        img = self._gama_trans(blurred, gama=1 / 2.2)
        return img

    def _op_blur_motion(self, img):
        img = self._gama_trans(img, 2.2)
        # load kernel
        blur_kernel = scipy.io.loadmat(random.choice(self.kernel_files))["kernel"]
        w, h = blur_kernel.shape

        ratio_ = random.choice(self.kernel_changes) * self.ratio
        new_shape = (int(w * ratio_), int(h * ratio_))
        blur_kernel = cv2.resize(blur_kernel, new_shape, interpolation=cv2.INTER_LINEAR)

        if random.random() > 0.5:
            blur_kernel = np.flip(blur_kernel, axis=0)
        if random.random() > 0.5:
            blur_kernel = np.flip(blur_kernel, axis=1)
        if random.random() > 0.5:
            blur_kernel = rotate(blur_kernel, random.choice(self.kernel_angles))

        blur_kernel /= blur_kernel.sum()
        img = cv2.filter2D(img, -1, blur_kernel)
        img = self._gama_trans(img, 1 / 2.2)
        return img.astype(np.uint8)

    def _op_ychannel(self, img, strength=2):

        image_ycbcr = cv2.cvtColor(img, cv2.COLOR_RGB2YCR_CB)
        left = self.distortion_strength.Noise[0]
        right = self.distortion_strength.Noise[1] * strength
        row, col, ch = image_ycbcr.shape

        mean = 0
        var = random.randint(left, right)
        sigma = var
        scale = random.randint(1, 3)
        norm_var = float(var - left) / float(right - left)
        if var == 0:
            return img

        gauss = np.random.normal(mean, sigma, (int(row / scale), int(col / scale), 1))
        gauss = gauss.reshape(int(row / scale), int(col / scale), 1)
        gauss = cv2.resize(gauss, (col, row), interpolation=cv2.INTER_LINEAR)

        noisy_y = image_ycbcr[:, :, 0] + gauss
        noisy_y = np.clip(noisy_y, 0, 255)
        image_ycbcr[:, :, 0] = noisy_y
        noisy = image_ycbcr

        noisy = cv2.cvtColor(noisy, cv2.COLOR_YCR_CB2RGB)

        noisy = np.clip(noisy, 0, 255)

        if hasattr(self.distortion_strength, 'non_local_mean') and self.distortion_strength.non_local_mean:
            pro = random.randint(0, 1)
            if pro == 0:
                noisy = self._op_non_local_mean(noisy, int(var * 5))

        return noisy.astype(np.uint8)

    def _op_jpeg(self, img):
        # 4/9 without jpeg compression
        _function_name = sys._getframe().f_code.co_name
        jpegq = random.randint(0, 45)

        if jpegq < 25 or jpegq > 45:
            self.distortion_record[_function_name] = 0.0
            return img

        self.distortion_record[_function_name] = 1.0 - (float(jpegq - 0) / float(100))
        if self.only_probability:
            return img

        encode_param = [int(cv2.IMWRITE_JPEG_QUALITY), jpegq]
        result, encimg = cv2.imencode('.jpg', img, encode_param)
        return cv2.imdecode(encimg, 1)

    def _op_beautyfy(self, img):
        # beatuify enhance:  r = 2-7,  eps=0.1^2-0.4^2, s=1, 2
        # 4/9 without jpeg compression
        # chance = random.randint(0, 1)
        # 0.05-0.4
        beauty_range = self.distortion_strength.Beautify
        _function_name = sys._getframe().f_code.co_name

        eps = (random.random() * (beauty_range[1] - beauty_range[0]) + beauty_range[0]) * 0.1
        self.distortion_record[_function_name] = float(eps * 10.0 - beauty_range[0]) / float(
            beauty_range[1] - beauty_range[0])
        if self.only_probability:
            return img
        r = random.randint(1, 4)
        # s = random.randint(1, 2)
        if random.random() > 0.3:
            return img
        img = img / 255.0
        return (cv2.ximgproc.guidedFilter(img.astype(np.float32),
                                          img.astype(np.float32), r, eps ** 2, -1) * 255.0).astype(np.uint8)

    @staticmethod
    def _op_non_local_mean(image, hstrength=None):

        # non local mean denoise after gaussian-noise
        # strength only limit [1~???]
        h = random.randint(5, 10)
        noisy_y = image[:, :, 0]
        y_denoised = cv2.fastNlMeansDenoising(noisy_y, h=h,
                                              templateWindowSize=7, searchWindowSize=21)
        image[:, :, 0] = y_denoised
        # image = hybrid_denoise(image, nr_method="nlm",
        # nr_parm=h, no_blend=True,
        # mask_level=-2,
        # blend_morph_kernel=1)

        return image

    @staticmethod
    def _op_frequency_denoise(image, var):

        _sigma = var * 0.01
        image = denoise_wavelet(
            image[..., ::-1] / 255.0, sigma=_sigma, method='VisuShrink', multichannel=True, convert2ycbcr=True)

        return (image[..., ::-1] * 255).astype(np.uint8)

    def get_real_strength(self):
        return self.distortion_record

    def reset_real_strength(self):
        del self.distortion_record
        self.distortion_record = {}


if __name__ == '__main__':
    """
        Temp test script for the hazing
    """
    import os
    import sys
    import glob
    from easydict import EasyDict as edict

    sys.path.insert(0, os.path.join(os.getcwd(), "HybridDenoise"))
    from HybridDenoise.hybrid_denoise import hybrid_denoise


    distortion = edict() 
    distortion.strength = edict()
    distortion.strength.iso_range = ["6400iso", "8600iso"]
    distort = Distortion_v2(distortion)

    img_path = "/data/Datasets/Face_Restoration/Newly_collected/Processed/high_res_longterm_from_0622/035412_crop_1792.jpg"
    # img_paths = glob.glob(os.path.join(root_path, "*.jpg"))

    radius = random.choice([3, 4])
    sr = 0.5 + random.random() * 1.5
    name = os.path.basename(img_path)

    img = cv2.imread(img_path)

    # x = distort._op_defocus(img)
    distort._op_random_awb()
    x = distort._op_noise_raw(img)
    cv2.imwrite("./test_results/noisy.png", x)
    # x = distort._op_median_denoise(x, radius=radius)
    x = hybrid_denoise(x, nr_method="bilateral", blend_alpha=0.8)
    x = distort._op_sharpening(x, sr=sr).astype(np.uint8)

    cv2.imwrite(os.path.join("./test_results/", "denoised_results.png"), x)
