# -*- coding: utf-8 -*-
# @Time    : 2017/9/2 13:40
# @Author  : 郑梓斌

import cv2
import numpy as np
import time
import os

import core


def transformation_points(src_img, src_points, dst_img, dst_points):
    src_points = src_points.astype(np.float64)
    dst_points = dst_points.astype(np.float64)

    c1 = np.mean(src_points, axis=0)
    c2 = np.mean(dst_points, axis=0)

    src_points -= c1
    dst_points -= c2

    s1 = np.std(src_points)
    s2 = np.std(dst_points)

    src_points /= s1
    dst_points /= s2

    u, s, vt = np.linalg.svd(src_points.T * dst_points)
    r = (u * vt).T

    m = np.vstack([np.hstack(((s2 / s1) * r, c2.T - (s2 / s1) * r * c1.T)), np.matrix([0., 0., 1.])])

    output = cv2.warpAffine(dst_img, m[:2],
                            (src_img.shape[1], src_img.shape[0]),
                            borderMode=cv2.BORDER_TRANSPARENT,
                            flags=cv2.WARP_INVERSE_MAP)

    return output


def tran_matrix(src_img, src_points, dst_img, dst_points):
    h = cv2.findHomography(dst_points, src_points)
    output = cv2.warpAffine(dst_img, h[0][:2], (src_img.shape[1], src_img.shape[0]),
                            borderMode=cv2.BORDER_TRANSPARENT,
                            flags=cv2.WARP_INVERSE_MAP)

    return output


def correct_color(img1, img2, landmark):
    blur_amount = 0.4 * np.linalg.norm(
        np.mean(landmark[core.LEFT_EYE_POINTS], axis=0)
        - np.mean(landmark[core.RIGHT_EYE_POINTS], axis=0)
    )
    blur_amount = int(blur_amount)

    if blur_amount % 2 == 0:
        blur_amount += 1

    img1_blur = cv2.GaussianBlur(img1, (blur_amount, blur_amount), 0)
    img2_blur = cv2.GaussianBlur(img2, (blur_amount, blur_amount), 0)

    img2_blur += (128 * (img2_blur <= 1.0)).astype(img2_blur.dtype)

    return img2.astype(np.float64) * img1_blur.astype(np.float64) / img2_blur.astype(np.float64)


def tran_src(src_img, src_points, dst_points, face_area=None):
    jaw = core.JAW_END

    dst_list = dst_points \
               + core.matrix_rectangle(face_area[0], face_area[1], face_area[2], face_area[3]) \
               + core.matrix_rectangle(0, 0, src_img.shape[1], src_img.shape[0])

    src_list = src_points \
               + core.matrix_rectangle(face_area[0], face_area[1], face_area[2], face_area[3]) \
               + core.matrix_rectangle(0, 0, src_img.shape[1], src_img.shape[0])

    jaw_points = []

    for i in range(0, jaw):
        jaw_points.append(dst_list[i])
        jaw_points.append(src_list[i])

    warp_jaw = cv2.convexHull(np.array(jaw_points), returnPoints=False)
    warp_jaw = warp_jaw.tolist()

    for i in range(0, len(warp_jaw)):
        warp_jaw[i] = warp_jaw[i][0]

    warp_jaw.sort()

    if len(warp_jaw) <= jaw:
        dst_list = dst_list[jaw - len(warp_jaw):]
        src_list = src_list[jaw - len(warp_jaw):]
        for i in range(0, len(warp_jaw)):
            dst_list[i] = jaw_points[int(warp_jaw[i])]
            src_list[i] = jaw_points[int(warp_jaw[i])]
    else:
        for i in range(0, jaw):
            if len(warp_jaw) > jaw and warp_jaw[i] == 2 * i and warp_jaw[i + 1] == 2 * i + 1:
                warp_jaw.remove(2 * i)

            dst_list[i] = jaw_points[int(warp_jaw[i])]

    dt = core.measure_triangle(src_img, dst_list)

    res_img = np.zeros(src_img.shape, dtype=src_img.dtype)

    for i in range(0, len(dt)):
        t_src = []
        t_dst = []

        for j in range(0, 3):
            t_src.append(src_list[dt[i][j]])
            t_dst.append(dst_list[dt[i][j]])

        core.affine_triangle(src_img, res_img, t_src, t_dst)

    return res_img


def merge_img(src_img, dst_img, dst_matrix, dst_points, blur_detail_x=None, blur_detail_y=None, mat_multiple=None):
    face_mask = np.zeros(src_img.shape, dtype=src_img.dtype)

    for group in core.OVERLAY_POINTS:
        cv2.fillConvexPoly(face_mask, cv2.convexHull(dst_matrix[group]), (255, 255, 255))

    r = cv2.boundingRect(np.float32([dst_points[:core.FACE_END]]))

    center = (r[0] + int(r[2] / 2), r[1] + int(r[3] / 2))

    if mat_multiple:
        mat = cv2.getRotationMatrix2D(center, 0, mat_multiple)
        face_mask = cv2.warpAffine(face_mask, mat, (face_mask.shape[1], face_mask.shape[0]))

    if blur_detail_x and blur_detail_y:
        face_mask = cv2.blur(face_mask, (blur_detail_x, blur_detail_y), center)

    return cv2.seamlessClone(np.uint8(dst_img), src_img, face_mask, center, cv2.NORMAL_CLONE)


def morph_img(src_img, src_points, dst_img, dst_points, alpha=0.5):
    morph_points = []

    src_img = src_img.astype(np.float32)
    dst_img = dst_img.astype(np.float32)

    res_img = np.zeros(src_img.shape, src_img.dtype)

    for i in range(0, len(src_points)):
        x = (1 - alpha) * src_points[i][0] + alpha * dst_points[i][0]
        y = (1 - alpha) * src_points[i][1] + alpha * dst_points[i][1]
        morph_points.append((x, y))

    dt = core.measure_triangle(src_img, morph_points)

    for i in range(0, len(dt)):
        t1 = []
        t2 = []
        t = []

        for j in range(0, 3):
            t1.append(src_points[dt[i][j]])
            t2.append(dst_points[dt[i][j]])
            t.append(morph_points[dt[i][j]])

        core.morph_triangle(src_img, dst_img, res_img, t1, t2, t, alpha)

    return res_img


def face_merge(dst_img, src_img, out_img,
               face_area, alpha=0.75,
               skin_buff=0, skin_detail=0, skin_p=0,
               blur_detail_x=None, blur_detail_y=None, mat_multiple=None):
    src_matrix, src_points, err = core.face_points(src_img)
    dst_matrix, dst_points, err = core.face_points(dst_img)

    src_img = cv2.imread(src_img, cv2.IMREAD_COLOR)
    dst_img = cv2.imread(dst_img, cv2.IMREAD_COLOR)

    dst_img = transformation_points(src_img, src_matrix[core.FACE_POINTS],
                                    dst_img, dst_matrix[core.FACE_POINTS])

    trans_file = 'images/' + str(int(time.time() * 1000)) + '.jpg'
    cv2.imwrite(trans_file, dst_img)
    _, dst_points, err = core.face_points(trans_file)

    dst_img = morph_img(src_img, src_points, dst_img, dst_points, alpha)

    morph_file = 'images/' + str(int(time.time() * 1000)) + '.jpg'
    cv2.imwrite(morph_file, dst_img)
    dst_matrix, dst_points, err = core.face_points(morph_file)

    src_img = tran_src(src_img, src_points, dst_points, face_area)

    dst_img = merge_img(src_img, dst_img, dst_matrix, dst_points, blur_detail_x, blur_detail_y, mat_multiple)

    os.remove(trans_file)
    os.remove(trans_file + '.txt')

    os.remove(morph_file)
    os.remove(morph_file + '.txt')

    cv2.imwrite(out_img, dst_img)

    return err
