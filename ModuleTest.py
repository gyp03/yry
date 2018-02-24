# -*- coding: utf-8 -*-
# @Time    : 2017/9/2 13:40
# @Author  : 郑梓斌

import core

if __name__ == '__main__':
    core.face_merge(src_img='images/model.jpg',
                    dst_img='images/20171030175254.jpg',
                    out_img='images/output.jpg',
                    face_area=[50, 30, 500, 485],
                    alpha=0.75,
                    blur_detail_x=15,
                    blur_detail_y=10,
                    mat_multiple=0.95)
