import numpy as np
import cfg as cfg
import cv2


def sigmoid(x):
    """`y = 1 / (1 + exp(-x))`"""
    return 1 / (1 + np.exp(-x))


def resize_image(im, max_img_size=cfg.image_size):
    im_width = np.minimum(im.width, max_img_size)
    if im_width == max_img_size < im.width:
        im_height = int((im_width / im.width) * im.height)
    else:
        im_height = im.height
    o_height = np.minimum(im_height, max_img_size)
    if o_height == max_img_size < im_height:
        o_width = int((o_height / im_height) * im_width)
    else:
        o_width = im_width
    d_wight = o_width - (o_width % 32)
    d_height = o_height - (o_height % 32)
    return d_wight, d_height


def should_merge(region, i, j):
    neighbor = {(i, j - 1)}
    return not region.isdisjoint(neighbor)


def region_neighbor(region_set):
    region_pixels = np.array(list(region_set))
    j_min = np.amin(region_pixels, axis=0)[1] - 1
    j_max = np.amax(region_pixels, axis=0)[1] + 1
    i_m = np.amin(region_pixels, axis=0)[0] + 1
    region_pixels[:, 0] += 1
    neighbor = {(region_pixels[n, 0], region_pixels[n, 1]) for n in
                range(len(region_pixels))}
    neighbor.add((i_m, j_min))
    neighbor.add((i_m, j_max))
    return neighbor


def region_group(region_list):
    S = [i for i in range(len(region_list))]
    D = []
    while len(S) > 0:
        m = S.pop(0)
        if len(S) == 0:
            # S has only one element, put it to D
            D.append([m])
        else:
            D.append(rec_region_merge(region_list, m, S))
    return D


def rec_region_merge(region_list, m, S):
    rows = [m]
    tmp = []
    for n in S:
        if not region_neighbor(region_list[m]).isdisjoint(region_list[n]) or \
                not region_neighbor(region_list[n]).isdisjoint(region_list[m]):
            # 第m与n相交
            tmp.append(n)
    for d in tmp:
        S.remove(d)
    for e in tmp:
        rows.extend(rec_region_merge(region_list, e, S))
    return rows


def nms(predict, activation_pixels, threshold=cfg.side_vertex_pixel_threshold):
    region_list = []
    for i, j in zip(activation_pixels[0], activation_pixels[1]):
        merge = False
        for k in range(len(region_list)):
            if should_merge(region_list[k], i, j):
                region_list[k].add((i, j))
                merge = True
        if not merge:
            region_list.append({(i, j)})
    D = region_group(region_list)
    quad_list = np.zeros((len(D), 4, 2))
    score_list = np.zeros((len(D), 4))
    for group, g_th in zip(D, range(len(D))):
        total_score = np.zeros((4, 2))
        for row in group:
            for ij in region_list[row]:
                score = predict[ij[0], ij[1], 1]
                if score >= threshold:
                    ith_score = predict[ij[0], ij[1], 2:3]
                    if not (cfg.trunc_threshold <= ith_score < 1 -
                            cfg.trunc_threshold):
                        ith = int(np.around(ith_score))
                        total_score[ith * 2:(ith + 1) * 2] += score
                        px = (ij[1] + 0.5) * cfg.pixel_size
                        py = (ij[0] + 0.5) * cfg.pixel_size
                        p_v = [px, py] + np.reshape(predict[ij[0], ij[1], 3:7],
                                              (2, 2))
                        quad_list[g_th, ith * 2:(ith + 1) * 2] += score * p_v
        score_list[g_th] = total_score[:, 0]
        quad_list[g_th] /= (total_score + cfg.epsilon)
    return score_list, quad_list


def crop_rectangle(img, geo):
    rect = cv2.minAreaRect(geo.astype(int))
    center, size, angle = rect[0], rect[1], rect[2]
    if(angle > -45):
        center = tuple(map(int, center))
        size = tuple([int(rect[1][0] + 10), int(rect[1][1] + 10)])
        height, width = img.shape[0], img.shape[1]
        M = cv2.getRotationMatrix2D(center, angle, 1)
        img_rot = cv2.warpAffine(img, M, (width, height))
        img_crop = cv2.getRectSubPix(img_rot, size, center)
    else:
        center = tuple(map(int, center))
        size = tuple([int(rect[1][1] + 10), int(rect[1][0]) + 10])
        angle -= 270
        height, width = img.shape[0], img.shape[1]
        M = cv2.getRotationMatrix2D(center, angle, 1)
        img_rot = cv2.warpAffine(img, M, (width, height))
        img_crop = cv2.getRectSubPix(img_rot, size, center)
    return img_crop