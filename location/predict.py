import numpy as np
from PIL import Image, ImageDraw
from keras.preprocessing import image
from keras.applications.vgg16 import preprocess_input
import cv2
import cfg
from network import East
from preprocess import resize_image
from nms import nms
import os


def sigmoid(x):
    """`y = 1 / (1 + exp(-x))`"""
    return 1 / (1 + np.exp(-x))


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


def predict(east_detect, img_path, pixel_threshold, quiet=False):
    img = image.load_img(img_path)
    d_wight, d_height = resize_image(img, cfg.image_size)
    img = img.resize((d_wight, d_height), Image.NEAREST).convert('RGB')
    img = image.img_to_array(img)
    img = preprocess_input(img, mode='tf')
    x = np.expand_dims(img, axis=0)
    y = east_detect.predict(x)
    y = np.squeeze(y, axis=0)
    y[:, :, :3] = sigmoid(y[:, :, :3])
    cond = np.greater_equal(y[:, :, 0], pixel_threshold)
    activation_pixels = np.where(cond)
    quad_scores, quad_after_nms = nms(y, activation_pixels)
    with Image.open(img_path) as im:
        im_array = image.img_to_array(im.convert('RGB'))
        d_wight, d_height = resize_image(im, cfg.image_size)
        scale_ratio_w = d_wight / im.width
        scale_ratio_h = d_height / im.height
        im = im.resize((d_wight, d_height), Image.NEAREST).convert('RGB')
        quad_im = im.copy()
        quad_draw = ImageDraw.Draw(quad_im)
        txt_items = []
        flag = False
        for score, geo, s in zip(quad_scores, quad_after_nms,
                                 range(len(quad_scores))):
            if np.amin(score) > 0:
                flag = True
                quad_draw.line([tuple(geo[0]),
                                tuple(geo[1]),
                                tuple(geo[2]),
                                tuple(geo[3]),
                                tuple(geo[0])], width=2, fill='blue')
                rescaled_geo = geo / [scale_ratio_w, scale_ratio_h]
                rescaled_geo_list = np.reshape(rescaled_geo, (8,)).tolist()
                txt_item = ','.join(map(str, rescaled_geo_list))
                txt_items.append(txt_item + '\n')
                if cfg.detection_box_crop:
                    img_crop = crop_rectangle(im_array, rescaled_geo)
                    cv2.imwrite(os.path.join('output_crop', img_path.split('/')[-1].split('.')[0] + '.jpg'), img_crop)
            elif not quiet:
                print('quad invalid with vertex num less then 4.')
        if flag:
            quad_im.save(os.path.join('output', img_path.split('/')[-1].split('.')[0] + '_predict.jpg'))
        if cfg.predict_write2txt and len(txt_items) > 0:
            with open(os.path.join("output_txt", img_path.split('/')[-1].split('.')[0] + '.txt'), 'w') as f_txt:
                f_txt.writelines(txt_items)


if __name__ == '__main__':
    east = East()
    east_detect = east.east_network()
    east_detect.summary()
    east_detect.load_weights(cfg.model_weights_path)
    img_list = os.listdir('test_imgs')
    for img_path in img_list:
        predict(east_detect, os.path.join('test_imgs', img_path), cfg.pixel_threshold)
