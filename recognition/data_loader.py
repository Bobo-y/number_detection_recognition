import cv2
import numpy as np
import linecache
import os
import cfg


train_imgs = open(os.path.join(cfg.ocr_dataset_path, "annotation_train.txt"), 'r').readlines()
train_imgs_num = len(train_imgs)

val_imgs = open(os.path.join(cfg.ocr_dataset_path, "annotation_val.txt"), 'r').readlines()
val_imgs_num = len(val_imgs)

lexicon_dic_path = os.path.join(cfg.ocr_dataset_path, "lexicon.txt")


def img_gen_lexicon(batch_size=50, input_shape=None):
    imgs = np.zeros((batch_size, cfg.width, cfg.height, 3), dtype=np.uint8)
    labels = np.zeros((batch_size, cfg.label_len), dtype=np.uint8)

    while True:
        for i in range(batch_size):
            while True:
                pick_index = np.random.randint(0, train_imgs_num - 1)
                train_imgs_split = [m for m in train_imgs[pick_index].split()]
                lexicon = linecache.getline(lexicon_dic_path, int(train_imgs_split[1]) + 1).strip("\n")
                img_path = cfg.ocr_dataset_path + train_imgs_split[0][1:]
                img = cv2.imread(img_path)
                if (img is not None) and len(lexicon) <= cfg.label_len:
                    img_size = img.shape  # (height, width, channels)
                    if img_size[1] > 2 and img_size[0] > 2:
                        break
            if (img_size[1]/(img_size[0]*1.0)) < 6.4:
                img_reshape = cv2.resize(img, (int(31.0/img_size[0]*img_size[1]), cfg.height))
                mat_ori = np.zeros((cfg.height, cfg.width - int(31.0/img_size[0]*img_size[1]), 3), dtype=np.uint8)
                out_img = np.concatenate([img_reshape, mat_ori], axis=1).transpose([1, 0, 2])
            else:
                out_img = cv2.resize(img, (cfg.width, cfg.height), interpolation=cv2.INTER_CUBIC)
                out_img = np.asarray(out_img).transpose([1, 0, 2])

            # due to the explanation of ctc_loss, try to not add "-" for blank
            while len(lexicon) < cfg.label_len:
                lexicon += "-"

            imgs[i] = out_img
            labels[i] = [cfg.characters.find(c) for c in lexicon]
        yield [imgs, labels, np.ones(batch_size) * int(input_shape[1] - 2), np.ones(batch_size) * cfg.label_len], labels


def img_gen_val_lexicon(batch_size=1000):
    imgs = np.zeros((batch_size, cfg.width, cfg.height, 3), dtype=np.uint8)
    labels = []

    while True:
        for i in range(batch_size):

            while True:
                pick_index = np.random.randint(0, val_imgs_num - 1)
                train_imgs_split = [m for m in val_imgs[pick_index].split()]
                lexicon = linecache.getline(lexicon_dic_path, int(train_imgs_split[1]) + 1).strip("\n")
                img_path = cfg.ocr_dataset_path + train_imgs_split[0][1:]
                img = cv2.imread(img_path)

                if (img is not None) and len(lexicon) <= cfg.label_len:
                    img_size = img.shape  # (height, width, channels)
                    if img_size[1] > 2 and img_size[0] > 2:
                        break
            if (img_size[1]/(img_size[0]*1.0)) < 6.4:
                img_reshape = cv2.resize(img, (int(31.0/img_size[0]*img_size[1]), cfg.height))
                mat_ori = np.zeros((cfg.height, cfg.width - int(31.0/img_size[0]*img_size[1]), 3), dtype=np.uint8)
                out_img = np.concatenate([img_reshape, mat_ori], axis=1).transpose([1, 0, 2])
            else:
                out_img = cv2.resize(img, (cfg.width, cfg.height), interpolation=cv2.INTER_CUBIC)
                out_img = np.asarray(out_img).transpose([1, 0, 2])

            imgs[i] = out_img
            labels.append(lexicon)
        yield imgs, labels

