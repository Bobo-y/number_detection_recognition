epoch_num = 24
lr = 1e-3
decay = 5e-4
patience = 5
load_weights = False
lambda_inside_score_loss = 4.0
lambda_side_vertex_code_loss = 1.0
lambda_side_vertex_coord_loss = 1.0

total_img = 10000
validation_split_ratio = 0.1
image_size = 512  # (height == width, in [256, 384, 512, 640, 736])
batch_size = 4
steps_per_epoch = total_img * (1 - validation_split_ratio) // batch_size
validation_steps = total_img * validation_split_ratio // batch_size

data_dir = 'icpr/'
origin_image_dir_name = 'image_10000/'
origin_txt_dir_name = 'txt_10000/'
train_image_dir_name = 'images_%s/' % image_size
train_label_dir_name = 'labels_%s/' % image_size
show_gt_image_dir_name = 'show_gt_images_%s/' % image_size
show_act_image_dir_name = 'show_act_images_%s/' % image_size
gen_origin_img = True
draw_gt_quad = True
draw_act_quad = True
val_fname = 'val_%s.txt' % image_size
train_fname = 'train_%s.txt' % image_size
# in paper it's 0.3, maybe to large to this problem
shrink_ratio = 0.2
# pixels between 0.2 and 0.6 are side pixels
shrink_side_ratio = 0.6
epsilon = 1e-4

num_channels = 3
feature_layers_range = range(5, 1, -1)
# feature_layers_range = range(3, 0, -1)
feature_layers_num = len(feature_layers_range)
# pixel_size = 4
pixel_size = 2 ** feature_layers_range[-1]

model_weights_path = 'saved_model/weights_%s.h5' % image_size
saved_model_file_path = 'saved_model/model_%s.h5' % image_size
saved_model_weights_file_path = 'model/weights_base.h5'

pixel_threshold = 0.9
side_vertex_pixel_threshold = 0.9
trunc_threshold = 0.1
predict_write2txt = False
detection_box_crop = True
