width = 200
height = 31
label_len = 11    # we recognition phone-number, and We think the maximum length is 11, it can be changed at will
# according to the actual needs
characters = '0123456789' + '-'  # recognition character 0 to 9, '-' for blank(ctc loss)
label_classes = len(characters)  # Number of categories requiring character recognition
ocr_dataset_path = "dataset/imgs"
save_model_path = "saved_model/weights.h5"
log_dir = "logs"
load_model = True
load_model_path = "model/weights_base.h5"
checkpoint_path = "save_model/val_model.h5"
