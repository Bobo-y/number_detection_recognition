from keras.callbacks import ModelCheckpoint, Callback
from keras.callbacks import TensorBoard
from data_loader import *
from network import *
import keras.backend as ktf
import cfg
import os
import tensorflow as tf


os.environ['CUDA_VISIBLE_DEVICES'] = '0'
tf_config = tf.ConfigProto()
tf_config.gpu_options.per_process_gpu_memory_fraction = 0.5
session = tf.Session(config=tf_config)
ktf.set_session(session)


class Evaluate(Callback):

    def on_epoch_end(self, epoch, logs=None):

        def evaluate(input_model):
            correct_prediction = 0
            generator = img_gen_val_lexicon()
            x_test, y_test = next(generator)
            y_pred = input_model.predict(x_test)
            shape = y_pred[:, 2:, :].shape
            ctc_decode = ktf.ctc_decode(y_pred[:, 2:, :], input_length=np.ones(shape[0]) * shape[1])[0][0]
            out = ktf.get_value(ctc_decode)[:, :cfg.label_len]

            for m in range(1000):
                result_str = ''.join([cfg.characters[k] for k in out[m]])
                result_str = result_str.replace('-', '')
                if result_str == y_test[m]:
                    correct_prediction += 1
                else:
                    print(result_str, y_test[m])

            return correct_prediction * 1.0 / 10
        acc = evaluate(infer_model)
        print('')
        print('acc:'+str(acc)+"%")


evaluator = Evaluate()


checkpoint = ModelCheckpoint(cfg.checkpoint_path, monitor='loss', verbose=1, save_best_only=True, mode='min')


train_model, infer_model = CRNN(cfg.width, cfg.height, cfg.label_len, cfg.characters).network()
if cfg.load_model:
    train_model.load_weights(cfg.load_model_path, by_name=True, skip_mismatch=True)
train_model.summary()
train_model.fit_generator(img_gen_lexicon(input_shape=(None, 50, 7, 512)), steps_per_epoch=2000, epochs=50, verbose=1,
                          callbacks=[evaluator,
                                     checkpoint,
                                     TensorBoard(log_dir=cfg.log_dir)]
                          )
infer_model.save(cfg.save_model_path)

