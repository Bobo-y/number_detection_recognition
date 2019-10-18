from keras import backend as ktf
from keras.layers import Conv2D, LSTM, Lambda, BatchNormalization, MaxPooling2D, Reshape, Dense, Dropout, add, concatenate, Bidirectional
from keras.models import Model, Input
from keras.optimizers import SGD


class CRNN:
    def __init__(self, width, height, label_len, characters):
        self.height = height
        self.width = width
        self.label_len = label_len
        self.characters = characters
        self.label_classes = len(self.characters)

    def ctc_loss(self, args):
        iy_pred, ilabels, iinput_length, ilabel_length = args
        # the 2 is critical here since the first couple outputs of the RNN
        # tend to be garbage:
        iy_pred = iy_pred[:, 2:, :]  # no such influence
        return ktf.ctc_batch_cost(ilabels, iy_pred, iinput_length, ilabel_length)

    def network(self):
        input_im = Input(shape=(self.width, self.height, 3))

        conv_1 = Conv2D(64, (3, 3), activation='relu', padding='same')(input_im)
        bn1 = BatchNormalization()(conv_1)

        conv_2_1 = Conv2D(128, (3, 3), activation='relu', padding='same')(bn1)
        conv_2_2 = Conv2D(256, (3, 3), activation='relu', padding='same')(conv_2_1)
        bn2 = BatchNormalization()(conv_2_2)
        pool_1 = MaxPooling2D(pool_size=(2, 2))(bn2)

        conv_3_1 = Conv2D(256, (3, 3), activation='relu', padding='same')(pool_1)
        conv_3_2 = Conv2D(512, (3, 3), activation='relu', padding='same')(conv_3_1)
        bn3 = BatchNormalization()(conv_3_2)
        pool_2 = MaxPooling2D(pool_size=(2, 2))(bn3)

        conv_4_1 = Conv2D(512, (3, 3), activation='relu', padding='same')(pool_2)
        conv_4_2 = Conv2D(512, (3, 3), activation='relu', padding='same')(conv_4_1)
        bn4 = BatchNormalization()(conv_4_2)

        bn_shape = bn4.get_shape()

        x_reshape = Reshape(target_shape=(int(bn_shape[1]), int(bn_shape[2] * bn_shape[3])))(bn4)

        fc_1 = Dense(128, activation='relu')(x_reshape)

        rnn_1 = LSTM(128, kernel_initializer="he_normal", return_sequences=True)(fc_1)
        rnn_1b = LSTM(128, kernel_initializer="he_normal", go_backwards=True, return_sequences=True)(fc_1)
        rnn1_merged = add([rnn_1, rnn_1b])
        # bi_lstm1 = Bidirectional(LSTM(128, kernel_initializer="he_normal", return_sequences=True), merge_mode='sum')(fc_1)
        #
        rnn_2 = LSTM(128, kernel_initializer="he_normal", return_sequences=True)(rnn1_merged)
        rnn_2b = LSTM(128, kernel_initializer="he_normal", go_backwards=True, return_sequences=True)(rnn1_merged)
        rnn2_merged = concatenate([rnn_2, rnn_2b])
        # bi_lstm2 = Bidirectional(LSTM(128, kernel_initializer="he_normal", return_sequences=True), merge_mode='concat')(bi_lstm1)

        drop_1 = Dropout(0.25)(rnn2_merged)
        # drop_1 = Dropout(0.25)(bi_lstm2)

        fc_2 = Dense(self.label_classes, kernel_initializer='he_normal', activation='softmax')(drop_1)

        infer_model = Model(inputs=input_im, outputs=fc_2)

        labels = Input(name='the_labels', shape=[self.label_len], dtype='float32')
        input_length = Input(name='input_length', shape=[1], dtype='int64')
        label_length = Input(name='label_length', shape=[1], dtype='int64')

        loss_out = Lambda(self.ctc_loss, output_shape=(1,), name='ctc')([fc_2, labels, input_length,
                                                                        label_length])

        train_model = Model(inputs=[input_im, labels, input_length, label_length], outputs=[loss_out])
        sgd = SGD(lr=0.0001, decay=1e-6, momentum=0.9, nesterov=True, clipnorm=5)

        train_model.compile(loss={'ctc': lambda y_true, y_pred: y_pred}, optimizer=sgd)
        infer_model.summary()
        train_model.summary()

        return train_model, infer_model


if __name__ == '__main__':
    import string
    CRNN(200, 31, 11, '0123456789'+'-').network()
