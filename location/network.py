# coding=utf-8
from keras import Input, Model
from keras.applications.vgg16 import VGG16
from keras.layers import Concatenate, Conv2D, UpSampling2D, BatchNormalization
import tensorflow as tf
import cfg
from keras.layers import Layer


class Att(Layer):
    def __init__(self, **kwargs):
        super(Att, self).__init__(**kwargs)

    def call(self, layer_input, **kwargs):
        x = layer_input
        attention = tf.reduce_mean(layer_input, axis=-1, keep_dims=True)
        importance_map = tf.sigmoid(attention)
        output = tf.multiply(x, importance_map)
        return output

    def compute_output_shape(self, input_shape):
        return input_shape


class East:

    def __init__(self):
        self.input_img = Input(name='input_img',
                               shape=(None, None, cfg.num_channels),
                               dtype='float32')
        vgg16 = VGG16(input_tensor=self.input_img,
                      weights=None,
                      include_top=False)
        self.vgg_pools = [vgg16.get_layer('block%d_pool' % i).output
                          for i in range(2, 6)]

    def east_network(self):
        
        def decoder(layer_input, skip_input, channel):
            concat = Concatenate(axis=-1)([layer_input, skip_input])
            bn1 = BatchNormalization()(concat)
            conv_1 = Conv2D(channel, 1,
                            activation='relu', padding='same')(bn1)
            bn2 = BatchNormalization()(conv_1)
            conv_3 = Conv2D(channel, 3,
                            activation='relu', padding='same')(bn2)
            return conv_3

        # d1 = Att()(self.vgg_pools[3])
        # d1 = decoder(UpSampling2D((2, 2))(d1), self.vgg_pools[2], 128)
        d1 = decoder(UpSampling2D((2, 2))(self.vgg_pools[3]), self.vgg_pools[2], 128)
        # d1 = Att()(d1)
        d2 = decoder(UpSampling2D((2, 2))(d1), self.vgg_pools[1], 64)
        # d2 = Att()(d2)
        d3 = decoder(UpSampling2D((2, 2))(d2), self.vgg_pools[0], 32)
        # d3 = Att()(d3)
        bn = BatchNormalization()(d3)
        before_output = Conv2D(32, 3, activation='relu', padding='same')(bn)
        inside_score = Conv2D(1, 1, padding='same', name='inside_score'
                              )(before_output)
        side_v_code = Conv2D(2, 1, padding='same', name='side_vertex_code'
                             )(before_output)
        side_v_coord = Conv2D(4, 1, padding='same', name='side_vertex_coord'
                              )(before_output)
        east_detect = Concatenate(axis=-1,
                                  name='east_detect')([inside_score,
                                                       side_v_code,
                                                       side_v_coord])
        model = Model(inputs=self.input_img, outputs=[east_detect])
        # model.summary()
        return model


if __name__ == '__main__':
    east = East()
    east_network = east.east_network()
    east_network.load_weights('saved_model/east_model_weights_origin.h5')
    east_network.summary()
