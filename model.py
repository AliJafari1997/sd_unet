from tensorflow.keras.layers import Conv2D, BatchNormalization, Activation, MaxPooling2D, Conv2DTranspose, Concatenate, Input, AveragePooling2D, UpSampling2D
from tensorflow.keras.models import Model
import tensorflow as tf
import tensorflow_addons as tfa

def conv_block(input, num_filters):
    x = Conv2D(num_filters, 3, padding="same")(input)
    x = Activation("leaky_relu")(x)
    x = tfa.layers.GroupNormalization(groups=32,axis=-1)(x)

    x = Conv2D(num_filters, 3, padding="same")(x)
    x = Activation("leaky_relu")(x)
    x = tfa.layers.GroupNormalization(groups=32,axis=-1)(x)
    
    return x

def SqueezeAttentionBlock(input, num_filters):
    x_res = conv_block(input, num_filters)

    y = AveragePooling2D((2, 2))(input)
    y = conv_block(y, num_filters)
    y = UpSampling2D(size=(2, 2))(y)

    return (y * x_res) + y

def encoder_block(input, num_filters):
    x = SqueezeAttentionBlock(input, num_filters)
    max_pool = MaxPooling2D((2, 2))(x)
    avg_pool = AveragePooling2D((2, 2))(x)
    cat = Concatenate()([max_pool, avg_pool])
    output = Conv2D(num_filters, (1, 1), padding='same')(cat)
    return x, output


def dense_aspp(x, filter):
    shape = x.shape

    # conv1*1 -> BN -> ReLU
    conv_1 = Conv2D(filter, 1, padding="same", use_bias=False)(x)
    conv_1 = Activation("leaky_relu")(conv_1)
    conv_1 = tfa.layers.GroupNormalization(groups=32,axis=-1)(conv_1)

    # Conv3*3 with dilation_rate=6 -> BN -> ReLU
    y0 = Conv2D(filter, 3, padding="same", dilation_rate=(6, 6), use_bias=False)(x)
    y0 = Activation("leaky_relu")(y0)
    y0 = tfa.layers.GroupNormalization(groups=32,axis=-1)(y0)

    concat_x_y0 = Concatenate()([x, y0])
    y1 = Conv2D(filter, 3, padding="same", dilation_rate=(12, 12), use_bias=False)(concat_x_y0)
    y1 = Activation("leaky_relu")(y1)
    y1 = tfa.layers.GroupNormalization(groups=32,axis=-1)(y1)

    concat_x_y0_y1 = Concatenate()([x, y0, y1])
    y2 = Conv2D(filter, 3, padding="same", dilation_rate=(18, 18), use_bias=False)(concat_x_y0_y1)
    y2 = Activation("leaky_relu")(y2)
    y2 = tfa.layers.GroupNormalization(groups=32,axis=-1)(y2)

    avg_pool = AveragePooling2D((2, 2))(x)
    avg_pool = UpSampling2D((2, 2))(avg_pool)
    y = Concatenate()([conv_1, y0, y1, y2, avg_pool])
    y = Conv2D(filter, 1, dilation_rate=1, padding="same", use_bias=False)(y)
    y = Activation("leaky_relu")(y)
    y = tfa.layers.GroupNormalization(groups=32,axis=-1)(y)

    return y

def decoder_block(input, skip_features, num_filters):
    x = Conv2DTranspose(num_filters, (2, 2), strides=2, padding="same")(input)
    x = Concatenate()([x, skip_features])
    x = conv_block(x, num_filters)
    return x


def build_model(input_shape):
    inputs = Input(input_shape)

    skip1, out1 = encoder_block(inputs, 32)
    skip2, out2 = encoder_block(out1, 64)
    skip3, out3 = encoder_block(out2, 128)
    skip4, out4 = encoder_block(out3, 256)
    out5 = SqueezeAttentionBlock(out4, 512)

    aspp_output = dense_aspp(out5, 256)

    d1 = decoder_block(aspp_output, skip4, 256)   # 256+256 -> 256
    d2 = decoder_block(d1, skip3, 128)            # 128+128 -> 128
    d3 = decoder_block(d2, skip2, 64)             # 64+64 -> 64
    d4 = decoder_block(d3, skip1, 32)             # 32+32 -> 32

    outputs = Conv2D(1, (3, 3), padding="same", activation="sigmoid")(d4)

    model = Model(inputs, outputs, name="SD-UNet")
    return model

if __name__ == "__main__":
    input_shape = (512, 512, 3)
    model = build_model(input_shape)
    model.summary()             
