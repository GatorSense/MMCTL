"""
Minimal model definitions for MMCTL training.

This file keeps only the architecture used by train_mmctl.py:
- encoder
- image_decoder
- image_reconstruction_multi_modalv1
"""

import tensorflow as tf
from tensorflow.keras.models import Model
from tensorflow.keras.layers import (
    Input,
    Dense,
    Conv2D,
    Conv2DTranspose,
    BatchNormalization,
    MaxPooling2D,
    GlobalMaxPooling2D,
    UpSampling2D,
    ReLU,
    Reshape,
    Cropping2D,
    Activation,
    concatenate,
)

KERNEL_SIZE = 3


def encoder(input_tensor, emb_dim=1024, dilation=1):
    x = Conv2D(
        16,
        (KERNEL_SIZE, KERNEL_SIZE),
        dilation_rate=dilation,
        activation="relu",
        kernel_initializer=tf.keras.initializers.GlorotNormal(),
        padding="valid",
        use_bias=True,
    )(input_tensor)
    x = BatchNormalization()(x)

    x = Conv2D(
        16,
        (KERNEL_SIZE, KERNEL_SIZE),
        dilation_rate=dilation,
        activation="relu",
        kernel_initializer=tf.keras.initializers.GlorotNormal(),
        padding="valid",
        use_bias=True,
    )(x)
    x = BatchNormalization()(x)
    x = MaxPooling2D((3, 3))(x)

    x = Conv2D(
        32,
        (KERNEL_SIZE, KERNEL_SIZE),
        dilation_rate=dilation,
        activation="relu",
        kernel_initializer=tf.keras.initializers.GlorotNormal(),
        padding="valid",
        use_bias=True,
    )(x)
    x = BatchNormalization()(x)

    x = Conv2D(
        32,
        (KERNEL_SIZE, KERNEL_SIZE),
        dilation_rate=dilation,
        activation="relu",
        kernel_initializer=tf.keras.initializers.GlorotNormal(),
        padding="valid",
        use_bias=True,
    )(x)
    x = BatchNormalization()(x)
    x = MaxPooling2D((3, 3))(x)

    x = Conv2D(
        64,
        (KERNEL_SIZE, KERNEL_SIZE),
        dilation_rate=dilation,
        activation="relu",
        kernel_initializer=tf.keras.initializers.GlorotNormal(),
        padding="valid",
        use_bias=True,
    )(x)
    x = BatchNormalization()(x)
    x = MaxPooling2D((3, 3))(x)

    x = Conv2D(
        128,
        (KERNEL_SIZE, KERNEL_SIZE),
        dilation_rate=dilation,
        activation="relu",
        kernel_initializer=tf.keras.initializers.GlorotNormal(),
        padding="valid",
        use_bias=True,
    )(x)
    x = BatchNormalization()(x)

    x = Conv2D(
        256,
        (KERNEL_SIZE, KERNEL_SIZE),
        dilation_rate=dilation,
        activation="relu",
        kernel_initializer=tf.keras.initializers.GlorotNormal(),
        padding="valid",
        use_bias=True,
    )(x)
    x = BatchNormalization()(x)

    x = Conv2D(
        512,
        (KERNEL_SIZE, KERNEL_SIZE),
        dilation_rate=dilation,
        activation="relu",
        kernel_initializer=tf.keras.initializers.GlorotNormal(),
        padding="valid",
        use_bias=True,
    )(x)
    x = BatchNormalization()(x)

    x = GlobalMaxPooling2D()(x)
    x = Dense(emb_dim, kernel_initializer=tf.keras.initializers.GlorotNormal())(x)

    return x


def image_decoder(input_tensor):
    x = Reshape((1, 1, 1024))(input_tensor)

    x = UpSampling2D((3, 3))(x)

    filters = [256, 256, 128, 128, 64]
    upsample_factors = [3, 3, 3, 2, 2]

    for i in range(5):
        x = Conv2DTranspose(
            filters[i],
            (3, 3),
            kernel_initializer=tf.keras.initializers.GlorotNormal(),
            strides=(1, 1),
            padding="same",
        )(x)
        x = ReLU()(x)

        x = Conv2DTranspose(
            filters[i],
            (3, 3),
            kernel_initializer=tf.keras.initializers.GlorotNormal(),
            strides=(1, 1),
            padding="same",
        )(x)
        x = ReLU()(x)

        x = UpSampling2D((upsample_factors[i], upsample_factors[i]))(x)

    x = Cropping2D(((0, 24), (0, 24)))(x)

    x = Conv2D(
        32,
        (3, 3),
        kernel_initializer=tf.keras.initializers.GlorotNormal(),
        strides=(1, 1),
        padding="same",
    )(x)
    x = ReLU()(x)

    x = Conv2D(
        32,
        (3, 3),
        kernel_initializer=tf.keras.initializers.GlorotNormal(),
        strides=(1, 1),
        padding="same",
    )(x)
    x = ReLU()(x)

    x = Conv2D(
        16,
        (3, 3),
        kernel_initializer=tf.keras.initializers.GlorotNormal(),
        strides=(1, 1),
        padding="same",
    )(x)
    x = ReLU()(x)

    x = Conv2D(
        3,
        (3, 3),
        kernel_initializer=tf.keras.initializers.GlorotNormal(),
        strides=(1, 1),
        padding="same",
    )(x)

    output = Activation("sigmoid")(x)
    return output


def image_reconstruction_multi_modalv1(rgb_dim, thermal_dim, emb_dim=1024):
    rgb_input = Input(rgb_dim, name="rgb_input")
    thermal_input = Input(thermal_dim, name="thermal_input")

    rgb_emb = encoder(rgb_input, emb_dim=emb_dim)
    thermal_emb = encoder(thermal_input, emb_dim=emb_dim)

    rgb_encoder = Model(rgb_input, rgb_emb, name="rgb_encoder")
    thermal_encoder = Model(thermal_input, thermal_emb, name="thermal_encoder")

    rgb_decoder_input = Input((emb_dim,), name="rgb_decoder_input")
    thermal_decoder_input = Input((emb_dim,), name="thermal_decoder_input")

    rgb_decoder_output = image_decoder(rgb_decoder_input)
    thermal_decoder_output = image_decoder(thermal_decoder_input)

    rgb_decoder = Model(rgb_decoder_input, rgb_decoder_output, name="rgb_decoder")
    thermal_decoder = Model(
        thermal_decoder_input,
        thermal_decoder_output,
        name="thermal_decoder",
    )

    rgb_features = rgb_encoder(rgb_input)
    thermal_features = thermal_encoder(thermal_input)

    joint_embedding = concatenate(
        [rgb_features, thermal_features],
        axis=-1,
        name="joint_embedding",
    )

    model = Model(
        [rgb_input, thermal_input],
        [
            joint_embedding,
            rgb_decoder(rgb_features),
            thermal_decoder(thermal_features),
        ],
        name="multimodal_autoencoder_triplet_v1",
    )

    return model