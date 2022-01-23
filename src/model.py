import tensorflow as tf
import numpy as np
import math
from tensorflow.keras import layers
from functools import partial
from typing import Dict, Any

class ViT(tf.keras.Model):
    def __init__(self, config: Dict[str, Any], output_size: int):
        super(ViT, self).__init__()
        # Data Preprocessing
        patch_size = config['model']['patch_size']
        num_patches = (config['model']['data_augmentation']['scale_size'] // patch_size) ** 2
        self.patcher = PatchImage(patch_size)

        # Encoders
        encoding_size = config['model']['encoding_size']
        n_encoders = config['model']['encoder']['layers']
        num_heads = config['model']['encoder']['multihead_attention']['heads']
        mha_dropout = config['model']['encoder']['multihead_attention']['dropout']
        mlp_units = config['model']['encoder']['mlp']['units']
        mlp_dropout = config['model']['encoder']['mlp']['dropout']
        self.positional_encoding = PositionalEncoding(num_patches, encoding_size)
        self.encoders = [Encoder(encoding_size, 
                                 num_heads,
                                 mha_dropout,
                                 mlp_units,
                                 mlp_dropout) for _ in range(n_encoders)]

        # Post-encoder layers
        self.layer_norm = tf.keras.layers.LayerNormalization(axis=0)
        self.flatten = tf.keras.layers.Flatten()
        self.dropout = tf.keras.layers.Dropout(config['model']['dropout'])
        self.logits = tf.keras.layers.Dense(output_size)
        self.mlp = MLP(units=config['model']['mlp']['units'], activation='relu', dropout=config['model']['mlp']['dropout'])
        self.data_augmentation = augmentation_layer(scale_size=config['model']['data_augmentation']['scale_size'],
                                                    rotation_factor=config['model']['data_augmentation']['rotation_factor'],
                                                    zoom_factor=config['model']['data_augmentation']['zoom_factor'])
        
    def call(self, inputs, training: bool=False):
        # Apply image augmentation to the input, split into square patches and apply the positional encoding
        augmented = self.data_augmentation(inputs)
        patches = self.patcher(augmented)
        x = self.positional_encoding(patches)

        # Run through all the encoder layers
        for i in range(len(self.encoders)):
            x = self.encoders[i](x)

        x = self.flatten(x)
        x = self.dropout(x)
        x = self.mlp(x)
        output = self.logits(x)
        return output


def augmentation_layer(scale_size: int, rotation_factor: float, zoom_factor: float):
    return tf.keras.models.Sequential(
    [
        layers.Normalization(),
        layers.Resizing(scale_size, scale_size),
        layers.RandomFlip("horizontal"),
        layers.RandomRotation(factor=rotation_factor),
        layers.RandomZoom(
            height_factor=zoom_factor, width_factor=zoom_factor
        ),
    ],
    name="data_augmentation",
)


class PositionalEncoding(tf.keras.Model):
    """ The positional encoding adds information to each input embedding on their position in the input image space.
        In this case, we use the sine cosine encoding method proposed in the "Attention is all you need" paper """
    def __init__(self, n_patches: int, embedding_size: int):
        super(PositionalEncoding, self).__init__()
        self.n_patches = n_patches
        self.dense = tf.keras.layers.Dense(units=embedding_size, activation='relu')
        self.positional_encoding = self.__class__.__generate_angles(np.arange(n_patches)[:, np.newaxis],
                                                                    np.arange(embedding_size)[np.newaxis, :],
                                                                    embedding_size)
        self.positional_encoding[:, 0::2] = np.sin(self.positional_encoding[:, 0::2])
        self.positional_encoding[:, 1::2] = np.cos(self.positional_encoding[:, 0::2])
        self.positional_encoding = self.positional_encoding[np.newaxis, ...]
        self.positional_encoding = tf.cast(self.positional_encoding, dtype=tf.float32)

    @staticmethod
    def __generate_angles(pos: int, i: int, d_model: int):
            angles = 1 / np.power(10000, (2 * (i // 2) / d_model))
            return pos * angles

    def call(self, input):
        return self.dense(input) + self.positional_encoding


class PatchImage(tf.keras.Model):
    """ The image patcher splits the original input image into square patches of the specified size """
    def __init__(self, patch_size):
        super(PatchImage, self).__init__()
        self.patch_size = patch_size
        self.stride_size = self.patch_size

    def call(self, input):
        batch_size = tf.shape(input)[0]
        patch_shape = [1, self.patch_size, self.patch_size, 1]
        stride_shape = [1, self.stride_size, self.stride_size, 1]
        patches = tf.image.extract_patches(
            images=input, 
            sizes=patch_shape,
            strides=stride_shape,
            rates=[1, 1, 1, 1],
            padding='VALID')
        patch_dims = patches.shape[-1]

        patches = tf.reshape(patches, [batch_size, -1, patch_dims])
        return patches
            

class Encoder(tf.keras.Model):
    def __init__(self, encoding_size, num_heads, dropout_encoder, mlp_units, dropout_mlp):
        super(Encoder, self).__init__()
        self.encoding_size = encoding_size
        self.layer_norm = tf.keras.layers.LayerNormalization()
        self.multihead_attention = tf.keras.layers.MultiHeadAttention(num_heads=num_heads,
                                                                      key_dim=encoding_size,
                                                                      dropout=dropout_encoder)
        self.mlp = MLP(units=mlp_units, activation='relu', dropout=dropout_mlp)

    def call(self, z_prev, training=False):
        z_prev_norm = self.layer_norm(z_prev)
        z_prime_l = self.multihead_attention(z_prev_norm, z_prev_norm) + z_prev
        z_prime_l_norm = self.layer_norm(z_prime_l)
        z_l = self.mlp(z_prime_l_norm) + z_prime_l
        return z_l


class MLP(tf.keras.Model):
    def __init__(self, units, activation, dropout=0.2):
        super(MLP, self).__init__()
        self.dense = [tf.keras.layers.Dense(units=units[i], activation=activation) for i in range(len(units))]
        self.dropout = tf.keras.layers.Dropout(rate=dropout)

    def call(self, x):
        for i in range(len(self.dense)):
            x = self.dense[i](x)
        return self.dropout(x)