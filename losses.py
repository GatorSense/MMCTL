"""
Minimal custom losses used by train_mmctl.py.
"""

import tensorflow as tf

from continuous_triplet_loss import batch_all_continuous_triplet_loss_final_multimodal


def keras_batch_all_triplet_continuous_loss_multimodal(margin=0.01, emb_dim=1024):
    def loss_fn(annotation_vector, embeddings):
        return batch_all_continuous_triplet_loss_final_multimodal(
            annotation_vector,
            embeddings,
            margin=margin,
            squared=False,
            EMB_DIM=emb_dim,
        )
    return loss_fn


def ssim_loss(alpha=0.1):
    def loss_fn(y_true, y_pred):
        ssim_term = 1.0 - tf.image.ssim(y_true, y_pred, max_val=1.0, filter_size=7)
        mse_term = tf.reduce_mean(tf.square(y_true - y_pred), axis=(-3, -2, -1))
        return tf.reduce_mean(alpha * ssim_term + (1.0 - alpha) * mse_term)
    return loss_fn