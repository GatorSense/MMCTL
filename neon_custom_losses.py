"""Losses for NEON multimodal reconstruction (from CTL/mmctl + HSI hybrid)."""

from __future__ import annotations

import tensorflow as tf
from tensorflow.keras import backend as K

from mmctl.continuous_triplet_loss import (
    batch_all_continuous_triplet_loss_final_multimodal as _mm_ctl_2mod,
)


def batch_all_continuous_triplet_loss_final_multimodal_nmod(
    annotation_vector,
    embeddings,
    margin=0.01,
    squared=False,
    EMB_DIM=1024,
    N_MOD=3,
):
    """Continuous CTL over N concatenated modality embeddings of size EMB_DIM."""
    del margin
    parts = [embeddings[:, i * EMB_DIM : (i + 1) * EMB_DIM] for i in range(N_MOD)]
    embeddings_all = tf.concat(parts, axis=0)
    annotation_all = tf.concat([annotation_vector] * N_MOD, axis=0)

    annotation_all = tf.cast(annotation_all, tf.float32)
    embeddings_all = tf.cast(embeddings_all, tf.float32)

    # cosine distances
    def _cos_dist(x):
        x = tf.nn.l2_normalize(x, axis=1)
        sim = tf.matmul(x, x, transpose_b=True)
        d = 1.0 - sim
        d = tf.maximum(d, 0.0)
        return tf.square(d) if squared else d

    cosine_dist = _cos_dist(annotation_all)
    emb_dist = _cos_dist(embeddings_all)

    ap_cos = tf.expand_dims(cosine_dist, 2)
    an_cos = tf.expand_dims(cosine_dist, 1)
    ap_emb = tf.expand_dims(emb_dist, 2)
    an_emb = tf.expand_dims(emb_dist, 1)

    gen_loss = tf.maximum(
        tf.square(ap_emb) - tf.square(an_emb) + (an_cos - ap_cos), 0.0
    )
    triplet_loss = gen_loss + tf.square(ap_emb - ap_cos) + tf.square(an_emb - an_cos)
    cosine_mask = tf.cast(tf.less(ap_cos, an_cos), tf.float32)
    triplet_loss = tf.multiply(cosine_mask, triplet_loss)
    num_positive = tf.reduce_sum(tf.cast(tf.greater(triplet_loss, 1e-16), tf.float32))
    loss = tf.reduce_sum(triplet_loss) / (num_positive + 1e-16)
    return loss


def keras_batch_all_triplet_continuous_loss_multimodal(margin=0.01, EMB_DIM=1024, N_MOD=3):
    def _loss(annotation_vector, embeddings):
        # Prefer N-mod when concat width matches N_MOD*EMB_DIM
        width = tf.shape(embeddings)[1]
        def three():
            return batch_all_continuous_triplet_loss_final_multimodal_nmod(
                annotation_vector, embeddings, margin=margin, EMB_DIM=EMB_DIM, N_MOD=3
            )

        def two():
            loss, _ = _mm_ctl_2mod(
                annotation_vector, embeddings, margin=margin, squared=False, EMB_DIM=EMB_DIM
            )
            return loss

        return tf.cond(tf.equal(width, 3 * EMB_DIM), three, two)

    return _loss


def ssim_loss(alpha=0.1):
    def _loss(y_true, y_pred):
        # filter_size must fit spatial dims (HSI/LiDAR are 6x6)
        spatial = tf.minimum(tf.shape(y_true)[1], tf.shape(y_true)[2])
        fs = tf.minimum(7, spatial)
        fs = tf.maximum(fs, 3)
        # tf.image.ssim needs static-ish filter; use 3 for small patches
        fs_int = 3
        ssim = 1.0 - tf.image.ssim(y_true, y_pred, 1.0, filter_size=fs_int)
        mse = tf.reduce_mean(tf.square(y_true - y_pred), axis=(-3, -2, -1))
        return tf.reduce_mean(alpha * ssim + (1.0 - alpha) * mse)

    return _loss


def rgb_ssim_loss(alpha=0.1):
    return ssim_loss(alpha=alpha)


def lidar_ssim_loss(alpha=0.1):
    return ssim_loss(alpha=alpha)


def hybrid_hsi_loss(y_true, y_pred):
    """MSE + light spectral cosine penalty for HSI reconstruction."""
    mse = tf.reduce_mean(tf.square(y_true - y_pred), axis=(-3, -2, -1))
    yt = tf.nn.l2_normalize(tf.reshape(y_true, [tf.shape(y_true)[0], -1]), axis=1)
    yp = tf.nn.l2_normalize(tf.reshape(y_pred, [tf.shape(y_pred)[0], -1]), axis=1)
    cos = 1.0 - tf.reduce_sum(yt * yp, axis=1)
    return tf.reduce_mean(mse + 0.1 * cos)


def mean_squared_loss(y_true, y_pred):
    return tf.reduce_mean(tf.square(y_true - y_pred))


def dice_loss(y_true, y_pred, smooth=1e-6):
    y_true_f = K.flatten(y_true)
    y_pred_f = K.flatten(y_pred)
    intersection = K.sum(y_true_f * y_pred_f)
    return 1.0 - (2.0 * intersection + smooth) / (
        K.sum(y_true_f) + K.sum(y_pred_f) + smooth
    )


def ciou_loss(y_true, y_pred, alpha=0.5):
    """Placeholder CIoU; not used by NEON recon script."""
    return tf.reduce_mean(tf.square(y_true - y_pred))
