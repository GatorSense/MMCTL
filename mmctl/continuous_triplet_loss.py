"""
Disclaimer:
This file is adapted from:
https://github.com/omoindrot/tensorflow-triplet-loss/blob/fc698369bb6c9acdc9f0e9e1ea00de0ddf782f12/model/triplet_loss.py

This version keeps only the functions required for MMCTL training in this repository.
"""

import tensorflow as tf


def _pairwise_distances_cosine(embeddings, squared=False):
    """
    Compute the pairwise cosine distance matrix.

    Args:
        embeddings: Tensor of shape (batch_size, embed_dim)
        squared: If True, square the cosine distances.

    Returns:
        Tensor of shape (batch_size, batch_size)
    """
    embeddings = tf.cast(embeddings, tf.float32)
    embeddings = tf.nn.l2_normalize(embeddings, axis=1)

    cosine_similarity = tf.matmul(embeddings, embeddings, transpose_b=True)
    distances = 1.0 - cosine_similarity
    distances = tf.maximum(distances, 0.0)

    if squared:
        distances = tf.square(distances)

    return distances


def batch_all_continuous_triplet_loss_final_multimodal(
    annotation_vector,
    embeddings,
    margin=0.01,
    squared=False,
    EMB_DIM=1024,
):
    """
    Continuous triplet loss for multimodal embeddings.

    The input embedding is assumed to be a concatenation of two modality-specific
    embeddings of size EMB_DIM each:
        embeddings = [embedding_modality_1, embedding_modality_2]

    The annotation vector is duplicated so both modality embeddings are compared
    against the same target annotation relationships.

    Args:
        annotation_vector: Tensor of shape (batch_size, feature_dim)
        embeddings: Tensor of shape (batch_size, 2 * EMB_DIM)
        margin: Margin parameter retained for interface compatibility
        squared: Whether to square cosine distances
        EMB_DIM: Dimension of each modality-specific embedding

    Returns:
        triplet_loss: Scalar tensor
        fraction_positive_triplets: Scalar tensor
    """
    del margin  # retained for API compatibility

    annotation_vector = tf.cast(annotation_vector, tf.float32)
    embeddings = tf.cast(embeddings, tf.float32)

    embeddings_1 = embeddings[:, :EMB_DIM]
    embeddings_2 = embeddings[:, EMB_DIM:2 * EMB_DIM]
    embeddings_all = tf.concat([embeddings_1, embeddings_2], axis=0)

    annotation_all = tf.concat([annotation_vector, annotation_vector], axis=0)

    annotation_dist = _pairwise_distances_cosine(annotation_all, squared=squared)
    embedding_dist = _pairwise_distances_cosine(embeddings_all, squared=squared)

    anchor_positive_annotation_dist = tf.expand_dims(annotation_dist, 2)
    anchor_negative_annotation_dist = tf.expand_dims(annotation_dist, 1)

    anchor_positive_embedding_dist = tf.expand_dims(embedding_dist, 2)
    anchor_negative_embedding_dist = tf.expand_dims(embedding_dist, 1)

    generalized_triplet_loss = (
        tf.square(anchor_positive_embedding_dist)
        - tf.square(anchor_negative_embedding_dist)
        + (anchor_negative_annotation_dist - anchor_positive_annotation_dist)
    )
    generalized_triplet_loss = tf.maximum(generalized_triplet_loss, 0.0)

    positive_regularization = tf.square(
        anchor_positive_embedding_dist - anchor_positive_annotation_dist
    )
    negative_regularization = tf.square(
        anchor_negative_embedding_dist - anchor_negative_annotation_dist
    )

    triplet_loss = (
        generalized_triplet_loss
        + positive_regularization
        + negative_regularization
    )

    valid_order_mask = tf.cast(
        anchor_positive_annotation_dist < anchor_negative_annotation_dist,
        tf.float32,
    )
    triplet_loss = triplet_loss * valid_order_mask

    positive_triplets = tf.cast(triplet_loss > 1e-16, tf.float32)
    num_positive_triplets = tf.reduce_sum(positive_triplets)
    num_valid_triplets = tf.reduce_sum(valid_order_mask)

    fraction_positive_triplets = num_positive_triplets / (num_valid_triplets + 1e-16)
    triplet_loss = tf.reduce_sum(triplet_loss) / (num_positive_triplets + 1e-16)

    return triplet_loss, fraction_positive_triplets