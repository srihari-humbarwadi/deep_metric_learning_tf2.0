def get_distances(embeddings):
    """embeddings : shape == [batch_size, EMBEDDING_DIM]
       return distances : shape == [batch_size, batch_size]
    """

    dot = tf.matmul(embeddings, embeddings, transpose_b=True)
    squared_norms = tf.expand_dims(tf.linalg.diag_part(dot), 0)
    squared_norms_t = tf.transpose(squared_norms)
    squared_distances = squared_norms + squared_norms_t - 2 * dot

    zero_mask = tf.equal(squared_distances, 0)
    squared_distances = squared_distances + tf.cast(zero_mask, dtype=tf.float32)*1e-14
    distances = tf.sqrt(squared_distances)
    distances = distances * (1 - tf.cast(zero_mask, dtype=tf.float32))
    return distances

def get_negative_mask(labels):
    """label(a) != label(b)"""
    labels = tf.expand_dims(labels, axis=1)
    labels_t = tf.transpose(labels)
    mask = tf.logical_not(tf.equal(labels, labels_t))
    return mask

def get_positive_mask(labels):
    """label(a) == label(b) && a != b """
    batch_shape = tf.shape(labels)[0]
    mask_1 = tf.logical_not(get_negative_mask(labels))
    mask_2 = tf.logical_not(tf.eye(batch_shape, dtype=tf.bool))
    return tf.logical_and(mask_1, mask_2)

def triplet_loss(labels, embeddings, margin=0.5):
    """embeddings : shape == [batch_size, EMBEDDING_DIM]
       labels : shape == [batch_size]
    """
    distances = get_distances(embeddings)
    positive_mask = get_positive_mask(labels)
    negative_mask = get_negative_mask(labels)

    ## hard positive samples
    positive_distances = tf.cast(positive_mask, dtype=tf.float32) * distances
    hard_positive_distances = tf.expand_dims(tf.reduce_max(positive_distances, axis=1), axis=1)

    ## hard negative samples
    max_distance = tf.expand_dims(tf.reduce_max(distances, axis=1), axis=1)
    hard_negative_distaces = tf.expand_dims(tf.reduce_min(distances + (1 - tf.cast(negative_mask, dtype=tf.float32)) * max_distance, axis=1), axis=1)

    ## final loss
    loss = hard_positive_distances - hard_negative_distaces + margin
    loss = tf.maximum(loss, 0)
    return tf.reduce_mean(loss, axis=0)
