import tensorflow as tf

class Index:
    def __init__(self, embeddings, labels):
        self.embeddings = embeddings
        self.dot_embeddings = tf.matmul(embeddings, embeddings, transpose_b=True)
        self.squared_norms_embeddings = tf.expand_dims(tf.linalg.diag_part(self.dot_embeddings), 0)
        self.labels = labels

    def search(self, query_vector, top_k=None):
        dot_query_vector = tf.matmul(query_vector, query_vector, transpose_b=True)
        squared_norms_query_vector = tf.expand_dims(tf.linalg.diag_part(dot_query_vector), 0)
        dot_product = tf.reduce_sum(self.embeddings * query_vector, axis=1)
        distances = tf.maximum(self.squared_norms_embeddings + squared_norms_query_vector - 2 * dot_product, 0)
        sorted_indices =  tf.argsort(distances)
        if top_k:
            sorted_indices = sorted_indices[..., :top_k]
        nearest_labels =  tf.reshape(tf.gather(self.labels, sorted_indices), shape=[-1, 1])
        nearest_distances = tf.reshape(tf.gather(distances[0], sorted_indices), shape=[-1, 1])
        return nearest_distances[..., 0], nearest_labels[..., 0], sorted_indices[0]
