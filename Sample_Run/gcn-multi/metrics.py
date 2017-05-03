import tensorflow as tf


def masked_sigmoid_binary_cross_entropy(preds, labels, mask):
    """Softmax cross-entropy loss with masking."""

    #label_sigmoid = tf.nn.sigmoid(preds)
    ##loss = tf.reduce_mean(tf.squared_difference(label_sigmoid, labels))
    #loss = -tf.reduce_sum(labels * tf.log(label_sigmoid + 1e-10) + (1-labels) * tf.log((1-label_sigmoid) + 1e-10), 1)

    #label_sigmoid = tf.nn.softmax(preds)
    #loss = -tf.reduce_sum(labels * tf.log(label_sigmoid + 1e-10) * tf.constant([.32, .68]), 1)
    #loss = -tf.reduce_sum(labels * tf.log(label_sigmoid + 1e-10), 1)

    #loss = tf.nn.sigmoid_cross_entropy_with_logits(preds, labels)
    #loss = tf.reduce_sum(loss, 1)
    loss = tf.nn.softmax_cross_entropy_with_logits(preds, labels)


    mask = tf.cast(mask, dtype=tf.float32)
    mask /= tf.reduce_mean(mask)
    loss *= mask
    return tf.reduce_mean(loss)

def masked_accuracy(preds, labels, mask):
    """Accuracy with masking."""
    correct_prediction = tf.equal(tf.argmax(preds, 1), tf.argmax(labels, 1))
    accuracy_all = tf.cast(correct_prediction, tf.float32)
    mask = tf.cast(mask, dtype=tf.float32)
    mask /= tf.reduce_mean(mask)
    accuracy_all *= mask
    return tf.reduce_mean(accuracy_all)