import tensorflow as tf
with tf.Session() as sess:
          new_saver = tf.train.import_meta_graph('/Users/samarth/Desktop/Luminosity Lab/Text-Classification-using-a-CNN/runs/1530861779/checkpoints/model-6700.meta')
          new_saver.restore(sess, tf.train.latest_checkpoint('/Users/samarth/Desktop/Luminosity Lab/Text-Classification-using-a-CNN/runs/1530861779/checkpoints/'))
