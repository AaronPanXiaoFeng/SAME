import tensorflow as tf
import math

class SearchGradient():
    def __init__(self, conf):
        self.conf = conf

    def get_optimizer(self, global_step):
        learning_rate = self.conf['learning_rate']
        tf.summary.scalar(name="Optimize/learning_rate", tensor=learning_rate)

        return tf.train.GradientDescentOptimizer(learning_rate)