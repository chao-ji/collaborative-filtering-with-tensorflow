import pandas as pd
import numpy as np
import tensorflow as tf

NUM_USER = 6040
NUM_ITEM = 3952

class MiniBatchFeeder(object):
    def __init__(self,
                 data,
                 batch_size=1000,
                 shuffle_data=False):

        self.data = data
        self.size = data.shape[0]
        self.batch_size = batch_size
        self.counter = 0
        self.shuffle_data = shuffle_data

        self.data[:, 0] = self.data[:, 0].astype(np.int32) - 1
        self.data[:, 1] = self.data[:, 1].astype(np.int32) - 1
        self.data[:, 2] = self.data[:, 2].astype(np.float32)

        self.reset_counter()

    def _shuffle(self, seed=42):
        self.data = self.data[np.random.permutation(self.size), :]

    def __iter__(self):
        return self

    def reset_counter(self, start=0):
        self.counter = start
        if self.shuffle_data:
            self._shuffle()

    def next(self):
        if self.counter < self.size:
            low = self.counter
            high = self.counter + self.batch_size
        
            self.counter += self.batch_size

            return self.data[low : high, :]
        else:
            raise StopIteration

class CollaborativeFilter(object):

    def __init__(self,
                 weight_initializer,
                 num_user,
                 num_item,
                 num_dim=20,
                 num_epoch=200,
                 device="/gpu:0",
                 learning_rate=0.001,
                 reg=0.05):

        self.weight_initializer = weight_initializer 
        self.num_user = num_user
        self.num_item = num_item
        self.num_dim = num_dim
        self.num_epoch = num_epoch
        self.device = device
        self.learning_rate = learning_rate
        self.reg = reg

        self.bias_, self.bias_user_, self.bias_item_, self.embd_user_, self.embd_item_ = None, None, None, None, None 

    def _create_param_tensors(self):
        with tf.device("/cpu:0") :
            bias = tf.get_variable(name="bias", shape=[], initializer=self.weight_initializer)
            bias_user = tf.get_variable(name="bias_user", shape=[self.num_user], initializer=self.weight_initializer)
            bias_item = tf.get_variable(name="bias_item", shape=[self.num_item], initializer=self.weight_initializer)
            embd_user = tf.get_variable(name="embd_user", shape=[self.num_user, self.num_dim], initializer=self.weight_initializer)
            embd_item = tf.get_variable(name="embd_item", shape=[self.num_item, self.num_dim], initializer=self.weight_initializer)

        return bias, bias_user, bias_item, embd_user, embd_item

    def _create_batch_tensors(self, user_batch_index, item_batch_index):
        with tf.device("/cpu:0"):
            bias_user_batch = tf.nn.embedding_lookup(params=self.bias_user_, ids=user_batch_index, name="bias_user_batch")
            bias_item_batch = tf.nn.embedding_lookup(params=self.bias_item_, ids=item_batch_index, name="bias_item_batch")

            embd_user_batch = tf.nn.embedding_lookup(params=self.embd_user_, ids=user_batch_index, name="embd_user_batch")
            embd_item_batch = tf.nn.embedding_lookup(params=self.embd_item_, ids=item_batch_index, name="embd_item_batch")

        return bias_user_batch, bias_item_batch, embd_user_batch, embd_item_batch

    def _create_op_tensors(self, user_batch_index, item_batch_index, rating_batch):
        global_step = tf.train.get_global_step()
        assert global_step is not None

        bias_user_batch, bias_item_batch, embd_user_batch, embd_item_batch = self._create_batch_tensors(user_batch_index, item_batch_index)

        with tf.device(self.device):
            rating_pred = tf.reduce_sum(tf.multiply(embd_user_batch, embd_item_batch), 1)
            rating_pred = tf.add(rating_pred, self.bias_)
            rating_pred = tf.add(rating_pred, bias_user_batch)
            rating_pred = tf.add(rating_pred, bias_item_batch, name="rating_pred")
            
            loss_error = tf.nn.l2_loss(tf.subtract(rating_pred, rating_batch))

            regularizer = tf.add(tf.nn.l2_loss(embd_user_batch), tf.nn.l2_loss(embd_item_batch))
            regularizer = tf.add(regularizer, tf.nn.l2_loss(bias_user_batch))
            regularizer = tf.add(regularizer, tf.nn.l2_loss(bias_item_batch), name="regularizer")

            loss_reg = tf.multiply(regularizer, tf.constant(self.reg))

            loss = tf.add(loss_error, loss_reg)
            train_op = tf.train.AdamOptimizer(self.learning_rate).minimize(loss, global_step=global_step)

        return loss, rating_pred, train_op
            
    def fit(self, train_data, test_data=None):

        user_batch_index = tf.placeholder(tf.int32, name="user_batch_index")
        item_batch_index = tf.placeholder(tf.int32, name="item_batch_index")
        rating_batch = tf.placeholder(tf.float32, name="rating_batch")
        
        global_step = tf.contrib.framework.get_or_create_global_step()

        self.bias_, self.bias_user_, self.bias_item_, self.embd_user_, self.embd_item_ = self._create_param_tensors()
        loss, rating_pred, train_op = self._create_op_tensors(user_batch_index, item_batch_index, rating_batch)
        init_op = tf.global_variables_initializer()
        
        with tf.Session() as sess:
            sess.run(init_op)
            for i in range(self.num_epoch):
                print i

                pred = []
                true = []

                train_data.reset_counter()
                for batch in train_data:
                    _, y_ = sess.run([train_op, rating_pred], feed_dict={user_batch_index: batch[:, 0],
                                                                 item_batch_index: batch[:, 1],
                                                                 rating_batch: batch[:, 2]})

                    pred.extend(y_)
                    true.extend(list(batch[:, 2]))

                
                pred = np.clip(np.array(pred), 1.0, 5.0)
                true = np.array(true)

                train_rmse = np.sqrt(np.mean(np.square(pred - true)))


                pred = []
                true = []

                test_data.reset_counter()
                for batch in test_data:
                    y_ = sess.run(rating_pred, feed_dict={user_batch_index: batch[:, 0],
                                                          item_batch_index: batch[:, 1],
                                                          rating_batch: batch[:, 2]})
                    pred.extend(y_)
                    true.extend(list(batch[:, 2]))

                pred = np.clip(np.array(pred), 1.0, 5.0)
                true = np.array(true)

                test_rmse = np.sqrt(np.mean(np.square(pred - true)))
                print "%f\t%f" % (train_rmse, test_rmse)

    def predict(self):
        pass


