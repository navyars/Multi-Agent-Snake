import tensorflow as tf

class NeuralNetwork:

    def __init__(self,data_length, size_of_hidden_layer=20, gamma=0.9, learning_rate=0.1):
        self.graph = tf.Graph()
        with self.graph.as_default():
            self.layers = self.create_model(data_length, size_of_hidden_layer) # defines the neural network architecture
            action = tf.placeholder(tf.int32, shape=[None, 1], name="action_selected")
            Q_value = tf.batch_gather(self.layers[-1], action, name="Q") # fetch the Q(s,a) value

            optimizer = tf.train.GradientDescentOptimizer(learning_rate=learning_rate)

            reward = tf.placeholder(tf.float32, shape=[None,1], name="reward")
            best_Q = tf.placeholder(tf.float32, shape=[None, 1], name="best_next_state_Q")
            t1 = gamma*best_Q
            t2 = reward + t1
            difference = t2 - Q_value
            loss = tf.square(difference, name="loss")

            trainable_vars = tf.trainable_variables()
            saver = tf.train.Saver(trainable_vars) # used for saving and restoring the weights of the hidden layers

            #list of TF variables. Each one corresponds to a layer weight, and is used to store the accumulated gradient of that layer weight
            accum_vars = [tf.Variable(tf.zeros_like(tv.initialized_value()), trainable=False) for tv in trainable_vars]
            zero_ops = [tv.assign(tf.zeros_like(tv)) for tv in accum_vars] # used to reset the accumulator
            update_ops = tf.get_collection(tf.GraphKeys.UPDATE_OPS)
            with tf.control_dependencies(update_ops):
            	# global_step = tf.Variable(0, name='global_step', trainable=False)
            	# learning_rate = tf_utils.poly_lr(global_step)
                grads_and_vars = optimizer.compute_gradients(loss, var_list=trainable_vars) # fetch the gradients
                accum_ops = [accum_vars[i].assign_add(gv[0]) for i, gv in enumerate(grads_and_vars)] # accumulate them
                train_op = optimizer.apply_gradients([(accum_vars[i], gv[1]) for i, gv in enumerate(grads_and_vars)]) # train with the accumulated gradients

            global_init = tf.global_variables_initializer()
        # Dictionary to access all these layers for running in session
        self.model = {}
        # all placeholders
        self.model["state"] = self.layers[0]
        self.model["action"] = action
        self.model["reward"] = reward
        self.model["best_Q"] = best_Q
        # all outputs
        self.model["softmax"] = self.layers[-1]
        self.model["Q_value"] = Q_value
        self.model["gradient"] = grads_and_vars
        self.model["update_gradient"] = accum_ops
        self.model["reset_accum"] = zero_ops
        self.model["train"] = train_op
        # saver
        self.model["saver"] = saver
        # initializer
        self.model["init"] = global_init
        return

    def create_model(self, data_length, size_of_hidden_layer):
        layers = [0,0,0,0,0]
        xavier_init = tf.contrib.layers.xavier_initializer()
        layers[0] = tf.placeholder(tf.float32, shape=[None, data_length, 2], name="data")
        layers[1] = tf.reshape(layers[0], [-1, data_length * 2])
        layers[2] = tf.layers.dense(layers[1], size_of_hidden_layer,
                                                            kernel_initializer=xavier_init, use_bias=True,
                                                            activation=tf.nn.relu, name="hidden")
        layers[3] = tf.layers.dense(layers[2], 4, # one for each action
                                                            kernel_initializer=xavier_init, use_bias=True,
                                                            activation=tf.nn.relu, name="out")
        layers[4] = tf.nn.softmax(layers[3], name="softmax")
        return layers

    def save_model(self, sess, path):
        save_path = self.model["saver"].save(sess, path)
        return save_path

    def restore_model(self, sess, path):
        self.model["saver"].restore(sess, path)
        return

    def init(self, sess):
        sess.run(self.model["init"])
        return

    def Q(self, sess, state, action):
        return sess.run(self.model["Q_value"], feed_dict={ self.model["state"] : state, self.model["action"] : action })

    def max_Q(self, sess, state):
        return np.max(sess.run(self.model["softmax"], feed_dict={ self.model["state"] : state }))

    def get_gradients(self, sess, state, action, reward, next_state_Q=[[0]]):
        arg_dict = {self.model["state"] : input_data,
                            self.model["action"] : action,
                            self.model["reward"] : reward,
                            self.model["best_Q"] : next_state_Q}
        return sess.run(self.model["gradient"], feed_dict=arg_dict)

    def update_gradient(self, sess, state, action, reward, next_state_Q=[[0]]):
        arg_dict = {self.model["state"] : input_data,
                            self.model["action"] : action,
                            self.model["reward"] : reward,
                            self.model["best_Q"] : next_state_Q}
        return sess.run(self.model["update_gradient"], feed_dict=arg_dict)

    def reset_accumulator(self, sess):
        return sess.run(self.model["reset_accum"])

    def train(self, sess, state, action, reward, next_state_Q=[[0]]):
        arg_dict = {self.model["state"] : input_data,
                            self.model["action"] : action,
                            self.model["reward"] : reward,
                            self.model["best_Q"] : next_state_Q}
        return sess.run(self.model["train"], feed_dict=arg_dict)
