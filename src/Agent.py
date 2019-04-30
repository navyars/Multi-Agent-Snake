import tensorflow as tf

class NeuralNetwork:

    def __init__(self,data_length, size_of_hidden_layer=20, gamma=0.9, learning_rate=0.1):
        self.layers = self.create_model(data_length, size_of_hidden_layer)
        action = tf.placeholder(tf.int32, shape=[None, 1], name="action_selected")
        Q_value = tf.batch_gather(self.layers[-1], action, name="Q")

        opt = tf.train.GradientDescentOptimizer(learning_rate=learning_rate)

        reward = tf.placeholder(tf.float32, shape=[None,1], name="reward")
        terminal_loss = reward - Q_value
        terminal_grads_and_vars = opt.compute_gradients(terminal_loss)

        best_Q = tf.placeholder(tf.float32, shape=[None, 1], name="best_next_state_Q")
        t1 = gamma*best_Q
        t2 = reward + t1
        loss = t2 - Q_value
        grads_and_vars = opt.compute_gradients(loss)

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
        self.model["terminal_gradient"] = terminal_grads_and_vars
        self.model["gradient"] = grads_and_vars
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
        layers[4] = tf.nn.softmax(layers[3])
        return layers

if __name__=="__main__":
    import numpy as np

    nn = NeuralNetwork(3)
    input_data = np.random.rand(1,3, 2)
    action = [[3]]
    reward = [[-1]]
    with tf.Session() as sess:
        sess.run(tf.global_variables_initializer())
        print "Output = "
        print sess.run(nn.model["Q_value"], feed_dict={nn.model["state"] : input_data, nn.model["action"] : action} )
        best_q_value = [[ np.max(sess.run(nn.model["softmax"], feed_dict={nn.model["state"] : input_data} )) ]]
        print "Terminal gradient = "
        print sess.run(nn.model["terminal_gradient"], feed_dict={nn.model["state"] : input_data, nn.model["reward"] : reward, nn.model["action"] : action})
        print "Normal gradient = "
        print sess.run(nn.model["gradient"], feed_dict={nn.model["state"] : input_data, nn.model["reward"] : reward, nn.model["best_Q"] : best_q_value, nn.model["action"] : action})
    print "Complete"
