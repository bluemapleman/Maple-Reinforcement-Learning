class Critic(object):
    def __init__(self, sess, state_dim, action_dim, learning_rate, discount_factor, action, next_action):
        self.sess = sess
        self.state_dimension = state_dim
        self.action_dimension = action_dim
        self.discount_factor = discount_factor
        self.learning_rate = learning_rate
        # Input (state, action) pair, output q values
        self.action = action
        self.q = self.build_nn(S, self.action, 'evaluation_nn', trainable=True)

        self.next_q = self.build_nn(NextState, next_action, 'target_nn', trainable=False)    

        self.evaluation_parameters = tf.get_collection(tf.GraphKeys.GLOBAL_VARIABLES)
        self.target_parameters = tf.get_collection(tf.GraphKeys.GLOBAL_VARIABLES)

        
        self.target_q = R + self.discount_factor * self.q_


        # temporal difference
        self.td = tf.abs(self.target_q - self.q)
        self.weights = tf.placeholder(tf.float32, [None, 1], name='weights')
        
        self.loss = tf.reduce_mean(self.weights * tf.squared_difference(self.target_q, self.q))
        self.train_optimizer = tf.train.AdamOptimizer(self.lr).minimize(self.loss, global_step=GLOBAL_STEP) # add global_step parameters to ensure increment of global_step
        self.action_gradients = tf.gradients(self.q, a)[0]   


    def learn(self, state, action, reward, nextState, weights):
        _, td = self.sess.run([self.train_op, self.td], feed_dict={State: state, self.action: action, R: reward, NextState: nextState, self.weights: weights})
        
        self.sess.run([tf.assign(t, e) for t, e in zip(self.target_parameters, self.evaluation_parameters)])
        return td

    def build_nn(self, s, a, scope, trainable):
        
        init_weights = tf.random_normal_initializer(0., 0.01)
        init_bias = tf.constant_initializer(0.01)

        w1_state = tf.get_variable('w1_state', [self.state_dimension, 500], initializer=init_weights, trainable=trainable)
        w1_action= tf.get_variable('w1_action', [self.state_dimension, 500], initializer=init_weights, trainable=trainable)
        b1 = tf.get_variable('b1', [1, 500], initializer=init_bias, trainable=trainable)
        nn = tf.nn.relu(tf.matmul(s, w1_s) + tf.matmul(a, w1_a) + b1)
        
        nn = tf.layers.dense(nn, 20, activation=tf.nn.relu, kernel_initializer=init_weights,
                                  bias_initializer=init_bias, name='l2', trainable=trainable)
        
        q = tf.layers.dense(nn, 1, kernel_initializer=init_weights, bias_initializer=init_bias, trainable=trainable)   # Q(s,a)
        return q
