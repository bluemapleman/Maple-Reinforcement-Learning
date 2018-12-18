class Actor:
    def __init__(self, sess, learning_rate,action_dim,action_bound):
        self.sess = sess
        self.action_dim = action_dim
        self.action_bound = action_bound
        self.learning_rate = learning_rate

        # input current state, output action to be taken
        self.a = self.build_neural_network(S, scope='eval_nn', trainable=True)

        self.a_ = self.build_neural_network(S_, scope='target_nn', trainable=False)

        self.eval_parameters = tf.get_collection(tf.GraphKeys.GLOBAL_VARIABLES, scope='Actor/eval_nn')
        self.target_parameters = tf.get_collection(tf.GraphKeys.GLOBAL_VARIABLES, scope='Actor/target_nn')


    def act(self, s):
        s = s[np.newaxis, :] 
        return self.sess.run(self.a, feed_dict={s: state})[0] 

    def learn(self, state):  # update parameters
        self.sess.run(self.train_op, feed_dict={s: state})
        if self.t_replace_counter % self.t_replace_iter == 0:
            self.sess.run([tf.assign(t, e) for t, e in zip(self.target_parameters, self.eval_parameters)])
        self.t_replace_counter += 1
    
    def build_neural_network(self, s, scope, trainable):
        
        init_weights = tf.random_normal_initializer(0., 0.1)
        init_bias = tf.constant_initializer(0.01)
        # three dense layer networks
        nn = tf.layers.dense(s, 500, activation=tf.nn.relu,
                              kernel_initializer=init_weights, bias_initializer=init_bias, name='l1', trainable=trainable)
        nn = tf.layers.dense(nn, 200, activation=tf.nn.relu,
                              kernel_initializer=init_weights, bias_initializer=init_bias, name='l2', trainable=trainable)
        actions = tf.layers.dense(nn, self.action_dim, activation=tf.nn.tanh, kernel_initializer=init_weights,
                                  bias_initializer=init_bias, name='a', trainable=trainable)
        scaled_actions = tf.multiply(actions, self.action_bound, name='scaled_actions')  

        return scaled_actions

    def add_gradient(self, a_gradients):
        self.policy_gradients_and_vars = tf.gradients(ys=self.a, xs=self.eval_parameters, grad_ys=a_gradients)
        opt = tf.train.RMSPropOptimizer(-self.learning_rate) # gradient ascent
        self.train_op = opt.apply_gradients(zip(self.policy_gradients_and_vars, self.eval_parameters), global_step=GLOBAL_STEP)

