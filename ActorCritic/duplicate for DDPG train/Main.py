import tensorflow as tf
import gym
import numpy as np
import shutil
import os

# reproducible results
np.random.seed(1)
tf.set_random_seed(1)

# Load Environment
ENV_NAME = 'BipedalWalker-v2'
env = gym.make(ENV_NAME)
# Reproducible environment parameters
env.seed(1)


STATE_DIMENSION = env.observation_space.shape[0] 
ACTION_DIMENSION = env.action_space.shape[0] 
ACTION_BOUND = env.action_space.high 

########################################  Hyperparameters  ########################################

# number of episodes to be trained
TRAIN_EPI_NUM=500
# Learning rate for actor and critic
ACTOR_LR=0.05
CRITIC_LR=0.05
R_DISCOUNT=0.9 # reward discount

MEMORY_CAPACITY=1000000

ACTOR_REP_ITE=1700 # after such many iterations, update ACTOR
CRITIC_REP_ITE=1500

BATCH=40 # size of batch used to learn

# Path used to store training result (parameters)
TRAIN_DATA_PATH='./train'


GLOBAL_STEP = tf.Variable(0, trainable=False) # record how many steps we have gone through
INCREASE_GLOBAL_STEP = GLOBAL_STEP.assign(tf.add(GLOBAL_STEP, 1))


# set automatically decaying learning rate to ensure convergence
ACTOR_LR = tf.train.exponential_decay(LR_A, GLOBAL_STEP, 10000, .95, staircase=True)
CRITIC_LR = tf.train.exponential_decay(LR_C, GLOBAL_STEP, 10000, .90, staircase=True)


END_POINT = (200 - 10) * (14/30)    # The end point of the game


##################################################
LOAD_MODEL = True # Whether to load trained model#
##################################################


with tf.Session() as sess:

    # Create actor and critic.
    actor = Actor(sess, ACTION_DIMENSION, ACTION_BOUND, ACTOR_LR, REPLACE_ITER_A)
    critic = Critic(sess, STATE_DIMENSION, ACTION_DIMENSION, CRITIC_LR, R_DISCOUNT, REPLACE_ITER_C, actor.a, actor.a_)

    actor.add_grad_to_graph(critic.a_grads)

    # Memory class implementation from: https://github.com/jaara/AI-blog/blob/master/Seaquest-DDQN-PER.py
    memory = Memory(MEMORY_CAPACITY)

    # saver is used to store or restore trained parameters
    saver = tf.train.Saver(max_to_keep=100)  # Maximum number of recent checkpoints to keep. Defaults to 5.


    ################################# Determine whether it's a new training or going-on training ###############3
    if LOAD_MODEL: # Returns CheckpointState proto from the "checkpoint" file.
        checkpoints = tf.train.get_checkpoint_state(TRAIN_DATA_PATH, 'checkpoint').all_model_checkpoint_paths
        saver.restore(sess, checkpoints[-1]) # reload trained parameters into the tf session
    else:
        if os.path.isdir(TRAIN_DATA_PATH): 
          shutil.rmtree(TRAIN_DATA_PATH) # recursively remove all files under directory
        os.mkdir(TRAIN_DATA_PATH)

        sess.run(tf.global_variables_initializer())

    explore_degree=0.1
    explore_degree_minimum=0.0001
    explore_decay_factor=0.99

    #################################  Main loop for training #################################
    for i_episode in range(MAX_EPISODES):
        
        state = env.reset()
        episode_reward = 0 # the episode reward
        
        while True:

            action = actor.act(s)

            action = np.clip(np.random.normal(action, explore_degree), -ACTION_BOUND, ACTION_BOUND)   # explore using randomness
            next_state, reward, done, _ = env.step(a) 

            trainsition = np.hstack((s, a, [r], s_))
            probability = np.max(memory.tree.tree[-memory.tree.capacity:])
            memory.store(probability, transition)  # stored for later learning

            # when r=-100, that means BipedalWalker has falled to the groud
            episode_reward += reward


            # when the training reaches stable stage, we lessen the probability of exploration
            if GLOBAL_STEP.eval(sess) > MEMORY_CAPACITY/20:
                explore_degree = max([explore_decay_factor*explore_degree, explore_degree_minimum])  # decay the action randomness
                tree_index, b_memory, weights = memory.prio_sample(BATCH)    # for critic update

                b_state = b_memory[:, :STATE_DIMENSION]
                b_action = b_memory[:, STATE_DIMENSION: STATE_DIMENSION + ACTION_DIMENSION]
                b_reward = b_memory[:, -STATE_DIMENSION - 1: -STATE_DIMENSION]
                b_next_state = b_memory[:, -STATE_DIMENSION:]
                
                td = critic.learn(b_state, b_action, b_reward, b_next_state, weights)
                actor.learn(b_state)
                
                for i in range(len(tree_index)):  # update priority
                    index = tree_idx[i]
                    memory.update(index, td[i])


            # if GLOBAL_STEP.eval(sess) % SAVE_MODEL_ITER == 0:
            #     ckpt_path = os.path.join(TRAIN_DATA_PATH, 'DDPG.ckpt')
            #     save_path = saver.save(sess, ckpt_path, global_step=GLOBAL_STEP, write_meta_graph=False)
            #     print("\nSave Model %s\n" % save_path)

            if done:
                if "running_reward" not in globals():
                    running_reward = episode_reward
                else:
                    running_reward = 0.95*running_r + 0.05*ep_r
                
                print('running reward: ',running_reward,', episode reward: ',episode_reward)
                break # start new episode

            state = nextState
            sess.run(INCREASE_GLOBAL_STEP)


