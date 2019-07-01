# This is the first approach to implementing a Meta Reinforcement Learning System
# The design is based on the work by Wang et al 2017/2018 ("Learning to Reinforcement Learn", "Prefrontal
# Cortex as a Meta Reinforcement Learning System")

# Another implementation is on: https://blog.floydhub.com/meta-rl/ ,
# however that code is very messy and badly documented (and there are very annoying gifs on the website...)

import threading
from AcNetwork import *
from WorkerTwoStepTask import *
from TwoStepTask import *
from helper import *

# encoding of the higher stages
S_1 = 0
S_2 = 1
S_3 = 2
nb_states = 3


# Hyperparameters for training/testing
gamma = .9
a_size = 2
n_seeds = 8
num_episode_train = 20000
num_episode_test = 300
learning_rate = 7e-4

collect_seed_transition_probs = []

# Do train and test for n_seeds different seeds
for seed_nb in range(n_seeds):
#
    for num_episodes in [num_episode_train]: # set this up to do first training and then testing?

        tf.reset_default_graph()

        with tf.device("/cpu:0"):
            #global_episodes = tf.Variable(0, dtype=tf.int32, name='global_episodes', trainable=False)
            trainer = tf.train.RMSPropOptimizer(learning_rate= learning_rate)
            master_network = AcNetwork(a_size, 'global', None)  # Generate global network
            # Create worker classes
            worker = Worker(TwoStepTask(), i, a_size, trainer, model_path = '', global_episodes = tf.Variable(0, dtype=tf.int32, name='global_episodes', trainable=False), make_gif=True)
            #workers.append(Worker(TwoStepTask(), i, a_size, trainer, model_path = '', global_episodes = tf.Variable(0, dtype=tf.int32, name='global_episodes', trainable=False), make_gif=True))
            saver = tf.train.Saver(max_to_keep=5)
#
        with tf.Session() as sess:
            # set the seed
            np.random.seed(seed_nb)
            tf.set_random_seed(seed_nb)

            coord = tf.train.Coordinator()
            # if load_model == True:
            #     print('Loading Model...')
            #     ckpt = tf.train.get_checkpoint_state(load_model_path)
            #     saver.restore(sess, ckpt.model_checkpoint_path)
            # else:
            sess.run(tf.global_variables_initializer())

            worker_threads = []

            worker_work = lambda: worker.work(gamma, sess, coord, saver, train = 1, num_episodes = num_episodes)
            thread = threading.Thread(target=(worker_work))
            thread.start()
            worker_threads.append(thread)
            coord.join(worker_threads)
#
#             # final plot of the different seeds
#
#             episode_count = 300
#             common_sum = np.array([0., 0.])
#             uncommon_sum = np.array([0., 0.])
#
#             fig, ax = plt.subplots()
#
#             for i in range(n_seeds):
#                 x = np.arange(2)
#                 ax.set_ylim([0.5, 1.0])
#                 ax.set_ylabel('Stay Probability')
#
#                 common, uncommon = collect_seed_transition_probs[i]
#
#                 common_sum += np.array(common)
#                 uncommon_sum += np.array(uncommon)
#
#                 ax.set_xticks([1.3, 3.3])
#                 ax.set_xticklabels(['Last trial rewarded', 'Last trial not rewarded'])
#
#                 plt.plot([1, 3], common, 'o', color='black');
#                 plt.plot([1.8, 3.8], uncommon, 'o', color='black');
#
#             c = plt.bar([1., 3.], (1. / n_seeds) * common_sum, color='b', width=0.5)
#             uc = plt.bar([1.8, 3.8], (1. / n_seeds) * uncommon_sum, color='r', width=0.5)
#             ax.legend((c[0], uc[0]), ('common', 'uncommon'))
#             plt.savefig(dir_name + "/final_plot.png")