import pickle
import time

from keras.datasets import mnist
from datetime import date

from functions import *
from AdvantageActorCritic import *


def classification_episode(num_units, learning_rate, swap):
    """ Run a Decision Network Classification with parameters set outside
    - does a classification after every episode
    (used to run a multiprocessing script with different configurations)"""

    # Todo: -  reset of state after every episode
    #

    filename = "data/autoencoder_mnist2019-07-11 12:34:09.098789.pickle"
    infile = open(filename, 'rb')
    autoencoderModel, encoderModel = pickle.load(infile)

    # Import data
    (x_train, y_train), (x_test, y_test) = mnist.load_data()

    # Prepare Input: Normalizing, Flatten Images
    x_train = x_train.astype('float32') / 255.
    x_test = x_test.astype('float32') / 255.
    x_train = np.reshape(x_train, (len(x_train), 28, 28, 1))
    x_test = np.reshape(x_test, (len(x_test), 28, 28, 1))

    # input shape
    input_shape = 2 * encoderModel.layers[-1].output_shape[1] * encoderModel.layers[-1].output_shape[2] * \
                  encoderModel.layers[-1].output_shape[3]

    # Hyperparameters for training/testing
    num_seeds = 1
    num_episode_train = 120000  # Wang: 120'000
    # num_episode_test = 300
    num_trial_per_episode = 10
    # optimizer = 'rmsprop' hardcoded
    loss = 'mean_squared_error'

    # initialize a train loss, a choice and a prediction matrix
    num_train_loss = num_trial_per_episode
    train_loss_matrix = np.zeros((num_seeds, num_episode_train))
    choice_matrix = np.zeros((num_seeds, num_episode_train, num_train_loss))
    prediction_matrix = np.zeros((num_seeds, num_episode_train, num_train_loss))
    target_matrix = np.zeros((num_seeds, num_episode_train, num_train_loss))

    time_seed = time.time()
    dm_network_array = []

    for idx_seed in range(num_seeds):
        # create network
        dm_network = create_dm_network(num_units, input_shape, learning_rate, loss)

        for idx_episode in range(num_episode_train):

            # initialize variables for training at the end of the episode
            episode_targets = np.zeros([num_trial_per_episode,1,1])
            episode_inputs = np.zeros([num_trial_per_episode, 1, input_shape])
            episode_choices = np.zeros([num_trial_per_episode])
            episode_predictions = np.zeros([num_trial_per_episode])

            # sample two hidden representations
            hidden_representations, labels \
                = create_sample_representation(encoderModel, x_train, y_train, is_training=1)

            # pick one of the samples to be the rewarded one
            q = np.random.random_sample()
            if q > 0.5:
                good_label = labels[1]
                target = 1
            else:
                good_label = labels[0]
                target = 0

            # perform the trials
            for idx_trial in range(num_trial_per_episode):

                # predict the from the current network and make a choice
                dm_input = np.concatenate((hidden_representations[0], hidden_representations[1]))
                dm_input = np.array([[dm_input]])
                prediction = dm_network.predict(dm_input, batch_size=None, verbose=0, steps=None)
                choice = int(round(prediction))

                # attach the input and the target values to the episode
                episode_targets[idx_trial,:,:] = target
                episode_inputs[idx_trial,:,:] = dm_input
                episode_choices[idx_trial] = choice
                episode_predictions[idx_trial] = prediction

                if swap == 1:
                    # mix up the representations into random order
                    hidden_rep1 = hidden_representations[0]
                    hidden_rep2 = hidden_representations[1]
                    hidden_representations, labels = mix_up(hidden_rep1, hidden_rep2, labels)

                    # reset the target value of the trial
                    if labels[0] == good_label:
                        target = 0
                    elif labels[1] == good_label:
                        target = 1

            # Todo: Change input structure for training
            # train the network using the episode target and input
            info = dm_network.fit(episode_inputs, episode_targets, verbose=0)
            current_loss = info.history.get('loss', '')

            # store loss, choice, prediction
            train_loss_matrix[idx_seed, idx_episode] = current_loss[0]
            choice_matrix[idx_seed, idx_episode, :] = episode_choices
            prediction_matrix[idx_seed, idx_episode, :] = episode_predictions
            target_matrix[idx_seed, idx_episode, :] = np.squeeze(episode_targets)

        # append the final network to the array of network
        dm_network_array.append(dm_network)
        time_seed_over = time.time()
        print("time for this seed: ")
        print(time_seed_over - time_seed)
        time_seed = time.time()


    """ Variable Saving and Plotting """
    today = date.today()
    month = today.month
    day = today.day
    filename = 'Class_episode_{0}{1}_nEpisode:{2}_nUnits:{3}_learning_rate:{4}_swap:{5}.pickle'\
        .format(month, day, num_episode_train, num_units, learning_rate, swap)
    with open('data/{0}'.format(filename), "w") as f:  # The w stands for write
        pickle.dump([dm_network_array, train_loss_matrix, prediction_matrix, choice_matrix, target_matrix], f)
    f.close()