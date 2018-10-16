import gym
import tensorflow as tf
import numpy as np
import random
from collections import deque

# General Parameters
# -- DO NOT MODIFY --
ENV_NAME = 'CartPole-v0'
EPISODE = 200000  # Episode limitation
STEP = 200  # Step limitation in an episode
TEST = 10  # The number of tests to run every TEST_FREQUENCY episodes
TEST_FREQUENCY = 100  # Num episodes to run before visualizing test accuracy

# TODO: HyperParameters
GAMMA = 0.99 # discount factor
INITIAL_EPSILON = 0.5 # starting value of epsilon
FINAL_EPSILON = 0.01 # final value of epsilon
EPSILON_DECAY_STEPS = 200 # decay period
SIZE_BUFFER = 10000 # size of the buffer
BATCH_SIZE = 64 # batch size

# Create environment
# -- DO NOT MODIFY --
env = gym.make(ENV_NAME)
epsilon = INITIAL_EPSILON
STATE_DIM = env.observation_space.shape[0]
ACTION_DIM = env.action_space.n

# Placeholders
# -- DO NOT MODIFY --
state_in = tf.placeholder("float", [None, STATE_DIM])
action_in = tf.placeholder("float", [None, ACTION_DIM])
target_in = tf.placeholder("float", [None])

# TODO: Define Network Graph
w1 = tf.Variable(tf.truncated_normal([STATE_DIM,ACTION_DIM]))
b1 = tf.Variable(tf.ones(shape = [ACTION_DIM]), dtype = tf.float32)
# w2 = tf.Variable(tf.random_normal([20,ACTION_DIM],stddev = 0.1))
# b2 = tf.Variable(tf.ones([ACTION_DIM]),dtype = tf.float32)
#
# hidden_layer = tf.nn.relu(tf.matmul(state_in,w1) + b1)

# TODO: Network outputs
q_values = tf.matmul(state_in,w1) + b1
q_action = tf.reduce_sum(tf.multiply(q_values,action_in),reduction_indices=1)

# TODO: Loss/Optimizer Definition
loss = tf.reduce_mean((q_action - target_in))
optimizer = tf.train.AdamOptimizer(0.0001).minimize(loss)

replay_buffer = deque()

# Start session - Tensorflow housekeeping
session = tf.InteractiveSession()
session.run(tf.global_variables_initializer())


# -- DO NOT MODIFY ---
def explore(state, epsilon):
    """
    Exploration function: given a state and an epsilon value,
    and assuming the network has already been defined, decide which action to
    take using e-greedy exploration based on the current q-value estimates.
    """
    Q_estimates = q_values.eval(feed_dict={
        state_in: [state]
    })
    if random.random() <= epsilon:
        action = random.randint(0, ACTION_DIM - 1)
    else:
        action = np.argmax(Q_estimates)

    one_hot_action = np.zeros(ACTION_DIM)
    one_hot_action[action] = 1

    return one_hot_action


# Main learning loop
for episode in range(EPISODE):

    # initialize task
    state = env.reset()


    # Update epsilon once per episode
    epsilon -= epsilon / EPSILON_DECAY_STEPS

    # Move through env according to e-greedy policy
    for step in range(STEP):
        action = explore(state, epsilon)

        next_state, reward, done, _ = env.step(np.argmax(action))


        replay_buffer.append((state, action, reward, next_state, done))

        if len(replay_buffer) > SIZE_BUFFER:
            replay_buffer.popleft()

        if len(replay_buffer) > BATCH_SIZE:

            minibatch = random.sample(replay_buffer, BATCH_SIZE)
            state_batch = [x[0] for x in minibatch]
            action_batch = [x[1] for x in minibatch]
            reward_batch = [x[2] for x in minibatch]
            next_state_batch = [x[3] for x in minibatch]

            #print(minibatch)

            target_batch = []

            nextstate_q_values = q_values.eval(feed_dict={
                state_in: next_state_batch
            })

            for i in range(BATCH_SIZE):
                done = minibatch[i][4]

                if done:
                    target = reward_batch[i]
                else:
                    # TODO: Calculate the target q-value.
                    # hint1: Bellman
                    # hint2: consider if the episode has terminated
                    target = GAMMA * np.max(nextstate_q_values[i]) + reward_batch[i]

                target_batch.append(target)



            # Do one training step
            session.run([optimizer], feed_dict={
                target_in: target_batch,
                action_in: action_batch,
                state_in: state_batch
            })

        # Update
        state = next_state
        if done:
            break


    # Test and view sample runs - can disable render to save time
    # -- DO NOT MODIFY --
    if (episode % TEST_FREQUENCY == 0 and episode != 0):
        total_reward = 0
        for i in range(TEST):
            state = env.reset()
            for j in range(STEP):
                env.render()
                action = np.argmax(q_values.eval(feed_dict={
                    state_in: [state]
                }))
                state, reward, done, _ = env.step(action)
                total_reward += reward
                if done:
                    break
        ave_reward = total_reward / TEST
        print('episode:', episode, 'epsilon:', epsilon, 'Evaluation '
                                                        'Average Reward:', ave_reward)

env.close()
