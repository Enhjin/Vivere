from __future__ import print_function
# ------------------------------------------------------------------------------------------------
# Copyright (c) 2016 Microsoft Corporation
#
# Permission is hereby granted, free of charge, to any person obtaining a copy of this software and
# associated documentation files (the "Software"), to deal in the Software without restriction,
# including without limitation the rights to use, copy, modify, merge, publish, distribute,
# sublicense, and/or sell copies of the Software, and to permit persons to whom the Software is
# furnished to do so, subject to the following conditions:
#
# The above copyright notice and this permission notice shall be included in all copies or
# substantial portions of the Software.
#
# THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR IMPLIED, INCLUDING BUT
# NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY, FITNESS FOR A PARTICULAR PURPOSE AND
# NONINFRINGEMENT. IN NO EVENT SHALL THE AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM,
# DAMAGES OR OTHER LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
# OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE SOFTWARE.
# ------------------------------------------------------------------------------------------------

# The "Cliff Walking" example using Q-learning.
# From pages 148-150 of:
# Richard S. Sutton and Andrews G. Barto
# Reinforcement Learning, An Introduction
# MIT Press, 1998

from future import standard_library

standard_library.install_aliases()
from builtins import range
import MalmoPython
import json
import os
import time
import malmoutils
import tensorflow as tf
import numpy as np
from random import randint
malmoutils.fix_print()

actionSet = ["movenorth 1", "movesouth 1", "movewest 1", "moveeast 1"]

# ===========================================================================================================================

'''

DEEP Learning code comes here

'''

class StateProcessor():
    """
    Processes a raw Atari images. Resizes it and converts it to grayscale.
    """

    def __init__(self):
        # Build the Tensorflow graph
        with tf.variable_scope("state_processor"):
            self.input_state = tf.placeholder(shape=[9, 9, 1], dtype=tf.uint8)
            # self.output = tf.image.rgb_to_grayscale(self.input_state)
            self.output = self.input_state
            # self.output = tf.image.crop_to_bounding_box(self.output, 0, 0, 10, 10)
            # self.output = tf.image.resize_images(
            #     self.output, [10, 10], method=tf.image.ResizeMethod.NEAREST_NEIGHBOR)
            self.output = tf.squeeze(self.output)

    def process(self, sess, state):
        """
        Args:
            sess: A Tensorflow session object
            state: A [10, 10, 1] Maze RGB State
        Returns:
            A processed [84, 84] state representing grayscale values.
        """
        val = sess.run(self.output, {self.input_state: state})
        return val


def gridProcess(state):
    msg = state.observations[-1].text
    observations = json.loads(msg)
    grid = observations.get(u'floor10x10', 0)
    obs = np.array(grid)
    obs = np.reshape(obs, [9, 9, 1])
    obs[obs == "carpet"] = 0
    obs[obs == "sea_lantern"] = 1
    obs[obs == "fire"] = 4
    obs[obs == "emerald_block"] = 5
    obs[obs == "beacon"] = 6
    obs[obs == "air"] = 7
    obs[obs == "grass"] = 8
    return obs


class Estimator():
    """Q-Value Estimator neural network.
    This network is used for both the Q-Network and the Target Network.
    """

    def __init__(self, scope="estimator", summaries_dir=None):
        self.scope = scope
        # Writes Tensorboard summaries to disk
        self.summary_writer = None
        with tf.variable_scope(scope):
            # Build the graph
            self._build_model()
            if summaries_dir:
                summary_dir = os.path.join(summaries_dir, "summaries_{}".format(scope))
                if not os.path.exists(summary_dir):
                    os.makedirs(summary_dir)
                self.summary_writer = tf.summary.FileWriter(summary_dir)

    def _build_model(self):
        """
        Builds the Tensorflow graph.
        """

        # Placeholders for our input
        # Our input are 1 RGB frames of shape 10, 10 each
        self.X_pl = tf.placeholder(shape=[None, 9, 9, 4], dtype=tf.uint8, name="X")
        # The TD target value
        self.y_pl = tf.placeholder(shape=[None], dtype=tf.float32, name="y")
        # Integer id of which action was selected
        self.actions_pl = tf.placeholder(shape=[None], dtype=tf.int32, name="actions")

        X = tf.to_float(self.X_pl) / 255.0
        batch_size = tf.shape(self.X_pl)[0]

        # Three convolutional layers
        conv1 = tf.contrib.layers.conv2d(
            X, 32, 8, 4, activation_fn=tf.nn.relu)
        conv2 = tf.contrib.layers.conv2d(
            conv1, 64, 4, 2, activation_fn=tf.nn.relu)
        conv3 = tf.contrib.layers.conv2d(
            conv2, 64, 3, 1, activation_fn=tf.nn.relu)

        # Fully connected layers
        flattened = tf.contrib.layers.flatten(conv3)
        fc1 = tf.contrib.layers.fully_connected(flattened, 512)
        self.predictions = tf.contrib.layers.fully_connected(fc1, len(actionSet))

        # Get the predictions for the chosen actions only
        gather_indices = tf.range(batch_size) * tf.shape(self.predictions)[1] + self.actions_pl
        self.action_predictions = tf.gather(tf.reshape(self.predictions, [-1]), gather_indices)

        # Calculate the loss
        self.losses = tf.squared_difference(self.y_pl, self.action_predictions)
        self.loss = tf.reduce_mean(self.losses)

        # Optimizer Parameters from original paper
        self.optimizer = tf.train.RMSPropOptimizer(0.00025, 0.99, 0.0, 1e-6)
        self.train_op = self.optimizer.minimize(self.loss, global_step=tf.contrib.framework.get_global_step())

        # Summaries for Tensorboard
        self.summaries = tf.summary.merge([
            tf.summary.scalar("loss", self.loss),
            tf.summary.histogram("loss_hist", self.losses),
            tf.summary.histogram("q_values_hist", self.predictions),
            tf.summary.scalar("max_q_value", tf.reduce_max(self.predictions))
        ])

    def predict(self, sess, s):
        """
        Predicts action values.
        Args:
          sess: Tensorflow session
          s: State input of shape [batch_size, 1, 10, 10, 1]
        Returns:
          Tensor of shape [batch_size, NUM_VALID_ACTIONS] containing the estimated
          action values.
        """
        # print("S's shape:",s.shape)
        return sess.run(self.predictions, {self.X_pl: s})

    def update(self, sess, s, a, y):
        """
        Updates the estimator towards the given targets.
        Args:
          sess: Tensorflow session object
          s: State input of shape [batch_size, 1, 10, 10, 1]
          a: Chosen actions of shape [batch_size]
          y: Targets of shape [batch_size]
        Returns:
          The calculated loss on the batch.
        """
        feed_dict = {self.X_pl: s, self.y_pl: y, self.actions_pl: a}
        summaries, global_step, _, loss = sess.run(
            [self.summaries, tf.contrib.framework.get_global_step(), self.train_op, self.loss],
            feed_dict)
        if self.summary_writer:
            self.summary_writer.add_summary(summaries, global_step)
        return loss


def copy_model_parameters(sess, estimator1, estimator2):
    """
    Copies the model parameters of one estimator to another.
    Args:
      sess: Tensorflow session instance
      estimator1: Estimator to copy the paramters from
      estimator2: Estimator to copy the parameters to
    """
    e1_params = [t for t in tf.trainable_variables() if t.name.startswith(estimator1.scope)]
    e1_params = sorted(e1_params, key=lambda v: v.name)
    e2_params = [t for t in tf.trainable_variables() if t.name.startswith(estimator2.scope)]
    e2_params = sorted(e2_params, key=lambda v: v.name)

    update_ops = []
    for e1_v, e2_v in zip(e1_params, e2_params):
        op = e2_v.assign(e1_v)
        update_ops.append(op)

    sess.run(update_ops)


def make_epsilon_greedy_policy(estimator, nA):
    """
    Creates an epsilon-greedy policy based on a given Q-function approximator and epsilon.
    Args:
        estimator: An estimator that returns q values for a given state
        nA: Number of actions in the environment.
    Returns:
        A function that takes the (sess, observation, epsilon) as an argument and returns
        the probabilities for each action in the form of a numpy array of length nA.
    """

    def policy_fn(sess, observation, epsilon):
        A = np.ones(nA, dtype=float) * epsilon / nA
        q_values = estimator.predict(sess, np.expand_dims(observation, 0))[0]
        best_action = np.argmax(q_values)
        A[best_action] += (1.0 - epsilon)

        return A

    return policy_fn


def deep_q_learning_run(sess,
                    agent_host,
                    q_estimator,
                    state_processor,
                    experiment_dir,
                    epsilon_start=1.0,
                    epsilon_end=0.1,
                    epsilon_decay_steps=8000):
    """
    Q-Learning algorithm for off-policy TD control using Function Approximation.
    Finds the optimal greedy policy while following an epsilon-greedy policy.
    Args:
        sess: Tensorflow Session object
        env: OpenAI environment
        q_estimator: Estimator object used for the q values
        target_estimator: Estimator object used for the targets
        state_processor: A StateProcessor object
        num_episodes: Number of episodes to run for
        experiment_dir: Directory to save Tensorflow summaries in
        replay_memory_size: Size of the replay memory
        replay_memory_init_size: Number of random experiences to sampel when initializing
          the reply memory.
        update_target_estimator_every: Copy parameters from the Q estimator to the
          target estimator every N steps
        discount_factor: Gamma discount factor
        epsilon_start: Chance to sample a random action when taking an action.
          Epsilon is decayed over time and this is the start value
        epsilon_end: The final minimum value of epsilon after decaying is done
        epsilon_decay_steps: Number of steps to decay epsilon over
        batch_size: Size of batches to sample from the replay memory
        record_video_every: Record a video every N episodes
    Returns:
        An EpisodeStats object with two numpy arrays for episode_lengths and episode_rewards.
    """
    mission_file = agent_host.getStringArgument('mission_file')
    with open(mission_file, 'r') as f:
        print("Loading mission from %s" % mission_file)
        mission_xml = f.read()
        my_mission = MalmoPython.MissionSpec(mission_xml, True)
    my_mission.removeAllCommandHandlers()
    my_mission.allowAllDiscreteMovementCommands()
    my_mission.setViewpoint(2)
    my_clients = MalmoPython.ClientPool()
    my_clients.add(MalmoPython.ClientInfo('127.0.0.1', 10000))  # add Minecraft machines here as available

    max_retries = 3
    agentID = 0
    expID = 'Deep_q_learning memory'


    # Create directories for checkpoints and summaries
    checkpoint_dir = os.path.join(experiment_dir, "checkpoints")
    print("Checkpoint dir is:", checkpoint_dir)
    saver = tf.train.Saver()
    # Load a previous checkpoint if we find one
    # latest_checkpoint = tf.train.latest_checkpoint(checkpoint_dir)
    # print("~~~~~~~~~~~~~~", latest_checkpoint)
    # exit(0)
    latest_checkpoint = os.path.join(checkpoint_dir, "model")
    if latest_checkpoint:
        print("Loading model checkpoint {}...\n".format(latest_checkpoint))
        saver.restore(sess, latest_checkpoint)

    total_t = sess.run(tf.contrib.framework.get_global_step())

    # The epsilon decay schedule
    epsilons = np.linspace(epsilon_start, epsilon_end, epsilon_decay_steps)

    # The policy we're following
    policy = make_epsilon_greedy_policy(
        q_estimator,
        len(actionSet))

    my_mission_record = malmoutils.get_default_recording_object(agent_host,
                                                                "save_%s-rep" % (expID))

    for retry in range(max_retries):
        try:
            agent_host.startMission(my_mission, my_clients, my_mission_record, agentID, "%s" % (expID))
            break
        except RuntimeError as e:
            if retry == max_retries - 1:
                print("Error starting mission:", e)
                exit(1)
            else:
                time.sleep(2.5)

    world_state = agent_host.getWorldState()
    while not world_state.has_mission_begun:
        print(".", end="")
        time.sleep(0.1)
        print("Sleeping")
        world_state = agent_host.getWorldState()
        for error in world_state.errors:
            print("Error:", error.text)
    print()

    agent_host.sendCommand("look -1")
    agent_host.sendCommand("look -1")

    while world_state.is_mission_running and all(e.text == '{}' for e in world_state.observations):
        print("Sleeping....")
        world_state = agent_host.peekWorldState()
    # Populate the replay memory with initial experience

    while world_state.number_of_observations_since_last_state <= 0 and world_state.is_mission_running:
        # print("Sleeping")
        time.sleep(0.1)
        world_state = agent_host.peekWorldState()

    state = gridProcess(world_state)  # MALMO ENVIRONMENT Grid world NEEDED HERE/ was env.reset()
    state = state_processor.process(sess, state)
    state = np.stack([state] * 4, axis=2)

    stepNum=0
    while world_state.is_mission_running:

        action = randint(0,3)
        print("actions:",action)
        # next_state, reward, done, _ = env.step(actionSet[action]) # Malmo send command for the action
        # print("Sending command: ", actionSet[action])
        print("Step %s"%stepNum)
        stepNum+=1
        agent_host.sendCommand(actionSet[action])

        world_state = agent_host.peekWorldState()

        num_frames_seen = world_state.number_of_video_frames_since_last_state

        while world_state.is_mission_running and world_state.number_of_video_frames_since_last_state == num_frames_seen:
            world_state = agent_host.peekWorldState()

        if world_state.is_mission_running:
            # Getting the reward from taking a step
            while world_state.number_of_observations_since_last_state <= 0:
                time.sleep(0.1)
                world_state = agent_host.peekWorldState()
            # world_state = agent_host.getWorldState()
            next_state = gridProcess(world_state)
            next_state = state_processor.process(sess, next_state)
            next_state = np.append(state[:, :, 1:], np.expand_dims(next_state, 2), axis=2)
            state = next_state
        # time.sleep(1)

    return None


# Main body=======================================================

agent_host = MalmoPython.AgentHost()

# Find the default mission file by looking next to the schemas folder:
schema_dir = None
try:
    schema_dir = os.environ['MALMO_XSD_PATH']
except KeyError:
    print("MALMO_XSD_PATH not set? Check environment.")
    exit(1)
mission_file = os.path.abspath(os.path.join(schema_dir, '..',
                                            'sample_missions', 'Maze0.xml'))  # Integration test path
if not os.path.exists(mission_file):
    mission_file = os.path.abspath(os.path.join(schema_dir, '..',
                                                'Sample_missions', 'Maze0.xml'))  # Install path
if not os.path.exists(mission_file):
    print("Could not find Maze.xml under MALMO_XSD_PATH")
    exit(1)

# add some args
agent_host.addOptionalStringArgument('mission_file',
                                     'Path/to/file from which to load the mission.', mission_file)
agent_host.addOptionalFlag('load_model', 'Load initial model from model_file.')
agent_host.addOptionalStringArgument('model_file', 'Path to the initial model file', '')
agent_host.addOptionalFlag('debug', 'Turn on debugging.')
agent_host.setRewardsPolicy(MalmoPython.RewardsPolicy.LATEST_REWARD_ONLY)
agent_host.setObservationsPolicy(MalmoPython.ObservationsPolicy.LATEST_OBSERVATION_ONLY)
malmoutils.parse_command_line(agent_host)

tf.reset_default_graph()

# Where we save our checkpoints and graphs
experiment_dir = os.path.abspath("./experiments/{}".format("DeepQLearning"))

# Create a glboal step variable
global_step = tf.Variable(0, name='global_step', trainable=False)

# Create estimators
q_estimator = Estimator(scope="q", summaries_dir=experiment_dir)
target_estimator = Estimator(scope="target_q")

# State processor
state_processor = StateProcessor()

with tf.Session() as sess:
    sess.run(tf.global_variables_initializer())
    deep_q_learning_run(sess,
                    agent_host,
                    q_estimator=q_estimator,
                    state_processor=state_processor,
                    experiment_dir=experiment_dir,
                    epsilon_start=1.0,
                    epsilon_end=0.1,
                    epsilon_decay_steps=8000)
    exit(0)

# ======================================================================================