import numpy as np
import tensorflow as tf
import os
import gym
from gym import spaces
from stable_baselines3 import PPO
from stable_baselines3.common.vec_env import DummyVecEnv
from .CNN import CNN

class PromoterOptimizationEnv(gym.Env):
    """
    Custom OpenAI Gym environment for optimizing promoter sequences using Reinforcement Learning.
    """

    def __init__(self, cnn_model_path, masked_sequence, target_expression, max_steps=20, seed=None):
        super(PromoterOptimizationEnv, self).__init__()
        
        if seed is not None:
            self._set_seed(seed)

        self.cnn = CNN(cnn_model_path)
        self.masked_sequence = np.array(self.cnn.one_hot_sequence(masked_sequence), dtype=np.float32)
        self.mask_indices = self._get_mask_indices(self.masked_sequence)
        self.target_expression = target_expression
        self.max_steps = max_steps
        self.current_step = 0

        # Action space: Choose a nucleotide (A, C, G, or T) for each masked position
        self.action_space = spaces.Discrete(4 * len(self.mask_indices))  # Each masked position has 4 choices
        
        # Observation space: One-hot-encoded sequence
        self.observation_space = spaces.Box(
            low=0, high=1, shape=self.masked_sequence.shape, dtype=np.float32
        )

        # Initialize state
        self.state = self.masked_sequence.copy()

    def _set_seed(self, seed):
        np.random.seed(seed)
        tf.random.set_seed(seed)
        os.environ['PYTHONHASHSEED'] = str(seed)

    def _get_mask_indices(self, masked_sequence):
        return [i for i, element in enumerate(masked_sequence) if np.allclose(element, 0.25, atol=1e-9)]

    def _decode_action(self, action):
        """
        Convert an action index into a (position, nucleotide) change.
        """
        position_index = action // 4
        nucleotide_index = action % 4

        if position_index >= len(self.mask_indices):
            position_index = len(self.mask_indices) - 1  # Safety check

        return self.mask_indices[position_index], nucleotide_index

    def step(self, action):
        """
        Apply the action (change one nucleotide) and compute the new reward.
        """
        self.current_step += 1

        # Decode action
        pos, nuc = self._decode_action(action)

        # Update sequence with new nucleotide
        new_nucleotide = [0, 0, 0, 0]
        new_nucleotide[nuc] = 1
        self.state[pos] = new_nucleotide

        # Predict transcription rate with CNN
        predicted_expression = self.cnn.predict([self.state], use_cache=False)[0]

        # Reward: Negative error from target
        error = np.abs(self.target_expression - predicted_expression)
        reward = -error  # Negative because we want to minimize error

        # Check termination conditions
        done = (self.current_step >= self.max_steps) or (error == 0)

        return self.state, reward, done, {}

    def reset(self):
        """
        Reset the environment to the initial masked sequence.
        Ensure that masked positions are initialized randomly.
        """
        self.current_step = 0
        self.state = self.masked_sequence.copy()
        
        # Randomly initialize the masked positions
        for pos in self.mask_indices:
            random_nuc = np.random.choice(4)  # Choose random nucleotide index (A=0, C=1, G=2, T=3)
            one_hot_nuc = [0, 0, 0, 0]
            one_hot_nuc[random_nuc] = 1
            self.state[pos] = one_hot_nuc  # Update the position
        
        return self.state


# Train RL Agent using PPO
def train_rl_agent(cnn_model_path, masked_sequence, target_expression, total_timesteps=50000):
    env = DummyVecEnv([lambda: PromoterOptimizationEnv(cnn_model_path, masked_sequence, target_expression)])

    model = PPO("MlpPolicy", env, verbose=1)
    model.learn(total_timesteps=total_timesteps)

    return model

# Optimize sequence using the trained RL agent
def optimize_sequence_with_rl(model, cnn_model_path, masked_sequence, target_expression):
    env = PromoterOptimizationEnv(cnn_model_path, masked_sequence, target_expression)
    obs = env.reset()
    
    done = False
    while not done:
        action, _ = model.predict(obs)
        obs, _, done, _ = env.step(action)

    # Decode optimized sequence
    optimized_sequence = env.cnn.reverse_one_hot_sequence(env.state)

    # Get final prediction and error
    prediction = env.cnn.predict([env.state], use_cache=False)[0]
    error = np.abs(target_expression - prediction)

    return optimized_sequence, prediction, error, env.state


def load_rl_agent(model_path):
    return PPO.load(model_path)
