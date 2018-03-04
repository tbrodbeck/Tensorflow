import tensorflow as tf
import gym
from Final_Task.network import Policy


class PPO_Agent:

    def __init__(self, env_name):
        env = gym.make(env_name)
        self.policy_old = Policy('policy_old', env)
        self.policy_new = Policy('policy_new', env)
        self.old_parameters = self.policy_old.parameters
        self.new_parameters = self.policy_new.parameters


# Test
PPO_Agent('Ant-v1')

