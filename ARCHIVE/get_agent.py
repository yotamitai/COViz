from os.path import abspath
from pathlib import Path

import gym



class MyEvaluation(Evaluation):
    def __init__(self, env, agent, output_dir='../agents', num_episodes=1000, display_env=False):
        self.OUTPUT_FOLDER = output_dir
        super(MyEvaluation, self).__init__(env, agent, num_episodes=num_episodes,
                                           display_env=display_env)


def get_agent(config, env_id, seed=0, offscreen_rendering=True):
    """Implement here for specific agent and environment loading scheme"""
    env = gym.make(env_id)
    env.seed(seed)
    env.configure({"offscreen_rendering": offscreen_rendering})
    # config agent agent
    agent = agent_factory(env, config)
    # implement deterministic greedy policy
    agent.exploration_policy = exploration_factory({'method': 'Greedy'}, env.action_space)
    # create evaluation
    evaluation = MyEvaluation(env, agent, display_env=False)
    agent_path = Path(abspath(config['path']))
    # load agent
    evaluation.load_agent_model(agent_path)
    agent = evaluation.agent

    return env, agent
