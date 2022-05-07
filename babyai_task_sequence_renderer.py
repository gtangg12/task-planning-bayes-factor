import torch
import gym
from gym_minigrid.window import Window
from babyai.utils.agent import BotAgent

from datasets.formats.task_sequence import *
from babyai_task_sequence import BabyaiFrame


def format_taskname_babyai(taskname: str) -> str:
    return f'BabyAI-{taskname}-v0'


def make_env(taskname: str) -> gym.Env:
    taskname = format_taskname_babyai(taskname)
    env = gym.make(taskname)
    env.reset()
    env.taskname = taskname # add taskname to env
    return env


def agent_solve(env: gym.Env) -> TaskSequence:
    agent = BotAgent(env)
    sequence = []
    while True:
        obs = env.gen_obs()
        action = agent.act(obs)['action']
        frame = BabyaiFrame(obs['image'], action, obs['direction'])
        sequence.append(frame)
        _, _, done, _ = env.step(action)
        if done:
            break
    return TaskSequence(Task(env.taskname, obs['mission']), sequence)


class BabyaiEnvRenderer:
    def __init__(self, env: gym.Env = None) -> None:
        self.env = env
        self.window = Window('gym_minigrid')

    def set_env(self, env: gym.Env) -> None:
        self.env = env
        
    def render_image(self, image: torch.Tensor) -> None:
        def key_handler(event):
            if event.key == 'q':
                self.window.close()
                
        self._render(image)
        self._setup_window(key_handler)

    def render_sequence(self, sequence: TaskSequence) -> None:
        step = 0
        def key_handler(event):
            nonlocal step
            if event.key == 'q':
                self.window.close()
            elif event.key == 'enter':
                step += 1
                if step == len(sequence):
                    self.window.close()
                    return
                self._render(sequence.frames[step].image)

        self._render(sequence.frames[step].image)
        self._setup_window(key_handler)

    def _render(self, image: torch.Tensor) -> None:
        display = self.env.get_obs_render(image, tile_size=32)
        self.window.show_img(display)
    
    def _setup_window(self, key_handler) -> None:
        self.window.reg_key_handler(key_handler)
        self.window.show(block=True)


if __name__ == '__main__':
    env = make_env('GoTo')
    sequence = agent_solve(env)
    engine = BabyaiEnvRenderer(env)
    engine.render_image(sequence.frames[0].image)