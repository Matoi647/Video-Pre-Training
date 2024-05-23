import sys
import os
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from argparse import ArgumentParser
import pickle
import cv2
import numpy as np
from pprint import pprint

from minerl.herobraine.env_specs.human_survival_specs import HumanSurvival

from agent import MineRLAgent, ENV_KWARGS

height, width = 360, 640
fourcc = cv2.VideoWriter_fourcc(*'mp4v')
video = cv2.VideoWriter('video.mp4', fourcc, 20.0, (width, height))
# video.set(cv2.VIDEOWRITER_PROP_QUALITY, 90)

def main(model, weights):
    env = HumanSurvival(**ENV_KWARGS).make()
    print("---Loading model---")
    agent_parameters = pickle.load(open(model, "rb"))
    policy_kwargs = agent_parameters["model"]["args"]["net"]["args"]
    pi_head_kwargs = agent_parameters["model"]["args"]["pi_head_opts"]
    pi_head_kwargs["temperature"] = float(pi_head_kwargs["temperature"])
    agent = MineRLAgent(env, policy_kwargs=policy_kwargs, pi_head_kwargs=pi_head_kwargs)
    agent.load_weights(weights)

    print("---Launching MineRL enviroment (be patient)---")
    obs = env.reset()

    # with open("obs.txt", "w") as f:
    # while True:
    for i in range(1200):
        minerl_action = agent.get_action(obs)
        if i % 20 == 0:
            minerl_action['chat'] = f'/me is exploring the world. {i//20}'
        if i % 1200 == 0:
            minerl_action['chat'] = '/effect give @p minecraft:night_vision 3600 1 false'
            
        obs, reward, done, info = env.step(minerl_action)
        # pprint(obs, stream=f)
        # env.render()
        print(i)
        # break
        img = obs['pov']
        video.write(cv2.cvtColor(img, cv2.COLOR_RGB2BGR))
    video.release()


if __name__ == "__main__":
    parser = ArgumentParser("Run pretrained models on MineRL environment")

    parser.add_argument("--weights", type=str, required=True, help="Path to the '.weights' file to be loaded.")
    parser.add_argument("--model", type=str, required=True, help="Path to the '.model' file to be loaded.")

    args = parser.parse_args()

    main(args.model, args.weights)
