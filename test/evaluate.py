import sys
import os
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from argparse import ArgumentParser
import pickle
import cv2
import numpy as np
from pprint import pprint
import datetime

from minerl.herobraine.env_specs.human_survival_specs import HumanSurvival

from agent import MineRLAgent, ENV_KWARGS

MAX_STEPS = 1000
NUM_EPISODES = 100

LOG_NAMES = ['acacia_log', 'birch_log', 'dark_oak_log', 
             'jungle_log', 'oak_log', 'spruce_log', 
             'stripped_acacia_log', 'stripped_birch_log', 'stripped_dark_oak_log',
             'stripped_jungle_log', 'stripped_oak_log', 'stripped_spruce_log']

# WOOD_NAMES = ['acacia_wood', 'birch_wood', 'dark_oak_wood', 
#              'jungle_wood', 'oak_wood', 'spruce_wood', 
#              'stripped_acacia_wood', 'stripped_birch_wood', 'stripped_dark_oak_wood',
#              'stripped_jungle_wood', 'stripped_oak_wood', 'stripped_spruce_wood']

PLANKS_NAMES = ['acacia_planks', 'birch_planks', 'crimson_planks',
                'dark_oak_planks', 'jungle_planks', 'oak_planks', 
                'spruce_planks', 'warped_planks']

COBBLESTONE_NAMES = ['cobblestone', 'infested_cobblestone', 'mossy_cobblestone']

GLASS_NAMES = ['black_stained_glass', 'blue_stained_glass', 'brown_stained_glass', 'cyan_stained_glass', 
               'glass', 'gray_stained_glass', 'green_stained_glass', 
               'light_blue_stained_glass', 'light_gray_stained_glass', 'lime_stained_glass', 
               'magenta_stained_glass', 'orange_stained_glass', 'pink_stained_glass', 
               'purple_stained_glass', 'red_stained_glass', 'white_stained_glass', 
               'yellow_stained_glass']

SINGLE_NAMES = ['crafting_table', 'wooden_pickaxe', 'stone_pickaxe',
                'iron_ore', 'furnace', 'iron_ingot', 'iron_pickaxe',
                'diamond', 'diamond_pickaxe', 'sand', 'torch']

ITEM_COUNT = {
    'log' : 0,              # multi names
    'planks' : 0,           # multi names
    'crafting_table' : 0,
    'wooden_pickaxe' : 0,
    'cobblestone' : 0,      # multi names
    'stone_pickaxe' : 0,
    'iron_ore' : 0,
    'furnace' : 0,
    'iron_ingot' : 0,
    'iron_pickaxe' : 0,
    'diamond' : 0,
    'diamond_pickaxe' : 0,
    'sand' : 0,
    'glass' : 0,            # multi names
    'torch' : 0
}

height, width = 360, 640
fourcc = cv2.VideoWriter_fourcc(*'mp4v')
video_dir = 'evaluate_videos'
# video_name = datetime.datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
# video = cv2.VideoWriter(f'{video_dir}\eval_{video_name}.mp4', fourcc, 20.0, (width, height))


def run_episode(agent, env, episode_idx):
    print(f"Running: episode {episode_idx}/{NUM_EPISODES}.")
    obs = env.reset()
    video_name = datetime.datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
    video = cv2.VideoWriter(f'{video_dir}/eval_episode_{episode_idx}_{video_name}.mp4', fourcc, 20.0, (width, height))
    item_count = ITEM_COUNT.copy()
    with open(f"{video_dir}/eval_episode_{episode_idx}_{video_name}.txt", "w") as f:
        for i in range(MAX_STEPS):
            minerl_action = agent.get_action(obs)
            obs, reward, done, info = env.step(minerl_action)
            inventory = obs['inventory']

            # count items
            for key in SINGLE_NAMES:
                item_count[key] = max(item_count[key], int(inventory[key]))
            
            log_count = sum([int(inventory[key]) for key in LOG_NAMES])
            item_count['log'] = max(item_count['log'], log_count)

            planks_count = sum([int(inventory[key]) for key in PLANKS_NAMES])
            item_count['planks'] = max(item_count['planks'], planks_count)

            cobblestone_count = sum([int(inventory[key]) for key in COBBLESTONE_NAMES])
            item_count['cobblestone'] = max(item_count['cobblestone'], cobblestone_count)

            glass_count = sum([int(inventory[key]) for key in GLASS_NAMES])
            item_count['glass'] = max(item_count['glass'], glass_count)

            # if i % 100 == 0:
            #     pprint(f'step {i}:\n{item_count}', stream=f)
            # env.render()
            img = obs['pov']
            video.write(cv2.cvtColor(img, cv2.COLOR_RGB2BGR))
        pprint(item_count, stream=f)
        video.release()


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
    # obs = env.reset()
    global video_name
    global video
    for i in range(NUM_EPISODES):
        run_episode(agent, env, i)


if __name__ == "__main__":
    parser = ArgumentParser("Run pretrained models on MineRL environment")

    parser.add_argument("--weights", type=str, required=True, help="Path to the '.weights' file to be loaded.")
    parser.add_argument("--model", type=str, required=True, help="Path to the '.model' file to be loaded.")

    args = parser.parse_args()

    main(args.model, args.weights)
