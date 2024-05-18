import gym
import minerl

# Uncomment to see more logs of the MineRL launch
# import coloredlogs
# coloredlogs.install(logging.DEBUG)

# python -m minerl.interactor 55555

env = gym.make("MineRLBasaltBuildVillageHouse-v0")
env.make_interactive(port=55555, realtime=True)
obs = env.reset()

for i in range(12000):
    ac = env.action_space.noop()
    # Spin around to see what is around us
    ac["camera"] = [0, 3]

    if i % 20 == 0:
        # ac['chat'] = f'hello {i//20}'
        ac['chat'] = f'/me is exploring the world. {i//20}'
    
    # if i == 0:
    #     ac['chat'] = '/time set night'
    if i == 1:
        ac['chat'] = '/effect give @p minecraft:night_vision 3600 1 false'

    obs, reward, done, info = env.step(ac)
    env.render()
env.close()