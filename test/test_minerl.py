import gym
import minerl
import cv2

# Uncomment to see more logs of the MineRL launch
# import coloredlogs
# coloredlogs.install(logging.DEBUG)

height, width = 360, 640
fourcc = cv2.VideoWriter_fourcc(*'mp4v')
video = cv2.VideoWriter('video.mp4', fourcc, 20.0, (width, height))
video.set(cv2.VIDEOWRITER_PROP_QUALITY, 90)

env = gym.make("MineRLBasaltBuildVillageHouse-v0")
obs = env.reset()

for i in range(600):
    ac = env.action_space.noop()
    # Spin around to see what is around us
    ac["camera"] = [0, 3]
    if i % 20 == 0:
        ac['chat'] = f'/me is exploring the world. {i//20}'
    if i % 1200 == 0:
        ac['chat'] = '/effect give @p minecraft:night_vision 3600 1 false'
        
    obs, reward, done, info = env.step(ac)
    img = obs['pov']
    video.write(cv2.cvtColor(img, cv2.COLOR_RGB2BGR))
    print(i)
    # env.render()
video.release()
env.close()