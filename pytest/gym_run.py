import gym

env = gym.make('hpsim-clz-v0')
env.reset()

for i in range(10):
  print '----- trial ', i , '-----'
  state, reward, done, _ = env.step(env.action_space.sample())
  env.render() # output state (beam survival rate)
  print reward # reward = sum(state)

env.reset()
