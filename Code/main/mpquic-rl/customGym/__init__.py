from gymnasium.envs.registration import register

register(
     id="NetworkEnv",
     entry_point="customGym.envs:NetworkEnv",
     max_episode_steps=9999999,
)