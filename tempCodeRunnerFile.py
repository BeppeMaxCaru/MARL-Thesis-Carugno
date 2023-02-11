ray.init()
algo = ppo.PPO(env=gymGraphEnv.GymGraphEnv, config={"env_config": {"render_mode": None, "size": 10}})
