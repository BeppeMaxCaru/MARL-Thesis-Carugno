    obs = env.reset()
    print(obs)
    print("ok")

    def create_env(args):
        return PettingZooEnv(PettingZooGraphEnv())