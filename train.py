import numpy as np
from stable_baselines3 import PPO
from stable_baselines3.common.env_checker import check_env
from stable_baselines3.common.vec_env import DummyVecEnv, VecNormalize
from stable_baselines3.common.callbacks import EvalCallback
from stable_baselines3.common.monitor import Monitor

from rl_implementation import DielsAlderSelectivityEnv

SEED = 42
N_ENVS = 4
TOTAL_TIMESTEPS = 1_000_000


def make_env(seed_offset=0):
    def _init():
        env = DielsAlderSelectivityEnv(seed=SEED + seed_offset)
        return Monitor(env)
    return _init


# Training env
env = DummyVecEnv([make_env(i) for i in range(N_ENVS)])
env = VecNormalize(env, norm_obs=True, norm_reward=True)

check_env(DielsAlderSelectivityEnv(seed=SEED), warn=True)


# Eval env
eval_env = DummyVecEnv([make_env(100)])
eval_env = VecNormalize(eval_env, norm_obs=True, norm_reward=False, training=False)

eval_callback = EvalCallback(
    eval_env,
    best_model_save_path="./logs/",
    log_path="./logs/",
    eval_freq=10_000,
    deterministic=True
)


model = PPO(
    "MlpPolicy",
    env,
    seed=SEED,
    verbose=1,
    learning_rate=1e-4,
    n_steps=4096,
    batch_size=128,
    gamma=0.98,
    ent_coef=0.01,
    policy_kwargs=dict(net_arch=[256, 256])
)


print("Training PPO agent...")
model.learn(
    total_timesteps=TOTAL_TIMESTEPS,
    callback=eval_callback
)


model.save("ppo_da_agent")
env.save("vecnormalize.pkl")
print("Saved model and VecNormalize stats")


# ------------------------
# Test rollout (FIXED)
# ------------------------
test_env = DummyVecEnv([make_env(999)])
test_env = VecNormalize.load("vecnormalize.pkl", test_env)
test_env.training = False
test_env.norm_reward = False

model = PPO.load("ppo_da_agent", env=test_env)

obs = test_env.reset()

print("\nTest rollout:\n")
for step in range(20):
    action, _ = model.predict(obs, deterministic=True)
    obs, reward, done, info = test_env.step(action)

    info = info[0]

    print(
        f"Step {step + 1:02d} | "
        f"Action: {action} | "
        f"Reward: {reward[0]:.3f} | "
        f"Endo: {info['endo']} | "
        f"Regio: {info['regio_correct']}"
    )

    if done:
        break
