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
        env = DielsAlderSelectivityEnv(seed=SEED + seed_offset, max_steps=20)
        return Monitor(env)
    return _init


# Training env
env = DummyVecEnv([make_env(i) for i in range(N_ENVS)])
env = VecNormalize(env, norm_obs=True, norm_reward=True)

print("Checking environment...")
check_env(DielsAlderSelectivityEnv(seed=SEED), warn=True)
print("Environment check passed!\n")


# Eval env
eval_env = DummyVecEnv([make_env(100)])
eval_env = VecNormalize(eval_env, norm_obs=True, norm_reward=False, training=False)

eval_callback = EvalCallback(
    eval_env,
    best_model_save_path="./logs/",
    log_path="./logs/",
    eval_freq=10_000,
    deterministic=True,
    verbose=1
)


# Improved PPO hyperparameters
model = PPO(
    "MlpPolicy",
    env,
    seed=SEED,
    verbose=1,
    learning_rate=3e-4,      # Higher learning rate
    n_steps=2048,            # Smaller for faster updates
    batch_size=64,           # Smaller batches
    gamma=0.95,              # Less future discounting
    ent_coef=0.05,           # More exploration
    clip_range=0.2,
    n_epochs=10,
    policy_kwargs=dict(
        net_arch=[128, 128]  # Simpler network
    )
)


print("=" * 60)
print("Training PPO agent with IMPROVED hyperparameters...")
print("=" * 60)
print(f"Total timesteps: {TOTAL_TIMESTEPS:,}")
print(f"Number of environments: {N_ENVS}")
print(f"Max episode length: 20 steps")
print(f"Learning rate: 3e-4")
print(f"Entropy coefficient: 0.05 (exploration)")
print("=" * 60)
print()

model.learn(
    total_timesteps=TOTAL_TIMESTEPS,
    callback=eval_callback,
    progress_bar=True
)


model.save("ppo_da_agent_improved")
env.save("vecnormalize_improved.pkl")
print("\n" + "=" * 60)
print("Saved model and VecNormalize stats")
print("=" * 60)


# ========================
# Detailed test rollout
# ========================
print("\n" + "=" * 60)
print("DETAILED TEST ROLLOUT")
print("=" * 60)

test_env = DummyVecEnv([make_env(999)])
test_env = VecNormalize.load("vecnormalize_improved.pkl", test_env)
test_env.training = False
test_env.norm_reward = False

model = PPO.load("ppo_da_agent_improved", env=test_env)

obs = test_env.reset()

action_names = {
    0: "Dienophile: C=CC#N",
    1: "Dienophile: CC=CC#N", 
    2: "Dienophile: C(C#N)=CC#N",
    3: "Dienophile: C(C(=O)N)=CC#N",
    4: "Solvent: hexane",
    5: "Solvent: DCM",
    6: "Solvent: acetonitrile",
    7: "Temp +10",
    8: "Temp -10"
}

print("\nStarting test episode...\n")

total_reward = 0
endo_count = 0
regio_count = 0

for step in range(20):
    action, _ = model.predict(obs, deterministic=True)
    obs, reward, done, info = test_env.step(action)

    info = info[0]
    action_int = int(action[0])
    
    total_reward += reward[0]
    endo_count += info['endo']
    regio_count += info['regio_correct']
    
    print(f"Step {step + 1:02d} | {action_names[action_int]:30s} | "
          f"R: {reward[0]:6.3f} | "
          f"Endo: {info['endo']} | "
          f"Regio: {info['regio_correct']} | "
          f"ΔG: {info['deltaG']:5.2f} | "
          f"T: {info['temperature']:3d}°C")

    if done:
        break

print("\n" + "=" * 60)
print("EPISODE SUMMARY")
print("=" * 60)
print(f"Total reward:        {total_reward:.3f}")
print(f"Endo selectivity:    {endo_count}/20 ({100*endo_count/20:.0f}%)")
print(f"Regioselectivity:    {regio_count}/20 ({100*regio_count/20:.0f}%)")
print(f"Final temperature:   {info['temperature']}°C")
print(f"Final solvent:       {info['solvent']}")
print(f"Final dienophile:    {info['dienophile']}")
print("=" * 60)

print("\n✅ Training complete! The agent should now show:")
print("   - Positive total rewards (vs negative before)")
print("   - High endo selectivity (acetonitrile/DCM preferred)")
print("   - Good regioselectivity")
print("   - Strategic use of strong EWG dienophiles")