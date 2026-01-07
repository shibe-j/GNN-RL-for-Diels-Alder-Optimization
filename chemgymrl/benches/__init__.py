from gymnasium.envs.registration import register

register(
    id="DielsAlderReact-v0",
    entry_point="chemgymrl.benches.da_bench:DielsAlderReact_v0",
)
