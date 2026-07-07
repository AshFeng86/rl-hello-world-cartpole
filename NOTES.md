# Commands I keep using:
- ./isaaclab.sh -p <path/to/script.py> [script args] --headless        # no visuals, faster
- ./isaaclab.sh -p <path/to/script.py> [script args] --livestream 2   # watch in /viewer/ tab
- ./isaaclab.sh -p scripts/tutorials/02_scene/create_scene.py --num_envs 32 --livestream 2
- cd /workspace/isaaclab
- ./isaaclab.sh -p scripts/tutorials/00_sim/spawn_prims.py --livestream 2  # smoke test (rendering check)
- ./isaaclab.sh -p scripts/reinforcement_learning/rsl_rl/train.py --task Isaac-Cartpole-v0 --num_envs 4096 --headless    # Train 150 iterations
- ./isaaclab.sh -p scripts/reinforcement_learning/rsl_rl/play.py --task Isaac-Cartpole-v0 --num_envs 32 --livestream 2   # Watch the trained policy

# Concept in my own words:
- policy: the neural network that decides. Give it a situation, it outputs one action. The dog's brain, not the dog's moves.
- reward: in cartpole, a frame that keeps the scene alive is a reward
- episode: one attempt, start to fail/finish

- Write / Step / Read:
-   Write: pushing one action into the scene
-   Step: the physics engine advances one tick
-   Read: understand the situation (outcome of the policy taken) in the scene

Whole process:

The policy (network) picks an action -> Write pushes it into the sim -> Step advances physics -> Read observes the result -> Repeat until the episode ends -> reward across the episode tells PPO how to improve the policy
