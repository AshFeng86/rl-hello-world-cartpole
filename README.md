# Issac Lab - Training Cartpole with PPO on Cloud GPU

Rung 3 of my CV -> robotics learning ladder. Goal: run NVIDIA Isaac Lab on a cloud GPU, train a policy with PPO, and actually understand what's happening - not just copy-paste commands.

## What I built
Trained a PPO policy on Isaac-Cartpole-v0 with 4096 parallel environments on a cloud L40S GPU. ~9.8 million timesteps in ~37 sec. Recorded the trained policy to video.

### Before (no policy)
poles flailing from random join efforts
### After (trained policy)
all poles balanced
<img width="1980" height="1103" alt="Screenshot 2026-07-07 at 8 46 54 PM" src="https://github.com/user-attachments/assets/e0261ace-d240-4530-8d7b-607df2dc9d49" />
<img width="1979" height="1114" alt="Screenshot 2026-07-07 at 8 47 17 PM" src="https://github.com/user-attachments/assets/d8ed436c-2939-4bc1-83ef-566c411c5220" />

## Concepts in my own words
- Policy: the neural network that decides. Give it a situation, it outputs one action. The dog's brain, not the dog's moves.
- Action: one decision the policy outputs (push left / push right).
- Reward: the score signal - and it's a design choice, not a fixed rule. In Gymnasium CartPole it's +1 per frame alive. In Isaac Lab it's composed: alive bonus - pole angle penalty - cart velocity penalty - pole velocity penalty. You can see each term in the training logs.
- Episode one attempt. Birth to death of one try.
- PPO: the training method. Improves the policy in small capped steps - progressive overload for neural networks, because one big jump wrecks what you've built.
- The loop: see (4 numbers) -> think (32->32) -> push (1 number) -> world moves -> repeat
    From the Simulator's view this is Write -> Step -> Read
    From the policy's view it's Read -> decide -> Write. Same wheel, different starting point.
- The policy's shape: 4->32->32->1. Four inputs (cart position, cart velocity, pole angle, pole velocity), two hidden layers, one output (push force). Training never changes the shape - it only tunes the weights on the connections. ~1,300 dials
- Vectorization: my Mac trained 1 CartPole. The GPU ran 4096 at once. Same task, 200x the experience per second. This is why Isaac Lab exists.

##Commands I keep using
every session starts with
cd /workspace/isaaclab

run any script: pick how to see it
./isaaclab.sh -p <script.py> [args] --headless        # no visuals, faster
./isaaclab.sh -p <script.py> [args] --livestream 2    # watch live in /viewer/

train (150 iterations, ~37s on L40S)
./isaaclab.sh -p scripts/reinforcement_learning/rsl_rl/train.py --task Isaac-Cartpole-v0 --num_envs 4096 --headless

record the trained policy to mp4 (more reliable than live streaming)
./isaaclab.sh -p scripts/reinforcement_learning/rsl_rl/play.py --task Isaac-Cartpole-v0 --num_envs 32 --headless --video --video_length 300

##Things that broke -> fixes
Cloud GPU marketplaces are less reliable than I expected. What I learned the hard way:
- AWS us-east-2 stalled twice at provisioning (empty logs 15+min)
- One deploy click = one billed machine
- Nebius and GCP hosts: training worked but rendering didn't. Compute working =/= graphics working - they're separate driver paths.
- Crusoe was the only provider where everything worked, at half the AWS price.
- Live streaming (WebRTC) is the flakiest link.


##Ladder context
- Rung 1: YOLO11n emptys-shelf detector (Project 1, 1.5)
- Rung 2: real-time gap tracking + VLM reasoning (Project 2, 2.5)
- Rung 3 pre-step: PPO on CartPole locally (rl-hellow-world-cartpole)
- Rung 3, Phase 1-3: This repo
- Rung 4 next: change one reward term, predict the behavior change, retrain, compare
