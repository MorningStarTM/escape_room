import os
import time
import numpy as np
from datetime import datetime
import torch
import torch.nn as nn
from torch.distributions import MultivariateNormal
from torch.distributions import Categorical
from src.escape_room.utils.logger import logger



class RolloutBuffer:
    def __init__(self):
        self.actions = []
        self.states = []
        self.logprobs = []
        self.rewards = []
        self.state_values = []
        self.is_terminals = []
    
    def clear(self):
        del self.actions[:]
        del self.states[:]
        del self.logprobs[:]
        del self.rewards[:]
        del self.state_values[:]
        del self.is_terminals[:]

    def __len__(self):
        return len(self.rewards)
    



class ActorCritic(nn.Module):
    def __init__(self, state_dim, action_dim):
        super(ActorCritic, self).__init__()
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
                
        self.actor = nn.Sequential(
                        nn.Linear(state_dim, 512),
                        nn.Tanh(),
                        nn.Linear(512, 256),
                        nn.Tanh(),
                        nn.Linear(256, 64),
                        nn.Tanh(),
                        nn.Linear(64, 64),
                        nn.Tanh(),
                        nn.Linear(64, action_dim),
                        nn.Softmax(dim=-1)
                    )

        # critic
        self.critic = nn.Sequential(
                        nn.Linear(state_dim, 512),
                        nn.Tanh(),
                        nn.Linear(512, 256),
                        nn.Tanh(),
                        nn.Linear(256, 128),
                        nn.Tanh(),
                        nn.Linear(128, 64),
                        nn.Tanh(),
                        nn.Linear(64, 1)
                    )



    def forward(self):
        raise NotImplementedError
    
    def act(self, state):
        action_probs = self.actor(state)
        dist = Categorical(action_probs)

        action = dist.sample()
        action_logprob = dist.log_prob(action)
        state_val = self.critic(state)
        return action.detach(), action_logprob.detach(), state_val.detach()
    
    
    def evaluate(self, state, action):
        action_probs = self.actor(state)
        dist = Categorical(action_probs)
        action_logprobs = dist.log_prob(action)
        dist_entropy = dist.entropy()
        state_values = self.critic(state)        
        return action_logprobs, state_values, dist_entropy





class PPO:
    def __init__(self, state_dim, action_dim, config):
        self.config = config
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        

        self.gamma = self.config['gamma']
        self.eps_clip = self.config['eps_clip']
        self.K_epochs = self.config['K_epochs']
        
        self.buffer = RolloutBuffer()

        self.policy = ActorCritic(state_dim, action_dim).to(self.device)
        self.optimizer = torch.optim.Adam([
                        {'params': self.policy.actor.parameters(), 'lr': self.config['lr_actor']},
                        {'params': self.policy.critic.parameters(), 'lr': self.config['lr_critic']}
                    ])

        self.policy_old = ActorCritic(state_dim, action_dim).to(self.device)
        self.policy_old.load_state_dict(self.policy.state_dict())
        
        self.MseLoss = nn.MSELoss()


    def get_buffer_size(self):
        return len(self.buffer)

    def select_action(self, state):

        with torch.no_grad():
            state = torch.FloatTensor(state).to(self.device)
            action, action_logprob, state_val = self.policy_old.act(state)
            
            self.buffer.states.append(state)
            self.buffer.actions.append(action)
            self.buffer.logprobs.append(action_logprob)
            self.buffer.state_values.append(state_val)

            action = action.item()
            return action, action_logprob, state_val

    def update(self):
        # Monte Carlo estimate of returns
        rewards = []
        discounted_reward = 0
        for reward, is_terminal in zip(reversed(self.buffer.rewards), reversed(self.buffer.is_terminals)):
            if is_terminal:
                discounted_reward = 0
            discounted_reward = reward + (self.gamma * discounted_reward)
            rewards.insert(0, discounted_reward)
            
        # Normalizing the rewards
        rewards = torch.tensor(rewards, dtype=torch.float32).to(self.device)
        rewards = (rewards - rewards.mean()) / (rewards.std() + 1e-7)

        # convert list to tensor
        old_states = torch.squeeze(torch.stack(self.buffer.states, dim=0)).detach().to(self.device)
        old_actions = torch.squeeze(torch.stack(self.buffer.actions, dim=0)).detach().to(self.device)
        old_logprobs = torch.squeeze(torch.stack(self.buffer.logprobs, dim=0)).detach().to(self.device)
        old_state_values = torch.squeeze(torch.stack(self.buffer.state_values, dim=0)).detach().to(self.device)

        # calculate advantages
        #logger.info(f"rewards shape: {rewards.shape}, old_state_values shape: {old_state_values.shape}")
        advantages = rewards.detach() - old_state_values.detach()

        # Optimize policy for K epochs
        for _ in range(self.K_epochs):

            # Evaluating old actions and values
            logprobs, state_values, dist_entropy = self.policy.evaluate(old_states, old_actions)

            # match state_values tensor dimensions with rewards tensor
            state_values = torch.squeeze(state_values)
            
            # Finding the ratio (pi_theta / pi_theta__old)
            ratios = torch.exp(logprobs - old_logprobs.detach())

            # Finding Surrogate Loss  
            surr1 = ratios * advantages
            surr2 = torch.clamp(ratios, 1-self.eps_clip, 1+self.eps_clip) * advantages

            # final loss of clipped objective PPO
            loss = -torch.min(surr1, surr2) + 0.5 * self.MseLoss(state_values, rewards) - 0.01 * dist_entropy
            
            # take gradient step
            self.optimizer.zero_grad()
            loss.mean().backward()
            self.optimizer.step()
            
        # Copy new weights into old policy
        self.policy_old.load_state_dict(self.policy.state_dict())

        # clear buffer
        self.buffer.clear()
    
    def save(self, checkpoint_path, filename="ppo_checkpoint.pth"):
        torch.save(self.policy_old.state_dict(), os.path.join(checkpoint_path, filename))
   
    def load(self, checkpoint_path, filename="ppo_checkpoint.pth"):
        file = os.path.join(checkpoint_path, filename)
        self.policy_old.load_state_dict(torch.load(file, map_location=lambda storage, loc: storage))
        self.policy.load_state_dict(torch.load(file, map_location=lambda storage, loc: storage))






import os
import csv
from datetime import datetime

class Trainer:
    def __init__(self, env, config):
        self.env = env
        self.env_name = "EscapeRoom"
        self.config = config

        # agent
        self.agent = PPO(
            state_dim=config["state_dim"],
            action_dim=config["action_dim"],
            config=config
        )

        self.best_score = 0.0
        self.score_history = []
        self.episode_rewards = []   # total reward per episode
        self.step_rewards = []      # reward per step

        # -----------------------------
        # Common root folder for everything
        # -----------------------------
        self.root_dir = os.path.join("runs", self.env_name)
        os.makedirs(self.root_dir, exist_ok=True)

        # Make run folders: run_000, run_001, ...
        run_num = self._next_run_number(self.root_dir)
        self.run_name = f"run_{run_num:03d}"

        self.run_dir = os.path.join(self.root_dir, self.run_name)
        self.models_dir = os.path.join(self.run_dir, "models")
        self.logs_dir = os.path.join(self.run_dir, "model_logs")
        self.rewards_dir = os.path.join(self.run_dir, "rewards")

        for d in (self.run_dir, self.models_dir, self.logs_dir, self.rewards_dir):
            os.makedirs(d, exist_ok=True)

        # log file path (inside model_logs/)
        self.log_f_name = os.path.join(
            self.logs_dir, f"PPO_{self.env_name}_log_{run_num:03d}.csv"
        )

        # reward file path (inside rewards/)
        self.reward_f_name = os.path.join(
            self.rewards_dir, f"PPO_{self.env_name}_rewards_{run_num:03d}.csv"
        )

        logger.info(f"Run: {self.run_name}")
        logger.info(f"Run folder: {self.run_dir}")
        logger.info(f"Logging at: {self.log_f_name}")
        logger.info(f"Rewards at: {self.reward_f_name}")

        # (optional) write CSV headers once
        self._init_csv(self.log_f_name, header=["episode", "timestep", "reward", "done"])
        self._init_csv(self.reward_f_name, header=["episode", "episode_reward"])

    @staticmethod
    def _next_run_number(root_dir: str) -> int:
        """
        Counts existing run_* folders and returns the next index.
        """
        if not os.path.isdir(root_dir):
            return 0
        existing = [
            name for name in os.listdir(root_dir)
            if os.path.isdir(os.path.join(root_dir, name)) and name.startswith("run_")
        ]
        nums = []
        for name in existing:
            try:
                nums.append(int(name.split("_")[1]))
            except Exception:
                pass
        return (max(nums) + 1) if nums else 0

    @staticmethod
    def _init_csv(path: str, header):
        """
        Create CSV with header if file doesn't exist or is empty.
        """
        need_header = (not os.path.exists(path)) or (os.path.getsize(path) == 0)
        if need_header:
            with open(path, "w", newline="", encoding="utf-8") as f:
                writer = csv.writer(f)
                writer.writerow(header)



    def train(self):
        start_time = datetime.now().replace(microsecond=0)
        logger.info("Started training at (GMT) : ", start_time)

        logger.info("============================================================================================")

        # logging file
        log_f = open(self.log_f_name,"w+")
        log_f.write('episode,timestep,reward\n')

        # printing and logging variables
        print_running_reward = 0
        print_running_episodes = 0

        log_running_reward = 0
        log_running_episodes = 0

        time_step = 0
        i_episode = 0
        
        while time_step <= self.config['max_training_timesteps']:

            state, _ = self.env.reset()
            current_ep_reward = 0

            for t in range(1, self.config['max_ep_len']+1):

                # select action with policy

                action, *_ = self.agent.select_action(state)
                state, reward, done, _, _ = self.env.step(action)
                self.step_rewards.append(reward)

                # saving reward and is_terminals
                self.agent.buffer.rewards.append(reward)
                self.agent.buffer.is_terminals.append(done)

                time_step +=1
                current_ep_reward += reward

                # update PPO agent
                if time_step % self.config['update_timestep'] == 0:
                    self.agent.update()


                # log in logging file
                if time_step % self.config['log_freq'] == 0:

                    # log average reward till last episode
                    log_avg_reward = log_running_reward / log_running_episodes
                    log_avg_reward = round(log_avg_reward, 4)

                    log_f.write('{},{},{}\n'.format(i_episode, time_step, log_avg_reward))
                    log_f.flush()

                    log_running_reward = 0
                    log_running_episodes = 0

                # printing average reward
                if time_step % self.config['print_freq'] == 0:

                    # print average reward till last episode
                    print_avg_reward = print_running_reward / print_running_episodes
                    print_avg_reward = round(print_avg_reward, 2)

                    logger.info("Episode : {} \t\t Timestep : {} \t\t Average Reward : {}".format(i_episode, time_step, print_avg_reward))

                    print_running_reward = 0
                    print_running_episodes = 0

                # save model weights
                if time_step % self.config['save_model_freq'] == 0:
                    logger.info("--------------------------------------------------------------------------------------------")
                    logger.info("saving model at : " + self.directory)
                    self.agent.save(self.directory)
                    logger.info("model saved")
                    logger.info("Elapsed Time  : ", datetime.now().replace(microsecond=0) - start_time)
                    logger.info("--------------------------------------------------------------------------------------------")

                # break; if the episode is over
                if done:
                    break
            
            self.episode_rewards.append(current_ep_reward)  
            print_running_reward += current_ep_reward
            print_running_episodes += 1

            log_running_reward += current_ep_reward
            log_running_episodes += 1

            i_episode += 1

        log_f.close()
        self.env.close()

        # print total training time
        logger.info("============================================================================================")
        end_time = datetime.now().replace(microsecond=0)
        logger.info("Started training at (GMT) : ", start_time)
        logger.info("Finished training at (GMT) : ", end_time)
        logger.info("Total training time  : ", end_time - start_time)
        logger.info("============================================================================================")

        np.save(os.path.join(self.reward_folder, f"ppo_{self.env_name}_step_rewards.npy"), np.array(self.step_rewards))
        np.save(os.path.join(self.reward_folder, f"ppo_{self.env_name}_episode_rewards.npy"), np.array(self.episode_rewards))
        logger.info(f"Saved step_rewards and episode_rewards to {self.log_dir}")
