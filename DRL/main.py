import numpy as np
import tensorflow as tf
from env.sfc_environment import SFCEnvironment
from env.dqn_model import DQNModel, ReplayMemory
from config import *
from utils import generate_sfc_requests
import matplotlib.pyplot as plt

class SFCTrainer:
    def __init__(self, num_dcs=4):
        self.env = SFCEnvironment(num_dcs=num_dcs)
        self.model = DQNModel(len(VNF_TYPES), len(SFC_TYPES))
        self.memory = ReplayMemory(TRAINING_CONFIG['memory_size'])
        
        self.epsilon = TRAINING_CONFIG['epsilon_start']
        self.epsilon_end = TRAINING_CONFIG['epsilon_end']
        self.epsilon_decay = TRAINING_CONFIG['epsilon_decay']
        
        self.metrics = {
            'acceptance_ratio': [],
            'avg_reward': [],
            'loss': []
        }
    
    def select_action(self, state, training=True):
        if training and np.random.random() < self.epsilon:
            return np.random.randint(0, self.model.num_actions)
        else:
            q_values = self.model.predict(state)
            return np.argmax(q_values)
    
    def train_episode(self):
        state, _ = self.env.reset()
        
        num_dcs = np.random.randint(2, 7)
        self.env = SFCEnvironment(num_dcs=num_dcs)
        state, _ = self.env.reset()
        
        requests = generate_sfc_requests()
        self.env.add_requests(requests)
        
        episode_reward = 0
        step_count = 0
        
        while True:
            action = self.select_action(state, training=True)
            next_state, reward, done, _, _ = self.env.step(action)
            
            self.memory.push(state, action, reward, next_state, done)
            
            episode_reward += reward
            state = next_state
            step_count += 1
            
            if step_count % TRAINING_CONFIG['request_generation_interval'] == 0:
                new_requests = generate_sfc_requests()
                self.env.add_requests(new_requests)
            
            if done or step_count >= TRAINING_CONFIG['max_actions_per_step']:
                break
        
        return episode_reward, self.env.get_acceptance_ratio()
    
    def train(self):
        update_count = 0
        best_acc_ratio = 0
        
        for update in range(TRAINING_CONFIG['num_updates']):
            episode_rewards = []
            acceptance_ratios = []
            
            for episode in range(TRAINING_CONFIG['episodes_per_update']):
                reward, acc_ratio = self.train_episode()
                episode_rewards.append(reward)
                acceptance_ratios.append(acc_ratio)
                
                if (episode + 1) % 5 == 0:
                    print(f"Update {update+1}/{TRAINING_CONFIG['num_updates']}, "
                          f"Episode {episode+1}/{TRAINING_CONFIG['episodes_per_update']}, "
                          f"Reward: {reward:.2f}, AccRatio: {acc_ratio:.3f}, "
                          f"Epsilon: {self.epsilon:.3f}")
            
            if len(self.memory) >= TRAINING_CONFIG['batch_size']:
                losses = []
                num_train_steps = max(10, len(self.memory) // TRAINING_CONFIG['batch_size'])
                
                for _ in range(min(num_train_steps, 50)):
                    states, actions, rewards, next_states, dones = \
                        self.memory.sample(TRAINING_CONFIG['batch_size'])
                    
                    loss = self.model.train_on_batch(
                        states, actions, 
                        np.array(rewards), 
                        next_states, 
                        np.array(dones)
                    )
                    losses.append(loss)
                
                avg_loss = np.mean(losses)
                self.metrics['loss'].append(avg_loss)
                
                if (update + 1) % 10 == 0:
                    print(f"  → Avg Loss: {avg_loss:.4f}")
            
            avg_acc = np.mean(acceptance_ratios)
            self.metrics['acceptance_ratio'].append(avg_acc)
            self.metrics['avg_reward'].append(np.mean(episode_rewards))
            
            if avg_acc > best_acc_ratio:
                best_acc_ratio = avg_acc
                self.model.save('checkpoints/best_model.weights.h5')
                print(f"  ★ New best model! AccRatio: {best_acc_ratio:.3f}")
            
            if self.epsilon > self.epsilon_end:
                self.epsilon *= self.epsilon_decay
            
            if (update + 1) % TRAINING_CONFIG['target_update_freq'] == 0:
                self.model.update_target_model()
                print(f"  → Target network updated at update {update+1}")
            
            if (update + 1) % 50 == 0:
                self.model.save(f'checkpoints/model_update_{update+1}.weights.h5')
                self.plot_metrics()
                print(f"\n{'='*60}")
                print(f"CHECKPOINT at Update {update+1}")
                print(f"Best AccRatio so far: {best_acc_ratio:.3f}")
                print(f"Current Avg AccRatio: {avg_acc:.3f}")
                print(f"{'='*60}\n")
        
        print("\n" + "="*60)
        print("TRAINING COMPLETED!")
        print(f"Best Acceptance Ratio: {best_acc_ratio:.3f}")
        print("="*60)
        self.model.save('checkpoints/final_model.weights.h5')
    
    def plot_metrics(self):
        fig, axes = plt.subplots(1, 3, figsize=(15, 4))
        
        axes[0].plot(self.metrics['acceptance_ratio'])
        axes[0].set_title('Acceptance Ratio')
        axes[0].set_xlabel('Update')
        axes[0].set_ylabel('Ratio')
        axes[0].grid(True)
        
        axes[1].plot(self.metrics['avg_reward'])
        axes[1].set_title('Average Reward')
        axes[1].set_xlabel('Update')
        axes[1].set_ylabel('Reward')
        axes[1].grid(True)
        
        if self.metrics['loss']:
            axes[2].plot(self.metrics['loss'])
            axes[2].set_title('Training Loss')
            axes[2].set_xlabel('Update')
            axes[2].set_ylabel('Loss')
            axes[2].grid(True)
        
        plt.tight_layout()
        plt.savefig('training_metrics.png')
        plt.close()
    
    def evaluate(self, num_episodes=10, num_dcs=4):
        self.env = SFCEnvironment(num_dcs=num_dcs)
        acceptance_ratios = []
        
        for ep in range(num_episodes):
            state, _ = self.env.reset()
            
            for i in range(3):
                requests = generate_sfc_requests()
                self.env.add_requests(requests)
            
            step_count = 0
            while True:
                action = self.select_action(state, training=False)
                next_state, _, done, _, _ = self.env.step(action)
                state = next_state
                step_count += 1
                
                if done or step_count >= TRAINING_CONFIG['max_actions_per_step']:
                    break
            
            acc_ratio = self.env.get_acceptance_ratio()
            acceptance_ratios.append(acc_ratio)
            print(f"Eval Episode {ep+1}: Acceptance Ratio = {acc_ratio:.3f}")
        
        avg_acc_ratio = np.mean(acceptance_ratios)
        print(f"\nAverage Acceptance Ratio: {avg_acc_ratio:.3f}")
        return avg_acc_ratio

def main():
    import os
    os.makedirs('checkpoints', exist_ok=True)
    
    tf.random.set_seed(42)
    np.random.seed(42)
    
    trainer = SFCTrainer(num_dcs=4)
    
    print("Starting training...")
    trainer.train()
    
    print("\nEvaluating trained model...")
    trainer.evaluate(num_episodes=20, num_dcs=4)

if __name__ == '__main__':
    main()