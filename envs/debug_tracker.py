from collections import defaultdict
import sys

class DebugTracker:
    def __init__(self):
        self.reset()
    
    def reset(self):
        self.action_counts = defaultdict(int)
        self.reward_counts = defaultdict(int)
        self.reward_sum = defaultdict(float)
        self.invalid_reasons = defaultdict(int)
        self.total_steps = 0
        self.total_reward = 0.0
    
    def track_action(self, action_type, reward, reason=None):
        self.action_counts[action_type] += 1
        self.reward_counts[reward] += 1
        self.reward_sum[action_type] += reward
        self.total_steps += 1
        self.total_reward += reward
        
        if reward == -1.0 and reason:
            self.invalid_reasons[reason] += 1
    
    def print_summary(self, episode):
        output = []
        output.append(f"\n{'='*60}")
        output.append(f"Episode {episode} Debug Summary")
        output.append(f"{'='*60}")
        output.append(f"Total Steps: {self.total_steps}, Total Reward: {self.total_reward:.1f}")
        
        output.append(f"\nAction Distribution:")
        for action_type, count in sorted(self.action_counts.items()):
            pct = count / self.total_steps * 100
            avg_r = self.reward_sum[action_type] / count
            output.append(f"  {action_type:12s}: {count:5d} ({pct:5.1f}%) | Avg R: {avg_r:6.2f}")
        
        output.append(f"\nReward Distribution:")
        for reward, count in sorted(self.reward_counts.items(), reverse=True):
            pct = count / self.total_steps * 100
            output.append(f"  R={reward:5.1f}: {count:5d} ({pct:5.1f}%)")
        
        if self.invalid_reasons:
            output.append(f"\nInvalid Action Reasons:")
            for reason, count in sorted(self.invalid_reasons.items(), key=lambda x: x[1], reverse=True):
                pct = count / self.reward_counts[-1.0] * 100 if self.reward_counts[-1.0] > 0 else 0
                output.append(f"  {reason:30s}: {count:5d} ({pct:5.1f}%)")
        
        output.append(f"{'='*60}\n")
        
        result = '\n'.join(output)
        print(result, flush=True)
        sys.stdout.flush()