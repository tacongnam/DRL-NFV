# runners/experiments.py
import numpy as np
import config
from runners.core import run_single_episode
from runners.visualization import plot_exp1_results, plot_exp2_results

def run_experiment_performance(env, agent, episodes=10, file_prefix=""):
    """
    Chạy Experiment 1: Performance Analysis
    Args:
        file_prefix: Tiền tố tên file ảnh (vd: "genai_" -> "genai_result_exp1.png")
    """
    print(f"\n{'='*80}")
    print(f"EXPERIMENT 1: Performance Analysis per SFC Type")
    print(f"Config: {episodes} episodes | Prefix: '{file_prefix}'")
    print(f"{'='*80}")
    
    total_completed = []
    total_dropped = []
    
    # 1. Chạy Loop
    for ep in range(episodes):
        # In tiến độ kiểu đè dòng (\r) để đỡ spam console -> Tăng tốc hiển thị
        print(f"  [Ep {ep+1}/{episodes}] Simulating...", end="\r", flush=True)
        
        env.reset(num_dcs=4) # Config cố định cho Exp 1
        run_single_episode(env, agent, epsilon=0.0, training_mode=False)
        
        total_completed.extend(env.sfc_manager.completed_history)
        total_dropped.extend(env.sfc_manager.dropped_history)
    
    print(f"  [Ep {episodes}/{episodes}] Completed!        ")
    print("\nProcessing results...")
    
    # 2. Phân tích kết quả
    sfc_types = config.SFC_TYPES
    acc_ratios = []
    e2e_delays = []
    
    for sfc_type in sfc_types:
        completed = [r for r in total_completed if r.type == sfc_type]
        dropped = [r for r in total_dropped if r.type == sfc_type]
        total = len(completed) + len(dropped)
        
        ar = (len(completed) / total * 100) if total > 0 else 0.0
        avg_delay = np.mean([r.get_total_e2e_delay() for r in completed]) if completed else 0.0
        
        acc_ratios.append(ar)
        e2e_delays.append(avg_delay)
        
        print(f"  {sfc_type:15s}: AR={ar:6.2f}%  |  E2E Delay={avg_delay:6.2f} ms")
    
    # 3. Vẽ biểu đồ
    save_path = f'fig/{file_prefix}result_exp1.png'
    plot_exp1_results(sfc_types, acc_ratios, e2e_delays, save_path=save_path)

def run_experiment_scalability(env, agent, episodes=10, file_prefix=""):
    """Chạy Experiment 2: Reconfigurability"""
    print(f"\n{'='*80}")
    print(f"EXPERIMENT 2: Reconfigurability & Scalability")
    print(f"{'='*80}")
    
    dc_counts = config.TEST_FIG3_DCS
    exp2_delays = []
    exp2_resources = []
    
    for n_dc in dc_counts:
        print(f"\n[Config: {n_dc} DCs]")
        current_completed = []
        cpu_usages = []
        
        for ep in range(episodes):
            print(f"  Running Ep {ep+1}/{episodes}...", end="\r", flush=True)
            env.reset(num_dcs=n_dc)
            
            # Custom loop để đo CPU usage
            state, _ = env._get_obs(), {}
            done = False
            
            while not done:
                mask = env._get_valid_actions_mask()
                action = agent.get_action(state, epsilon=0.0, valid_actions_mask=mask)
                state, _, done, _, _ = env.step(action)
                
                # Đo Resource mỗi step
                total_cap = n_dc * config.DC_CPU_CYCLES
                used_cap = sum(config.DC_CPU_CYCLES - dc.cpu for dc in env.dcs)
                cpu_usages.append((used_cap / total_cap * 100) if total_cap > 0 else 0)
            
            current_completed.extend(env.sfc_manager.completed_history)
        
        # Tổng hợp kết quả mỗi config
        avg_delay = np.mean([r.get_total_e2e_delay() for r in current_completed]) if current_completed else 0.0
        avg_cpu = np.mean(cpu_usages) if cpu_usages else 0.0
        
        exp2_delays.append(avg_delay)
        exp2_resources.append(avg_cpu)
        
        print(f"  → Done. Avg E2E: {avg_delay:.2f} ms | Avg CPU: {avg_cpu:.2f}%     ")
    
    # Vẽ biểu đồ
    save_path = f'fig/{file_prefix}result_exp2.png'
    plot_exp2_results(dc_counts, exp2_delays, exp2_resources, save_path=save_path)