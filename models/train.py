import math
import copy
import numpy as np
import config
from env.env import Env
from models.model import ReplayBuffer, VGAENetwork, HighLevelAgent, LowLevelAgent
from strategy.hrl import HRL_VGAE_Strategy

def train_hrl_vgae(env: Env):
    
    # 1. KHỞI TẠO (INITIALIZATION)
    
    buffer_HL = ReplayBuffer(capacity=5000)
    buffer_LL = ReplayBuffer(capacity=10000)
    buffer_Graph = ReplayBuffer(capacity=2000)
    
    vgae_net = VGAENetwork()
    hl_agent = HighLevelAgent(gamma=0.95)
    ll_agent = LowLevelAgent(gamma=0.95)
    
    strategy = HRL_VGAE_Strategy(env, vgae_net, ll_agent)
    env.set_strategy(strategy)
    
    # Siêu tham số (Slide 19)
    EPISODES = 300
    BATCH_SIZE = 32
    MAX_TIME_MS = 70.0 # Thời gian sống tối đa của mạng theo Slide
    total_steps = 0
    max_total_steps = EPISODES * (MAX_TIME_MS / config.TIMESTEP)
    
    
    # 2. VÒNG LẶP HUẤN LUYỆN CHÍNH
    
    for episode in range(1, EPISODES + 1):
        env.reset()
        
        # Load tất cả các request theo thứ tự thời gian vào waitlist tổng
        waitlist = sorted(env.requests, key=lambda r: r.arrival_time)
        queue =[]
        
        t_step = 0
        done = False
        
        while not done:
            env.t = t_step * config.TIMESTEP
            
            # Cập nhật Queue: Thêm request đến tại thời điểm t
            while len(waitlist) > 0 and waitlist[0].arrival_time <= env.t:
                queue.append(waitlist.pop(0))
                
            # Cập nhật Queue: Bỏ các request đã quá hạn (Drop)
            queue =[sfc for sfc in queue if env.t <= sfc.end_time]
            
            # Kiểm tra kết thúc Episode
            if len(queue) == 0 and len(waitlist) == 0:
                done = True
                break

            total_steps += 1
            
            # Epsilon Decay (Công thức Slide 19)
            decay_rate = (-total_steps * 3) / max_total_steps
            epsilon = 0.01 + (1.0 - 0.01) * math.exp(decay_rate)
            
            
            # BƯỚC 1: STATE REPRESENTATION
            
            t_end_eval = int(env.t / config.TIMESTEP) + 10 
            X, A, dc_mapping = strategy.build_dc_full_graph(int(env.t / config.TIMESTEP), t_end_eval)
            Z_t = vgae_net.encode(X, A)
            
            if len(queue) > 0:
                # Trích xuất State cho HL Agent
                Z_mean_t = np.mean(Z_t, axis=0, keepdims=True)
                sfc_feats_t = hl_agent.extract_sfc_features(queue)
                
                
                # BƯỚC 2: HIGH-LEVEL FORWARD (Chọn SFC)
                
                sfc_idx = hl_agent.act(Z_t, queue, epsilon)
                selected_sfc = queue.pop(sfc_idx)
                
                # Backup môi trường trước khi LL đặt VNF
                S_backup = copy.deepcopy(env.network)
                
                
                # BƯỚC 3 & BƯỚC 4: LOW-LEVEL FORWARD & REWARDS
                
                plan = strategy.get_placement(selected_sfc, env.t, Z_t, dc_mapping, epsilon)
                success, rewards, score = env.step(plan)
                
                if success:
                    # Slide 15 & 17: Tính thưởng thành công
                    time_rem = selected_sfc.end_time - env.t
                    tMax = selected_sfc.request.delay_max
                    
                    # rewards[1] chính là -cost trả về từ env.step
                    R_HL = [1.0, rewards[1]] #[BaseAR, -TotalCost]
                    R_LL = 1.0 + 0.5 * max(0, time_rem / tMax) + rewards[1]
                else:
                    # Slide 18: Rollback & Waitlist
                    env.network = S_backup
                    R_HL =[-1.0, 0.0]
                    R_LL = -1.0
                    
                    # Nếu chưa quá hạn, trả lại vào queue để thử lại ở timeslot sau
                    if env.t < selected_sfc.end_time:
                        queue.append(selected_sfc)
                
                
                # BƯỚC 5: LƯU TRỮ REPLAY BUFFER
                
                # Lấy State của bước tiếp theo (Z_next)
                X_next, A_next, _ = strategy.build_dc_full_graph(int(env.t / config.TIMESTEP), t_end_eval)
                Z_next = vgae_net.encode(X_next, A_next)
                
                # 5.1 Push to HL Buffer
                Z_mean_next = np.mean(Z_next, axis=0, keepdims=True)
                sfc_feats_next = hl_agent.extract_sfc_features(queue) # Queue hiện tại đại diện cho state next
                buffer_HL.push((Z_mean_t, sfc_feats_t, sfc_idx, R_HL, Z_mean_next, sfc_feats_next, done))
                
                # 5.2 Push to LL Buffer (Lấy từ trajectory sinh ra trong get_placement)
                for i, step_log in enumerate(strategy.last_ll_trajectory):
                    step_z = step_log['Z_t']
                    step_vnf = np.array([step_log['vnf_feat']])
                    step_action = step_log['action_idx']
                    
                    # Để đơn giản hóa trong RL, bước Next_Mask của VNF lấy là [] nếu là VNF cuối cùng
                    next_valid_mask =[]
                    if i + 1 < len(strategy.last_ll_trajectory):
                        next_valid_mask = strategy.last_ll_trajectory[i+1]['valid_mask']
                        
                    buffer_LL.push((step_z, step_vnf, step_action, R_LL, Z_next, next_valid_mask, done))
                
            # Lưu đồ thị định kỳ cho GenAI
            if total_steps % 100 == 0:
                buffer_Graph.push((X, A))
            
            
            # BƯỚC 6: TỐI ƯU MÔ HÌNH (BACKPROPAGATION)
            
            if total_steps % 100 == 0 and len(buffer_Graph) > BATCH_SIZE:
                vgae_net.train(buffer_Graph, epochs=1)
                
            if len(buffer_LL) > BATCH_SIZE:
                ll_agent.train(buffer_LL, BATCH_SIZE)
                
            if len(buffer_HL) > BATCH_SIZE:
                hl_agent.train(buffer_HL, BATCH_SIZE)
                
            # Cập nhật Target Networks
            if total_steps % 500 == 0:
                ll_agent.update_target_network()
                hl_agent.update_target_network()
                
            t_step += 1
            
        print(f"Episode {episode}/{EPISODES} completed. Total steps executed: {total_steps}, epsilon: {epsilon:.4f}")

if __name__ == "__main__":
    from main import load_from_dict
    import os
    os.environ['TF_ENABLE_ONEDNN_OPTS'] = '0'
    os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2' # Ẩn warning rác của TensorFlow
    
    # Chạy mô phỏng huấn luyện
    env = load_from_dict("data/test.json")
    train_hrl_vgae(env)