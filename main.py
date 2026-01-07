#!/usr/bin/env python
import sys, argparse, os
from envs import PrioritySelector, Observer, VAESelector
from runners import Runner
from agents import VAEAgent
import config, numpy as np

def train_dqn(args):
    print("Starting DQN training...")
    print(f"Data file: {args.data}")
    
    try:
        Runner.load_from(args.data)
        print("Data loaded successfully")
    except Exception as e:
        print(f"Error loading data: {e}")
        import traceback
        traceback.print_exc()
        return
    
    try:
        Runner.train_dqn(dc_selector=PrioritySelector(), save_prefix="", num_updates=args.updates)
    except Exception as e:
        print(f"Error during training: {e}")
        import traceback
        traceback.print_exc()

def train_vae_dqn(args):
    print("Starting VAE-DQN training...")
    Runner.load_from(args.data)
    
    vae_model = None
    if not args.skip_vae:
        print("\nStep 1/2: Training VAE")
        vae_model = Runner.collect_vae_data(num_episodes=args.vae_episode)
    else:
        print("\nLoading existing VAE model...")
        fake_env = Runner.create_env(dc_selector=PrioritySelector())
        server_dc = fake_env.get_first_dc()
        dc_state_dim = Observer.get_dc_state(server_dc, fake_env.sfc_manager, None).shape[0]

        vae_model = VAEAgent(state_dim=dc_state_dim, latent_dim=config.GENAI_LATENT_DIM)
        vae_model.load_weights("models/vae_model")

        norm_path = "models/vae_model_norm.npz"
        if os.path.exists(norm_path):
            data = np.load(norm_path)
            vae_model.set_normalization_params(float(data['mean']), float(data['std']))
        else:
            print(f"Warning: Normalization parameters not found at {norm_path}. "
                  "VAE predictions might be uncalibrated.")
            
    print("\nStep 2/2: Training DQN with VAE selector")
    selector = VAESelector(vae_model)
    Runner.train_dqn(selector, save_prefix="vae_", num_updates=args.updates)

def eval_dqn(args):
    print("Evaluating DQN...")
    Runner.load_from(args.data)
    
    from agents import DQNAgent
    from envs import PrioritySelector
    
    env = Runner.create_env(dc_selector=PrioritySelector())
    
    if hasattr(env.observation_space, 'spaces'):
        state_shapes = [s.shape for s in env.observation_space.spaces]
    else:
        state_shapes = [env.observation_space.shape]
    
    agent = DQNAgent(state_shapes, env.action_space.n)
    
    model_path = args.model if args.model else "models/best_model"
    agent.load(model_path)
    
    num_eps = args.episodes if args.episodes else config.TEST_EPISODES
    Runner.evaluate(agent, PrioritySelector(), num_eps, "dqn_")

def eval_vae_dqn(args):
    print("Evaluating VAE-DQN...")
    Runner.load_from(args.data)
    
    from agents import DQNAgent, VAEAgent
    from envs import VAESelector, PrioritySelector, Observer
    
    fake_env = Runner.create_env(dc_selector=PrioritySelector())
    server_dc = fake_env.get_first_dc()
    dc_state_dim = Observer.get_dc_state(server_dc, fake_env.sfc_manager, None).shape[0]
    
    vae_model = VAEAgent(state_dim=dc_state_dim, latent_dim=config.GENAI_LATENT_DIM)
    vae_path = args.model if args.model else "models/vae_model"
    vae_model.load_weights(vae_path)
    
    norm_path = f"{vae_path}_norm.npz"
    if os.path.exists(norm_path):
        data = np.load(norm_path)
        vae_model.set_normalization_params(float(data['mean']), float(data['std']))
    
    env = Runner.create_env(dc_selector=VAESelector(vae_model))
    
    if hasattr(env.observation_space, 'spaces'):
        state_shapes = [s.shape for s in env.observation_space.spaces]
    else:
        state_shapes = [env.observation_space.shape]
    
    agent = DQNAgent(state_shapes, env.action_space.n)
    dqn_path = args.model if args.model else "models/best_vae_model"
    agent.load(dqn_path)
    
    num_eps = args.episodes if args.episodes else config.TEST_EPISODES
    Runner.evaluate(agent, VAESelector(vae_model), num_eps, "vae_dqn_")

def eval_compare(args):
    print("Comparing DQN vs VAE-DQN...")
    Runner.load_from(args.data)
    
    from agents import DQNAgent, VAEAgent
    from envs import PrioritySelector, VAESelector, Observer
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    
    fake_env = Runner.create_env(dc_selector=PrioritySelector())
    if hasattr(fake_env.observation_space, 'spaces'):
        state_shapes = [s.shape for s in fake_env.observation_space.spaces]
    else:
        state_shapes = [fake_env.observation_space.shape]
    
    dqn_agent = DQNAgent(state_shapes, fake_env.action_space.n)
    dqn_path = args.dqn_model if args.dqn_model else "models/best_model"
    dqn_agent.load(dqn_path)
    
    server_dc = fake_env.get_first_dc()
    dc_state_dim = Observer.get_dc_state(server_dc, fake_env.sfc_manager, None).shape[0]
    vae_model = VAEAgent(state_dim=dc_state_dim, latent_dim=config.GENAI_LATENT_DIM)
    vae_path = args.vae_model if args.vae_model else "models/vae_model"
    vae_model.load_weights(vae_path)
    
    norm_path = f"{vae_path}_norm.npz"
    if os.path.exists(norm_path):
        data = np.load(norm_path)
        vae_model.set_normalization_params(float(data['mean']), float(data['std']))
    
    vae_dqn_agent = DQNAgent(state_shapes, fake_env.action_space.n)
    vae_dqn_path = args.vae_model if args.vae_model else "models/best_vae_model"
    vae_dqn_agent.load(vae_dqn_path)
    
    num_eps = args.episodes if args.episodes else config.TEST_EPISODES
    
    print(f"\n{'='*80}\nComparing DQN vs VAE-DQN: {num_eps} episodes\n{'='*80}")
    
    dqn_ars, dqn_delays = [], []
    vae_ars, vae_delays = [], []
    
    for ep in range(num_eps):
        print(f"\rEpisode {ep+1}/{num_eps}", end='', flush=True)
        
        env_dqn = Runner.create_env(dc_selector=PrioritySelector())
        state, _ = env_dqn.reset()
        done = False
        while not done:
            mask = env_dqn._get_valid_actions_mask()
            action = dqn_agent.select_action(state, 0.0, mask)
            state, _, done, _, _ = env_dqn.step(action)
        stats = env_dqn.sfc_manager.get_statistics()
        dqn_ars.append(stats['acceptance_ratio'])
        dqn_delays.append(stats['avg_e2e_delay'])
        
        env_vae = Runner.create_env(dc_selector=VAESelector(vae_model))
        state, _ = env_vae.reset()
        done = False
        while not done:
            mask = env_vae._get_valid_actions_mask()
            action = vae_dqn_agent.select_action(state, 0.0, mask)
            state, _, done, _, _ = env_vae.step(action)
        stats = env_vae.sfc_manager.get_statistics()
        vae_ars.append(stats['acceptance_ratio'])
        vae_delays.append(stats['avg_e2e_delay'])
    
    print(f"\n\nResults:")
    print(f"DQN - AR: {np.mean(dqn_ars):.2f}% ± {np.std(dqn_ars):.2f}%")
    print(f"      Delay: {np.mean(dqn_delays):.2f} ± {np.std(dqn_delays):.2f} ms")
    print(f"VAE-DQN - AR: {np.mean(vae_ars):.2f}% ± {np.std(vae_ars):.2f}%")
    print(f"          Delay: {np.mean(vae_delays):.2f} ± {np.std(vae_delays):.2f} ms")
    
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5))
    
    episodes = range(1, num_eps + 1)
    ax1.plot(episodes, dqn_ars, 'o-', label='DQN', alpha=0.7)
    ax1.plot(episodes, vae_ars, 's-', label='VAE-DQN', alpha=0.7)
    ax1.set_xlabel('Episode')
    ax1.set_ylabel('Acceptance Ratio (%)')
    ax1.set_title('Acceptance Ratio Comparison')
    ax1.legend()
    ax1.grid(alpha=0.3)
    
    ax2.plot(episodes, dqn_delays, 'o-', label='DQN', alpha=0.7)
    ax2.plot(episodes, vae_delays, 's-', label='VAE-DQN', alpha=0.7)
    ax2.set_xlabel('Episode')
    ax2.set_ylabel('E2E Delay (ms)')
    ax2.set_title('E2E Delay Comparison')
    ax2.legend()
    ax2.grid(alpha=0.3)
    
    plt.tight_layout()
    os.makedirs("fig", exist_ok=True)
    plt.savefig("fig/comparison.png", dpi=150)
    plt.close()
    print(f"Plot saved: fig/comparison.png")

def main(): 
    print("Starting main...")
    print(f"Arguments: {sys.argv}")
    
    parser = argparse.ArgumentParser(
        description="DRL-based NFV Placement",
        formatter_class=argparse.RawDescriptionHelpFormatter
    )
    
    subparsers = parser.add_subparsers(dest='command', help='Command to run')
    
    train_parser = subparsers.add_parser('train', help='Train models')
    train_parser.add_argument('mode', choices=['dqn', 'vae'],
                             help='dqn: Priority selector | vae: VAE-assisted')
    train_parser.add_argument('--data', default='data/cogent_centers_easy_s1.json',
                             help='Training data path')
    train_parser.add_argument('--updates', type=int,
                             help='Training updates (overrides config)')
    train_parser.add_argument('--vae-episodes', type=int, dest='vae_episode',
                             help='VAE collection episodes')
    train_parser.add_argument('--skip-vae', action='store_true',
                             help='Skip VAE training (use existing)')
    
    eval_parser = subparsers.add_parser('eval', help='Evaluate models')
    eval_parser.add_argument('mode', choices=['dqn', 'vae', 'compare'],
                            help='Model to evaluate')
    eval_parser.add_argument('--data', default='data/cogent_centers_easy_s1.json',
                            help='Test data path')
    eval_parser.add_argument('--episodes', type=int,
                            help='Evaluation episodes')
    eval_parser.add_argument('--model', help='Model path')
    eval_parser.add_argument('--dqn-model', help='DQN model (compare mode)')
    eval_parser.add_argument('--vae-model', help='VAE-DQN model (compare mode)')
    
    args = parser.parse_args()
    print(f"Parsed command: {args.command}")
    
    if not args.command:
        parser.print_help()
        sys.exit(1)
    
    if args.command == 'train':
        print(f"Training mode: {args.mode}")
        if args.mode == 'dqn':
            train_dqn(args)
        else:
            train_vae_dqn(args)
    elif args.command == 'eval':
        if args.mode == 'dqn':
            eval_dqn(args)
        elif args.mode == 'vae':
            eval_vae_dqn(args)
        else:
            eval_compare(args)

if __name__ == "__main__":
    try:
        main()
    except Exception as e:
        print(f"Fatal error: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)