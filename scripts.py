import sys
import argparse
import os

def run_train(mode='drl'):
    """Chạy training"""
    if mode == 'genai':
        print("\n[INFO] Starting GenAI-DRL Training...")
        try:
            from runners import train_vae
            train_vae.main()
        except ImportError as e:
            print(f"[ERROR] Import failed: {e}")
    else:
        print("\n[INFO] Starting DRL Training...")
        try:
            from runners import train_drl
            train_drl.main()
        except ImportError as e:
            print(f"[ERROR] Import failed: {e}")

def run_eval(mode='drl'):
    """Chạy evaluation"""
    if mode == 'genai':
        print("\n[INFO] Starting GenAI-DRL Evaluation...")
        try:
            from runners import eval_vae
            eval_vae.main()
        except ImportError as e:
            print(f"[ERROR] Import failed: {e}")
    else:
        print("\n[INFO] Starting DRL Evaluation...")
        try:
            from runners import eval_drl
            eval_drl.main()
        except ImportError as e:
            print(f"[ERROR] Import failed: {e}")

def run_collect_data():
    """Chạy data collection cho GenAI"""
    print("\n[INFO] Starting Data Collection for GenAI...")
    try:
        from runners import collect_data
        collect_data.main()
    except ImportError as e:
        print(f"[ERROR] Import failed: {e}")

def main():
    sys.path.append(os.getcwd())
    
    parser = argparse.ArgumentParser(
        description="DRL SFC Provisioning Project Manager",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  python scripts.py train               # Train DRL
  python scripts.py train --mode genai  # Train GenAI-DRL
  python scripts.py eval                # Evaluate DRL
  python scripts.py eval --mode genai   # Evaluate GenAI-DRL
  python scripts.py collect             # Collect data for GenAI
        """
    )
    parser.add_argument(
        'command', 
        nargs='?', 
        choices=['train', 'eval', 'collect'],
        help="Command to execute"
    )
    parser.add_argument(
        '--mode',
        choices=['drl', 'genai'],
        default='drl',
        help="Training/Evaluation mode (default: drl)"
    )
    
    args = parser.parse_args()
    
    if args.command == 'train':
        run_train(mode=args.mode)
    elif args.command == 'eval':
        run_eval(mode=args.mode)
    elif args.command == 'collect':
        run_collect_data()

if __name__ == "__main__":
    main()