import sys
import argparse
import os

def run_train(mode='drl'):
    """Chạy training"""
    if mode == 'genai':
        print("\n[INFO] Starting GenAI-DRL Training...")
        try:
            from runners import train_genai
            train_genai.main()
        except ImportError as e:
            print(f"[ERROR] Import failed: {e}")
    else:
        print("\n[INFO] Starting DRL Training...")
        try:
            from runners import train
            train.main()
        except ImportError as e:
            print(f"[ERROR] Import failed: {e}")

def run_eval(mode='drl'):
    """Chạy evaluation"""
    if mode == 'genai':
        print("\n[INFO] Starting GenAI-DRL Evaluation...")
        try:
            from runners import evaluate_genai
            evaluate_genai.main()
        except ImportError as e:
            print(f"[ERROR] Import failed: {e}")
    else:
        print("\n[INFO] Starting DRL Evaluation...")
        try:
            from runners import evaluate
            evaluate.main()
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

def run_demo():
    """Chạy demo tests"""
    print("\n[INFO] Starting Demo & Validation Tests...")
    try:
        from runners import demo
        demo.main()
    except ImportError as e:
        print(f"[ERROR] Import failed: {e}")

def run_debug():
    """Chạy debug mode"""
    print("\n[INFO] Starting Debug Mode...")
    try:
        from runners import debug
        debug.main()
    except ImportError as e:
        print(f"[ERROR] Import failed: {e}")

def interactive_menu():
    """Menu tương tác"""
    while True:
        print("\n" + "="*60)
        print("   DRL SFC PROVISIONING - PROJECT MANAGER")
        print("="*60)
        print("Standard DRL:")
        print("  1. Train DRL Model")
        print("  2. Evaluate DRL Model")
        print()
        print("GenAI-DRL:")
        print("  3. Collect GenAI Data")
        print("  4. Train GenAI-DRL Model")
        print("  5. Evaluate GenAI-DRL Model")
        print()
        print("Others:")
        print("  6. Run Demo Tests")
        print("  7. Debug Mode")
        print("  0. Exit")
        print("-" * 60)
        
        choice = input("Select (0-7): ").strip()
        
        if choice == '1':
            run_train(mode='drl')
        elif choice == '2':
            run_eval(mode='drl')
        elif choice == '3':
            run_collect_data()
        elif choice == '4':
            run_train(mode='genai')
        elif choice == '5':
            run_eval(mode='genai')
        elif choice == '6':
            run_demo()
        elif choice == '7':
            run_debug()
        elif choice == '0':
            print("\nExiting. Goodbye!")
            sys.exit(0)
        else:
            print("\n[ERROR] Invalid choice.")

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
  python scripts.py demo                # Run tests
        """
    )
    parser.add_argument(
        'command', 
        nargs='?', 
        choices=['train', 'eval', 'collect', 'demo', 'debug'],
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
    elif args.command == 'demo':
        run_demo()
    elif args.command == 'debug':
        run_debug()
    else:
        interactive_menu()

if __name__ == "__main__":
    main()