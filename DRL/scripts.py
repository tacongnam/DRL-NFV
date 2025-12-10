# scripts.py
"""
Main entry point cho DRL SFC Provisioning Project

Usage:
    python scripts.py train    # Train model
    python scripts.py eval     # Evaluate model
    python scripts.py demo     # Run demo tests
"""

import sys
import argparse
import os

def run_train():
    """Chạy training"""
    print("\n[INFO] Starting Training Process...")
    try:
        from runners import train
        train.main()
    except ImportError as e:
        print(f"[ERROR] Import failed: {e}")
        print("Make sure all dependencies are installed and folder structure is correct.")
    except Exception as e:
        print(f"[ERROR] Runtime error: {e}")
        import traceback
        traceback.print_exc()

def run_eval():
    """Chạy evaluation"""
    print("\n[INFO] Starting Evaluation Process...")
    try:
        from runners import evaluate
        evaluate.main()
    except ImportError as e:
        print(f"[ERROR] Import failed: {e}")
        print("Make sure all dependencies are installed and folder structure is correct.")
    except Exception as e:
        print(f"[ERROR] Runtime error: {e}")
        import traceback
        traceback.print_exc()

def run_demo():
    """Chạy demo tests"""
    print("\n[INFO] Starting Demo & Validation Tests...")
    try:
        from runners import demo
        demo.main()
    except ImportError as e:
        print(f"[ERROR] Import failed: {e}")
        print("Make sure all dependencies are installed and folder structure is correct.")
    except Exception as e:
        print(f"[ERROR] Runtime error: {e}")
        import traceback
        traceback.print_exc()

def run_debug():
    """Chạy debug mode"""
    print("\n[INFO] Starting Debug Mode...")
    try:
        from runners import debug
        debug.main()
    except ImportError as e:
        print(f"[ERROR] Import failed: {e}")
        print("Make sure all dependencies are installed and folder structure is correct.")
    except Exception as e:
        print(f"[ERROR] Runtime error: {e}")
        import traceback
        traceback.print_exc()

def interactive_menu():
    """Menu tương tác"""
    while True:
        print("\n" + "="*60)
        print("   DRL SFC PROVISIONING - PROJECT MANAGER")
        print("="*60)
        print("1. Train Model       (runners/train.py)")
        print("2. Evaluate Model    (runners/evaluate.py)")
        print("3. Run Demo Tests    (runners/demo.py)")
        print("4. Debug Mode        (runners/debug.py)")
        print("0. Exit")
        print("-" * 60)
        
        choice = input("Select option (0-4): ").strip()
        
        if choice == '1':
            run_train()
        elif choice == '2':
            run_eval()
        elif choice == '3':
            run_demo()
        elif choice == '4':
            run_debug()
        elif choice == '0':
            print("\nExiting. Goodbye!")
            sys.exit(0)
        else:
            print("\n[ERROR] Invalid choice. Please try again.")

def main():
    # Add current directory to path
    sys.path.append(os.getcwd())
    
    # Parse arguments
    parser = argparse.ArgumentParser(
        description="DRL SFC Provisioning Project Manager",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  python scripts.py train    # Train the DQN model
  python scripts.py eval     # Evaluate trained model
  python scripts.py demo     # Run validation tests
        """
    )
    parser.add_argument(
        'mode', 
        nargs='?', 
        choices=['train', 'eval', 'demo', 'debug'],
        help="Execution mode"
    )
    
    args = parser.parse_args()
    
    # Execute based on mode
    if args.mode == 'train':
        run_train()
    elif args.mode == 'eval':
        run_eval()
    elif args.mode == 'demo':
        run_demo()
    elif args.mode == 'debug':
        run_debug()
    else:
        # No arguments: show interactive menu
        interactive_menu()

if __name__ == "__main__":
    main()