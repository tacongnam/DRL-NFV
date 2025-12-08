import sys
import argparse
import os

def run_train():
    """Import và chạy module Training"""
    print("\n[INFO] Initializing Training Process...")
    try:
        from runners import train
        train.main()
    except ImportError as e:
        print(f"[ERROR] Could not import 'runners.train'. Check your folder structure. Details: {e}")
    except Exception as e:
        print(f"[ERROR] Runtime error in Training: {e}")

def run_eval():
    """Import và chạy module Evaluation (Test)"""
    print("\n[INFO] Initializing Evaluation Process...")
    try:
        from runners import evaluate
        evaluate.main()
    except ImportError as e:
        print(f"[ERROR] Could not import 'runners.evaluate'. Check your folder structure. Details: {e}")
    except Exception as e:
        print(f"[ERROR] Runtime error in Evaluation: {e}")

def run_demo():
    """Import và chạy module Demo"""
    print("\n[INFO] Initializing Demo Mode...")
    try:
        from runners import demo
        demo.main()
    except ImportError as e:
        print(f"[ERROR] Could not import 'runners.demo'. Check your folder structure. Details: {e}")
    except Exception as e:
        print(f"[ERROR] Runtime error in Demo: {e}")

def interactive_menu():
    """Hiển thị menu nếu người dùng không truyền tham số"""
    while True:
        print("\n" + "="*30)
        print("   DRL SFC PROJECT MANAGER")
        print("="*30)
        print("1. Train Model  (runners/train.py)")
        print("2. Evaluate     (runners/evaluate.py)")
        print("3. Run Demo     (runners/demo.py)")
        print("0. Exit")
        print("-" * 30)
        
        choice = input("Select an option (0-3): ").strip()
        
        if choice == '1':
            run_train()
        elif choice == '2':
            run_eval()
        elif choice == '3':
            run_demo()
        elif choice == '0':
            print("Exiting.")
            sys.exit(0)
        else:
            print("Invalid selection. Please try again.")

if __name__ == "__main__":
    # Cấu hình Argument Parser
    parser = argparse.ArgumentParser(description="Run DRL scripts for NFV-SFC project.")
    parser.add_argument('mode', nargs='?', choices=['train', 'eval', 'demo'], 
                        help="Mode to run: 'train', 'eval' (evaluate), or 'demo'")

    args = parser.parse_args()

    # Thêm thư mục hiện tại vào sys.path để đảm bảo python tìm thấy các packages
    sys.path.append(os.getcwd())

    # Điều hướng dựa trên tham số dòng lệnh
    if args.mode == 'train':
        run_train()
    elif args.mode == 'eval':
        run_eval()
    elif args.mode == 'demo':
        run_demo()
    else:
        # Nếu không có tham số, hiện menu
        interactive_menu()