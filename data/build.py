import shutil
from pathlib import Path


def copy_easy_json_files(source_dir, target_dir):
    # Chuyển đổi đường dẫn sang đối tượng Path
    src = Path(source_dir)
    tgt = Path(target_dir)

    # Tạo thư mục đích nếu nó chưa tồn tại
    tgt.mkdir(parents=True, exist_ok=True)

    # Tìm các file thỏa mãn định dạng *_easy_*.json
    # Cách này sẽ khớp với tất cả các file có dạng chữ_chữ_easy_chữ.json
    search_pattern = "*_easy_*.json"
    files_to_copy = list(src.glob(search_pattern))

    if not files_to_copy:
        print(
            f"Không tìm thấy file nào có định dạng '{search_pattern}' trong thư mục '{source_dir}'."
        )
        return

    print(f"Tìm thấy {len(files_to_copy)} file. Đang tiến hành copy...")

    # Tiến hành copy từng file
    for file_path in files_to_copy:
        # Định nghĩa đường dẫn mới cho file tại thư mục đích
        destination_path = tgt / file_path.name

        # Copy file (giữ nguyên metadata nếu có thể)
        shutil.copy2(file_path, destination_path)
        print(f"✔ Đã copy: {file_path.name} -> {target_dir}")

    print("--- Hoàn thành! ---")


# --- Cấu hình đường dẫn của bạn ở đây ---
SOURCE_DIRECTORY = "data/test"  # Thư mục gốc chứa file
TARGET_DIRECTORY = "data/easy"  # Thư mục mới muốn chuyển đến

# Chạy hàm
copy_easy_json_files(SOURCE_DIRECTORY, TARGET_DIRECTORY)