import os
from pathlib import Path
from ultralytics import YOLO
import yaml
import shutil
import torch

# Kiểm tra xem GPU có sẵn không
print(f"Cuda có sẵn: {torch.cuda.is_available()}")
if torch.cuda.is_available():
    num_gpus = torch.cuda.device_count()
    gpu_name = torch.cuda.get_device_name(0) if num_gpus > 0 else "N/A"
    print(f"Số lượng GPU: {num_gpus}")
    print(f"GPU đang sử dụng: {gpu_name}")
else:
    print("Không tìm thấy GPU, sẽ sử dụng CPU")

# Thiết lập đường dẫn cơ sở
BASE_PATH = Path(r"D:\ĐATN\Phát hiện rác")
DATA_YAML = BASE_PATH / "data.yaml"
MODEL_SAVE_PATH = BASE_PATH / "models"

# Tạo thư mục để lưu mô hình nếu chưa tồn tại
MODEL_SAVE_PATH.mkdir(parents=True, exist_ok=True)

# Kiểm tra tồn tại file data.yaml
if not DATA_YAML.exists():
    raise FileNotFoundError(f"Không tìm thấy file data.yaml tại: {DATA_YAML}")

# Đọc file data.yaml để hiển thị thông tin
with open(DATA_YAML, 'r') as f:
    data_config = yaml.safe_load(f)

print("Thông tin từ file data.yaml:")
print(f"- Số lượng lớp: {data_config.get('nc', 'Không có thông tin')}")
print(f"- Tên các lớp: {data_config.get('names', 'Không có thông tin')}")
print(f"- Đường dẫn train: {data_config.get('train', 'Không có thông tin')}")
print(f"- Đường dẫn val: {data_config.get('val', 'Không có thông tin')}")
print(f"- Đường dẫn test: {data_config.get('test', 'Không có thông tin')}")

# Hàm để đếm số lượng ảnh và nhãn
def count_files(path, ext=".jpg"):
    if not Path(path).exists():
        return 0
    return len(list(Path(path).glob(f"*{ext}")))

# Kiểm tra số lượng file trong thư mục train và val
if 'train' in data_config:
    train_path = Path(data_config['train'])
    if train_path.exists():
        print(f"Số lượng ảnh trong tập train: {count_files(train_path / 'images')}")
        print(f"Số lượng nhãn trong tập train: {count_files(train_path / 'labels', '.txt')}")

if 'val' in data_config:
    val_path = Path(data_config['val'])
    if val_path.exists():
        print(f"Số lượng ảnh trong tập val: {count_files(val_path / 'images')}")
        print(f"Số lượng nhãn trong tập val: {count_files(val_path / 'labels', '.txt')}")

# Hàm huấn luyện và lưu mô hình
def train_and_save_model(yaml_path, model_size="n", epochs=50, img_size=640, batch_size=16, save_path=None):
    """
    Huấn luyện và lưu mô hình YOLOv8
    
    Args:
        yaml_path: Đường dẫn đến file data.yaml
        model_size: Kích thước mô hình (n, s, m, l, x)
        epochs: Số epoch huấn luyện
        img_size: Kích thước ảnh đầu vào
        batch_size: Kích thước batch
        save_path: Thư mục để lưu mô hình
        
    Returns:
        model: Mô hình đã huấn luyện
        best_model_path: Đường dẫn đến mô hình tốt nhất
    """
    print(f"\n{'='*50}")
    print(f"Bắt đầu huấn luyện mô hình YOLOv8{model_size}")
    print(f"{'='*50}")
    
    # Tải mô hình YOLOv8 pre-trained
    model = YOLO(f"yolov8{model_size}.pt")
    
    # Xác định device sử dụng
    device = '0' if torch.cuda.is_available() else 'cpu'
    print(f"Huấn luyện trên thiết bị: {device}")
    
    # Huấn luyện mô hình
    results = model.train(
        data=yaml_path,
        epochs=epochs,
        imgsz=img_size,
        batch=batch_size,
        name=f"waste_detection_yolov8{model_size}",
        device=device,
        verbose=True,
        patience=15,  # Early stopping sau 15 epochs không cải thiện
        workers=8 if torch.cuda.is_available() else 4,  # Số lượng worker tối ưu
        optimizer='AdamW'  # Tối ưu hóa hiệu suất
    )
    
    print("\nHuấn luyện hoàn tất!")
    
    # Lưu mô hình vào thư mục được chỉ định
    best_model_path = None
    if save_path:
        # Đường dẫn đến mô hình đã được huấn luyện (best.pt và last.pt)
        run_dir = Path(results.save_dir)
        best_model_path = run_dir / "weights" / "best.pt"
        last_model_path = run_dir / "weights" / "last.pt"
        
        # Kiểm tra và sao chép các mô hình
        if best_model_path.exists():
            dest_best = save_path / f"waste_detection_yolov8{model_size}_best.pt"
            shutil.copy(best_model_path, dest_best)
            best_model_path = dest_best
            print(f"Đã lưu mô hình tốt nhất tại: {dest_best}")
        
        if last_model_path.exists():
            dest_last = save_path / f"waste_detection_yolov8{model_size}_last.pt"
            shutil.copy(last_model_path, dest_last)
            print(f"Đã lưu mô hình cuối cùng tại: {dest_last}")
    
    # Thực hiện đánh giá trên tập validation
    print("\nĐánh giá mô hình trên tập validation:")
    val_results = model.val(device=device)
    print(f"mAP50-95: {val_results.box.map:.4f}")
    print(f"mAP50: {val_results.box.map50:.4f}")
    
    return model, best_model_path

# Hàm chính
def main():
    # Thiết lập tham số huấn luyện
    model_size = "n"  # Các lựa chọn: "n", "s", "m", "l", "x" (từ nhỏ đến lớn)
    epochs = 30
    img_size = 640
    
    # Điều chỉnh batch size dựa vào GPU
    if torch.cuda.is_available():
        # Tự động điều chỉnh batch size dựa trên dung lượng VRAM
        gpu_mem = torch.cuda.get_device_properties(0).total_memory / 1e9  # GB
        if gpu_mem > 12:  # GPU > 12GB VRAM
            batch_size = 32
        elif gpu_mem > 8:  # GPU 8-12GB VRAM
            batch_size = 24
        elif gpu_mem > 4:  # GPU 4-8GB VRAM
            batch_size = 16
        else:  # GPU < 4GB VRAM
            batch_size = 8
        print(f"Dung lượng GPU: {gpu_mem:.2f} GB, Batch size tự động điều chỉnh: {batch_size}")
    else:
        batch_size = 8
        print(f"Sử dụng CPU, batch size: {batch_size}")
    
    # Huấn luyện và lưu mô hình
    model, best_model_path = train_and_save_model(
        yaml_path=str(DATA_YAML),
        model_size=model_size,
        epochs=epochs,
        img_size=img_size,
        batch_size=batch_size,
        save_path=MODEL_SAVE_PATH
    )
    
    print("\nThông tin mô hình:")
    print(f"- Đường dẫn mô hình tốt nhất: {best_model_path}")
    #print(f"- Thông tin chi tiết về quá trình huấn luyện được lưu trong thư mục: {model.ckpt_path.parent.parent}")
    print("\nBạn có thể sử dụng mô hình này trong file khác bằng cách:")
    print(f"from ultralytics import YOLO")
    print(f"model = YOLO('{best_model_path}')")
    print(f"results = model.predict('path_to_image.jpg')")

if __name__ == "__main__":
    main()