import sys
import os
from pathlib import Path
import torch
from ultralytics import YOLO
from PyQt5.QtWidgets import (QApplication, QMainWindow, QLabel, QPushButton, 
                             QVBoxLayout, QHBoxLayout, QWidget, QFileDialog, 
                             QComboBox, QSlider, QMessageBox, QProgressBar, QSplashScreen)
from PyQt5.QtGui import QPixmap, QImage, QPainter, QPen, QColor, QFont
from PyQt5.QtCore import Qt, QThread, pyqtSignal, QTimer

class InferenceThread(QThread):
    finished = pyqtSignal(object)
    progress = pyqtSignal(int)
    
    def __init__(self, model_path, image_path, conf_thresh):
        super().__init__()
        self.model_path = model_path
        self.image_path = image_path
        self.conf_thresh = conf_thresh
        
    def run(self):
        try:
            # Load model
            model = YOLO(self.model_path)
            
            # Update progress
            self.progress.emit(30)
            
            # Run inference
            results = model.predict(
                source=self.image_path,
                conf=self.conf_thresh,
                save=False,
                verbose=False
            )
            
            # Update progress
            self.progress.emit(90)
            
            # Emit results
            self.finished.emit(results[0])
            
        except Exception as e:
            self.finished.emit(None)
            print(f"Lỗi khi thực hiện dự đoán: {str(e)}")

class WasteDetectionGUI(QMainWindow):
    def __init__(self):
        super().__init__()
        self.initUI()
        self.model = None
        self.results = None
        self.inference_thread = None
        
    def initUI(self):
        # Thiết lập cửa sổ chính
        self.setWindowTitle("Phát hiện rác bằng YOLOv8")
        self.setGeometry(100, 100, 1200, 800)
        
        # Widget chính
        main_widget = QWidget()
        self.setCentralWidget(main_widget)
        
        # Layout chính
        main_layout = QVBoxLayout()
        main_widget.setLayout(main_layout)
        
        # Layout trên
        top_layout = QHBoxLayout()
        
        # Khung hiển thị ảnh
        self.image_frame = QLabel("Chưa có ảnh nào được tải lên")
        self.image_frame.setAlignment(Qt.AlignCenter)
        self.image_frame.setStyleSheet("border: 2px solid gray; background-color: #f0f0f0;")
        self.image_frame.setMinimumSize(800, 600)
        
        # Layout phải (các nút điều khiển)
        right_layout = QVBoxLayout()
        
        # Nút chọn mô hình
        self.model_label = QLabel("Chọn mô hình:")
        self.model_combo = QComboBox()
        self.model_refresh_btn = QPushButton("Làm mới")
        self.model_refresh_btn.clicked.connect(self.update_model_list)
        
        # Hiển thị thông tin
        self.info_label = QLabel("Sẵn sàng")
        self.info_label.setAlignment(Qt.AlignCenter)
        self.info_label.setStyleSheet("font-weight: bold;")
        right_layout.addWidget(self.info_label)
        
        self.update_model_list()
        
        model_layout = QHBoxLayout()
        model_layout.addWidget(self.model_label)
        model_layout.addWidget(self.model_combo)
        model_layout.addWidget(self.model_refresh_btn)
        right_layout.addLayout(model_layout)
        
        # Nút chọn ảnh
        self.select_image_btn = QPushButton("Chọn ảnh")
        self.select_image_btn.clicked.connect(self.select_image)
        right_layout.addWidget(self.select_image_btn)
        
        # Thanh trượt ngưỡng tin cậy
        self.conf_label = QLabel("Ngưỡng tin cậy: 0.3")
        self.conf_slider = QSlider(Qt.Horizontal)
        self.conf_slider.setMinimum(1)
        self.conf_slider.setMaximum(99)
        self.conf_slider.setValue(30)
        self.conf_slider.setTickPosition(QSlider.TicksBelow)
        self.conf_slider.setTickInterval(10)
        self.conf_slider.valueChanged.connect(self.update_conf_label)
        right_layout.addWidget(self.conf_label)
        right_layout.addWidget(self.conf_slider)
        
        # Nút dự đoán
        self.predict_btn = QPushButton("Dự đoán")
        self.predict_btn.clicked.connect(self.predict)
        self.predict_btn.setEnabled(False)
        right_layout.addWidget(self.predict_btn)
        
        # Nút lưu kết quả
        self.save_btn = QPushButton("Lưu kết quả")
        self.save_btn.clicked.connect(self.save_result)
        self.save_btn.setEnabled(False)
        right_layout.addWidget(self.save_btn)
        
        # Thanh tiến trình
        self.progress_bar = QProgressBar()
        self.progress_bar.setVisible(False)
        right_layout.addWidget(self.progress_bar)
        
        # Hiển thị thông tin
        self.info_label = QLabel("Sẵn sàng")
        self.info_label.setAlignment(Qt.AlignCenter)
        self.info_label.setStyleSheet("font-weight: bold;")
        right_layout.addWidget(self.info_label)
        
        # Thêm vào layout chính
        top_layout.addWidget(self.image_frame, 4)
        top_layout.addLayout(right_layout, 1)
        main_layout.addLayout(top_layout)
        
        # Thông tin về các đối tượng được phát hiện
        self.result_label = QLabel("Kết quả dự đoán sẽ hiển thị ở đây")
        self.result_label.setAlignment(Qt.AlignCenter)
        main_layout.addWidget(self.result_label)
        
        # Biến lưu trữ đường dẫn ảnh hiện tại
        self.current_image_path = None
        
        # Thông tin thiết bị
        device_info = "CPU"
        if torch.cuda.is_available():
            device_info = f"GPU ({torch.cuda.get_device_name(0)})"
        
        self.status_label = QLabel(f"Thiết bị: {device_info}")
        main_layout.addWidget(self.status_label)
    
    def update_model_list(self):
        """Cập nhật danh sách mô hình từ thư mục 'models'"""
        self.model_combo.clear()
        base_path = Path("D:/ĐATN/Phát hiện rác/models")
        
        if base_path.exists():
            model_files = list(base_path.glob("*.pt"))
            if model_files:
                for model_file in model_files:
                    self.model_combo.addItem(model_file.name, str(model_file))
                
                self.info_label.setText(f"Đã tìm thấy {len(model_files)} mô hình")
            else:
                self.info_label.setText("Không tìm thấy mô hình nào")
                self.model_combo.addItem("-- Chưa có mô hình --")
        else:
            self.info_label.setText("Thư mục 'models' không tồn tại")
            self.model_combo.addItem("-- Thư mục không tồn tại --")
    
    def update_conf_label(self):
        """Cập nhật nhãn khi thay đổi ngưỡng tin cậy"""
        conf_value = self.conf_slider.value() / 100.0
        self.conf_label.setText(f"Ngưỡng tin cậy: {conf_value:.2f}")
    
    def select_image(self):
        """Mở hộp thoại chọn ảnh"""
        file_dialog = QFileDialog()
        image_path, _ = file_dialog.getOpenFileName(
            self, "Chọn ảnh", "", "Ảnh (*.png *.jpg *.jpeg *.bmp)"
        )
        
        if image_path:
            try:
                # Hiển thị ảnh
                pixmap = QPixmap(image_path)
                
                # Thay đổi kích thước để vừa với khung hiển thị
                scaled_pixmap = pixmap.scaled(
                    self.image_frame.width(), 
                    self.image_frame.height(),
                    Qt.KeepAspectRatio, 
                    Qt.SmoothTransformation
                )
                
                self.image_frame.setPixmap(scaled_pixmap)
                self.current_image_path = image_path
                self.predict_btn.setEnabled(True)
                self.info_label.setText(f"Đã tải ảnh: {os.path.basename(image_path)}")
                self.save_btn.setEnabled(False)
                self.result_label.setText("Nhấn 'Dự đoán' để phát hiện đối tượng")
                
            except Exception as e:
                QMessageBox.critical(self, "Lỗi", f"Không thể tải ảnh: {str(e)}")
    
    def predict(self):
        """Thực hiện dự đoán trên ảnh đã chọn"""
        if not self.current_image_path:
            QMessageBox.warning(self, "Cảnh báo", "Vui lòng chọn ảnh trước")
            return
            
        model_path = self.model_combo.currentData()
        if not model_path or not Path(model_path).exists():
            QMessageBox.warning(self, "Cảnh báo", "Vui lòng chọn mô hình hợp lệ")
            return
            
        # Hiển thị thanh tiến trình
        self.progress_bar.setValue(0)
        self.progress_bar.setVisible(True)
        self.info_label.setText("Đang xử lý...")
        self.predict_btn.setEnabled(False)
        
        # Lấy ngưỡng tin cậy
        conf_thresh = self.conf_slider.value() / 100.0
        
        # Tạo và chạy thread dự đoán
        self.inference_thread = InferenceThread(model_path, self.current_image_path, conf_thresh)
        self.inference_thread.progress.connect(self.update_progress)
        self.inference_thread.finished.connect(self.process_results)
        self.inference_thread.start()
        
    def update_progress(self, value):
        """Cập nhật thanh tiến trình"""
        self.progress_bar.setValue(value)
    
    def process_results(self, results):
        """Xử lý kết quả dự đoán"""
        self.progress_bar.setValue(100)
        
        if results is None:
            self.info_label.setText("Dự đoán thất bại")
            self.predict_btn.setEnabled(True)
            QTimer.singleShot(2000, lambda: self.progress_bar.setVisible(False))
            return
            
        self.results = results
        self.info_label.setText("Dự đoán hoàn tất")
        self.predict_btn.setEnabled(True)
        self.save_btn.setEnabled(True)
        
        # Vẽ kết quả lên ảnh
        self.draw_results()
        
        # Ẩn thanh tiến trình sau 2 giây
        QTimer.singleShot(2000, lambda: self.progress_bar.setVisible(False))
        
    def draw_results(self):
        """Vẽ kết quả dự đoán lên ảnh"""
        if not self.results or not self.current_image_path:
            return
            
        # Đọc ảnh gốc
        image = QImage(self.current_image_path)
        pixmap = QPixmap.fromImage(image)
        
        # Tạo bản sao để vẽ lên
        result_pixmap = pixmap.copy()
        painter = QPainter(result_pixmap)
        
        # Thiết lập bút vẽ
        pen = QPen(QColor(0, 255, 0))  # Màu xanh lá
        pen.setWidth(3)
        painter.setPen(pen)
        
        # Thiết lập font chữ
        font = QFont()
        font.setPointSize(10)
        font.setBold(True)
        painter.setFont(font)
        
        # Lấy các đối tượng đã phát hiện
        boxes = self.results.boxes
        class_names = self.results.names
        
        detection_text = []
        
        # Vẽ các hộp giới hạn và nhãn
        for i, box in enumerate(boxes):
            # Lấy tọa độ
            x1, y1, x2, y2 = box.xyxy[0].cpu().numpy().astype(int)
            
            # Lấy lớp và độ tin cậy
            cls_id = int(box.cls[0].item())
            cls_name = class_names[cls_id]
            conf = box.conf[0].item()
            
            # Vẽ hộp giới hạn
            painter.drawRect(x1, y1, x2-x1, y2-y1)
            
            # Vẽ nhãn
            label_text = f"{cls_name}: {conf:.2f}"
            painter.fillRect(x1, y1-20, len(label_text)*8, 20, QColor(0, 255, 0, 128))
            painter.setPen(QColor(0, 0, 0))
            painter.drawText(x1+5, y1-5, label_text)
            painter.setPen(pen)
            
            # Thêm vào danh sách phát hiện
            detection_text.append(f"{cls_name} (tin cậy: {conf:.2f})")
        
        painter.end()
        
        # Hiển thị kết quả
        scaled_pixmap = result_pixmap.scaled(
            self.image_frame.width(), 
            self.image_frame.height(),
            Qt.KeepAspectRatio, 
            Qt.SmoothTransformation
        )
        self.image_frame.setPixmap(scaled_pixmap)
        
        # Hiển thị thông tin phát hiện
        if detection_text:
            self.result_label.setText(f"Đã phát hiện {len(detection_text)} đối tượng: " + ", ".join(detection_text))
        else:
            self.result_label.setText("Không phát hiện đối tượng nào")
    
    def save_result(self):
        """Lưu kết quả dự đoán"""
        if not self.results:
            return
            
        file_dialog = QFileDialog()
        save_path, _ = file_dialog.getSaveFileName(
            self, "Lưu kết quả", "", "Ảnh (*.png *.jpg)"
        )
        
        if save_path:
            try:
                # Lưu ảnh với các đối tượng được vẽ
                self.image_frame.pixmap().save(save_path)
                self.info_label.setText(f"Đã lưu kết quả tại: {os.path.basename(save_path)}")
            except Exception as e:
                QMessageBox.critical(self, "Lỗi", f"Không thể lưu ảnh: {str(e)}")

class SplashScreen(QSplashScreen):
    def __init__(self):
        super().__init__()
        # Tạo ảnh splash screen
        pixmap = QPixmap(400, 200)
        pixmap.fill(QColor(240, 240, 240))
        
        # Vẽ văn bản lên splash screen
        painter = QPainter(pixmap)
        painter.setPen(QColor(0, 0, 0))
        font = QFont()
        font.setPointSize(16)
        font.setBold(True)
        painter.setFont(font)
        painter.drawText(pixmap.rect(), Qt.AlignCenter, "Đang khởi động\nỨng dụng phát hiện rác")
        painter.end()
        
        self.setPixmap(pixmap)

def main():
    app = QApplication(sys.argv)
    
    # Hiển thị splash screen
    splash = SplashScreen()
    splash.show()
    
    # Kiểm tra cài đặt
    if not torch.cuda.is_available():
        splash.showMessage("Đang chạy trên CPU, hiệu suất có thể bị giảm", Qt.AlignBottom | Qt.AlignCenter)
    else:
        splash.showMessage(f"Đã phát hiện GPU: {torch.cuda.get_device_name(0)}", Qt.AlignBottom | Qt.AlignCenter)
    
    app.processEvents()
    
    # Tạo đường dẫn đến thư mục models nếu chưa tồn tại
    Path("D:/ĐATN/Phát hiện rác/models").mkdir(parents=True, exist_ok=True)
    
    # Khởi tạo cửa sổ chính
    window = WasteDetectionGUI()
    
    # Đóng splash screen và hiển thị cửa sổ chính sau 2 giây
    QTimer.singleShot(2000, splash.close)
    QTimer.singleShot(2000, window.show)
    
    sys.exit(app.exec_())

if __name__ == "__main__":
    main()