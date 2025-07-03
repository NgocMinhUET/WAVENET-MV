# Hướng dẫn sửa lỗi cú pháp trong compressor_improved.py

## Vấn đề
File `models/compressor_improved.py` gặp lỗi cú pháp:
```
SyntaxError: unterminated string literal (detected at line 323)
```

## Giải pháp

### Bước 1: Tạo file fix_compressor_server.py trên server
Tạo file `fix_compressor_server.py` trên server với nội dung sau:

```python
"""
Fix syntax error in compressor_improved.py
"""
import os

def fix_compressor_improved():
    print("🔧 Đang sửa file compressor_improved.py...")
    
    # Nội dung đúng cho phương thức to() của ImprovedCompressorVNVC
    improved_to_method = """    def to(self, device):
        super().to(device)
        if hasattr(self, 'analysis_transform'):
            self.analysis_transform.to(device)
        if hasattr(self, 'synthesis_transform'):
            self.synthesis_transform.to(device)
        if hasattr(self, 'quantizer'):
            self.quantizer.to(device)
        if hasattr(self, 'entropy_bottleneck'):
            self.entropy_bottleneck.to(device)
        return self
        
"""

    # Nội dung đầy đủ cho class ImprovedMultiLambdaCompressorVNVC
    multi_lambda_class = """
class ImprovedMultiLambdaCompressorVNVC(nn.Module):
    \"\"\"
    Multi-lambda version of improved compressor
    Maintains compatibility with existing training scripts
    \"\"\"
    
    def __init__(self, input_channels=128, latent_channels=192):
        super().__init__()
        
        self.input_channels = input_channels
        self.latent_channels = latent_channels
        
        # Create compressor instances for different lambdas
        self.compressors = nn.ModuleDict({
            '64': ImprovedCompressorVNVC(input_channels, latent_channels, 64),
            '128': ImprovedCompressorVNVC(input_channels, latent_channels, 128),
            '256': ImprovedCompressorVNVC(input_channels, latent_channels, 256),
            '512': ImprovedCompressorVNVC(input_channels, latent_channels, 512),
            '1024': ImprovedCompressorVNVC(input_channels, latent_channels, 1024),
            '2048': ImprovedCompressorVNVC(input_channels, latent_channels, 2048),
            '4096': ImprovedCompressorVNVC(input_channels, latent_channels, 4096)
        })
        
        self.current_lambda = 128
        
    def to(self, device):
        super().to(device)
        if hasattr(self, 'compressors'):
            for lambda_key, compressor in self.compressors.items():
                if compressor is not None:
                    compressor.to(device)
        return self
        
    def set_lambda(self, lambda_value):
        self.current_lambda = lambda_value
        
    def forward(self, x):
        compressor = self.compressors[str(self.current_lambda)]
        return compressor(x)
        
    def compress(self, x, lambda_value=None):
        if lambda_value is None:
            lambda_value = self.current_lambda
        compressor = self.compressors[str(lambda_value)]
        return compressor.compress(x)
        
    def decompress(self, bitstream, lambda_value=None):
        if lambda_value is None:
            lambda_value = self.current_lambda
        compressor = self.compressors[str(lambda_value)]
        return compressor.decompress(bitstream)
        
    def compute_rate_distortion_loss(self, x, x_hat, likelihoods, original_shape):
        compressor = self.compressors[str(self.current_lambda)]
        return compressor.compute_rate_distortion_loss(x, x_hat, likelihoods, original_shape)
        
    def update(self):
        for lambda_key, compressor in self.compressors.items():
            try:
                if hasattr(compressor, 'entropy_bottleneck') and hasattr(compressor.entropy_bottleneck, 'gaussian_conditional'):
                    compressor.entropy_bottleneck.gaussian_conditional.update()
            except Exception as e:
                print(f\"Warning: Failed to update entropy model for lambda={lambda_key}: {e}\")

"""

    # Tạo file mới hoàn toàn (phòng trường hợp file gốc bị hỏng)
    file_path = 'models/compressor_improved.py'
    
    if os.path.exists(file_path):
        # Đọc nội dung file hiện tại
        with open(file_path, 'r') as f:
            content = f.read()
        
        # Kiểm tra xem đã có phương thức to() cho ImprovedCompressorVNVC chưa
        improved_class_pos = content.find("class ImprovedCompressorVNVC")
        if improved_class_pos > 0:
            # Tìm vị trí để thêm phương thức to()
            forward_method_pos = content.find("def forward(self, x)", improved_class_pos)
            if forward_method_pos > 0:
                # Thêm phương thức to() trước forward()
                # Kiểm tra nếu chưa có
                if "def to(self, device)" not in content[improved_class_pos:forward_method_pos]:
                    content_before = content[:forward_method_pos]
                    content_after = content[forward_method_pos:]
                    content = content_before + improved_to_method + content_after
                    print("✅ Đã thêm phương thức to() cho ImprovedCompressorVNVC")
        
        # Kiểm tra xem có class ImprovedMultiLambdaCompressorVNVC chưa
        if "class ImprovedMultiLambdaCompressorVNVC" not in content:
            # Tìm vị trí để thêm class mới (cuối file)
            test_func_pos = content.find("def test_improved_compressor")
            if test_func_pos > 0:
                content_before = content[:test_func_pos]
                content_after = content[test_func_pos:]
                content = content_before + multi_lambda_class + "\n\n" + content_after
                print("✅ Đã thêm class ImprovedMultiLambdaCompressorVNVC")
            else:
                # Thêm vào cuối file nếu không tìm thấy hàm test
                content += "\n\n" + multi_lambda_class
                print("✅ Đã thêm class ImprovedMultiLambdaCompressorVNVC vào cuối file")
        
        # Lưu file đã sửa
        with open(file_path, 'w') as f:
            f.write(content)
        
        print("✅ Đã sửa xong file compressor_improved.py")
        
        # Kiểm tra cú pháp của file đã sửa
        try:
            with open(file_path, 'r') as f:
                code = f.read()
            compile(code, file_path, 'exec')
            print("✅ Kiểm tra cú pháp: OK!")
        except SyntaxError as e:
            print(f"❌ Lỗi cú pháp: {e}")
            print("🔄 Thử lại với phương pháp thứ hai...")
            
            # Nếu vẫn lỗi, ghi đè hoàn toàn file
            # Đọc nội dung từ đầu file đến class ImprovedCompressorVNVC (giữ nguyên các class đầu tiên)
            with open(file_path, 'r') as f:
                lines = f.readlines()
            
            new_content = []
            in_improved_compressor = False
            for line in lines:
                new_content.append(line)
                if "class ImprovedCompressorVNVC" in line:
                    in_improved_compressor = True
                if in_improved_compressor and line.strip() == "def forward(self, x):":
                    # Thêm phương thức to() ngay trước forward()
                    new_content.insert(len(new_content) - 1, improved_to_method)
            
            # Tìm vị trí kết thúc của class ImprovedCompressorVNVC
            test_func_idx = -1
            for i, line in enumerate(lines):
                if "def test_improved_compressor" in line:
                    test_func_idx = i
                    break
            
            if test_func_idx >= 0:
                # Thêm class ImprovedMultiLambdaCompressorVNVC trước hàm test
                for i in range(test_func_idx, len(lines)):
                    new_content.append(lines[i])
            
            # Lưu file mới
            with open(file_path, 'w') as f:
                f.writelines(new_content)
            
            print("✅ Đã sửa file với phương pháp thứ hai")
    else:
        print(f"❌ Không tìm thấy file {file_path}")
    
    print("\n🚀 Đề xuất chạy lệnh sau để kiểm tra:")
    print("python evaluation/codec_metrics.py --checkpoint checkpoints/stage3_ai_heads_coco_best.pth --dataset coco --data_dir datasets/COCO --split val --lambdas 128 --batch_size 1 --max_samples 10 --output_csv results/test_fixed.csv")

if __name__ == "__main__":
    fix_compressor_improved() 