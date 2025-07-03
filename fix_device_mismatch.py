"""
Sửa lỗi device mismatch trong compressor cải tiến
Lỗi: Input type (torch.cuda.FloatTensor) and weight type (torch.FloatTensor) should be the same
"""

import torch
import os

def patch_improved_compressor():
    """
    Sửa lỗi device mismatch trong compressor_improved.py
    """
    print("🔧 ĐANG SỬA LỖI DEVICE MISMATCH")
    print("="*50)
    
    # Đọc file
    with open('models/compressor_improved.py', 'r', encoding='utf-8') as f:
        content = f.read()
    
    # Thêm hàm to() để đảm bảo tất cả modules cùng device
    if "def to(self, device):" not in content:
        # Thêm phương thức to() cho ImprovedCompressorVNVC
        improved_compressor_to_method = '''
    def to(self, device):
        """Chuyển toàn bộ model sang device chỉ định"""
        super().to(device)
        self.analysis_transform.to(device)
        self.synthesis_transform.to(device)
        if hasattr(self, 'entropy_bottleneck'):
            self.entropy_bottleneck.to(device)
        return self
'''
        
        # Tìm vị trí để thêm phương thức
        insert_pos = content.find("def compute_rate_distortion_loss(self, x, x_hat, likelihoods, original_shape):")
        if insert_pos > 0:
            # Tìm dòng trước để thêm vào
            last_def_end = content.rfind("}", 0, insert_pos)
            if last_def_end > 0:
                # Thêm phương thức mới
                new_content = content[:last_def_end+1] + improved_compressor_to_method + content[last_def_end+1:]
                
                # Lưu file
                with open('models/compressor_improved.py', 'w', encoding='utf-8') as f:
                    f.write(new_content)
                    
                print("✅ Đã thêm phương thức to() cho ImprovedCompressorVNVC")
            else:
                print("❌ Không tìm thấy vị trí phù hợp để thêm to()")
        else:
            print("❌ Không tìm thấy compute_rate_distortion_loss")
    else:
        print("✅ Phương thức to() đã tồn tại")
    
    # Thêm phương thức to() cho ImprovedMultiLambdaCompressorVNVC
    if "def to(self, device):" not in content or "def to(self, device):" not in content.split("ImprovedMultiLambdaCompressorVNVC")[1]:
        multilambda_to_method = '''
    def to(self, device):
        """Chuyển toàn bộ model sang device chỉ định"""
        super().to(device)
        for lambda_key, compressor in self.compressors.items():
            compressor.to(device)
        return self
'''
        
        # Tìm vị trí để thêm phương thức
        multilambda_pos = content.find("class ImprovedMultiLambdaCompressorVNVC")
        if multilambda_pos > 0:
            insert_pos = content.find("def update(self):", multilambda_pos)
            if insert_pos > 0:
                # Tìm dòng trước để thêm vào
                last_def_end = content.rfind("}", 0, insert_pos)
                if last_def_end > 0:
                    # Chèn phương thức to() trước update()
                    new_content = content[:last_def_end+1] + multilambda_to_method + content[last_def_end+1:]
                    
                    # Lưu file
                    with open('models/compressor_improved.py', 'w', encoding='utf-8') as f:
                        f.write(new_content)
                        
                    print("✅ Đã thêm phương thức to() cho ImprovedMultiLambdaCompressorVNVC")
                else:
                    print("❌ Không tìm thấy vị trí phù hợp để thêm to()")
            else:
                print("❌ Không tìm thấy update()")
        else:
            print("❌ Không tìm thấy ImprovedMultiLambdaCompressorVNVC")
    else:
        print("✅ Phương thức to() cho ImprovedMultiLambdaCompressorVNVC đã tồn tại")
    
    # Sửa file đánh giá để đảm bảo to(device) được gọi
    eval_file = 'evaluation/codec_metrics.py'
    if os.path.exists(eval_file):
        with open(eval_file, 'r', encoding='utf-8') as f:
            eval_content = f.read()
        
        # Kiểm tra nếu cần thêm .to(device) sau khi khởi tạo model
        if ".to(self.device)" not in eval_content or ".to(self.device)" not in eval_content.split("self.compressor = MultiLambdaCompressorVNVC")[1]:
            # Tìm vị trí để thêm .to(self.device)
            init_pos = eval_content.find("self.compressor = MultiLambdaCompressorVNVC")
            if init_pos > 0:
                # Tìm dấu ) để thêm .to(self.device)
                end_pos = eval_content.find(")", init_pos)
                if end_pos > 0:
                    # Thêm .to(self.device)
                    new_eval_content = eval_content[:end_pos+1] + ".to(self.device)" + eval_content[end_pos+1:]
                    
                    # Lưu file
                    with open(eval_file, 'w', encoding='utf-8') as f:
                        f.write(new_eval_content)
                        
                    print(f"✅ Đã thêm .to(self.device) vào {eval_file}")
                else:
                    print(f"❌ Không tìm thấy dấu ) trong {eval_file}")
            else:
                print(f"❌ Không tìm thấy MultiLambdaCompressorVNVC trong {eval_file}")
        else:
            print(f"✅ .to(self.device) đã tồn tại trong {eval_file}")
    
    print("\n✅ ĐÃ SỬA XONG LỖI DEVICE MISMATCH")
    print("Hãy chạy lại đánh giá!")

if __name__ == "__main__":
    patch_improved_compressor() 