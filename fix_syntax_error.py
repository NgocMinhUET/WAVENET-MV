"""
Sửa lỗi cú pháp trong compressor_improved.py
Lỗi: SyntaxError: unterminated string literal (detected at line 323)
"""

import os

def fix_syntax_error():
    """Sửa lỗi cú pháp trong compressor_improved.py"""
    print("🔧 ĐANG SỬA LỖI CÚ PHÁP")
    print("="*50)
    
    file_path = 'models/compressor_improved.py'
    if os.path.exists(file_path):
        with open(file_path, 'r', encoding='utf-8') as f:
            content = f.read()
            
        # Tìm vị trí của ImprovedMultiLambdaCompressorVNVC
        multi_improved_class = content.find("class ImprovedMultiLambdaCompressorVNVC")
        if multi_improved_class > 0:
            # Tìm phương thức to() trong class này
            to_method_start = content.find("def to(self, device)", multi_improved_class)
            if to_method_start > 0:
                # Thay thế toàn bộ phương thức to() với phiên bản chính xác
                correct_to_method = """
    def to(self, device):
        super().to(device)
        if hasattr(self, 'compressors'):
            for lambda_key, compressor in self.compressors.items():
                if compressor is not None:
                    compressor.to(device)
        return self
    
"""
                # Tìm vị trí kết thúc của phương thức to() hiện tại
                next_def = content.find("def ", to_method_start + 10)
                if next_def > 0:
                    # Cắt nội dung trước phương thức to()
                    content_before = content[:to_method_start]
                    # Cắt nội dung sau phương thức to()
                    content_after = content[next_def:]
                    
                    # Nối lại với phiên bản đúng của phương thức to()
                    new_content = content_before + correct_to_method + content_after
                    
                    # Lưu file
                    with open(file_path, 'w', encoding='utf-8') as f:
                        f.write(new_content)
                        
                    print("✅ Đã sửa phương thức to() cho ImprovedMultiLambdaCompressorVNVC")
                else:
                    print("❌ Không thể tìm phương thức tiếp theo sau to()")
            else:
                # Nếu không tìm thấy phương thức to(), thêm mới
                update_method = content.find("def update(self):", multi_improved_class)
                if update_method > 0:
                    correct_to_method = """
    def to(self, device):
        super().to(device)
        if hasattr(self, 'compressors'):
            for lambda_key, compressor in self.compressors.items():
                if compressor is not None:
                    compressor.to(device)
        return self
    
"""
                    # Tìm vị trí để thêm phương thức to() vào trước update()
                    before_update = content.rfind("}", multi_improved_class, update_method)
                    if before_update > 0:
                        new_content = content[:before_update+1] + correct_to_method + content[before_update+1:]
                        
                        # Lưu file
                        with open(file_path, 'w', encoding='utf-8') as f:
                            f.write(new_content)
                            
                        print("✅ Đã thêm phương thức to() cho ImprovedMultiLambdaCompressorVNVC")
                    else:
                        print("❌ Không thể tìm vị trí phù hợp để thêm phương thức to()")
                else:
                    print("❌ Không thể tìm phương thức update(self) để định vị")
        else:
            print("❌ Không thể tìm class ImprovedMultiLambdaCompressorVNVC")
            
        # Sửa lỗi cú pháp khác nếu còn
        content = ""
        with open(file_path, 'r', encoding='utf-8') as f:
            for i, line in enumerate(f, 1):
                # Dòng lỗi cú pháp tại 323 
                if i == 323 and '"' in line and not line.strip().endswith('"'):
                    # Fix unterminated string
                    if line.strip().startswith('"'):
                        fixed_line = line.rstrip() + '"\n'
                        content += fixed_line
                        print(f"✅ Đã sửa dòng {i}: {line.strip()} -> {fixed_line.strip()}")
                    else:
                        # Nếu chuỗi ở giữa dòng
                        parts = line.split('"')
                        if len(parts) % 2 == 0:  # Số lượng dấu " lẻ -> thiếu đóng
                            fixed_line = line.rstrip() + '"\n'
                            content += fixed_line
                            print(f"✅ Đã sửa dòng {i}: {line.strip()} -> {fixed_line.strip()}")
                        else:
                            content += line
                else:
                    content += line
                    
        # Lưu lại file sau khi sửa
        with open(file_path, 'w', encoding='utf-8') as f:
            f.write(content)
        
        print("\n✅ ĐÃ SỬA XONG LỖI CÚ PHÁP")
        print("Hãy chạy lại đánh giá")
    else:
        print(f"❌ Không tìm thấy file {file_path}")

if __name__ == "__main__":
    fix_syntax_error() 