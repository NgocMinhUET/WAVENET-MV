"""
Sửa lỗi cú pháp trực tiếp trong models/compressor_improved.py
"""
import re

def fix_syntax_error():
    print("🔧 Đang tìm và sửa lỗi cú pháp...")
    
    # Đọc file
    file_path = 'models/compressor_improved.py'
    try:
        with open(file_path, 'r') as f:
            content = f.read()
        print(f"✅ Đã đọc file {file_path}")
    except Exception as e:
        print(f"❌ Lỗi khi đọc file: {e}")
        return
    
    # 1. Sửa lỗi chuỗi không được đóng đúng cách (unterminated string literal)
    # Tìm tất cả các chuỗi bắt đầu bằng ' hoặc " mà không có kết thúc đúng
    # Pattern cho chuỗi bắt đầu bằng " hoặc ' và kết thúc ở cuối dòng
    pattern = r'["\'][^"\'\n]*$'
    lines = content.split('\n')
    
    fixed = False
    for i, line in enumerate(lines):
        match = re.search(pattern, line)
        if match:
            print(f"🔍 Tìm thấy lỗi chuỗi không kết thúc ở dòng {i+1}:")
            print(f"   {line}")
            
            # Sửa bằng cách thêm dấu đóng chuỗi
            quote = match.group()[0]  # Lấy dấu nháy mở
            lines[i] = line + quote
            print(f"✅ Đã sửa thành: {lines[i]}")
            fixed = True
    
    if fixed:
        content = '\n'.join(lines)
    
    # 2. Thêm phương thức to() cho class ImprovedCompressorVNVC
    to_method = """    def to(self, device):
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
    
    # Tìm vị trí class ImprovedCompressorVNVC
    improved_class_pos = content.find("class ImprovedCompressorVNVC")
    if improved_class_pos > 0:
        # Tìm vị trí để thêm phương thức to()
        forward_method_pos = content.find("def forward(self, x)", improved_class_pos)
        if forward_method_pos > 0:
            # Thêm phương thức to() trước forward()
            if "def to(self, device)" not in content[improved_class_pos:forward_method_pos]:
                content_before = content[:forward_method_pos]
                content_after = content[forward_method_pos:]
                content = content_before + to_method + content_after
                print("✅ Đã thêm phương thức to() cho ImprovedCompressorVNVC")
    
    # 3. Kiểm tra và sửa lỗi docstring
    # Tìm tất cả docstring không được đóng đúng cách
    pattern = r'"""[^"]*"""'
    matches = re.finditer(pattern, content)
    
    # Kiểm tra từng docstring
    for match in matches:
        docstring = match.group()
        if docstring.count('"""') != 2:
            print(f"🔍 Tìm thấy docstring không đóng đúng: {docstring[:50]}...")
            fixed_docstring = docstring + '"""'
            content = content.replace(docstring, fixed_docstring)
            print("✅ Đã sửa docstring")
    
    # 4. Lưu file đã sửa
    try:
        with open(file_path, 'w') as f:
            f.write(content)
        print(f"✅ Đã lưu thành công file {file_path}")
    except Exception as e:
        print(f"❌ Lỗi khi lưu file: {e}")
        return
    
    # 5. Kiểm tra cú pháp
    try:
        compile(content, file_path, 'exec')
        print("✅ Kiểm tra cú pháp: OK!")
    except SyntaxError as e:
        print(f"❌ Vẫn còn lỗi cú pháp: {e}")
        print(f"   Dòng {e.lineno}, cột {e.offset}: {e.text}")
        
        # Hiển thị ngữ cảnh xung quanh lỗi
        if hasattr(e, 'lineno') and e.lineno:
            lines = content.split('\n')
            start = max(0, e.lineno - 5)
            end = min(len(lines), e.lineno + 5)
            
            print("\nNgữ cảnh xung quanh lỗi:")
            for i in range(start, end):
                prefix = ">> " if i == e.lineno - 1 else "   "
                print(f"{prefix}{i+1}: {lines[i]}")

if __name__ == "__main__":
    fix_syntax_error() 