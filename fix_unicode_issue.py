"""
Script để kiểm tra và sửa các ký tự Unicode đặc biệt trong file Python
Giải quyết lỗi 'SyntaxError: unterminated string literal'
"""

import os
import re
import sys

def fix_unicode_issues(file_path):
    print(f"🔍 Kiểm tra file: {file_path}")
    
    # Đọc nội dung file
    try:
        with open(file_path, 'r', encoding='utf-8') as f:
            content = f.read()
        print("✅ Đã đọc file thành công")
    except Exception as e:
        print(f"❌ Lỗi khi đọc file: {e}")
        return False
    
    # Danh sách các ký tự đặc biệt cần thay thế
    replacements = {
        '→': '->',         # Mũi tên
        '←': '<-',         # Mũi tên ngược
        '⇒': '=>',         # Mũi tên kép
        '⇐': '<=',         # Mũi tên kép ngược
        '≈': '~=',         # Xấp xỉ
        '≠': '!=',         # Không bằng
        '≤': '<=',         # Nhỏ hơn hoặc bằng
        '≥': '>=',         # Lớn hơn hoặc bằng
        '×': 'x',          # Dấu nhân
        '÷': '/',          # Dấu chia
        '…': '...',        # Dấu ba chấm
        ''': "'",          # Nháy đơn kiểu
        ''': "'",          # Nháy đơn kiểu
        '"': '"',          # Nháy kép kiểu
        '"': '"',          # Nháy kép kiểu
        '–': '-',          # Gạch ngang dài
        '—': '--',         # Gạch ngang dài hơn
        '•': '*',          # Dấu chấm đầu dòng
        '·': '.',          # Dấu chấm giữa
        '′': "'",          # Dấu phẩy trên
        '″': '"',          # Dấu nháy kép trên
        '‐': '-',          # Dấu gạch nối
        '‑': '-',          # Dấu gạch nối không ngắt
        '‒': '-',          # Dấu gạch ngang
        '–': '-',          # Dấu gạch ngang en
        '—': '--',         # Dấu gạch ngang em
        '―': '--',         # Dấu gạch ngang ngang
    }
    
    # Đếm số lần thay thế
    replacement_count = 0
    
    # Thực hiện thay thế
    for special_char, replacement in replacements.items():
        if special_char in content:
            count = content.count(special_char)
            if count > 0:
                print(f"🔄 Thay thế '{special_char}' thành '{replacement}' ({count} lần)")
                content = content.replace(special_char, replacement)
                replacement_count += count
    
    # Sửa lỗi f-string có chứa ký tự đặc biệt
    # Tìm các f-string có dạng f"..." hoặc f'...'
    f_string_pattern = r'f["\']([^"\']*?)["\']'
    f_strings = re.findall(f_string_pattern, content)
    
    # Kiểm tra và sửa f-string có thể gây lỗi
    for f_string in f_strings:
        # Nếu f-string chứa ký tự đặc biệt hoặc có dấu ngoặc nhọn lồng nhau
        if any(char in f_string for char in replacements.keys()) or re.search(r'\{[^}]*\{', f_string):
            # Tìm f-string đầy đủ
            full_f_string = re.search(rf'f["\']({re.escape(f_string)})["\']', content)
            if full_f_string:
                # Lấy toàn bộ f-string bao gồm cả f"..." hoặc f'...'
                original = full_f_string.group(0)
                
                # Chuyển đổi sang chuỗi thông thường + phép nối
                # Ví dụ: f"Hello {name}" -> "Hello " + str(name)
                parts = re.split(r'\{(.*?)\}', f_string)
                new_string = ""
                
                # Xây dựng chuỗi thay thế
                for i, part in enumerate(parts):
                    if i % 2 == 0:  # Phần text thường
                        if part:
                            new_string += f'"{part}" + '
                    else:  # Phần biểu thức trong {}
                        new_string += f'str({part}) + '
                
                # Loại bỏ dấu + cuối cùng
                if new_string.endswith(' + '):
                    new_string = new_string[:-3]
                
                # Nếu chuỗi rỗng
                if not new_string:
                    new_string = '""'
                
                # Thay thế f-string gốc
                content = content.replace(original, new_string)
                print(f"🔄 Đã chuyển đổi f-string: {original} -> {new_string}")
                replacement_count += 1
    
    # Nếu không có thay đổi nào
    if replacement_count == 0:
        print("✅ Không tìm thấy ký tự Unicode đặc biệt cần thay thế")
        return True
    
    # Lưu nội dung đã sửa
    try:
        with open(file_path, 'w', encoding='utf-8') as f:
            f.write(content)
        print(f"✅ Đã sửa {replacement_count} ký tự đặc biệt và lưu file")
        return True
    except Exception as e:
        print(f"❌ Lỗi khi lưu file: {e}")
        return False

def main():
    # Kiểm tra tham số dòng lệnh
    if len(sys.argv) > 1:
        file_path = sys.argv[1]
        if not os.path.exists(file_path):
            print(f"❌ File không tồn tại: {file_path}")
            return
        fix_unicode_issues(file_path)
    else:
        # Mặc định kiểm tra file compressor_improved.py
        default_path = 'models/compressor_improved.py'
        if os.path.exists(default_path):
            fix_unicode_issues(default_path)
        else:
            print(f"❌ File mặc định không tồn tại: {default_path}")
            print("Cách sử dụng: python fix_unicode_issue.py [đường_dẫn_file]")

if __name__ == "__main__":
    main() 