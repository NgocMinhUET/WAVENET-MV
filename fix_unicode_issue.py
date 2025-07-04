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