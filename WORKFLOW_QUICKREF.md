# 🚀 WAVENET-MV: Quick Git Workflow Reference

## 📋 **WORKFLOW CHECKLIST**

### **Máy A (Windows - Development)**
```bash
git status                    # Kiểm tra thay đổi
git add .                     # Thêm tất cả files
git commit -m "Fix: [lỗi]"    # Commit với message rõ ràng  
git push origin master        # Push lên remote
```

### **Máy B (Server - Training)**
```bash
git pull origin master       # Pull latest changes
python training/stage1_train_wavelet.py    # Chạy training
```

### **Error Handling**
1. **Copy error** từ Máy B (full traceback)
2. **Paste vào Cursor AI** ở Máy A 
3. **Fix và push:**
   ```bash
   git add .
   git commit -m "Fix: [mô tả lỗi cụ thể]"
   git push origin master
   ```
4. **Pull ở Máy B:** `git pull origin master`

---

## 🔗 **Repository:** 
https://github.com/NgocMinhUET/WAVENET-MV.git

## 📝 **Commit Message Format:**
- `Fix: [mô tả lỗi được sửa]`
- `Feature: [tính năng mới]` 
- `Update: [cập nhật nào]`

## 🚨 **Error Report Template:**
```
Environment: [OS, Python, CUDA version]
Command: [exact command]  
Error: [full traceback]
Git commit: [git rev-parse HEAD]
``` 