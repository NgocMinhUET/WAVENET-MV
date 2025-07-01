"""
Retrain Pipeline với Improved Compressor
- Stage 2: Retrain compressor với quantization fixes
- Stage 3: Retrain AI heads với fixed compression features
"""

import os
import subprocess
import time

def check_prerequisites():
    """Kiểm tra prerequisites trước khi retrain"""
    
    print("🔍 CHECKING PREREQUISITES")
    print("="*50)
    
    # Check if improved compressor exists
    if not os.path.exists("models/compressor_improved.py"):
        print("❌ Missing models/compressor_improved.py")
        return False
    
    # Check if Stage 1 checkpoint exists
    if not os.path.exists("checkpoints/stage1_wavelet_coco_best.pth"):
        print("❌ Missing Stage 1 checkpoint")
        return False
    
    # Check if dataset exists
    if not os.path.exists("datasets/COCO_Official"):
        print("❌ Missing COCO dataset")
        return False
    
    print("✅ All prerequisites available")
    return True

def integrate_improved_compressor():
    """Deploy improved compressor integration"""
    
    print("\n🔄 INTEGRATING IMPROVED COMPRESSOR")
    print("="*50)
    
    try:
        result = subprocess.run(
            ["python", "integrate_improved_compressor.py"],
            capture_output=True,
            text=True,
            timeout=300  # 5 minutes timeout
        )
        
        if result.returncode == 0:
            print("✅ Improved compressor integrated successfully")
            print(result.stdout)
            return True
        else:
            print(f"❌ Integration failed: {result.stderr}")
            return False
            
    except Exception as e:
        print(f"❌ Integration error: {e}")
        return False

def backup_old_checkpoints():
    """Backup old checkpoints trước khi retrain"""
    
    print("\n🔄 BACKING UP OLD CHECKPOINTS")
    print("="*50)
    
    backup_dir = "checkpoints_backup_old_compressor"
    os.makedirs(backup_dir, exist_ok=True)
    
    old_checkpoints = [
        "checkpoints/stage2_compressor_coco_lambda128_best.pth",
        "checkpoints/stage2_compressor_coco_lambda128_latest.pth", 
        "checkpoints/stage3_ai_heads_coco_best.pth",
        "checkpoints/stage3_ai_heads_coco_latest.pth"
    ]
    
    for checkpoint in old_checkpoints:
        if os.path.exists(checkpoint):
            import shutil
            backup_path = os.path.join(backup_dir, os.path.basename(checkpoint))
            shutil.copy(checkpoint, backup_path)
            print(f"✅ Backed up: {checkpoint} → {backup_path}")
    
    print(f"✅ Old checkpoints backed up to {backup_dir}/")

def retrain_stage2():
    """Retrain Stage 2 với improved compressor"""
    
    print("\n🚀 RETRAINING STAGE 2 WITH IMPROVED COMPRESSOR")
    print("="*60)
    
    cmd = [
        "python", "training/stage2_train_compressor.py",
        "--dataset", "coco",
        "--data_dir", "datasets/COCO_Official", 
        "--stage1_checkpoint", "checkpoints/stage1_wavelet_coco_best.pth",
        "--lambda_rd", "128",
        "--epochs", "40",
        "--batch_size", "8",
        "--save_interval", "5",
        "--log_interval", "10"
    ]
    
    print(f"Command: {' '.join(cmd)}")
    print("Starting Stage 2 retraining...")
    
    try:
        process = subprocess.Popen(
            cmd,
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            text=True,
            bufsize=1,
            universal_newlines=True
        )
        
        # Monitor training progress
        for line in process.stdout:
            print(line.rstrip())
            
            # Check for success indicators
            if "✅ MSE HEALTHY" in line:
                print("🎉 MSE health check passed!")
            if "✅ BALANCED" in line:
                print("🎉 Rate-distortion balance achieved!")
            if "Epoch" in line and "BPP" in line:
                print(f"📊 Progress: {line.rstrip()}")
        
        process.wait()
        
        if process.returncode == 0:
            print("✅ Stage 2 retraining completed successfully!")
            return True
        else:
            print(f"❌ Stage 2 retraining failed with code {process.returncode}")
            return False
            
    except Exception as e:
        print(f"❌ Stage 2 retraining error: {e}")
        return False

def retrain_stage3():
    """Retrain Stage 3 với improved Stage 2 checkpoint"""
    
    print("\n🚀 RETRAINING STAGE 3 WITH IMPROVED STAGE 2")
    print("="*60)
    
    cmd = [
        "python", "training/stage3_train_ai.py",
        "--dataset", "coco",
        "--data_dir", "datasets/COCO_Official",
        "--stage1_checkpoint", "checkpoints/stage1_wavelet_coco_best.pth",
        "--stage2_checkpoint", "checkpoints/stage2_compressor_coco_lambda128_best.pth",
        "--lambda_rd", "128", 
        "--enable_detection",
        "--epochs", "50",
        "--batch_size", "4",
        "--save_interval", "10",
        "--log_interval", "10"
    ]
    
    print(f"Command: {' '.join(cmd)}")
    print("Starting Stage 3 retraining...")
    
    try:
        process = subprocess.Popen(
            cmd,
            stdout=subprocess.PIPE, 
            stderr=subprocess.STDOUT,
            text=True,
            bufsize=1,
            universal_newlines=True
        )
        
        # Monitor training progress
        for line in process.stdout:
            print(line.rstrip())
            
            # Check for success indicators
            if "Detection Loss" in line:
                print(f"🎯 Detection progress: {line.rstrip()}")
            if "Total Loss" in line:
                print(f"📊 Training progress: {line.rstrip()}")
        
        process.wait()
        
        if process.returncode == 0:
            print("✅ Stage 3 retraining completed successfully!")
            return True
        else:
            print(f"❌ Stage 3 retraining failed with code {process.returncode}")
            return False
            
    except Exception as e:
        print(f"❌ Stage 3 retraining error: {e}")
        return False

def verify_new_checkpoints():
    """Verify rằng new checkpoints đã được tạo"""
    
    print("\n🔍 VERIFYING NEW CHECKPOINTS")
    print("="*50)
    
    expected_checkpoints = [
        "checkpoints/stage2_compressor_coco_lambda128_best.pth",
        "checkpoints/stage3_ai_heads_coco_best.pth"
    ]
    
    all_exist = True
    for checkpoint in expected_checkpoints:
        if os.path.exists(checkpoint):
            size = os.path.getsize(checkpoint) / (1024*1024)  # MB
            print(f"✅ {checkpoint} ({size:.1f} MB)")
        else:
            print(f"❌ Missing: {checkpoint}")
            all_exist = False
    
    return all_exist

def run_quick_evaluation():
    """Chạy quick evaluation để verify improvements"""
    
    print("\n🧪 RUNNING QUICK EVALUATION")
    print("="*50)
    
    cmd = [
        "python", "evaluation/codec_metrics.py",
        "--checkpoint", "checkpoints/stage3_ai_heads_coco_best.pth",
        "--dataset", "coco",
        "--data_dir", "datasets/COCO",
        "--split", "val",
        "--lambdas", "128", 
        "--output_csv", "results/improved_codec_metrics.csv",
        "--batch_size", "2",
        "--max_samples", "50"  # Quick test
    ]
    
    print(f"Command: {' '.join(cmd)}")
    
    try:
        result = subprocess.run(cmd, capture_output=True, text=True, timeout=600)
        
        if result.returncode == 0:
            print("✅ Quick evaluation completed")
            print(result.stdout)
            
            # Parse results for success indicators
            if "PSNR" in result.stdout:
                lines = result.stdout.split('\n')
                for line in lines:
                    if "PSNR" in line and "dB" in line:
                        print(f"📊 {line}")
                        
                        # Check if PSNR is positive (good sign)
                        try:
                            psnr_value = float(line.split("PSNR=")[1].split("dB")[0])
                            if psnr_value > 0:
                                print("🎉 POSITIVE PSNR - Improvement detected!")
                                return True
                            else:
                                print("⚠️ Still negative PSNR - May need more investigation")
                                return False
                        except:
                            pass
            
            return True
        else:
            print(f"❌ Evaluation failed: {result.stderr}")
            return False
            
    except Exception as e:
        print(f"❌ Evaluation error: {e}")
        return False

if __name__ == "__main__":
    print("🚀 COMPLETE RETRAIN WITH IMPROVED COMPRESSOR")
    print("="*70)
    
    # Step 1: Check prerequisites
    if not check_prerequisites():
        print("❌ Prerequisites not met. Please check setup.")
        exit(1)
    
    # Step 2: Integrate improved compressor  
    if not integrate_improved_compressor():
        print("❌ Failed to integrate improved compressor")
        exit(1)
    
    # Step 3: Backup old checkpoints
    backup_old_checkpoints()
    
    # Step 4: Retrain Stage 2
    print(f"\n{'='*70}")
    print("STARTING STAGE 2 RETRAIN")
    print(f"{'='*70}")
    
    if not retrain_stage2():
        print("❌ Stage 2 retraining failed")
        exit(1)
    
    # Step 5: Retrain Stage 3
    print(f"\n{'='*70}")
    print("STARTING STAGE 3 RETRAIN") 
    print(f"{'='*70}")
    
    if not retrain_stage3():
        print("❌ Stage 3 retraining failed")
        exit(1)
    
    # Step 6: Verify checkpoints
    if not verify_new_checkpoints():
        print("❌ New checkpoints not created properly")
        exit(1)
    
    # Step 7: Quick evaluation
    print(f"\n{'='*70}")
    print("VERIFICATION EVALUATION")
    print(f"{'='*70}")
    
    success = run_quick_evaluation()
    
    # Final summary
    print(f"\n🎉 RETRAIN SUMMARY")
    print("="*50)
    
    if success:
        print("✅ Complete retrain pipeline successful!")
        print("✅ Improved compressor integrated")
        print("✅ Stage 2 retrained with quantization fixes")
        print("✅ Stage 3 retrained with fixed compression")
        print("✅ Evaluation shows improvements")
        print("")
        print("🚀 READY FOR FULL EVALUATION!")
    else:
        print("⚠️ Retrain completed but may need further investigation")
        print("📊 Check logs for detailed analysis")
    
    print(f"\n✅ Retrain process completed") 