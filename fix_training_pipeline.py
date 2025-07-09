#!/usr/bin/env python3
"""
COMPREHENSIVE TRAINING PIPELINE DIAGNOSTIC AND FIX SCRIPT
Kiểm tra và sửa chữa toàn diện các vấn đề trong WAVENET-MV training pipeline
"""

import os
import sys
import torch
import subprocess
import importlib.util
from pathlib import Path
import json
import shutil
from tqdm import tqdm
import numpy as np
import warnings
warnings.filterwarnings('ignore')


class TrainingPipelineDiagnostic:
    """
    Comprehensive diagnostic and fix system for WAVENET-MV training pipeline
    """
    
    def __init__(self):
        self.project_root = Path(__file__).parent
        self.issues_found = []
        self.fixes_applied = []
        
    def log_issue(self, issue_type, description, severity="WARNING"):
        """Log an issue found during diagnostic"""
        self.issues_found.append({
            'type': issue_type,
            'description': description,
            'severity': severity
        })
        print(f"{'❌' if severity == 'ERROR' else '⚠️'} [{issue_type}] {description}")
        
    def log_fix(self, fix_type, description):
        """Log a fix that was applied"""
        self.fixes_applied.append({
            'type': fix_type,
            'description': description
        })
        print(f"🔧 [FIX] {description}")
    
    def check_environment(self):
        """Check Python environment and dependencies"""
        print("\n🔍 CHECKING ENVIRONMENT")
        print("=" * 50)
        
        # Check Python version
        python_version = sys.version_info
        print(f"Python version: {python_version.major}.{python_version.minor}.{python_version.micro}")
        
        if python_version < (3, 8):
            self.log_issue("ENVIRONMENT", "Python version too old (< 3.8)", "ERROR")
            return False
            
        # Check critical dependencies
        critical_deps = [
            'torch', 'torchvision', 'numpy', 'pandas', 'tqdm', 
            'cv2', 'PIL', 'compressai', 'albumentations'
        ]
        
        missing_deps = []
        for dep in critical_deps:
            try:
                if dep == 'cv2':
                    import cv2
                elif dep == 'PIL':
                    import PIL
                elif dep == 'compressai':
                    import compressai
                elif dep == 'albumentations':
                    import albumentations
                else:
                    __import__(dep)
                print(f"✅ {dep} - OK")
            except ImportError:
                missing_deps.append(dep)
                print(f"❌ {dep} - MISSING")
        
        if missing_deps:
            self.log_issue("DEPENDENCIES", f"Missing dependencies: {missing_deps}", "ERROR")
            return False
        
        # Check CUDA availability
        if torch.cuda.is_available():
            print(f"✅ CUDA available - {torch.cuda.get_device_name(0)}")
        else:
            self.log_issue("CUDA", "CUDA not available - will use CPU", "WARNING")
        
        return True
    
    def check_dataset(self):
        """Check dataset availability and structure"""
        print("\n🔍 CHECKING DATASET")
        print("=" * 50)
        
        # Check COCO dataset
        coco_paths = [
            "datasets/COCO_Official/val2017",
            "datasets/COCO_Official/annotations",
            "datasets/COCO/val2017",
            "datasets/COCO/annotations"
        ]
        
        coco_available = False
        for path in coco_paths:
            if os.path.exists(path):
                print(f"✅ Found COCO dataset at: {path}")
                coco_available = True
                break
        
        if not coco_available:
            self.log_issue("DATASET", "COCO dataset not found", "ERROR")
            return False
        
        # Check dataset loading
        try:
            from datasets.dataset_loaders import COCODatasetLoader
            dataset = COCODatasetLoader(
                data_dir="datasets/COCO_Official" if os.path.exists("datasets/COCO_Official") else "datasets/COCO",
                subset='val',
                image_size=256
            )
            print(f"✅ Dataset loader works - {len(dataset)} images found")
        except Exception as e:
            self.log_issue("DATASET", f"Dataset loader failed: {e}", "ERROR")
            return False
        
        return True
    
    def check_models(self):
        """Check model architectures and compatibility"""
        print("\n🔍 CHECKING MODEL ARCHITECTURES")
        print("=" * 50)
        
        model_files = [
            "models/wavelet_transform_cnn.py",
            "models/adamixnet.py", 
            "models/compressor_vnvc.py",
            "models/ai_heads.py"
        ]
        
        for model_file in model_files:
            if os.path.exists(model_file):
                print(f"✅ {model_file} exists")
                
                # Try to import and instantiate
                try:
                    if "wavelet" in model_file:
                        from models.wavelet_transform_cnn import WaveletTransformCNN
                        model = WaveletTransformCNN(3, 64, 64)
                        print(f"  ✅ WaveletTransformCNN instantiated successfully")
                        
                    elif "adamixnet" in model_file:
                        from models.adamixnet import AdaMixNet
                        model = AdaMixNet(256, 64, 128, 4)
                        print(f"  ✅ AdaMixNet instantiated successfully")
                        
                    elif "compressor" in model_file:
                        from models.compressor_vnvc import CompressorVNVC
                        model = CompressorVNVC(128, 256)
                        print(f"  ✅ CompressorVNVC instantiated successfully")
                        
                    elif "ai_heads" in model_file:
                        from models.ai_heads import YOLOTinyHead
                        model = YOLOTinyHead(128, 80)
                        print(f"  ✅ YOLOTinyHead instantiated successfully")
                        
                except Exception as e:
                    self.log_issue("MODEL", f"Failed to instantiate model from {model_file}: {e}", "ERROR")
                    return False
            else:
                self.log_issue("MODEL", f"Model file missing: {model_file}", "ERROR")
                return False
        
        return True
    
    def check_training_scripts(self):
        """Check training scripts and their compatibility"""
        print("\n🔍 CHECKING TRAINING SCRIPTS")
        print("=" * 50)
        
        training_scripts = [
            "training/stage1_train_wavelet.py",
            "training/stage2_train_compressor.py",
            "training/stage3_train_ai.py"
        ]
        
        for script in training_scripts:
            if os.path.exists(script):
                print(f"✅ {script} exists")
                
                # Check script syntax
                try:
                    with open(script, 'r') as f:
                        content = f.read()
                    
                    # Compile to check syntax
                    compile(content, script, 'exec')
                    print(f"  ✅ Script syntax OK")
                    
                except Exception as e:
                    self.log_issue("TRAINING", f"Script syntax error in {script}: {e}", "ERROR")
                    return False
                    
            else:
                self.log_issue("TRAINING", f"Training script missing: {script}", "ERROR")
                return False
        
        return True
    
    def check_checkpoints(self):
        """Check checkpoint directory and existing checkpoints"""
        print("\n🔍 CHECKING CHECKPOINTS")
        print("=" * 50)
        
        checkpoints_dir = "checkpoints"
        if not os.path.exists(checkpoints_dir):
            os.makedirs(checkpoints_dir)
            self.log_fix("CHECKPOINTS", "Created checkpoints directory")
        
        checkpoint_files = list(Path(checkpoints_dir).glob("*.pth"))
        
        if not checkpoint_files:
            self.log_issue("CHECKPOINTS", "No checkpoints found - training has never completed", "WARNING")
            return False
        else:
            print(f"✅ Found {len(checkpoint_files)} checkpoint files:")
            for ckpt in checkpoint_files:
                print(f"  - {ckpt.name}")
        
        # Check checkpoint integrity
        for ckpt_file in checkpoint_files:
            try:
                checkpoint = torch.load(ckpt_file, map_location='cpu')
                print(f"  ✅ {ckpt_file.name} - loadable")
            except Exception as e:
                self.log_issue("CHECKPOINTS", f"Corrupted checkpoint {ckpt_file.name}: {e}", "ERROR")
        
        return True
    
    def fix_common_issues(self):
        """Apply fixes for common issues"""
        print("\n🔧 APPLYING COMMON FIXES")
        print("=" * 50)
        
        # Fix 1: Create missing directories
        required_dirs = ["checkpoints", "results", "runs", "fig", "tables"]
        for directory in required_dirs:
            if not os.path.exists(directory):
                os.makedirs(directory)
                self.log_fix("DIRECTORY", f"Created missing directory: {directory}")
        
        # Fix 2: Fix dataset loader import issues
        dataset_loader_path = "datasets/dataset_loaders.py"
        if os.path.exists(dataset_loader_path):
            try:
                with open(dataset_loader_path, 'r') as f:
                    content = f.read()
                
                # Fix common import issues
                if "from pycocotools.coco import COCO" in content:
                    # Check if pycocotools is available
                    try:
                        import pycocotools
                        print("  ✅ pycocotools available")
                    except ImportError:
                        self.log_issue("DATASET", "pycocotools not available - will use fallback", "WARNING")
                        
            except Exception as e:
                self.log_issue("DATASET", f"Error checking dataset loader: {e}", "WARNING")
        
        # Fix 3: Create a simple training verification script
        self.create_training_verification_script()
        
        return True
    
    def create_training_verification_script(self):
        """Create a simple script to verify training pipeline"""
        
        verification_script = """#!/usr/bin/env python3
'''
Simple Training Pipeline Verification Script
'''
import torch
import sys
from pathlib import Path

sys.path.append(str(Path(__file__).parent))

def verify_stage1():
    '''Verify Stage 1 can run'''
    try:
        from models.wavelet_transform_cnn import WaveletTransformCNN
        model = WaveletTransformCNN(3, 64, 64)
        
        # Test forward pass
        x = torch.randn(1, 3, 256, 256)
        y = model(x)
        
        print(f"✅ Stage 1 verification passed - output shape: {y.shape}")
        return True
    except Exception as e:
        print(f"❌ Stage 1 verification failed: {e}")
        return False

def verify_stage2():
    '''Verify Stage 2 can run'''
    try:
        from models.adamixnet import AdaMixNet
        from models.compressor_vnvc import CompressorVNVC
        
        adamix = AdaMixNet(256, 64, 128, 4)
        compressor = CompressorVNVC(128, 256)
        
        # Test forward pass
        x = torch.randn(1, 256, 64, 64)
        y = adamix(x)
        z = compressor(y)
        
        print(f"✅ Stage 2 verification passed - compressed shape: {z['x_hat'].shape}")
        return True
    except Exception as e:
        print(f"❌ Stage 2 verification failed: {e}")
        return False

def main():
    print("🔍 TRAINING PIPELINE VERIFICATION")
    print("=" * 40)
    
    stage1_ok = verify_stage1()
    stage2_ok = verify_stage2()
    
    if stage1_ok and stage2_ok:
        print("\\n✅ All stages verified successfully!")
        print("🚀 Training pipeline is ready to run")
    else:
        print("\\n❌ Some stages failed verification")
        print("🔧 Please check the issues above")

if __name__ == "__main__":
    main()
"""
        
        with open("verify_training_pipeline.py", 'w') as f:
            f.write(verification_script)
        
        self.log_fix("VERIFICATION", "Created training pipeline verification script")
    
    def generate_diagnostic_report(self):
        """Generate comprehensive diagnostic report"""
        print("\n📊 DIAGNOSTIC REPORT")
        print("=" * 50)
        
        # Summary
        total_issues = len(self.issues_found)
        error_count = sum(1 for issue in self.issues_found if issue['severity'] == 'ERROR')
        warning_count = total_issues - error_count
        
        print(f"Total issues found: {total_issues}")
        print(f"  - Errors: {error_count}")
        print(f"  - Warnings: {warning_count}")
        print(f"Fixes applied: {len(self.fixes_applied)}")
        
        # Detailed issues
        if self.issues_found:
            print("\n🔍 DETAILED ISSUES:")
            for i, issue in enumerate(self.issues_found, 1):
                print(f"  {i}. [{issue['type']}] {issue['description']} ({issue['severity']})")
        
        # Applied fixes
        if self.fixes_applied:
            print("\n🔧 APPLIED FIXES:")
            for i, fix in enumerate(self.fixes_applied, 1):
                print(f"  {i}. [{fix['type']}] {fix['description']}")
        
        # Recommendations
        print("\n💡 RECOMMENDATIONS:")
        if error_count > 0:
            print("  ❌ Critical errors found - training will likely fail")
            print("  🔧 Please fix all ERROR-level issues before running training")
        elif warning_count > 0:
            print("  ⚠️ Some warnings found - training may have issues")
            print("  🔧 Consider addressing WARNING-level issues")
        else:
            print("  ✅ No critical issues found - training pipeline looks good")
        
        # Save report to file
        report_data = {
            'issues_found': self.issues_found,
            'fixes_applied': self.fixes_applied,
            'summary': {
                'total_issues': total_issues,
                'error_count': error_count,
                'warning_count': warning_count,
                'fixes_count': len(self.fixes_applied)
            }
        }
        
        with open('diagnostic_report.json', 'w') as f:
            json.dump(report_data, f, indent=2)
        
        print(f"\n📄 Detailed report saved to: diagnostic_report.json")
        
        return error_count == 0  # Return True if no errors
    
    def run_full_diagnostic(self):
        """Run complete diagnostic and fix procedure"""
        print("🔍 WAVENET-MV TRAINING PIPELINE DIAGNOSTIC")
        print("=" * 60)
        
        # Run all checks
        env_ok = self.check_environment()
        dataset_ok = self.check_dataset()
        models_ok = self.check_models()
        training_ok = self.check_training_scripts()
        checkpoints_ok = self.check_checkpoints()
        
        # Apply fixes
        self.fix_common_issues()
        
        # Generate report
        success = self.generate_diagnostic_report()
        
        if success:
            print("\n🎉 DIAGNOSTIC COMPLETED SUCCESSFULLY!")
            print("✅ Training pipeline is ready to run")
        else:
            print("\n❌ DIAGNOSTIC FOUND CRITICAL ISSUES")
            print("🔧 Please fix the issues before running training")
        
        return success


def main():
    """Main function"""
    diagnostic = TrainingPipelineDiagnostic()
    success = diagnostic.run_full_diagnostic()
    
    if success:
        print("\n🚀 NEXT STEPS:")
        print("1. Run training pipeline verification:")
        print("   python verify_training_pipeline.py")
        print("2. If verification passes, run full training:")
        print("   bash server_training.sh")
        print("3. Run JPEG baseline evaluation:")
        print("   bash run_jpeg_evaluation.sh")
    else:
        print("\n🔧 REQUIRED ACTIONS:")
        print("1. Fix all ERROR-level issues shown above")
        print("2. Re-run this diagnostic script")
        print("3. Only proceed with training after all errors are fixed")


if __name__ == "__main__":
    main() 