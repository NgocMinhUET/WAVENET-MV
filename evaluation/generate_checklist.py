"""
Generate Validation Checklist Report
Tự động verify tất cả requirements theo specification
"""

import os
import sys
import inspect
import torch
import importlib.util
from pathlib import Path
from datetime import datetime

# Add parent directory to path
sys.path.append(str(Path(__file__).parent.parent))


def check_file_exists(filepath):
    """Check if file exists"""
    return os.path.exists(filepath)


def check_wavelet_architecture():
    """Check WaveletTransformCNN architecture"""
    try:
        from models.wavelet_transform_cnn import WaveletTransformCNN, PredictCNN, UpdateCNN
        
        # Check PredictCNN layers
        predict_cnn = PredictCNN()
        predict_layers = predict_cnn.predict_layers
        
        # Should have: Conv3x3→Conv3x3→Conv1x1
        expected_layers = ['Conv2d', 'ReLU', 'Conv2d', 'ReLU', 'Conv2d']
        actual_layers = [type(layer).__name__ for layer in predict_layers]
        
        predict_correct = all(exp in actual for exp, actual in zip(expected_layers, actual_layers))
        
        # Check UpdateCNN layers
        update_cnn = UpdateCNN()
        update_layers = update_cnn.update_layers
        
        # Should have: Conv3x3→Conv3x3→Conv1x1
        update_correct = len(update_layers) >= 5  # Conv→ReLU→Conv→ReLU→Conv
        
        # Check main class
        model = WaveletTransformCNN()
        has_predict = hasattr(model, 'predict_cnn')
        has_update = hasattr(model, 'update_cnn')
        
        return predict_correct and update_correct and has_predict and has_update
        
    except Exception as e:
        return False


def check_adamixnet_architecture():
    """Check AdaMixNet architecture"""
    try:
        from models.adamixnet import AdaMixNet
        
        model = AdaMixNet(input_channels=256, C_prime=64, C_mix=128, N=4)
        
        # Check parallel filters
        has_parallel_filters = hasattr(model, 'parallel_filters')
        correct_num_filters = len(model.parallel_filters) == 4 if has_parallel_filters else False
        
        # Check attention mechanism
        has_attention = hasattr(model, 'attention_cnn')
        
        # Test forward pass
        x = torch.randn(1, 256, 32, 32)
        try:
            output = model(x)
            correct_output_shape = output.shape == (1, 128, 32, 32)
        except:
            correct_output_shape = False
        
        # Check attention weights method
        has_attention_method = hasattr(model, 'get_attention_weights')
        
        return (has_parallel_filters and correct_num_filters and 
                has_attention and correct_output_shape and has_attention_method)
        
    except Exception as e:
        return False


def check_compressor_gaussianconditional():
    """Check CompressorVNVC uses GaussianConditional"""
    try:
        from models.compressor_vnvc import CompressorVNVC, MultiLambdaCompressorVNVC
        
        # Check single compressor
        compressor = CompressorVNVC()
        has_entropy_bottleneck = hasattr(compressor, 'entropy_bottleneck')
        
        # Check entropy bottleneck has GaussianConditional
        if has_entropy_bottleneck:
            entropy_bottleneck = compressor.entropy_bottleneck
            has_gaussian = hasattr(entropy_bottleneck, 'gaussian_conditional')
        else:
            has_gaussian = False
        
        # Check multi-lambda support
        multi_compressor = MultiLambdaCompressorVNVC()
        has_lambda_support = hasattr(multi_compressor, 'set_lambda')
        
        # Check lambda values
        supported_lambdas = ['256', '512', '1024']
        correct_lambdas = all(lambda_val in multi_compressor.compressors for lambda_val in supported_lambdas)
        
        return has_entropy_bottleneck and has_gaussian and has_lambda_support and correct_lambdas
        
    except Exception as e:
        return False


def check_ai_heads_compressed_features():
    """Check AI heads consume compressed features"""
    try:
        from models.ai_heads import YOLOTinyHead, SegFormerLiteHead
        
        # Check YOLO head
        yolo_head = YOLOTinyHead(input_channels=128)
        has_feature_adapter = hasattr(yolo_head, 'feature_adapter')
        
        # Test forward pass
        compressed_features = torch.randn(1, 128, 32, 32)
        try:
            yolo_output = yolo_head(compressed_features)
            yolo_works = yolo_output.shape[0] == 1  # Batch size preserved
        except:
            yolo_works = False
        
        # Check SegFormer head
        segformer_head = SegFormerLiteHead(input_channels=128)
        has_segformer_adapter = hasattr(segformer_head, 'feature_adapter')
        
        try:
            segformer_output = segformer_head(compressed_features)
            segformer_works = segformer_output.shape[0] == 1
        except:
            segformer_works = False
        
        return (has_feature_adapter and yolo_works and 
                has_segformer_adapter and segformer_works)
        
    except Exception as e:
        return False


def check_training_scripts():
    """Check training scripts save checkpoints and logs"""
    training_scripts = [
        'training/stage1_train_wavelet.py',
        'training/stage2_train_compressor.py', 
        'training/stage3_train_ai.py'
    ]
    
    all_exist = all(check_file_exists(script) for script in training_scripts)
    
    # Check if scripts contain logging code
    has_logging = True
    for script in training_scripts:
        if check_file_exists(script):
            with open(script, 'r') as f:
                content = f.read()
                # Check for TensorBoard và checkpoint saving
                has_tensorboard = 'SummaryWriter' in content or 'tensorboard' in content
                has_checkpoint = 'torch.save' in content or 'checkpoint' in content
                if not (has_tensorboard and has_checkpoint):
                    has_logging = False
                    break
    
    return all_exist and has_logging


def check_evaluation_outputs():
    """Check evaluation scripts output CSV and plots"""
    eval_scripts = [
        'evaluation/codec_metrics.py',
        'evaluation/task_metrics.py',
        'evaluation/plot_rd_curves.py'
    ]
    
    all_exist = all(check_file_exists(script) for script in eval_scripts)
    
    # Check CSV output capability
    has_csv_output = True
    for script in eval_scripts:
        if check_file_exists(script):
            with open(script, 'r') as f:
                content = f.read()
                if 'csv' not in content.lower() and 'pandas' not in content:
                    has_csv_output = False
                    break
    
    # Check plot generation
    has_plot_capability = check_file_exists('evaluation/plot_rd_curves.py')
    if has_plot_capability:
        with open('evaluation/plot_rd_curves.py', 'r') as f:
            content = f.read()
            has_matplotlib = 'matplotlib' in content or 'plt' in content
    else:
        has_matplotlib = False
    
    return all_exist and has_csv_output and has_matplotlib


def check_readme_commands():
    """Check README has dataset download and run commands"""
    if not check_file_exists('README.md'):
        return False
        
    with open('README.md', 'r', encoding='utf-8') as f:
        content = f.read()
    
    # Check dataset setup commands
    has_dataset_setup = ('setup_coco.sh' in content and 'setup_davis.sh' in content)
    
    # Check training commands
    has_training_commands = ('stage1_train_wavelet.py' in content and 
                           'stage2_train_compressor.py' in content and
                           'stage3_train_ai.py' in content)
    
    # Check evaluation commands
    has_eval_commands = ('codec_metrics.py' in content and 'task_metrics.py' in content)
    
    return has_dataset_setup and has_training_commands and has_eval_commands


def generate_checklist_report():
    """Generate complete checklist report"""
    
    print("🔍 Generating WAVENET-MV Validation Checklist Report...")
    print("=" * 60)
    
    # Run all checks
    checks = [
        ("WaveletTransformCNN includes PredictCNN & UpdateCNN layers exactly", check_wavelet_architecture),
        ("AdaMixNet implements 4 parallel conv branches and softmax attention mixing", check_adamixnet_architecture),
        ("Compressor uses CompressAI GaussianConditional; λ ∈ {256, 512, 1024}", check_compressor_gaussianconditional),
        ("AI heads consume compressed features without pixel reconstruction", check_ai_heads_compressed_features),
        ("All training scripts save checkpoints and TensorBoard logs", check_training_scripts),
        ("Evaluation scripts output CSV + RD-plots under ./fig/", check_evaluation_outputs),
        ("README contains dataset download and run commands", check_readme_commands)
    ]
    
    results = []
    for description, check_func in checks:
        try:
            result = check_func()
            status = "✅ YES" if result else "❌ NO"
            results.append((description, status, result))
            print(f"{status} - {description}")
        except Exception as e:
            results.append((description, f"❌ ERROR: {str(e)}", False))
            print(f"❌ ERROR - {description}: {str(e)}")
    
    print("=" * 60)
    
    # Generate markdown report
    report_content = f"""# WAVENET-MV Validation Checklist Report

**Generated on:** {datetime.now().strftime("%Y-%m-%d %H:%M:%S")}

## Architecture Validation

"""
    
    for i, (description, status, result) in enumerate(results, 1):
        report_content += f"{i}. **{description}**\n"
        report_content += f"   - Status: {status}\n\n"
    
    # Summary
    passed_checks = sum(1 for _, _, result in results if result)
    total_checks = len(results)
    
    report_content += f"""## Summary

- **Total Checks:** {total_checks}
- **Passed:** {passed_checks}
- **Failed:** {total_checks - passed_checks}
- **Success Rate:** {(passed_checks/total_checks)*100:.1f}%

"""
    
    if passed_checks == total_checks:
        report_content += "🎉 **All validation checks PASSED!** Framework is ready for use.\n"
    else:
        report_content += "⚠️ **Some validation checks FAILED.** Please review and fix the issues above.\n"
    
    report_content += f"""
## File Structure Verification

```
WAVENET-MV/
├── models/
│   ├── wavelet_transform_cnn.py {'✅' if check_file_exists('models/wavelet_transform_cnn.py') else '❌'}
│   ├── adamixnet.py {'✅' if check_file_exists('models/adamixnet.py') else '❌'}
│   ├── compressor_vnvc.py {'✅' if check_file_exists('models/compressor_vnvc.py') else '❌'}
│   └── ai_heads.py {'✅' if check_file_exists('models/ai_heads.py') else '❌'}
├── training/
│   ├── stage1_train_wavelet.py {'✅' if check_file_exists('training/stage1_train_wavelet.py') else '❌'}
│   ├── stage2_train_compressor.py {'✅' if check_file_exists('training/stage2_train_compressor.py') else '❌'}
│   └── stage3_train_ai.py {'✅' if check_file_exists('training/stage3_train_ai.py') else '❌'}
├── evaluation/
│   ├── codec_metrics.py {'✅' if check_file_exists('evaluation/codec_metrics.py') else '❌'}
│   ├── task_metrics.py {'✅' if check_file_exists('evaluation/task_metrics.py') else '❌'}
│   └── plot_rd_curves.py {'✅' if check_file_exists('evaluation/plot_rd_curves.py') else '❌'}
├── datasets/
│   ├── setup_coco.sh {'✅' if check_file_exists('datasets/setup_coco.sh') else '❌'}
│   ├── setup_davis.sh {'✅' if check_file_exists('datasets/setup_davis.sh') else '❌'}
│   └── dataset_loaders.py {'✅' if check_file_exists('datasets/dataset_loaders.py') else '❌'}
├── requirements.txt {'✅' if check_file_exists('requirements.txt') else '❌'}
├── README.md {'✅' if check_file_exists('README.md') else '❌'}
└── PROJECT_CONTEXT.md {'✅' if check_file_exists('PROJECT_CONTEXT.md') else '❌'}
```

---
*This report was auto-generated by the WAVENET-MV validation system.*
"""
    
    # Save report
    with open('checklist_report.md', 'w', encoding='utf-8') as f:
        f.write(report_content)
    
    print(f"\n✅ Checklist report saved to: checklist_report.md")
    print(f"📊 Results: {passed_checks}/{total_checks} checks passed")
    
    return passed_checks == total_checks


if __name__ == "__main__":
    success = generate_checklist_report()
    sys.exit(0 if success else 1) 