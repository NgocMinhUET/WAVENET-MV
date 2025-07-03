import torch
import os
import re

def deep_fix_device_mismatch():
    print("🔍 BẮT ĐẦU SỬA TRIỆT ĐỂ LỖI DEVICE MISMATCH")
    print("="*50)
    
    eval_file = 'evaluation/codec_metrics.py'
    if os.path.exists(eval_file):
        with open(eval_file, 'r', encoding='utf-8') as f:
            content = f.read()
        
        load_models_func = content.find("def load_models(self):")
        if load_models_func > 0:
            next_def = content.find("def ", load_models_func + 20)
            if next_def > 0:
                device_check_code = """
        print(f"Moving all models to {self.device}")
        self.wavelet_cnn = self.wavelet_cnn.to(self.device)
        self.adamixnet = self.adamixnet.to(self.device)
        self.compressor = self.compressor.to(self.device)
        
        wavelet_device = next(self.wavelet_cnn.parameters()).device
        adamixnet_device = next(self.adamixnet.parameters()).device
        compressor_device = next(self.compressor.parameters()).device
        print(f"Devices: wavelet={wavelet_device}, adamixnet={adamixnet_device}, compressor={compressor_device}")
        
"""
                end_of_load = content.rfind("print", load_models_func, next_def)
                if end_of_load > 0:
                    line_end = content.find("\n", end_of_load)
                    if line_end > 0:
                        modified_content = content[:line_end+1] + device_check_code + content[line_end+1:]
                        
                        with open(eval_file, 'w', encoding='utf-8') as f:
                            f.write(modified_content)
                        print(f"✅ Đã thêm code kiểm tra device vào {eval_file}")
    
    eb_file = 'models/compressor_vnvc.py'
    if os.path.exists(eb_file):
        with open(eb_file, 'r', encoding='utf-8') as f:
            eb_content = f.read()
        
        eb_class = eb_content.find("class EntropyBottleneck")
        if eb_class > 0:
            if "def to(self, device)" not in eb_content[eb_class:]:
                compress_method = eb_content.find("def compress", eb_class)
                if compress_method > 0:
                    eb_to_method = """
    def to(self, device):
        super().to(device)
        if hasattr(self, 'gaussian_conditional'):
            self.gaussian_conditional.to(device)
        if hasattr(self, 'context_prediction'):
            self.context_prediction.to(device)
        return self
        
"""
                    before_compress = eb_content.rfind("}", eb_class, compress_method)
                    if before_compress > 0:
                        modified_eb_content = eb_content[:before_compress+1] + eb_to_method + eb_content[before_compress+1:]
                        
                        with open(eb_file, 'w', encoding='utf-8') as f:
                            f.write(modified_eb_content)
                        print(f"✅ Đã thêm phương thức to() cho EntropyBottleneck trong {eb_file}")
    
    with open(eval_file, 'r', encoding='utf-8') as f:
        eval_content = f.read()
    
    modified_eval_content = eval_content
    
    patterns = [
        (r"self\.wavelet_cnn = WaveletTransformCNN\((.*?)\)", r"self.wavelet_cnn = WaveletTransformCNN(\1).to(self.device)"),
        (r"self\.adamixnet = AdaMixNet\((.*?)\)", r"self.adamixnet = AdaMixNet(\1).to(self.device)"),
        (r"self\.compressor = MultiLambdaCompressorVNVC\((.*?)\)", r"self.compressor = MultiLambdaCompressorVNVC(\1).to(self.device)")
    ]
    
    for pattern, replacement in patterns:
        modified_eval_content = re.sub(pattern, replacement, modified_eval_content)
    
    with open(eval_file, 'w', encoding='utf-8') as f:
        f.write(modified_eval_content)
    print(f"✅ Đã sửa instantiation patterns trong {eval_file}")
    
    with open(eb_file, 'r', encoding='utf-8') as f:
        compressor_content = f.read()
    
    compressor_class = compressor_content.find("class CompressorVNVC")
    if compressor_class > 0:
        if "def to(self, device)" not in compressor_content[compressor_class:compressor_class+5000]:
            compute_rd_method = compressor_content.find("def compute_rate_distortion_loss", compressor_class)
            if compute_rd_method > 0:
                compressor_to_method = """
    def to(self, device):
        super().to(device)
        self.analysis_transform.to(device)
        self.synthesis_transform.to(device)
        self.quantizer.to(device)
        self.entropy_bottleneck.to(device)
        return self
        
"""
                before_compute = compressor_content.rfind("}", compressor_class, compute_rd_method)
                if before_compute > 0:
                    modified_compressor_content = compressor_content[:before_compute+1] + compressor_to_method + compressor_content[before_compute+1:]
                    
                    with open(eb_file, 'w', encoding='utf-8') as f:
                        f.write(modified_compressor_content)
                    print(f"✅ Đã thêm phương thức to() cho CompressorVNVC trong {eb_file}")
    
    with open(eb_file, 'r', encoding='utf-8') as f:
        multi_content = f.read()
    
    multi_class = multi_content.find("class MultiLambdaCompressorVNVC")
    if multi_class > 0:
        if "def to(self, device)" not in multi_content[multi_class:multi_class+5000]:
            update_method = multi_content.find("def update", multi_class)
            if update_method > 0:
                multi_to_method = """
    def to(self, device):
        super().to(device)
        for lambda_key, compressor in self.compressors.items():
            compressor.to(device)
        return self
        
"""
                before_update = multi_content.rfind("}", multi_class, update_method)
                if before_update > 0:
                    modified_multi_content = multi_content[:before_update+1] + multi_to_method + multi_content[before_update+1:]
                    
                    with open(eb_file, 'w', encoding='utf-8') as f:
                        f.write(modified_multi_content)
                    print(f"✅ Đã thêm phương thức to() cho MultiLambdaCompressorVNVC trong {eb_file}")
    
    improved_file = 'models/compressor_improved.py'
    if os.path.exists(improved_file):
        with open(improved_file, 'r', encoding='utf-8') as f:
            improved_content = f.read()
        
        improved_class = improved_content.find("class ImprovedCompressorVNVC")
        if improved_class > 0:
            to_method = improved_content.find("def to(self, device)", improved_class)
            rd_method = improved_content.find("def compute_rate_distortion_loss", improved_class)
            
            if to_method < 0 or to_method > rd_method:
                improved_to_method = """
    def to(self, device):
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
                if rd_method > 0:
                    before_rd = improved_content.rfind("}", improved_class, rd_method)
                    if before_rd > 0:
                        if to_method > 0 and to_method < rd_method:
                            to_end = improved_content.find("}", to_method)
                            if to_end > 0:
                                improved_content = improved_content[:to_method] + improved_content[to_end+1:]
                                before_rd = improved_content.rfind("}", improved_class, rd_method)
                        
                        improved_content = improved_content[:before_rd+1] + improved_to_method + improved_content[before_rd+1:]
                        
                        with open(improved_file, 'w', encoding='utf-8') as f:
                            f.write(improved_content)
                        print(f"✅ Đã thêm/sửa phương thức to() cho ImprovedCompressorVNVC trong {improved_file}")
        
        multi_improved_class = improved_content.find("class ImprovedMultiLambdaCompressorVNVC")
        if multi_improved_class > 0:
            update_method = improved_content.find("def update(self):", multi_improved_class)
            if update_method > 0:
                multi_improved_to_method = """
    def to(self, device):
        super().to(device)
        if hasattr(self, 'compressors'):
            for lambda_key, compressor in self.compressors.items():
                if compressor is not None:
                    compressor.to(device)
        return self
    
"""
                before_update = improved_content.rfind("}", multi_improved_class, update_method)
                if before_update > 0:
                    to_method = improved_content.find("def to(self, device)", multi_improved_class)
                    if to_method > 0 and to_method < update_method:
                        to_end = improved_content.find("}", to_method)
                        if to_end > 0:
                            improved_content = improved_content[:to_method] + improved_content[to_end+1:]
                            before_update = improved_content.rfind("}", multi_improved_class, update_method)
                    
                    improved_content = improved_content[:before_update+1] + multi_improved_to_method + improved_content[before_update+1:]
                    
                    with open(improved_file, 'w', encoding='utf-8') as f:
                        f.write(improved_content)
                    print(f"✅ Đã thêm/sửa phương thức to() cho ImprovedMultiLambdaCompressorVNVC trong {improved_file}")
    
    print("\n✅ HOÀN THÀNH SỬA TRIỆT ĐỂ")
    print("="*50)
    print("🚀 Bây giờ hãy thử chạy đánh giá với batch_size=1:")
    print("python evaluation/codec_metrics.py --checkpoint checkpoints/stage3_ai_heads_coco_best.pth --dataset coco --data_dir datasets/COCO --split val --lambdas 128 --batch_size 1 --max_samples 10 --output_csv results/test_fixed.csv")

if __name__ == "__main__":
    deep_fix_device_mismatch() 