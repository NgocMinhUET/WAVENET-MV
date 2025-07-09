#!/usr/bin/env python3
"""
JPEG/JPEG2000 Codec Installation and Testing Script
Kiểm tra và cài đặt codec JPEG/JPEG2000 cần thiết cho evaluation
"""

import os
import sys
import subprocess
import cv2
import numpy as np
from pathlib import Path
import platform
import shutil
import urllib.request
import zipfile
import tarfile


def check_system_info():
    """Kiểm tra thông tin hệ thống"""
    print("🔍 CHECKING SYSTEM INFORMATION")
    print("=" * 50)
    
    print(f"OS: {platform.system()} {platform.release()}")
    print(f"Python: {sys.version}")
    print(f"OpenCV: {cv2.__version__}")
    
    # Check OpenCV build info
    build_info = cv2.getBuildInformation()
    
    # Check JPEG support
    jpeg_support = "JPEG:" in build_info and "YES" in build_info
    print(f"JPEG Support: {'✅ YES' if jpeg_support else '❌ NO'}")
    
    # Check JPEG2000 support
    jpeg2000_support = "JPEG 2000:" in build_info and "YES" in build_info
    print(f"JPEG2000 Support: {'✅ YES' if jpeg2000_support else '❌ NO'}")
    
    return jpeg_support, jpeg2000_support


def install_opencv_full():
    """Cài đặt OpenCV với full codec support"""
    print("\n🔧 INSTALLING OPENCV WITH FULL CODEC SUPPORT")
    print("=" * 50)
    
    try:
        # Uninstall existing opencv
        print("Uninstalling existing OpenCV...")
        subprocess.run([sys.executable, "-m", "pip", "uninstall", "opencv-python", "-y"], 
                      capture_output=True)
        subprocess.run([sys.executable, "-m", "pip", "uninstall", "opencv-contrib-python", "-y"], 
                      capture_output=True)
        
        # Install OpenCV with contrib (has more codec support)
        print("Installing OpenCV with contrib...")
        subprocess.run([sys.executable, "-m", "pip", "install", "opencv-contrib-python"], 
                      check=True)
        
        print("✅ OpenCV with contrib installed successfully")
        return True
        
    except subprocess.CalledProcessError as e:
        print(f"❌ Failed to install OpenCV: {e}")
        return False


def install_pillow_with_jpeg2000():
    """Cài đặt Pillow với JPEG2000 support"""
    print("\n🔧 INSTALLING PILLOW WITH JPEG2000 SUPPORT")
    print("=" * 50)
    
    try:
        # Install Pillow with JPEG2000 support
        subprocess.run([sys.executable, "-m", "pip", "install", "Pillow[jpeg2000]"], 
                      check=True)
        
        # Also install OpenJPEG for better JPEG2000 support
        if platform.system() == "Linux":
            try:
                subprocess.run(["sudo", "apt-get", "update"], check=True)
                subprocess.run(["sudo", "apt-get", "install", "-y", "libopenjp2-7-dev"], check=True)
                print("✅ OpenJPEG installed on Linux")
            except:
                print("⚠️ Could not install OpenJPEG via apt-get")
        
        print("✅ Pillow with JPEG2000 support installed")
        return True
        
    except subprocess.CalledProcessError as e:
        print(f"❌ Failed to install Pillow: {e}")
        return False


def install_imageio_with_codecs():
    """Cài đặt imageio với codec plugins"""
    print("\n🔧 INSTALLING IMAGEIO WITH CODEC PLUGINS")
    print("=" * 50)
    
    try:
        # Install imageio
        subprocess.run([sys.executable, "-m", "pip", "install", "imageio"], check=True)
        
        # Install imageio plugins
        subprocess.run([sys.executable, "-m", "pip", "install", "imageio-ffmpeg"], check=True)
        
        print("✅ ImageIO with codecs installed")
        return True
        
    except subprocess.CalledProcessError as e:
        print(f"❌ Failed to install ImageIO: {e}")
        return False


def test_jpeg_codec():
    """Test JPEG codec functionality"""
    print("\n🧪 TESTING JPEG CODEC")
    print("=" * 30)
    
    try:
        # Create test image
        test_image = np.random.randint(0, 255, (256, 256, 3), dtype=np.uint8)
        
        # Test OpenCV JPEG
        temp_file = "test_jpeg.jpg"
        success = cv2.imwrite(temp_file, test_image, [cv2.IMWRITE_JPEG_QUALITY, 80])
        
        if success and os.path.exists(temp_file):
            # Try to read back
            loaded_image = cv2.imread(temp_file)
            if loaded_image is not None:
                print("✅ OpenCV JPEG: Working")
                os.remove(temp_file)
                return True
        
        print("❌ OpenCV JPEG: Failed")
        return False
        
    except Exception as e:
        print(f"❌ JPEG test failed: {e}")
        return False


def test_jpeg2000_codec():
    """Test JPEG2000 codec functionality"""
    print("\n🧪 TESTING JPEG2000 CODEC")
    print("=" * 30)
    
    # Test OpenCV JPEG2000
    opencv_success = test_opencv_jpeg2000()
    
    # Test Pillow JPEG2000
    pillow_success = test_pillow_jpeg2000()
    
    # Test ImageIO JPEG2000
    imageio_success = test_imageio_jpeg2000()
    
    if opencv_success:
        print("✅ Will use OpenCV for JPEG2000")
        return "opencv"
    elif pillow_success:
        print("✅ Will use Pillow for JPEG2000")
        return "pillow"
    elif imageio_success:
        print("✅ Will use ImageIO for JPEG2000")
        return "imageio"
    else:
        print("❌ No working JPEG2000 codec found")
        return None


def test_opencv_jpeg2000():
    """Test OpenCV JPEG2000"""
    try:
        test_image = np.random.randint(0, 255, (256, 256, 3), dtype=np.uint8)
        temp_file = "test_opencv.jp2"
        
        success = cv2.imwrite(temp_file, test_image, [cv2.IMWRITE_JPEG2000_COMPRESSION_X1000, 10000])
        
        if success and os.path.exists(temp_file):
            loaded_image = cv2.imread(temp_file)
            if loaded_image is not None:
                print("  ✅ OpenCV JPEG2000: Working")
                os.remove(temp_file)
                return True
        
        print("  ❌ OpenCV JPEG2000: Failed")
        return False
        
    except Exception as e:
        print(f"  ❌ OpenCV JPEG2000 test failed: {e}")
        return False


def test_pillow_jpeg2000():
    """Test Pillow JPEG2000"""
    try:
        from PIL import Image
        
        test_image = np.random.randint(0, 255, (256, 256, 3), dtype=np.uint8)
        pil_image = Image.fromarray(test_image)
        temp_file = "test_pillow.jp2"
        
        pil_image.save(temp_file, "JPEG2000", quality_mode="rates", quality_layers=[10])
        
        if os.path.exists(temp_file):
            loaded_image = Image.open(temp_file)
            if loaded_image is not None:
                print("  ✅ Pillow JPEG2000: Working")
                os.remove(temp_file)
                return True
        
        print("  ❌ Pillow JPEG2000: Failed")
        return False
        
    except Exception as e:
        print(f"  ❌ Pillow JPEG2000 test failed: {e}")
        return False


def test_imageio_jpeg2000():
    """Test ImageIO JPEG2000"""
    try:
        import imageio
        
        test_image = np.random.randint(0, 255, (256, 256, 3), dtype=np.uint8)
        temp_file = "test_imageio.jp2"
        
        imageio.imwrite(temp_file, test_image, format='JPEG2000')
        
        if os.path.exists(temp_file):
            loaded_image = imageio.imread(temp_file)
            if loaded_image is not None:
                print("  ✅ ImageIO JPEG2000: Working")
                os.remove(temp_file)
                return True
        
        print("  ❌ ImageIO JPEG2000: Failed")
        return False
        
    except Exception as e:
        print(f"  ❌ ImageIO JPEG2000 test failed: {e}")
        return False


def create_improved_evaluation_script(jpeg2000_method):
    """Tạo script evaluation cải tiến với codec tốt nhất"""
    
    script_content = f'''#!/usr/bin/env python3
"""
IMPROVED JPEG/JPEG2000 Baseline Evaluation Script
Sử dụng codec tốt nhất được phát hiện: {jpeg2000_method}
"""

import os
import sys
import cv2
import numpy as np
import pandas as pd
from tqdm import tqdm
import argparse
from pathlib import Path
import time
import warnings
warnings.filterwarnings('ignore')

# Import codec libraries
{"from PIL import Image" if jpeg2000_method == "pillow" else ""}
{"import imageio" if jpeg2000_method == "imageio" else ""}

# Add parent directory to path
sys.path.append(str(Path(__file__).parent))

from skimage.metrics import structural_similarity as ssim
from skimage.metrics import peak_signal_noise_ratio as psnr


def calculate_metrics(original, compressed):
    """Calculate PSNR and SSIM between original and compressed images"""
    # Ensure same dtype and range
    original = original.astype(np.float32)
    compressed = compressed.astype(np.float32)
    
    # Calculate PSNR
    psnr_value = psnr(original, compressed, data_range=255)
    
    # Calculate SSIM
    if len(original.shape) == 3:
        ssim_value = ssim(original, compressed, data_range=255, channel_axis=2)
    else:
        ssim_value = ssim(original, compressed, data_range=255)
    
    return psnr_value, ssim_value


def evaluate_jpeg(image_path, quality, output_dir):
    """Evaluate JPEG compression"""
    try:
        # Load original image
        original = cv2.imread(image_path)
        if original is None:
            return None
        
        original_rgb = cv2.cvtColor(original, cv2.COLOR_BGR2RGB)
        
        # Compress with JPEG
        temp_path = os.path.join(output_dir, f'temp_jpeg_{{quality}}.jpg')
        cv2.imwrite(temp_path, original, [cv2.IMWRITE_JPEG_QUALITY, quality])
        
        # Load compressed image
        compressed = cv2.imread(temp_path)
        compressed_rgb = cv2.cvtColor(compressed, cv2.COLOR_BGR2RGB)
        
        # Calculate metrics
        psnr_value, ssim_value = calculate_metrics(original_rgb, compressed_rgb)
        
        # Calculate file size and BPP
        file_size = os.path.getsize(temp_path)
        H, W = original.shape[:2]
        bpp = (file_size * 8) / (H * W)
        
        # Clean up
        os.remove(temp_path)
        
        return {{
            'codec': 'JPEG',
            'quality': quality,
            'psnr': psnr_value,
            'ssim': ssim_value,
            'bpp': bpp,
            'file_size': file_size,
            'image_path': image_path
        }}
        
    except Exception as e:
        print(f"Error processing JPEG {{image_path}}: {{e}}")
        return None


def evaluate_jpeg2000(image_path, quality, output_dir):
    """Evaluate JPEG2000 compression using {jpeg2000_method}"""
    try:
        # Load original image
        original = cv2.imread(image_path)
        if original is None:
            return None
        
        original_rgb = cv2.cvtColor(original, cv2.COLOR_BGR2RGB)
        
        # Compress with JPEG2000
        temp_path = os.path.join(output_dir, f'temp_jp2_{{quality}}.jp2')
        
        {"# Using OpenCV" if jpeg2000_method == "opencv" else ""}
        {"compression_ratio = max(1, int(100 - quality + 1))" if jpeg2000_method == "opencv" else ""}
        {"cv2.imwrite(temp_path, original, [cv2.IMWRITE_JPEG2000_COMPRESSION_X1000, compression_ratio * 1000])" if jpeg2000_method == "opencv" else ""}
        
        {"# Using Pillow" if jpeg2000_method == "pillow" else ""}
        {"pil_image = Image.fromarray(original_rgb)" if jpeg2000_method == "pillow" else ""}
        {"quality_layers = [max(1, int(quality/10))]" if jpeg2000_method == "pillow" else ""}
        {"pil_image.save(temp_path, 'JPEG2000', quality_mode='rates', quality_layers=quality_layers)" if jpeg2000_method == "pillow" else ""}
        
        {"# Using ImageIO" if jpeg2000_method == "imageio" else ""}
        {"imageio.imwrite(temp_path, original_rgb, format='JPEG2000')" if jpeg2000_method == "imageio" else ""}
        
        # Load compressed image
        {"compressed = cv2.imread(temp_path)" if jpeg2000_method == "opencv" else ""}
        {"compressed_rgb = cv2.cvtColor(compressed, cv2.COLOR_BGR2RGB)" if jpeg2000_method == "opencv" else ""}
        
        {"compressed_rgb = np.array(Image.open(temp_path))" if jpeg2000_method == "pillow" else ""}
        
        {"compressed_rgb = imageio.imread(temp_path)" if jpeg2000_method == "imageio" else ""}
        
        # Calculate metrics
        psnr_value, ssim_value = calculate_metrics(original_rgb, compressed_rgb)
        
        # Calculate file size and BPP
        file_size = os.path.getsize(temp_path)
        H, W = original.shape[:2]
        bpp = (file_size * 8) / (H * W)
        
        # Clean up
        os.remove(temp_path)
        
        return {{
            'codec': 'JPEG2000',
            'quality': quality,
            'psnr': psnr_value,
            'ssim': ssim_value,
            'bpp': bpp,
            'file_size': file_size,
            'image_path': image_path
        }}
        
    except Exception as e:
        print(f"Error processing JPEG2000 {{image_path}}: {{e}}")
        return None


def main():
    parser = argparse.ArgumentParser(description='Improved JPEG/JPEG2000 Evaluation')
    parser.add_argument('--data_dir', type=str, default='datasets/COCO_Official', 
                        help='Path to COCO dataset')
    parser.add_argument('--max_images', type=int, default=100,
                        help='Maximum number of images to evaluate')
    parser.add_argument('--quality_levels', type=int, nargs='+', 
                        default=[10, 20, 30, 40, 50, 60, 70, 80, 90, 95],
                        help='Quality levels to test')
    parser.add_argument('--output_dir', type=str, default='results',
                        help='Output directory for results')
    parser.add_argument('--output_file', type=str, default='improved_jpeg_results.csv',
                        help='Output CSV file')
    
    args = parser.parse_args()
    
    # Create output directory
    os.makedirs(args.output_dir, exist_ok=True)
    temp_dir = os.path.join(args.output_dir, 'temp')
    os.makedirs(temp_dir, exist_ok=True)
    
    print(f"🔄 Starting Improved JPEG/JPEG2000 Evaluation")
    print(f"📂 Dataset: {{args.data_dir}}")
    print(f"🎯 Max images: {{args.max_images}}")
    print(f"⚙️ Quality levels: {{args.quality_levels}}")
    print(f"🔧 JPEG2000 method: {jpeg2000_method}")
    
    # Get image list
    image_dir = os.path.join(args.data_dir, 'val2017')
    if not os.path.exists(image_dir):
        print(f"❌ Image directory not found: {{image_dir}}")
        return
    
    image_files = list(Path(image_dir).glob('*.jpg'))[:args.max_images]
    print(f"📊 Processing {{len(image_files)}} images...")
    
    # Process images
    all_results = []
    
    for img_path in tqdm(image_files, desc="Processing images"):
        for quality in args.quality_levels:
            # JPEG
            jpeg_result = evaluate_jpeg(str(img_path), quality, temp_dir)
            if jpeg_result:
                all_results.append(jpeg_result)
            
            # JPEG2000
            jp2_result = evaluate_jpeg2000(str(img_path), quality, temp_dir)
            if jp2_result:
                all_results.append(jp2_result)
    
    # Save results
    if all_results:
        df = pd.DataFrame(all_results)
        output_path = os.path.join(args.output_dir, args.output_file)
        df.to_csv(output_path, index=False)
        
        print(f"\\n✅ Results saved to: {{output_path}}")
        print(f"📊 Total results: {{len(all_results)}}")
    else:
        print("❌ No results obtained")
    
    # Clean up temp directory
    try:
        for temp_file in os.listdir(temp_dir):
            os.remove(os.path.join(temp_dir, temp_file))
        os.rmdir(temp_dir)
    except:
        pass


if __name__ == "__main__":
    main()
'''
    
    with open("improved_jpeg_evaluation.py", "w") as f:
        f.write(script_content)
    
    print(f"✅ Created improved evaluation script: improved_jpeg_evaluation.py")
    print(f"🔧 Using {jpeg2000_method} for JPEG2000 compression")


def main():
    """Main function"""
    print("🚀 JPEG/JPEG2000 CODEC INSTALLATION AND TESTING")
    print("=" * 60)
    
    # Check current system
    jpeg_ok, jpeg2000_ok = check_system_info()
    
    # Install codecs if needed
    if not jpeg_ok or not jpeg2000_ok:
        print("\n⚠️ Missing codec support, installing...")
        
        # Install OpenCV with full support
        install_opencv_full()
        
        # Install Pillow with JPEG2000
        install_pillow_with_jpeg2000()
        
        # Install ImageIO with codecs
        install_imageio_with_codecs()
        
        print("\n🔄 Rechecking after installation...")
        jpeg_ok, jpeg2000_ok = check_system_info()
    
    # Test codecs
    jpeg_working = test_jpeg_codec()
    jpeg2000_method = test_jpeg2000_codec()
    
    if jpeg_working and jpeg2000_method:
        print("\n✅ ALL CODECS WORKING!")
        print("🔧 Creating improved evaluation script...")
        create_improved_evaluation_script(jpeg2000_method)
        
        print("\n🚀 READY TO RUN EVALUATION:")
        print("python3 improved_jpeg_evaluation.py --max_images 50")
        
    else:
        print("\n❌ SOME CODECS NOT WORKING")
        print("🔧 Please check the errors above and try manual installation")
        
        if not jpeg_working:
            print("  - JPEG codec failed")
        if not jpeg2000_method:
            print("  - JPEG2000 codec failed")


if __name__ == "__main__":
    main() 