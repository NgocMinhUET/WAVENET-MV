#!/usr/bin/env python3
"""
Quick JPEG/JPEG2000 Codec Test Script
Test nhanh xem codec có hoạt động không
"""

import os
import sys
import cv2
import numpy as np
import traceback


def test_basic_imports():
    """Test basic imports"""
    print("🔍 Testing basic imports...")
    
    try:
        import cv2
        print(f"✅ OpenCV {cv2.__version__} imported successfully")
    except ImportError as e:
        print(f"❌ OpenCV import failed: {e}")
        return False
    
    try:
        import numpy as np
        print(f"✅ NumPy {np.__version__} imported successfully")
    except ImportError as e:
        print(f"❌ NumPy import failed: {e}")
        return False
    
    try:
        from PIL import Image
        print(f"✅ Pillow imported successfully")
    except ImportError as e:
        print(f"❌ Pillow import failed: {e}")
        return False
    
    return True


def test_jpeg_simple():
    """Test JPEG compression and decompression"""
    print("\n🧪 Testing JPEG codec...")
    
    try:
        # Create test image
        test_image = np.random.randint(0, 255, (128, 128, 3), dtype=np.uint8)
        
        # Test JPEG compression
        temp_file = "test_simple.jpg"
        
        # Write JPEG
        success = cv2.imwrite(temp_file, test_image, [cv2.IMWRITE_JPEG_QUALITY, 90])
        
        if not success:
            print("❌ JPEG write failed")
            return False
        
        if not os.path.exists(temp_file):
            print("❌ JPEG file not created")
            return False
        
        # Read back
        loaded_image = cv2.imread(temp_file)
        
        if loaded_image is None:
            print("❌ JPEG read failed")
            return False
        
        # Check dimensions
        if loaded_image.shape != test_image.shape:
            print(f"❌ JPEG dimension mismatch: {loaded_image.shape} vs {test_image.shape}")
            return False
        
        # Clean up
        os.remove(temp_file)
        
        print("✅ JPEG codec working properly")
        return True
        
    except Exception as e:
        print(f"❌ JPEG test failed: {e}")
        traceback.print_exc()
        return False


def test_jpeg2000_opencv():
    """Test JPEG2000 with OpenCV"""
    print("\n🧪 Testing JPEG2000 with OpenCV...")
    
    try:
        # Create test image
        test_image = np.random.randint(0, 255, (128, 128, 3), dtype=np.uint8)
        
        # Test JPEG2000 compression
        temp_file = "test_opencv.jp2"
        
        # Write JPEG2000
        success = cv2.imwrite(temp_file, test_image, [cv2.IMWRITE_JPEG2000_COMPRESSION_X1000, 10000])
        
        if not success:
            print("❌ OpenCV JPEG2000 write failed")
            return False
        
        if not os.path.exists(temp_file):
            print("❌ OpenCV JPEG2000 file not created")
            return False
        
        # Read back
        loaded_image = cv2.imread(temp_file)
        
        if loaded_image is None:
            print("❌ OpenCV JPEG2000 read failed")
            return False
        
        # Check dimensions
        if loaded_image.shape != test_image.shape:
            print(f"❌ OpenCV JPEG2000 dimension mismatch: {loaded_image.shape} vs {test_image.shape}")
            return False
        
        # Clean up
        os.remove(temp_file)
        
        print("✅ OpenCV JPEG2000 codec working properly")
        return True
        
    except Exception as e:
        print(f"❌ OpenCV JPEG2000 test failed: {e}")
        return False


def test_jpeg2000_pillow():
    """Test JPEG2000 with Pillow"""
    print("\n🧪 Testing JPEG2000 with Pillow...")
    
    try:
        from PIL import Image
        
        # Create test image
        test_image = np.random.randint(0, 255, (128, 128, 3), dtype=np.uint8)
        pil_image = Image.fromarray(test_image)
        
        # Test JPEG2000 compression
        temp_file = "test_pillow.jp2"
        
        # Write JPEG2000
        pil_image.save(temp_file, "JPEG2000", quality_mode="rates", quality_layers=[10])
        
        if not os.path.exists(temp_file):
            print("❌ Pillow JPEG2000 file not created")
            return False
        
        # Read back
        loaded_image = Image.open(temp_file)
        loaded_array = np.array(loaded_image)
        
        if loaded_array is None:
            print("❌ Pillow JPEG2000 read failed")
            return False
        
        # Check dimensions
        if loaded_array.shape != test_image.shape:
            print(f"❌ Pillow JPEG2000 dimension mismatch: {loaded_array.shape} vs {test_image.shape}")
            return False
        
        # Clean up
        os.remove(temp_file)
        
        print("✅ Pillow JPEG2000 codec working properly")
        return True
        
    except Exception as e:
        print(f"❌ Pillow JPEG2000 test failed: {e}")
        return False


def test_imageio_jpeg2000():
    """Test JPEG2000 with ImageIO"""
    print("\n🧪 Testing JPEG2000 with ImageIO...")
    
    try:
        import imageio
        
        # Create test image
        test_image = np.random.randint(0, 255, (128, 128, 3), dtype=np.uint8)
        
        # Test JPEG2000 compression
        temp_file = "test_imageio.jp2"
        
        # Write JPEG2000
        imageio.imwrite(temp_file, test_image, format='JPEG2000')
        
        if not os.path.exists(temp_file):
            print("❌ ImageIO JPEG2000 file not created")
            return False
        
        # Read back
        loaded_image = imageio.imread(temp_file)
        
        if loaded_image is None:
            print("❌ ImageIO JPEG2000 read failed")
            return False
        
        # Check dimensions
        if loaded_image.shape != test_image.shape:
            print(f"❌ ImageIO JPEG2000 dimension mismatch: {loaded_image.shape} vs {test_image.shape}")
            return False
        
        # Clean up
        os.remove(temp_file)
        
        print("✅ ImageIO JPEG2000 codec working properly")
        return True
        
    except Exception as e:
        print(f"❌ ImageIO JPEG2000 test failed: {e}")
        return False


def main():
    """Main test function"""
    print("🚀 QUICK CODEC TEST")
    print("=" * 40)
    
    # Test basic imports
    if not test_basic_imports():
        print("\n❌ Basic imports failed - please install required packages")
        return False
    
    # Test JPEG
    jpeg_ok = test_jpeg_simple()
    
    # Test JPEG2000 with different methods
    opencv_jp2_ok = test_jpeg2000_opencv()
    pillow_jp2_ok = test_jpeg2000_pillow()
    imageio_jp2_ok = test_jpeg2000_imageio()
    
    # Summary
    print("\n📊 CODEC TEST SUMMARY")
    print("=" * 40)
    print(f"JPEG (OpenCV):     {'✅ OK' if jpeg_ok else '❌ FAILED'}")
    print(f"JPEG2000 (OpenCV): {'✅ OK' if opencv_jp2_ok else '❌ FAILED'}")
    print(f"JPEG2000 (Pillow): {'✅ OK' if pillow_jp2_ok else '❌ FAILED'}")
    print(f"JPEG2000 (ImageIO): {'✅ OK' if imageio_jp2_ok else '❌ FAILED'}")
    
    # Recommendations
    print("\n💡 RECOMMENDATIONS")
    print("=" * 40)
    
    if jpeg_ok:
        print("✅ JPEG codec ready for evaluation")
    else:
        print("❌ JPEG codec not working - please install opencv-python")
    
    if opencv_jp2_ok:
        print("✅ Recommend using OpenCV for JPEG2000 (best performance)")
    elif pillow_jp2_ok:
        print("✅ Recommend using Pillow for JPEG2000 (fallback)")
    elif imageio_jp2_ok:
        print("✅ Recommend using ImageIO for JPEG2000 (fallback)")
    else:
        print("❌ No working JPEG2000 codec found")
        print("🔧 Try installing: pip install opencv-contrib-python pillow[jpeg2000] imageio")
    
    # Overall status
    any_jp2_ok = opencv_jp2_ok or pillow_jp2_ok or imageio_jp2_ok
    
    if jpeg_ok and any_jp2_ok:
        print("\n🎉 ALL CODECS READY FOR EVALUATION!")
        print("🚀 You can now run: python server_jpeg_evaluation.py")
        return True
    else:
        print("\n❌ Some codecs are not working")
        print("🔧 Please fix the issues above before running evaluation")
        return False


if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1) 