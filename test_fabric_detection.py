"""
Test script for Fabric Defect Detection

This script verifies that the model can be instantiated and basic functions work
without actually downloading data or training (which would take too long).
"""

import sys
import numpy as np
from unittest.mock import Mock, patch


def test_imports():
    """Test that all required modules can be imported"""
    print("Testing imports...")
    try:
        import tensorflow as tf
        print(f"  ✓ TensorFlow {tf.__version__}")
    except ImportError as e:
        print(f"  ✗ TensorFlow import failed: {e}")
        return False
    
    try:
        from tensorflow import keras
        print(f"  ✓ Keras")
    except ImportError as e:
        print(f"  ✗ Keras import failed: {e}")
        return False
    
    try:
        import numpy as np
        print(f"  ✓ NumPy {np.__version__}")
    except ImportError as e:
        print(f"  ✗ NumPy import failed: {e}")
        return False
    
    try:
        import pandas as pd
        print(f"  ✓ Pandas {pd.__version__}")
    except ImportError as e:
        print(f"  ✗ Pandas import failed: {e}")
        return False
    
    try:
        import matplotlib.pyplot as plt
        print(f"  ✓ Matplotlib")
    except ImportError as e:
        print(f"  ✗ Matplotlib import failed: {e}")
        return False
    
    try:
        import seaborn as sns
        print(f"  ✓ Seaborn")
    except ImportError as e:
        print(f"  ✗ Seaborn import failed: {e}")
        return False
    
    try:
        from sklearn.metrics import classification_report, confusion_matrix
        print(f"  ✓ Scikit-learn")
    except ImportError as e:
        print(f"  ✗ Scikit-learn import failed: {e}")
        return False
    
    return True


def test_class_instantiation():
    """Test that FabricDefectDetector can be instantiated"""
    print("\nTesting FabricDefectDetector instantiation...")
    try:
        from fabric_defect_detection import FabricDefectDetector
        
        detector = FabricDefectDetector(img_size=(224, 224), batch_size=32)
        print(f"  ✓ FabricDefectDetector created")
        print(f"    - Image size: {detector.img_size}")
        print(f"    - Batch size: {detector.batch_size}")
        return True
    except Exception as e:
        print(f"  ✗ Failed to instantiate: {e}")
        import traceback
        traceback.print_exc()
        return False


def test_model_building():
    """Test that the model can be built"""
    print("\nTesting model building...")
    try:
        from fabric_defect_detection import FabricDefectDetector
        
        detector = FabricDefectDetector(img_size=(224, 224), batch_size=32)
        detector.class_names = ['defect1', 'defect2', 'no_defect']  # Mock class names
        
        model = detector.build_resnet_model(num_classes=3)
        
        print(f"  ✓ Model built successfully")
        print(f"    - Input shape: {model.input_shape}")
        print(f"    - Output shape: {model.output_shape}")
        print(f"    - Total parameters: {model.count_params():,}")
        
        return True
    except Exception as e:
        print(f"  ✗ Failed to build model: {e}")
        import traceback
        traceback.print_exc()
        return False


def test_model_prediction():
    """Test that the model can make predictions on dummy data"""
    print("\nTesting model prediction...")
    try:
        from fabric_defect_detection import FabricDefectDetector
        import numpy as np
        
        detector = FabricDefectDetector(img_size=(224, 224), batch_size=32)
        detector.class_names = ['defect1', 'defect2', 'no_defect']
        
        model = detector.build_resnet_model(num_classes=3)
        
        # Create dummy input
        dummy_input = np.random.rand(1, 224, 224, 3).astype(np.float32)
        
        # Make prediction
        prediction = model.predict(dummy_input, verbose=0)
        
        print(f"  ✓ Model prediction successful")
        print(f"    - Input shape: {dummy_input.shape}")
        print(f"    - Output shape: {prediction.shape}")
        print(f"    - Prediction probabilities: {prediction[0]}")
        print(f"    - Predicted class: {np.argmax(prediction[0])}")
        
        return True
    except Exception as e:
        print(f"  ✗ Failed to predict: {e}")
        import traceback
        traceback.print_exc()
        return False


def test_confusion_matrix_plotting():
    """Test confusion matrix plotting with dummy data"""
    print("\nTesting confusion matrix plotting...")
    try:
        from fabric_defect_detection import FabricDefectDetector
        import numpy as np
        import matplotlib
        matplotlib.use('Agg')  # Use non-interactive backend
        
        detector = FabricDefectDetector(img_size=(224, 224), batch_size=32)
        detector.class_names = ['defect1', 'defect2', 'no_defect']
        
        # Create dummy confusion matrix
        cm = np.array([[50, 5, 3], [7, 45, 2], [4, 3, 60]])
        
        # Plot (but don't show)
        detector.plot_confusion_matrix(cm, save_path='/tmp/test_confusion_matrix.png')
        
        print(f"  ✓ Confusion matrix plotting successful")
        print(f"    - Saved to: /tmp/test_confusion_matrix.png")
        
        return True
    except Exception as e:
        print(f"  ✗ Failed to plot confusion matrix: {e}")
        import traceback
        traceback.print_exc()
        return False


def run_all_tests():
    """Run all tests and report results"""
    print("="*70)
    print("FABRIC DEFECT DETECTION - TEST SUITE")
    print("="*70)
    
    tests = [
        ("Import Test", test_imports),
        ("Class Instantiation Test", test_class_instantiation),
        ("Model Building Test", test_model_building),
        ("Model Prediction Test", test_model_prediction),
        ("Confusion Matrix Plotting Test", test_confusion_matrix_plotting),
    ]
    
    results = []
    for test_name, test_func in tests:
        try:
            result = test_func()
            results.append((test_name, result))
        except Exception as e:
            print(f"\n✗ {test_name} crashed: {e}")
            import traceback
            traceback.print_exc()
            results.append((test_name, False))
    
    # Print summary
    print("\n" + "="*70)
    print("TEST SUMMARY")
    print("="*70)
    
    passed = sum(1 for _, result in results if result)
    total = len(results)
    
    for test_name, result in results:
        status = "✓ PASSED" if result else "✗ FAILED"
        print(f"{status}: {test_name}")
    
    print(f"\nTotal: {passed}/{total} tests passed")
    
    if passed == total:
        print("\n✓ All tests passed!")
        return 0
    else:
        print(f"\n✗ {total - passed} test(s) failed")
        return 1


if __name__ == "__main__":
    sys.exit(run_all_tests())
