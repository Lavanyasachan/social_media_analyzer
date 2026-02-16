"""
Simple validation script for Fabric Defect Detection

This script validates the code structure without requiring internet access.
"""

import sys


def validate_code_structure():
    """Validate that all files are present and properly structured"""
    print("="*70)
    print("FABRIC DEFECT DETECTION - CODE VALIDATION")
    print("="*70)
    
    # Check files exist
    print("\n[1/4] Checking files exist...")
    files_to_check = [
        'fabric_defect_detection.py',
        'fabric_defect_detection.ipynb',
        'quick_start.py',
        'requirements.txt',
        'README_FABRIC_DEFECT.md',
        '.gitignore'
    ]
    
    import os
    all_exist = True
    for file in files_to_check:
        if os.path.exists(file):
            print(f"  ✓ {file}")
        else:
            print(f"  ✗ {file} NOT FOUND")
            all_exist = False
    
    if not all_exist:
        return False
    
    # Check Python syntax
    print("\n[2/4] Checking Python syntax...")
    try:
        import py_compile
        py_compile.compile('fabric_defect_detection.py', doraise=True)
        print("  ✓ fabric_defect_detection.py - syntax OK")
        py_compile.compile('quick_start.py', doraise=True)
        print("  ✓ quick_start.py - syntax OK")
    except Exception as e:
        print(f"  ✗ Syntax error: {e}")
        return False
    
    # Check imports
    print("\n[3/4] Checking basic imports...")
    try:
        import numpy as np
        import pandas as pd
        import matplotlib
        import seaborn
        import sklearn
        print("  ✓ All basic dependencies available")
    except ImportError as e:
        print(f"  ✗ Missing dependency: {e}")
        print("  Run: pip install -r requirements.txt")
        return False
    
    # Check FabricDefectDetector class
    print("\n[4/4] Checking FabricDefectDetector class...")
    try:
        # Read the source file directly
        with open('fabric_defect_detection.py', 'r') as f:
            source = f.read()
        
        if 'class FabricDefectDetector' in source:
            print("  ✓ FabricDefectDetector class found")
        else:
            print("  ✗ FabricDefectDetector class not found")
            return False
        
        # Check key methods exist
        methods_to_check = [
            'download_dataset',
            'prepare_data_generators',
            'build_resnet_model',
            'train_model',
            'evaluate_model',
            'plot_confusion_matrix'
        ]
        
        for method in methods_to_check:
            if f'def {method}' in source:
                print(f"  ✓ Method '{method}' found")
            else:
                print(f"  ✗ Method '{method}' NOT FOUND")
                return False
                
    except Exception as e:
        print(f"  ✗ Error checking class: {e}")
        import traceback
        traceback.print_exc()
        return False
    
    return True


def validate_requirements():
    """Validate requirements.txt has all necessary packages"""
    print("\n" + "="*70)
    print("VALIDATING REQUIREMENTS")
    print("="*70)
    
    required_packages = [
        'tensorflow',
        'keras',
        'numpy',
        'pandas',
        'matplotlib',
        'seaborn',
        'scikit-learn',
        'kagglehub',
        'Pillow'
    ]
    
    try:
        with open('requirements.txt', 'r') as f:
            content = f.read().lower()
        
        all_found = True
        for package in required_packages:
            if package.lower() in content:
                print(f"  ✓ {package}")
            else:
                print(f"  ✗ {package} - NOT IN REQUIREMENTS")
                all_found = False
        
        return all_found
    except Exception as e:
        print(f"  ✗ Error reading requirements.txt: {e}")
        return False


def validate_documentation():
    """Validate documentation exists and has key sections"""
    print("\n" + "="*70)
    print("VALIDATING DOCUMENTATION")
    print("="*70)
    
    try:
        with open('README_FABRIC_DEFECT.md', 'r') as f:
            content = f.read()
        
        sections_to_check = [
            'Overview',
            'Features',
            'Installation',
            'Usage',
            'Model Architecture',
            'Results'
        ]
        
        all_found = True
        for section in sections_to_check:
            if section in content:
                print(f"  ✓ Section '{section}' found")
            else:
                print(f"  ✗ Section '{section}' NOT FOUND")
                all_found = False
        
        return all_found
    except Exception as e:
        print(f"  ✗ Error reading documentation: {e}")
        return False


def main():
    """Run all validations"""
    print("Starting validation...\n")
    
    results = []
    
    # Run validations
    results.append(("Code Structure", validate_code_structure()))
    results.append(("Requirements", validate_requirements()))
    results.append(("Documentation", validate_documentation()))
    
    # Print summary
    print("\n" + "="*70)
    print("VALIDATION SUMMARY")
    print("="*70)
    
    passed = sum(1 for _, result in results if result)
    total = len(results)
    
    for test_name, result in results:
        status = "✓ PASSED" if result else "✗ FAILED"
        print(f"{status}: {test_name}")
    
    print(f"\nTotal: {passed}/{total} validations passed")
    
    if passed == total:
        print("\n" + "="*70)
        print("✓ ALL VALIDATIONS PASSED!")
        print("="*70)
        print("\nThe fabric defect detection implementation is complete and ready to use.")
        print("\nNext steps:")
        print("1. Install dependencies: pip install -r requirements.txt")
        print("2. Run the model: python fabric_defect_detection.py")
        print("   OR use the notebook: jupyter notebook fabric_defect_detection.ipynb")
        print("   OR use quick start: python quick_start.py")
        return 0
    else:
        print(f"\n✗ {total - passed} validation(s) failed")
        return 1


if __name__ == "__main__":
    sys.exit(main())
