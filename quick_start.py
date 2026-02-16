"""
Quick Start Guide for Fabric Defect Detection

This script provides a simple interface to quickly train and evaluate
the fabric defect detection model with minimal configuration.
"""

import sys
from fabric_defect_detection import FabricDefectDetector


def quick_train(epochs=20, batch_size=32, img_size=224):
    """
    Quick training pipeline with default settings
    
    Args:
        epochs (int): Number of training epochs
        batch_size (int): Batch size for training
        img_size (int): Image size (will be img_size x img_size)
    """
    print("="*70)
    print("FABRIC DEFECT DETECTION - QUICK START")
    print("="*70)
    print(f"\nConfiguration:")
    print(f"  - Epochs: {epochs}")
    print(f"  - Batch Size: {batch_size}")
    print(f"  - Image Size: {img_size}x{img_size}")
    print(f"  - Model: ResNet50 with Transfer Learning")
    print()
    
    # Initialize detector
    detector = FabricDefectDetector(img_size=(img_size, img_size), batch_size=batch_size)
    
    try:
        # Step 1: Download dataset
        print("\n[Step 1/6] Downloading dataset...")
        dataset_path = detector.download_dataset()
        print(f"✓ Dataset downloaded to: {dataset_path}")
        
        # Step 2: Prepare data
        print("\n[Step 2/6] Preparing data generators...")
        train_gen, val_gen = detector.prepare_data_generators(dataset_path, validation_split=0.2)
        print(f"✓ Data prepared - {train_gen.samples} training samples, {val_gen.samples} validation samples")
        
        # Step 3: Build model
        print("\n[Step 3/6] Building ResNet model...")
        num_classes = len(detector.class_names)
        detector.build_resnet_model(num_classes)
        print(f"✓ Model built with {num_classes} output classes")
        
        # Step 4: Train model
        print(f"\n[Step 4/6] Training model for {epochs} epochs...")
        print("(This may take a while depending on your hardware)")
        history = detector.train_model(train_gen, val_gen, epochs=epochs)
        print("✓ Training completed!")
        
        # Step 5: Evaluate model
        print("\n[Step 5/6] Evaluating model...")
        results = detector.evaluate_model(val_gen)
        print(f"✓ Evaluation completed - Accuracy: {results['accuracy']:.4f}")
        
        # Step 6: Generate visualizations
        print("\n[Step 6/6] Generating visualizations...")
        detector.plot_training_history()
        detector.plot_confusion_matrix(results['confusion_matrix'])
        detector.save_model('fabric_defect_quick_model.h5')
        print("✓ All visualizations and model saved!")
        
        # Summary
        print("\n" + "="*70)
        print("TRAINING COMPLETED SUCCESSFULLY!")
        print("="*70)
        print(f"\nFinal Results:")
        print(f"  - Accuracy: {results['accuracy']:.4f} ({results['accuracy']*100:.2f}%)")
        print(f"  - Classes detected: {num_classes}")
        print(f"\nGenerated Files:")
        print(f"  - Model: fabric_defect_quick_model.h5")
        print(f"  - Best checkpoint: best_fabric_defect_model.h5")
        print(f"  - Training history: training_history.png")
        print(f"  - Confusion matrix: confusion_matrix.png")
        print("\n" + "="*70)
        
        return detector, results
        
    except Exception as e:
        print(f"\n✗ Error occurred: {str(e)}")
        import traceback
        traceback.print_exc()
        return None, None


def print_usage():
    """Print usage instructions"""
    print("Usage: python quick_start.py [epochs] [batch_size] [img_size]")
    print("\nExamples:")
    print("  python quick_start.py                # Default: 20 epochs, batch 32, 224x224")
    print("  python quick_start.py 10             # 10 epochs, batch 32, 224x224")
    print("  python quick_start.py 30 16          # 30 epochs, batch 16, 224x224")
    print("  python quick_start.py 50 32 128      # 50 epochs, batch 32, 128x128")
    print("\nNote: Image size must be at least 32 and is recommended to be 224 for ResNet50")


if __name__ == "__main__":
    # Parse command line arguments
    epochs = 20
    batch_size = 32
    img_size = 224
    
    if len(sys.argv) > 1:
        if sys.argv[1] in ['-h', '--help', 'help']:
            print_usage()
            sys.exit(0)
        try:
            epochs = int(sys.argv[1])
        except ValueError:
            print(f"Error: epochs must be an integer, got '{sys.argv[1]}'")
            print_usage()
            sys.exit(1)
    
    if len(sys.argv) > 2:
        try:
            batch_size = int(sys.argv[2])
        except ValueError:
            print(f"Error: batch_size must be an integer, got '{sys.argv[2]}'")
            print_usage()
            sys.exit(1)
    
    if len(sys.argv) > 3:
        try:
            img_size = int(sys.argv[3])
            if img_size < 32:
                print("Error: img_size must be at least 32")
                sys.exit(1)
        except ValueError:
            print(f"Error: img_size must be an integer, got '{sys.argv[3]}'")
            print_usage()
            sys.exit(1)
    
    # Run quick training
    detector, results = quick_train(epochs=epochs, batch_size=batch_size, img_size=img_size)
    
    if detector is None:
        sys.exit(1)
