#!/usr/bin/env python3
"""
Complete automated training script with dependency installation and full pipeline execution.
"""

import subprocess
import sys
import os
from pathlib import Path

def install_dependencies():
    """Install all required dependencies."""
    print("\n" + "="*70)
    print("INSTALLING DEPENDENCIES")
    print("="*70)
    
    packages = [
        ("numpy", "numpy"),
        ("pandas", "pandas"),
        ("matplotlib", "matplotlib"),
        ("seaborn", "seaborn"),
        ("scikit-learn", "sklearn"),
        ("pillow", "PIL"),
        ("opencv-python", "cv2"),
        ("fastai", "fastai"),
    ]
    
    for package_name, import_name in packages:
        try:
            __import__(import_name)
            print(f"✓ {package_name} (already installed)")
        except ImportError:
            print(f"Installing {package_name}...")
            subprocess.check_call(
                [sys.executable, "-m", "pip", "install", "-q", package_name],
                timeout=120
            )
            print(f"✓ {package_name}")

def run_training():
    """Run the training script."""
    print("\n" + "="*70)
    print("STARTING TRAINING PIPELINE")
    print("="*70 + "\n")
    
    os.chdir('/home/mahasri123/Pneumonia_Detection')
    
    # Import and run inline
    import subprocess
    result = subprocess.run(
        [sys.executable, "train.py", 
         "--data_path", "dataset/chest_xray",
         "--epochs", "10",
         "--bs", "32",
         "--output", "models"],
        timeout=600  # 10 minutes timeout
    )
    
    return result.returncode == 0

def verify_outputs():
    """Verify all outputs exist."""
    print("\n" + "="*70)
    print("VERIFYING OUTPUTS")
    print("="*70)
    
    required_files = [
        'models/pneumonia_model.pkl',
        'outputs/confusion_matrix.png',
        'outputs/classification_report.txt',
        'outputs/metrics.json',
        'outputs/roc_curve.png'
    ]
    
    print("\nChecking required files:")
    all_exist = True
    for file_path in required_files:
        path = Path(file_path)
        exists = path.exists()
        status = "✓" if exists else "✗"
        print(f"  {status} {file_path}")
        all_exist = all_exist and exists
    
    return all_exist

if __name__ == "__main__":
    os.chdir('/home/mahasri123/Pneumonia_Detection')
    
    try:
        # Install dependencies
        install_dependencies()
        
        # Run training
        success = run_training()
        
        if success:
            # Verify outputs
            all_files_exist = verify_outputs()
            
            print("\n" + "="*70)
            if all_files_exist:
                print("✓✓✓ PROJECT COMPLETED SUCCESSFULLY ✓✓✓")
            else:
                print("⚠ Training completed but some output files missing")
            print("="*70)
        else:
            print("\n✗ Training failed")
            sys.exit(1)
            
    except Exception as e:
        print(f"\n✗ Error: {e}")
        sys.exit(1)
