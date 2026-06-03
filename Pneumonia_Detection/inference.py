#!/usr/bin/env python3
"""
inference.py - Simple Inference Script

Usage:
    python3 inference.py path/to/image.jpg --model models/pneumonia_model.pkl
"""

import argparse
import logging
import sys
from pathlib import Path

try:
    from fastai.vision.all import load_learner, PILImage
except ImportError:
    import subprocess
    subprocess.check_call([sys.executable, "-m", "pip", "install", "-q", "fastai"])
    from fastai.vision.all import load_learner, PILImage

logging.basicConfig(level=logging.INFO, format='%(asctime)s %(levelname)s: %(message)s')
logger = logging.getLogger('inference')


def get_args():
    p = argparse.ArgumentParser(description='Run inference on a chest X-ray image')
    p.add_argument('image', type=Path, help='Path to chest X-ray image file')
    p.add_argument('--model', type=Path, default=Path('models/pneumonia_model.pkl'), help='Path to trained model (pneumonia_model.pkl)')
    return p.parse_args()


def main():
    args = get_args()
    
    # Validate files
    if not args.model.exists():
        logger.error(f'Model file not found: {args.model}')
        logger.info('Train the model first using: python3 train.py')
        sys.exit(1)
    
    if not args.image.exists():
        logger.error(f'Image file not found: {args.image}')
        sys.exit(1)
    
    # Load model
    logger.info(f'Loading model from {args.model}...')
    learn = load_learner(args.model)
    
    # Make prediction
    logger.info(f'Making prediction on {args.image}...')
    img = PILImage.create(args.image)
    pred, idx, probs = learn.predict(img)
    
    confidence = float(probs[idx])
    
    logger.info(f'Prediction: {pred}')
    logger.info(f'Confidence: {confidence:.4f} ({confidence*100:.2f}%)')
    
    # Print result in structured format
    print(f"\n{'='*50}")
    print(f"Prediction Results:")
    print(f"{'='*50}")
    print(f"Image: {args.image}")
    print(f"Predicted Class: {pred}")
    print(f"Confidence Score: {confidence:.4f}")
    print(f"Confidence Percentage: {confidence*100:.2f}%")
    print(f"{'='*50}\n")
    
    # Print all probabilities
    print(f"All Class Probabilities:")
    for i, class_name in enumerate(learn.dls.vocab):
        print(f"  {class_name}: {float(probs[i]):.4f} ({float(probs[i])*100:.2f}%)")


if __name__ == '__main__':
    main()
