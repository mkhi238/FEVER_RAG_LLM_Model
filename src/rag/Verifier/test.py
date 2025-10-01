import json
from pathlib import Path
from collections import Counter
from config import config


def check_label_balance():
    # Load your training data
    train_file = config.verifier_output_dir / "training_data.json" / "training_data.json"
    
    with open(train_file, 'r') as f:
        examples = json.load(f)
    
    # Count labels
    labels = [ex['label'] for ex in examples]
    label_counts = Counter(labels)
    
    total = len(examples)
    
    print("=" * 50)
    print("TRAINING DATA LABEL DISTRIBUTION")
    print("=" * 50)
    print(f"Total examples: {total}\n")
    
    for label, count in label_counts.items():
        percentage = (count / total) * 100
        print(f"{label:20s}: {count:6d} ({percentage:5.1f}%)")
    
    print("\n" + "=" * 50)
    print("BALANCE ANALYSIS")
    print("=" * 50)
    
    # Check if severely imbalanced
    max_count = max(label_counts.values())
    min_count = min(label_counts.values())
    imbalance_ratio = max_count / min_count
    
    print(f"Imbalance ratio: {imbalance_ratio:.2f}x")
    
    if imbalance_ratio > 3:
        print("\n⚠️  SEVERE IMBALANCE DETECTED")
        print("   Your model likely learned to always predict the majority class")
        print("   Recommendation: Balance your dataset or use class weights")
    elif imbalance_ratio > 1.5:
        print("\n⚠️  MODERATE IMBALANCE")
        print("   Consider using class weights during training")
    else:
        print("\n✓  Dataset is reasonably balanced")
    
    # Show ideal vs actual
    print("\n" + "=" * 50)
    print("IDEAL DISTRIBUTION (33.3% each)")
    print("=" * 50)
    ideal_per_class = total / 3
    
    for label in label_counts.keys():
        actual = label_counts[label]
        ideal = ideal_per_class
        diff = actual - ideal
        print(f"{label:20s}: {actual:6d} (should be ~{ideal:.0f}, diff: {diff:+.0f})")

if __name__ == "__main__":
    check_label_balance()