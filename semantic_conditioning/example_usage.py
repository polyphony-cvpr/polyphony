"""
Example: How to use the semantic conditioning module
"""

import torch
import numpy as np
from pathlib import Path

# ============================================================================
# Example 1: Training from scratch
# ============================================================================

def train_from_scratch():
    """Train the model from scratch with default settings"""
    print("Example 1: Training from scratch")
    print("=" * 80)
    
    # Simply run the main script
    import main
    
    # The main script will:
    # 1. Load data from config paths
    # 2. Train the model
    # 3. Save checkpoints
    # 4. Generate visualizations
    # 5. Export enhanced features
    
    print("To train, just run: python main.py")
    print("Or customize config.py first!")


# ============================================================================
# Example 2: Loading a trained model
# ============================================================================

def load_trained_model():
    """Load a trained model for inference"""
    print("\nExample 2: Loading a trained model")
    print("=" * 80)
    
    from main import SemanticFeatureAlignmentModel
    import config
    
    # Initialize model
    model = SemanticFeatureAlignmentModel(
        visual_dim=config.VISUAL_DIM,
        semantic_dim=config.SEMANTIC_DIM,
        tcn_hidden_dims=config.TCN_HIDDEN_DIMS
    )
    
    # Load checkpoint
    checkpoint_path = './checkpoints/best_model.pth'
    if Path(checkpoint_path).exists():
        checkpoint = torch.load(checkpoint_path, map_location='cpu')
        model.load_state_dict(checkpoint['model_state_dict'])
        model.eval()
        print(f"✓ Model loaded from {checkpoint_path}")
        print(f"  Epoch: {checkpoint.get('epoch', 'N/A')}")
        print(f"  Val Loss: {checkpoint.get('val_loss', 'N/A'):.6f}")
    else:
        print(f"✗ Checkpoint not found: {checkpoint_path}")
        print("  Train the model first!")
    
    return model


# ============================================================================
# Example 3: Inference on new video
# ============================================================================

def inference_on_video(video_features_path: str):
    """Run inference on a single video"""
    print("\nExample 3: Inference on new video")
    print("=" * 80)
    
    # Load model
    model = load_trained_model()
    
    # Load visual features
    visual_features = np.load(video_features_path)  # Shape: (feature_dim, seq_len)
    visual_features = torch.from_numpy(visual_features.T).float()  # Transpose to (seq_len, feature_dim)
    visual_features = visual_features.unsqueeze(0)  # Add batch dimension: (1, seq_len, feature_dim)
    
    print(f"\nInput shape: {visual_features.shape}")
    
    # Run inference
    with torch.no_grad():
        outputs = model(visual_features, return_intermediate=True)
    
    # Get enhanced features
    enhanced_features = outputs['aligned_features']  # Shape: (1, seq_len, semantic_dim)
    tcn_features = outputs['tcn_features']  # Shape: (1, seq_len, tcn_hidden_dims[-1])
    
    print(f"Enhanced features shape: {enhanced_features.shape}")
    print(f"TCN features shape: {tcn_features.shape}")
    
    # Save enhanced features
    enhanced_np = enhanced_features.squeeze(0).cpu().numpy()  # Remove batch dim
    output_path = 'enhanced_video_features.npy'
    np.save(output_path, enhanced_np.T)  # Transpose to (feature_dim, seq_len)
    print(f"✓ Enhanced features saved to {output_path}")
    
    return enhanced_features


# ============================================================================
# Example 4: Batch inference
# ============================================================================

def batch_inference(video_dir: str, output_dir: str):
    """Run inference on multiple videos"""
    print("\nExample 4: Batch inference")
    print("=" * 80)
    
    from main import SemanticAlignmentEvaluator, create_data_loaders
    import config
    
    # Load model
    model = load_trained_model()
    
    # Create data loader (for test set)
    _, test_loader = create_data_loaders(
        data_root=config.DATA_ROOT,
        train_split=config.TRAIN_SPLIT,
        test_split=config.TEST_SPLIT,
        feature_path=config.FEATURE_PATH,
        annotation_path=config.ANNOTATION_PATH,
        batch_size=4,
        num_workers=0,
        semantic_embeddings_path=config.SEMANTIC_EMBEDDINGS_PATH
    )
    
    # Get action mapping
    action_mapping = test_loader.dataset.action_mapping
    label_to_idx = test_loader.dataset.label_to_idx
    
    # Create evaluator
    evaluator = SemanticAlignmentEvaluator(
        model, torch.device(config.DEVICE), action_mapping, label_to_idx
    )
    
    # Save enhanced features for all videos
    evaluator.save_enhanced_features(test_loader, output_dir)
    
    print(f"✓ Enhanced features saved to {output_dir}/")


# ============================================================================
# Example 5: Using enhanced features for action segmentation
# ============================================================================

def use_enhanced_features_for_segmentation():
    """Show how to use enhanced features with segmentation models"""
    print("\nExample 5: Using enhanced features for action segmentation")
    print("=" * 80)
    
    # Load enhanced features
    enhanced_features_path = 'enhanced_features/video_001.npy'
    
    if not Path(enhanced_features_path).exists():
        print(f"✗ Enhanced features not found: {enhanced_features_path}")
        print("  Run training first to generate enhanced features!")
        return
    
    features = np.load(enhanced_features_path)  # Shape: (feature_dim, seq_len)
    
    print(f"Enhanced features shape: {features.shape}")
    print(f"  Feature dimension: {features.shape[0]}")
    print(f"  Sequence length: {features.shape[1]}")
    
    # These features can now be used with action segmentation models:
    print("\nYou can now use these features with:")
    print("  • MS-TCN")
    print("  • ASFormer")
    print("  • C2F-TCN")
    print("  • FACT")
    print("  • Other temporal segmentation models")
    
    print("\nExample usage with MS-TCN:")
    print("  1. Replace original features with enhanced features")
    print("  2. Run MS-TCN training: python main.py --config ms-tcn_config.json")
    print("  3. Enhanced features should improve segmentation accuracy!")


# ============================================================================
# Example 6: Custom configuration
# ============================================================================

def custom_configuration():
    """Show how to customize configuration"""
    print("\nExample 6: Custom configuration")
    print("=" * 80)
    
    print("\nTo customize training, edit config.py:")
    print("\n# For larger model:")
    print("TCN_HIDDEN_DIMS = [640, 512, 384]")
    print("\n# For smaller GPU:")
    print("BATCH_SIZE = 2")
    print("DOWNSAMPLE_RATE = 2")
    print("\n# For different loss:")
    print("LOSS_TYPE = 'smooth_l1'")
    print("\n# For faster training:")
    print("NUM_EPOCHS = 50")
    print("PATIENCE = 10")
    
    print("\nThen run: python main.py")


# ============================================================================
# Example 7: Evaluation only
# ============================================================================

def evaluate_model():
    """Evaluate a trained model without training"""
    print("\nExample 7: Evaluation only")
    print("=" * 80)
    
    from main import (
        SemanticAlignmentEvaluator, 
        AlignmentVisualizer,
        create_data_loaders
    )
    import config
    
    # Load model
    model = load_trained_model()
    
    # Create data loader
    _, test_loader = create_data_loaders(
        data_root=config.DATA_ROOT,
        train_split=config.TRAIN_SPLIT,
        test_split=config.TEST_SPLIT,
        feature_path=config.FEATURE_PATH,
        annotation_path=config.ANNOTATION_PATH,
        batch_size=4,
        num_workers=0,
        semantic_embeddings_path=config.SEMANTIC_EMBEDDINGS_PATH
    )
    
    # Get action mapping
    action_mapping = test_loader.dataset.action_mapping
    label_to_idx = test_loader.dataset.label_to_idx
    
    # Create evaluator
    evaluator = SemanticAlignmentEvaluator(
        model, torch.device(config.DEVICE), action_mapping, label_to_idx
    )
    
    # Evaluate
    print("\nEvaluating model...")
    eval_results = evaluator.evaluate_alignment_quality(test_loader)
    
    print("\nEvaluation Results:")
    for metric, value in eval_results.items():
        if isinstance(value, float):
            print(f"  {metric}: {value:.6f}")
        else:
            print(f"  {metric}: {value}")
    
    # Generate visualizations
    print("\nGenerating visualizations...")
    visualizer = AlignmentVisualizer(save_dir='./evaluation_results')
    visualizer.generate_full_report(
        model=model,
        data_loader=test_loader,
        device=torch.device(config.DEVICE),
        eval_results=eval_results,
        report_name='evaluation'
    )
    
    print("✓ Evaluation complete!")


# ============================================================================
# Main
# ============================================================================

if __name__ == "__main__":
    print("╔" + "═" * 78 + "╗")
    print("║" + " " * 78 + "║")
    print("║" + "Semantic Conditioning - Example Usage".center(78) + "║")
    print("║" + " " * 78 + "║")
    print("╚" + "═" * 78 + "╝")
    
    print("\nThis file demonstrates various ways to use the semantic conditioning module.")
    print("\nAvailable examples:")
    print("  1. Training from scratch")
    print("  2. Loading a trained model")
    print("  3. Inference on a single video")
    print("  4. Batch inference on multiple videos")
    print("  5. Using enhanced features for action segmentation")
    print("  6. Custom configuration")
    print("  7. Evaluation only (no training)")
    
    print("\n" + "=" * 80)
    print("Note: Uncomment the example you want to run below")
    print("=" * 80)
    
    # Uncomment the example you want to run:
    
    # train_from_scratch()
    # model = load_trained_model()
    # inference_on_video('path/to/video_features.npy')
    # batch_inference('path/to/videos', 'output_enhanced_features')
    # use_enhanced_features_for_segmentation()
    # custom_configuration()
    # evaluate_model()
    
    print("\nFor quick start, simply run:")
    print("  python main.py")
    print("\nFor more details, see:")
    print("  • README.md - Full documentation")
    print("  • QUICK_START.md - 5-step quick start guide")
    print("  • config.py - All configuration options")

