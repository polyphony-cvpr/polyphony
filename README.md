# Polyphony: Diffusion-based Dual-Hand Action Segmentation

### [Paper](link) | [arXiv](link)

**Official PyTorch implementation of "Polyphony: Diffusion-based Dual-Hand Action Segmentation with Alternating Vision Transformer and Semantic Conditioning"**

---

## Overview

Polyphony is a unified three-stage method for dual-hand action segmentation that addresses the unique challenges of dual-hand action segmentation: complex inter-hand dependencies, visual asymmetry, representation conflicts, and semantic ambiguity.

<p align="center">
  <img src="asset/Overall_framework.png" width="70%" alt="Polyphony Framework Overview">
</p>
<p align="center"><i>Figure 1: Overview of Polyphony, a three-stage dual-hand action segmentation method. Stage 1 extracts dual-hand features via a shared ViT and hand-specific classification heads; Stage 2 performs semantic feature conditioning by aligning visual features with compositional action descriptions; Stage 3 conducts diffusion-based segmentation with cross-hand feature fusion. The modular architecture enables: (1) flexible deployment—handles both dual-hand and single-stream tasks with potential extension to multi-agent scenarios; (2) versatile application—the ViT in Stage 1 can operates as a standalone action recognition model; (3) modular design—each component can be improved independently.</i></p>


**Key Features:**
- 🎯 **Unified Model**: Single model with shared backbone outperforms separate per-hand models
- 🔄 **Alternating Training**: Prevents gradient domination and ensures balanced learning for both hands
- 📝 **Semantic Conditioning**: Aligns visual features with structured compositional action descriptions
- 🎨 **Diffusion-based Segmentation**: Cross-hand feature fusion with adaptive loss weighting
- 🚀 **State-of-the-Art**: Best results on HA-ViD and ATTACH dual-hand benchmarks
- 🔧 **Versatile**: ADH-ViT operates standalone for action recognition; extends to single-stream tasks

**Performance Highlights:**
- HA-ViD: 57.1% (LH) / 60.6% (RH) accuracy (+12.0 / +16.8 over previous SOTA)
- ATTACH: 52.8% (LH) / 47.3% (RH) accuracy (+5.3 / +4.8 over previous SOTA)
- Breakfast: 81.8% accuracy (competitive with 10× larger models)

---
