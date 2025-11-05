# Polyphony: Diffusion-based Dual-Hand Action Segmentation

### [Paper](link) | [arXiv](link)

**Official PyTorch implementation of "Polyphony: Diffusion-based Dual-Hand Action Segmentation with Alternating Vision Transformer and Semantic Conditioning"**

---

## Overview

Polyphony is a unified three-stage framework for dual-hand action segmentation that addresses the unique challenges of modeling bimanual activities: complex inter-hand dependencies, visual asymmetry, representation conflicts, and semantic ambiguity.

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
