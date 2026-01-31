# Quantum Autoencoder (QAE) Training Report

## 1. Overview
This report summarizes the QAE model training process, key metric trends, and final performance.

## 2. Training and Validation
- Training set size: 56000
- Validation set size: 12000
- Test set size: 12000
- Samples used per epoch: 1000
- Total epochs: 240

## 3. Key Metrics
| Metric | Min | Max | Final | Best Epoch |
|--------|-----|-----|-------|------------|
| Avg Train Loss | 0.1929 | 0.9789 | 0.1929 | - |
| Val Fidelity   | 0.0464 | 0.8071 | 0.8071 | 238 |

## 4. Loss and Fidelity Curves

![Train Loss](qae_avg_loss.png)

![Validation Fidelity](qae_val_fidelity.png)

![Loss & Fidelity](qae_loss_fidelity.png)

## 5. Batch Loss Curve

![Batch Loss](qae_batch_loss.png)

## 6. Detailed Data
- Epoch metrics: qae_epoch_metrics.csv
- Batch losses: qae_batch_losses.csv
- Summary: qae_summary.csv

---
Report auto-generated on 2026-01-30
