# Indian Traffic Sign Dataset Leaderboard

This table tracks the performance of different experiments on the validation set (15% stratified split).

| Rank | Model | Input Size | Pretrained | Mixup | CutMix | Val Accuracy | Confused Pairs (Top 3) | Notes |
| :---: | --- | :---: | :---: | :---: | :---: | :---: | --- | --- |
| **1** | **ResNet-50** | 128x128 | GTSRB (Champion) | Yes (0.3) | Yes (1.0) | **91.85%** | 37 ↔ 36 (22)<br>24 ↔ 23 (20)<br>42 ↔ 43 (17) | Transfer learning from GTSRB champion checkpoint. Strong feature extraction but directional confusion persists. |
| **2** | **Custom CNN** | 64x64 | None | Yes (0.3) | Yes (1.0) | **86.28%** | - | Trained from scratch. |
