# Indian Traffic Sign Dataset — Quality & Leakage Audit Report

**Audit Timestamp**: 2026-09-10 11:53:22
**Total Scanned Files**: `12925` images across `58` classes

## 1. Executive Metric Summary

| Audit Dimension | Detected Count | Severity Level |
| :--- | :---: | :---: |
| **Corrupted / Zero-Byte Files** | `0` | 🔴 CRITICAL |
| **Cross-Class Exact Duplicates (Direct Label Conflicts)** | `0` | 🔴 CRITICAL |
| **Train ↔ Validation Exact Leakage** | `0` | 🔴 CRITICAL |
| **Train ↔ Validation Perceptual Leakage** | `1250` | 🟠 HIGH |
| **Cross-Class Perceptual Matches (Review Flags)** | `4170` | 🟠 HIGH |
| **Severe Blur Samples** (Laplacian var < 50.0) | `16` | 🟠 HIGH |
| **Severe Class Imbalance (< 40 samples)** | `0 classes` | 🟠 HIGH |
| **Low Contrast Samples** ($\text{std} < 18.0$) | `45` | 🟡 MEDIUM |
| **Anomalous Dimensions / Aspect Ratios** | `0` | 🟡 MEDIUM |
| **Intra-Class Exact Duplicate Groups** | `0` | 🟡 MEDIUM |

---

## 2. Ranked Action List

### 🔴 CRITICAL ACTIONS
1. **Cross-Class Label Conflicts**: None found ✅
2. **Train ↔ Validation Exact Leakage**: None found ✅
3. **Corrupted Images**: None found ✅

### 🟠 HIGH PRIORITY ACTIONS
5. **Review 16 Severely Blurry Samples**:
   - Images with extremely low Laplacian variance are unreadable and corrupt arrow/number recognition.
   - Top 10 worst blur: `3/3_original_46_original_46.png_a6798576-768b-4ab9-9b90-c4079e8a5c2c.png_9ab0ebad-2ce9-4ad2-abd5-a7f52bd41981.png` (0.0), `5/5_original_48_original_48.png_8b109244-b265-499a-9ee5-1795bdbefb2d.png_c7ceb7ca-9471-4402-a597-c6441950d051.png` (0.0), `5/5_original_48_original_48.png_8b109244-b265-499a-9ee5-1795bdbefb2d.png_d8e50020-a6d5-4b30-9bec-181d8a365ce3.png` (0.0), `8/8_original_51_original_51.png_6a9df54a-15c3-4843-b72d-b12af7f3bca5.png_3eb0f414-bbc0-4ad2-807d-ade8f6e0072b.png` (0.0), `8/8_original_51_original_51.png_6a9df54a-15c3-4843-b72d-b12af7f3bca5.png_6a40fad2-2872-4fbb-b324-641b38a831cf.png` (0.0), `8/8_original_51_original_51.png_6a9df54a-15c3-4843-b72d-b12af7f3bca5.png_6cfc0c8e-6c98-4bda-b261-41bbdbc82ded.png` (0.0), `16/16_original_59_original_59.png_1f816118-19ee-4cba-a989-d09e10739b1f.png_2c47b4aa-e519-4bb1-a717-0de03446b226.png` (0.0), `20/20_original_63_original_63.png_72672e43-fd55-4471-9de2-d42fcf8e77ae.png_6d5ffe89-242d-4f20-b8cd-52a479b8cd95.png` (0.0), `20/20_original_63_original_63.png_811d2758-1889-4466-9a45-6ae84b7ba9d7.png_cc95cb6e-25a6-4486-9229-56b0e7b6c58e.png` (0.0), `22/22_original_65_original_65.png_45721ed2-259d-43c8-931d-bf44e3cbb4dc.png_0bfb76ab-3eb4-4c9f-8d97-4eb2b8c32017.png` (0.0)
6. **Review 4170 Perceptual Cross-Class Matches**:
   - Visually near-identical signs assigned to different classes. Check if these are directional mirror mismatches:
   - Dist `0`: `53/53_original_96_original_96.png_2ad2b06c-091d-4e43-8f77-136fdd537ddb.png_9762f100-c3ee-41fc-a803-fd3829240af7.png` (First aid post) ↔ `55/55_original_98_original_98.png_3ff15261-2cf2-4dbe-b18d-9e92897abfde.png_8df62e9e-bdf6-4728-ba70-1b42bc6d867c.png` (Filling station)
   - Dist `0`: `23/66_original_66.png_4e2b57cb-982b-4ef2-b3f0-24527eb38085.png` (Turn left) ↔ `24/67_original_67.png_65d044a6-8070-4644-9b47-bacb88621f49.png` (Turn right)
   - Dist `0`: `53/53_original_96_original_96.png_5db185b3-4d4f-401f-8272-8427ff093b88.png_7b536a4f-b8ec-40f8-a233-2580a00b6b4f.png` (First aid post) ↔ `55/98_original_98.png_c8879b04-4e92-4c81-aee0-855b12bcade9.png` (Filling station)
   - Dist `0`: `23/23_original_66_original_66.png_4121f891-61ef-4f0e-8fe8-3df649f276a8.png_00f39e40-bb70-44b8-b068-cd81321fba95.png` (Turn left) ↔ `24/67_original_67.png_57353309-9f0c-45b4-a2db-2edf2b59316e.png` (Turn right)
   - Dist `0`: `42/85_original_85.png_bfc24c09-c84e-4669-b108-5c8f47d6ce25.png` (Staggered side road junction) ↔ `43/86_original_86.png_9df1db23-88f1-4c21-9261-8a83398af2eb.png` (Staggered side road junction)
   - Dist `0`: `53/53_original_96_original_96.png_2ad2b06c-091d-4e43-8f77-136fdd537ddb.png_e004d65a-bbe5-4b0e-a902-bc4964b35ca0.png` (First aid post) ↔ `55/55_original_98_original_98.png_cf8c437f-46f1-4720-b5dd-f58b1b0a7f35.png_4806ca05-5d31-491c-9686-0ccef3174a10.png` (Filling station)
   - Dist `0`: `15/15_original_58_original_58.png_5c8b8933-8f49-4db2-a988-5faffd800fc1.png_dc5e4518-65b1-4cc6-8eaa-9f91fedc81cb.png` (No left turn) ↔ `16/16_original_59_original_59.png_1f816118-19ee-4cba-a989-d09e10739b1f.png_2c47b4aa-e519-4bb1-a717-0de03446b226.png` (No right turn)
   - Dist `0`: `53/53_original_96_original_96.png_5db185b3-4d4f-401f-8272-8427ff093b88.png_7b536a4f-b8ec-40f8-a233-2580a00b6b4f.png` (First aid post) ↔ `55/98_original_98.png_d5574538-1368-45b9-900e-f00eede0b27d.png` (Filling station)
   - Dist `0`: `42/42_original_85_original_85.png_1a6a4566-3c6b-4b38-9e22-8d65921b4fd4.png_2890c39a-9d44-4453-8285-fc2b0dbbfb89.png` (Staggered side road junction) ↔ `43/86_original_86.png_b0277c8b-a7da-488f-9da1-19ef50787d17.png` (Staggered side road junction)
   - Dist `0`: `53/53_original_96_original_96.png_9401da51-bb16-44ca-8aa8-b978f49065d8.png_02b17860-9f8e-4a93-a5ef-7efaf1e7faa9.png` (First aid post) ↔ `55/55_original_98_original_98.png_d5574538-1368-45b9-900e-f00eede0b27d.png_0dc1696e-fc4b-406c-97d3-1d125df5aeb4.png` (Filling station)
   - Dist `0`: `47/47_original_90_original_90.png_36c28a97-56f5-480f-b919-b566cc1d3a59.png_de0a6de2-3403-47c6-a343-a3010375c060.png` (Level crossing countdown marker) ↔ `48/91_original_91.png_58265f18-9e00-4af7-91c1-833ad826f071.png` (Level crossing countdown marker)
   - Dist `0`: `53/96_original_96.png_2ad2b06c-091d-4e43-8f77-136fdd537ddb.png` (First aid post) ↔ `55/55_original_98_original_98.png_cf8c437f-46f1-4720-b5dd-f58b1b0a7f35.png_4806ca05-5d31-491c-9686-0ccef3174a10.png` (Filling station)
   - Dist `0`: `53/96_original_96.png_5ed14c18-50d7-4e32-a0d7-94537d332b58.png` (First aid post) ↔ `55/98_original_98.png_d5574538-1368-45b9-900e-f00eede0b27d.png` (Filling station)
   - Dist `0`: `53/53_original_96_original_96.png_8378ddba-9fc1-49f3-802e-1ea14ce36b30.png_56134611-0f9a-4998-b7a4-e2a0e75f7c08.png` (First aid post) ↔ `55/98_original_98.png_24067a15-73a3-4f24-ac26-2b7e6162d9bc.png` (Filling station)
   - Dist `0`: `53/53_original_96_original_96.png_8ce7456a-0c72-4556-91b5-6b4dd9b71b18.png_044c36ea-703a-4ee0-8099-279f6f394827.png` (First aid post) ↔ `55/55_original_98_original_98.png_7f2d6381-3f33-447b-b801-9a7c021ce0bf.png_6711641a-7756-4813-a64a-0d0b7ee95aae.png` (Filling station)

### 🟡 MEDIUM PRIORITY ACTIONS
7. **Review 0 Intra-Class Duplicate Groups**: Identical redundant copies inside the same class.
8. **Review 45 Low Contrast Samples** and **0 Extreme Aspect Ratio Crops**.

---

## 3. Class Distribution by Stratified Split

| Class ID | Class Name | Total Samples | Train (85%) | Val (15%) |
| :---: | :--- | :---: | :---: | :---: |
| 49 | Level crossing countdown marker | 49 | 42 | 7 |
| 47 | Level crossing countdown marker | 134 | 114 | 20 |
| 22 | No stopping | 153 | 131 | 22 |
| 48 | Level crossing countdown marker | 156 | 133 | 23 |
| 50 | Level crossing countdown marker | 176 | 150 | 26 |
| 17 | No overtaking | 179 | 153 | 26 |
| 24 | Turn right | 179 | 153 | 26 |
| 29 | Unprotected quay | 179 | 153 | 26 |
| 42 | Staggered side road junction | 180 | 153 | 27 |
| 28 | Narrow bridge | 181 | 154 | 27 |
| 1 | No entry | 183 | 156 | 27 |
| 21 | No parking | 183 | 156 | 27 |
| 45 | Guarded level crossing ahead | 183 | 156 | 27 |
| 56 | Hotel | 183 | 156 | 27 |
| 58 | Refreshments | 183 | 156 | 27 |
| 36 | Side road junction | 184 | 157 | 27 |
| 9 | No entry for hand carts | 185 | 158 | 27 |
| 11 | Height limit | 185 | 158 | 27 |
| 12 | Weight limit | 185 | 158 | 27 |
| 16 | No right turn | 185 | 158 | 27 |
| 41 | Y-junction | 185 | 158 | 27 |
| 4 | No vehicles in both directions | 186 | 159 | 27 |
| 13 | Axle weight limit | 186 | 159 | 27 |
| 26 | Steep ascent | 186 | 159 | 27 |
| 27 | Narrow road | 186 | 159 | 27 |
| 32 | Loose gravel | 186 | 159 | 27 |
| 35 | Crossroads | 186 | 159 | 27 |
| 3 | One-way traffic | 187 | 159 | 28 |
| 15 | No left turn | 188 | 160 | 28 |
| 18 | Maximum speed limit (90 km/h) | 188 | 160 | 28 |
| 23 | Turn left | 188 | 160 | 28 |
| 25 | Steep descent | 188 | 160 | 28 |
| 34 | Cattle | 188 | 160 | 28 |
| 46 | Unguarded level crossing ahead | 188 | 160 | 28 |
| 0 | Give way | 189 | 161 | 28 |
| 6 | No entry for goods vehicles | 189 | 161 | 28 |
| 10 | No entry for motor vehicles | 189 | 161 | 28 |
| 19 | Maximum speed limit (110 km/h) | 189 | 161 | 28 |
| 30 | Road hump | 190 | 162 | 28 |
| 14 | Length limit | 191 | 163 | 28 |
| 37 | Side road junction | 191 | 163 | 28 |
| 31 | Dip | 192 | 164 | 28 |
| 38 | Oblique side road junction | 192 | 164 | 28 |
| 57 | Restaurant | 192 | 164 | 28 |
| 40 | T-junction | 193 | 165 | 28 |
| 44 | Roundabout | 193 | 165 | 28 |
| 7 | No entry for pedestrians | 194 | 165 | 29 |
| 33 | Falling rocks | 194 | 165 | 29 |
| 43 | Staggered side road junction | 194 | 165 | 29 |
| 2 | One-way traffic | 195 | 166 | 29 |
| 52 | Bus stop | 202 | 172 | 30 |
| 5 | No entry for cycles | 290 | 247 | 43 |
| 51 | Parking | 516 | 439 | 77 |
| 20 | Horn prohibited | 551 | 469 | 82 |
| 54 | Telephone | 558 | 475 | 83 |
| 8 | No entry for bullock carts | 569 | 484 | 85 |
| 53 | First aid post | 571 | 486 | 85 |
| 55 | Filling station | 580 | 493 | 87 |
