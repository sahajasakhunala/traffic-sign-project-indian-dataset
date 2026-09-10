# Indian Traffic Sign Dataset — Cleaning & Pruning Summary Report

**Source Directory (Preserved)**: `data/Indian_Dataset`
**Clean Target Directory**: `data/Indian_Dataset_Clean`

## 1. Executive Accounting Summary

| Category | Count | Action Taken |
| :--- | :---: | :--- |
| **Original Scanned Images** | `13971` | Baseline input count |
| **Solid Black Dummy Images** | `300` | ❌ Excluded (Conflicting labels across 49 classes) |
| **Ambiguous Cross-Class Conflicts** | `4` | ❌ Excluded (Same image in 2 contradictory classes) |
| **Intra-Class Byte-Duplicate Copies** | `742` | ✂️ Pruned (1 canonical copy preserved per class) |
| **Final Clean Exported Dataset** | **`12925`** | ✅ Exported to `data/Indian_Dataset_Clean` |
| **Difficult / Blurry / Low Contrast Images** | `All retained` | 🛡️ Kept intact (Realistic road challenge samples) |

---

## 2. Impact on Train ↔ Validation Integrity

- **Zero Black Images**: The network will never receive conflicting class gradients for blank black inputs.
- **Zero Exact Train/Val Leakage**: By pruning intra-class byte-for-byte duplicates down to 1 canonical image, the randomized stratified split will have zero overlap between train and validation images.
- **Pure Traffic Sign Representation**: Every single image in `data/Indian_Dataset_Clean` is a genuine, non-synthetic crop.

---

## 3. Per-Class Sample Distribution (Before vs After Cleaning)

| Class ID | Class Name | Original | Black Removed | Dups Pruned | Clean Total |
| :---: | :--- | :---: | :---: | :---: | :---: |
|  0 | Give way                           |   201 |             3 |           9 | **  189** |
|  1 | No entry                           |   201 |             2 |          16 | **  183** |
|  2 | One-way traffic                    |   201 |             5 |           1 | **  195** |
|  3 | One-way traffic                    |   201 |             3 |          11 | **  187** |
|  4 | No vehicles in both directions     |   201 |             3 |          12 | **  186** |
|  5 | No entry for cycles                |   321 |            16 |          14 | **  290** |
|  6 | No entry for goods vehicles        |   201 |             3 |           9 | **  189** |
|  7 | No entry for pedestrians           |   201 |             3 |           4 | **  194** |
|  8 | No entry for bullock carts         |   601 |             9 |          23 | **  569** |
|  9 | No entry for hand carts            |   201 |             0 |          16 | **  185** |
| 10 | No entry for motor vehicles        |   201 |             5 |           7 | **  189** |
| 11 | Height limit                       |   201 |             6 |          10 | **  185** |
| 12 | Weight limit                       |   201 |             3 |          13 | **  185** |
| 13 | Axle weight limit                  |   201 |             3 |          12 | **  186** |
| 14 | Length limit                       |   201 |             5 |           5 | **  191** |
| 15 | No left turn                       |   201 |             3 |          10 | **  188** |
| 16 | No right turn                      |   201 |            12 |           4 | **  185** |
| 17 | No overtaking                      |   201 |             6 |          16 | **  179** |
| 18 | Maximum speed limit (90 km/h)      |   201 |             3 |          10 | **  188** |
| 19 | Maximum speed limit (110 km/h)     |   201 |             1 |          11 | **  189** |
| 20 | Horn prohibited                    |   601 |            36 |          14 | **  551** |
| 21 | No parking                         |   201 |             4 |          14 | **  183** |
| 22 | No stopping                        |   201 |            31 |          16 | **  153** |
| 23 | Turn left                          |   201 |             2 |          11 | **  188** |
| 24 | Turn right                         |   201 |            13 |           9 | **  179** |
| 25 | Steep descent                      |   201 |             1 |          12 | **  188** |
| 26 | Steep ascent                       |   201 |             4 |          11 | **  186** |
| 27 | Narrow road                        |   201 |             2 |          13 | **  186** |
| 28 | Narrow bridge                      |   201 |             8 |          12 | **  181** |
| 29 | Unprotected quay                   |   201 |            12 |          10 | **  179** |
| 30 | Road hump                          |   201 |             0 |          11 | **  190** |
| 31 | Dip                                |   201 |             2 |           7 | **  192** |
| 32 | Loose gravel                       |   201 |             3 |          11 | **  186** |
| 33 | Falling rocks                      |   201 |             3 |           4 | **  194** |
| 34 | Cattle                             |   201 |             1 |          12 | **  188** |
| 35 | Crossroads                         |   201 |             3 |          12 | **  186** |
| 36 | Side road junction                 |   201 |             4 |          13 | **  184** |
| 37 | Side road junction                 |   201 |             0 |          10 | **  191** |
| 38 | Oblique side road junction         |   201 |             4 |           5 | **  192** |
| 40 | T-junction                         |   201 |             2 |           6 | **  193** |
| 41 | Y-junction                         |   201 |             7 |           9 | **  185** |
| 42 | Staggered side road junction       |   201 |             9 |          11 | **  180** |
| 43 | Staggered side road junction       |   201 |             3 |           4 | **  194** |
| 44 | Roundabout                         |   201 |             3 |           5 | **  193** |
| 45 | Guarded level crossing ahead       |   201 |             4 |          14 | **  183** |
| 46 | Unguarded level crossing ahead     |   201 |             3 |          10 | **  188** |
| 47 | Level crossing countdown marker    |   144 |             0 |          10 | **  134** |
| 48 | Level crossing countdown marker    |   168 |             2 |          10 | **  156** |
| 49 | Level crossing countdown marker    |    58 |             0 |           9 | **   49** |
| 50 | Level crossing countdown marker    |   193 |             1 |          16 | **  176** |
| 51 | Parking                            |   600 |             7 |          77 | **  516** |
| 52 | Bus stop                           |   220 |             4 |          14 | **  202** |
| 53 | First aid post                     |   607 |             7 |          29 | **  571** |
| 54 | Telephone                          |   602 |             6 |          38 | **  558** |
| 55 | Filling station                    |   616 |             7 |          29 | **  580** |
| 56 | Hotel                              |   200 |             2 |          15 | **  183** |
| 57 | Restaurant                         |   201 |             4 |           5 | **  192** |
| 58 | Refreshments                       |   196 |             2 |          11 | **  183** |
