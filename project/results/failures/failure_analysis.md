# Phase 4: Failure Analysis


## SIFT
### Best Case: `v_bricks` (Idx 2)
- **Inlier Ratio**: 1.00
- **Detected**: 5000 / 5000
- **Matches**: 2537
- **Inliers (RANSAC)**: 2534
![Best](failures/SIFT_Best_v_bricks_2.png)


### Worst Case: `v_there` (Idx 6)
- **Inlier Ratio**: 0.00
- **Detected**: 634 / 2871
- **Matches**: 11
- **Inliers (RANSAC)**: 4
![Worst](failures/SIFT_Worst_v_there_6.png)


## ORB
### Best Case: `i_contruction` (Idx 2)
- **Inlier Ratio**: 0.98
- **Detected**: 5000 / 5000
- **Matches**: 3869
- **Inliers (RANSAC)**: 3796
![Best](failures/ORB_Best_i_contruction_2.png)


### Worst Case: `i_pencils` (Idx 4)
- **Inlier Ratio**: 0.00
- **Detected**: 5000 / 5000
- **Matches**: 1415
- **Inliers (RANSAC)**: 448
![Worst](failures/ORB_Worst_i_pencils_4.png)


## BRISK
### Best Case: `i_contruction` (Idx 3)
- **Inlier Ratio**: 0.97
- **Detected**: 5983 / 5730
- **Matches**: 4156
- **Inliers (RANSAC)**: 4034
![Best](failures/BRISK_Best_i_contruction_3.png)


### Worst Case: `v_there` (Idx 6)
- **Inlier Ratio**: 0.00
- **Detected**: 285 / 2275
- **Matches**: 134
- **Inliers (RANSAC)**: 5
![Worst](failures/BRISK_Worst_v_there_6.png)


## LATCH
### Best Case: `i_contruction` (Idx 2)
- **Inlier Ratio**: 0.99
- **Detected**: 5000 / 5000
- **Matches**: 3638
- **Inliers (RANSAC)**: 3601
![Best](failures/LATCH_Best_i_contruction_2.png)


### Worst Case: `i_pool` (Idx 4)
- **Inlier Ratio**: 0.00
- **Detected**: 4185 / 1149
- **Matches**: 275
- **Inliers (RANSAC)**: 11
![Worst](failures/LATCH_Worst_i_pool_4.png)

