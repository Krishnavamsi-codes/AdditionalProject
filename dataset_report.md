# HPatches Dataset Report

## Overview
- **Dataset Path:** `d:\AdditionalProject\hpatches-sequences-release`
- **Total Sequences:** 116 sequences

## Sequence Types
- **Illumination Sequences (`i_*`):** 57 sequences (e.g., `i_ajuntament`, `i_autannes`). These sequences contain image variations under different lighting conditions. The homography between images in an illumination sequence is typically the identity matrix.
- **Viewpoint Sequences (`v_*`):** 59 sequences (e.g., `v_abstract`, `v_adam`). These sequences contain variations in camera viewpoint, causing perspective distortions.

## File Structure per Sequence
Each sequence directory contains exactly 12 files:
- **Images:** 6 `.ppm` image files (`1.ppm`, `2.ppm`, `3.ppm`, `4.ppm`, `5.ppm`, `6.ppm`).
  - `1.ppm` is always the reference image.
  - `2.ppm` to `6.ppm` are the target images exhibiting either illumination or viewpoint changes.
- **Homographies:** 5 text files (`H_1_2`, `H_1_3`, `H_1_4`, `H_1_5`, `H_1_6`).
  - These files contain the ground-truth 3x3 homography matrix mapping points from the reference image (`1.ppm`) to the respective target image.
- **Attributes:** 1 `attribs.txt` file (typically containing metadata flags).

## Homography File Format
The homography matrices are stored as 3 lines of space-separated float values (a 3x3 matrix). 
Example from `v_abstract/H_1_2`:
```
0.7088 -0.010965 -26.07
-0.13602 0.83489 103.19
-0.00023352 -1.5615e-05 1.0004
```

## Image Resolutions
The image resolutions in the dataset vary significantly across sequences. A small sampling shows shapes like:
- 600 x 800
- 480 x 640
- 700 x 1280
- 820 x 1280
- 870 x 1280

The benchmarking framework must correctly handle arbitrary and varying image resolutions natively.
