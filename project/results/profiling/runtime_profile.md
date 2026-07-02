# Phase 6: Runtime Profiling

This document breaks down the computational cost of the benchmark pipeline on a subset of 5 sequences.

| Method   |   Image Loading (ms) |   Detection (ms) |   Description (ms) |   Matching (ms) |   Homography Est (ms) |   Metric Comp (ms) |
|:---------|---------------------:|-----------------:|-------------------:|----------------:|----------------------:|-------------------:|
| BRISK    |              3.68674 |          77.3398 |            45.3097 |         32.07   |              18.995   |            34.3757 |
| LATCH    |              3.56356 |          14.1134 |           249.871  |         38.6585 |              21.6913  |            45.01   |
| ORB      |              3.35406 |          19.4018 |            15.4474 |         38.7747 |              22.1079  |            60.9954 |
| SIFT     |              3.94494 |         153.979  |           120.318  |         20.3881 |               2.43058 |            24.8696 |