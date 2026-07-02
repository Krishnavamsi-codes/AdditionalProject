# Phase 7: Parameter Study

Evaluated on a 5-sequence subset to determine hyperparameter sensitivity.

| Method   | Parameter         |   Value |   MMA@3px |
|:---------|:------------------|--------:|----------:|
| SIFT     | contrastThreshold |    0.02 |  0.763448 |
| SIFT     | contrastThreshold |    0.04 |  0.75812  |
| SIFT     | contrastThreshold |    0.08 |  0.708351 |
| ORB      | nfeatures         |  500    |  0.43022  |
| ORB      | nfeatures         | 2000    |  0.438425 |
| ORB      | nfeatures         | 5000    |  0.462355 |
| BRISK    | thresh            |   15    |  0.492004 |
| BRISK    | thresh            |   30    |  0.489982 |
| BRISK    | thresh            |   60    |  0.365746 |