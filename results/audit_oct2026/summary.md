### Table A. Headline recomputed, floors and output-side judges (pooled OOF AUC, mean of 5 seeds)

| run | side | n+/n- | probe | conf | paper floor | struct floor | conf LR | text(call) | text(call+user) | output judge |
|---|---|---|---|---|---|---|---|---|---|---|
| LlamaOneBGlaive | internals | 221/359 | 0.963 | 0.576 | 0.539 | 0.557 | 0.493 | 0.865 | 0.916 | 0.910 |
| LlamaOneBBfcl | internals | 161/339 | 0.831 | 0.651 | 0.635 | 0.684 | 0.682 | 0.736 | 0.704 | 0.755 |
| LlamaOneBLive | internals | 208/157 | 0.782 | 0.663 | 0.709 | 0.685 | 0.653 | 0.676 | 0.680 | 0.682 |
| LlamaThreeBGlaive | internals | 90/555 | 0.911 | 0.785 | 0.522 | 0.415 | 0.785 | 0.670 | 0.671 | 0.684 |
| LlamaThreeBBfcl | internals | 120/576 | 0.831 | 0.707 | 0.618 | 0.681 | 0.702 | 0.589 | 0.627 | 0.633 |
| GemmaGlaive | internals | 158/460 | 0.883 | 0.650 | 0.704 | 0.599 | 0.519 | 0.802 | 0.869 | 0.872 |
| GemmaBfcl | internals | 364/150 | 0.948 | 0.711 | 0.842 | 0.916 | 0.856 | 0.875 | 0.882 | 0.900 |
| QwenThreeBfcl | confidence | 52/639 | 0.745 | 0.799 | 0.628 | 0.662 | 0.735 | 0.532 | 0.551 | 0.558 |
| MiniCpmBfcl | confidence | 54/640 | 0.775 | 0.823 | 0.578 | 0.641 | 0.779 | 0.512 | 0.557 | 0.585 |
| MiniCpmLive | confidence | 120/475 | 0.744 | 0.797 | 0.545 | 0.507 | 0.756 | 0.611 | 0.623 | 0.642 |
| QwenThreeFiveBfcl | confidence | 86/398 | 0.682 | 0.792 | 0.601 | 0.663 | 0.838 | 0.622 | 0.660 | 0.678 |

### Table B. Paired, tool-resampled differences (95% percentile interval, draws pooled over 5 seeds)

| run | side | probe - conf (paper) | within-fold gap | fusion(probe,conf) - conf | probe - text(call+user) | probe - output judge | fusion(probe,output) - output | gap without schema echo |
|---|---|---|---|---|---|---|---|---|
| LlamaOneBGlaive | internals | +0.387 [+0.18, +0.59] | +0.403 | +0.290 [+0.17, +0.45] | +0.047 [-0.03, +0.30] | +0.053 [-0.03, +0.33] | +0.042 [-0.00, +0.23] | +0.349 [+0.01, +0.64] |
| LlamaOneBBfcl | internals | +0.180 [+0.09, +0.27] | +0.179 | +0.162 [+0.11, +0.21] | +0.127 [+0.06, +0.20] | +0.077 [+0.01, +0.14] | +0.066 [+0.03, +0.10] | +0.197 [+0.10, +0.29] |
| LlamaOneBLive | internals | +0.119 [-0.00, +0.24] | +0.140 | +0.126 [+0.06, +0.20] | +0.102 [+0.02, +0.18] | +0.101 [+0.03, +0.18] | +0.075 [+0.03, +0.12] | -0.020 [-0.16, +0.10] |
| LlamaThreeBGlaive | internals | +0.126 [+0.02, +0.28] | +0.212 | +0.108 [+0.04, +0.22] | +0.240 [-0.00, +0.49] | +0.227 [-0.02, +0.47] | +0.161 [+0.03, +0.32] | +0.126 [+0.02, +0.28] |
| LlamaThreeBBfcl | internals | +0.125 [+0.05, +0.20] | +0.143 | +0.133 [+0.09, +0.17] | +0.205 [+0.13, +0.28] | +0.199 [+0.12, +0.27] | +0.142 [+0.10, +0.18] | +0.125 [+0.05, +0.19] |
| GemmaGlaive | internals | +0.233 [+0.11, +0.34] | +0.236 | +0.160 [+0.06, +0.22] | +0.014 [-0.13, +0.13] | +0.011 [-0.14, +0.13] | +0.021 [-0.05, +0.08] | +0.293 [+0.16, +0.48] |
| GemmaBfcl | internals | +0.237 [+0.19, +0.29] | +0.238 | +0.177 [+0.15, +0.21] | +0.066 [+0.03, +0.10] | +0.048 [+0.01, +0.08] | +0.039 [+0.02, +0.06] | +0.339 [+0.24, +0.44] |
| QwenThreeBfcl | confidence | -0.053 [-0.16, +0.05] | -0.036 | +0.029 [-0.02, +0.08] | +0.194 [+0.08, +0.32] | +0.187 [+0.05, +0.32] | +0.131 [+0.07, +0.19] | -0.053 [-0.15, +0.05] |
| MiniCpmBfcl | confidence | -0.048 [-0.15, +0.05] | -0.034 | +0.034 [-0.02, +0.08] | +0.218 [+0.10, +0.34] | +0.190 [+0.09, +0.29] | +0.131 [+0.08, +0.19] | -0.048 [-0.15, +0.05] |
| MiniCpmLive | confidence | -0.053 [-0.15, +0.02] | -0.041 | +0.016 [-0.04, +0.05] | +0.122 [+0.03, +0.22] | +0.102 [+0.01, +0.20] | +0.091 [+0.03, +0.15] | -0.053 [-0.16, +0.02] |
| QwenThreeFiveBfcl | confidence | -0.110 [-0.19, -0.03] | -0.102 | -0.003 [-0.05, +0.04] | +0.022 [-0.06, +0.11] | +0.004 [-0.08, +0.08] | +0.031 [-0.01, +0.08] | -0.110 [-0.19, -0.03] |

### Table C. Within one failure mode (wrong argument values vs valid), stored scores

| run | side | n+ | probe | conf | gap |
|---|---|---|---|---|---|
| GemmaBfcl | internals | 33 | 0.710 | 0.552 | +0.158 [-0.04, +0.33] |
| GemmaGlaive | internals | 109 | 0.868 | 0.642 | +0.226 [+0.02, +0.38] |
| LlamaOneBBfcl | internals | 127 | 0.828 | 0.587 | +0.240 [+0.14, +0.34] |
| LlamaOneBGlaive | internals | 132 | 0.946 | 0.589 | +0.357 [-0.01, +0.68] |
| LlamaOneBLive | internals | 89 | 0.664 | 0.675 | -0.010 [-0.19, +0.16] |
| LlamaThreeBBfcl | internals | 88 | 0.818 | 0.684 | +0.134 [+0.05, +0.22] |
| LlamaThreeBGlaive | internals | 85 | 0.926 | 0.781 | +0.145 [+0.05, +0.31] |
| MiniCpmBfcl | confidence | 22 | 0.715 | 0.877 | -0.162 [-0.31, -0.02] |
| MiniCpmLive | confidence | 84 | 0.731 | 0.779 | -0.048 [-0.16, +0.04] |
| QwenThreeBfcl | confidence | 33 | 0.717 | 0.841 | -0.124 [-0.25, +0.01] |
| QwenThreeFiveBfcl | confidence | 54 | 0.668 | 0.789 | -0.121 [-0.22, -0.02] |

### Table D. Schema echo and failure-mode mix of the scored population

| run | side | echo share of failures | echo share of correct | regex AUC | modes |
|---|---|---|---|---|---|
| LlamaOneBGlaive | internals | 0.38 | 0.01 | 0.684 | missing_args 86, valid 359, wrong_arg_values 132, wrong_name 3 |
| LlamaOneBBfcl | internals | 0.11 | 0.00 | 0.556 | missing_args 21, missing_calls 8, valid 339, wrong_arg_values 127, wrong_name 5 |
| LlamaOneBLive | internals | 0.34 | 0.06 | 0.636 | missing_args 46, missing_calls 5, valid 157, wrong_arg_values 89, wrong_name 68 |
| LlamaThreeBGlaive | internals | 0.00 | 0.00 | 0.500 | missing_args 5, valid 555, wrong_arg_values 85 |
| LlamaThreeBBfcl | internals | 0.00 | 0.00 | 0.500 | missing_args 4, missing_calls 25, valid 576, wrong_arg_values 88, wrong_name 3 |
| GemmaGlaive | internals | 0.22 | 0.01 | 0.603 | missing_args 47, valid 460, wrong_arg_values 109, wrong_name 2 |
| GemmaBfcl | internals | 0.73 | 0.07 | 0.827 | missing_args 219, missing_calls 99, valid 150, wrong_arg_values 33, wrong_name 13 |
| QwenThreeBfcl | confidence | 0.00 | 0.00 | 0.500 | missing_args 7, missing_calls 11, valid 639, wrong_arg_values 33, wrong_name 1 |
| MiniCpmBfcl | confidence | 0.00 | 0.00 | 0.500 | missing_args 14, missing_calls 16, valid 640, wrong_arg_values 22, wrong_name 2 |
| MiniCpmLive | confidence | 0.00 | 0.00 | 0.500 | missing_args 24, missing_calls 2, valid 475, wrong_arg_values 84, wrong_name 10 |
| QwenThreeFiveBfcl | confidence | 0.00 | 0.00 | 0.500 | missing_args 5, missing_calls 19, valid 398, wrong_arg_values 54, wrong_name 8 |

Reverse checks: GemmaBfcl: floor max|diff| 1.0e-07, folds match True, GemmaGlaive: floor max|diff| 1.3e-04, folds match True, LlamaOneBBfcl: floor max|diff| 3.8e-04, folds match True, LlamaOneBGlaive: floor max|diff| 6.3e-05, folds match True, LlamaOneBLive: floor max|diff| 2.9e-05, folds match True, LlamaThreeBBfcl: floor max|diff| 2.1e-04, folds match True, LlamaThreeBGlaive: floor max|diff| 5.4e-04, folds match True, MiniCpmBfcl: floor max|diff| 9.2e-08, folds match True, MiniCpmLive: floor max|diff| 5.3e-04, folds match True, QwenThreeBfcl: floor max|diff| 4.0e-03, folds match True, QwenThreeFiveBfcl: floor max|diff| 9.1e-05, folds match True

### Table E. Reader-model probe (Qwen3.5-0.8B reads request + call text, no schemas)

| run | side | reader | own probe | conf | probe - reader | reader - conf |
|---|---|---|---|---|---|---|
| LlamaOneBGlaive | internals | 0.949 | 0.963 | 0.576 | +0.015 [-0.02, +0.11] | +0.373 [+0.12, +0.60] |
| LlamaOneBBfcl | internals | 0.818 | 0.831 | 0.651 | +0.014 [-0.05, +0.07] | +0.167 [+0.07, +0.26] |
| LlamaOneBLive | internals | 0.795 | 0.782 | 0.663 | -0.012 [-0.09, +0.06] | +0.131 [+0.03, +0.23] |
