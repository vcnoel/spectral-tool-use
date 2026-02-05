# Comprehensive Statistical Analysis - T=0.3 Detectors

## Best 2-feature (Overall)

**Configuration:** L20 + L22

### Dataset
- Total samples: 1000
- Hallucinations: 200 (20.0%)
- Valid: 800 (80.0%)

### Performance Metrics
- **AUC**: 0.8259 [95% CI: 0.7914, 0.8582]
- **Recall (TPR)**: 0.7350 (147/200 hallucinations detected)
- **Precision**: 0.4820 (147/305 predictions correct)
- **F1 Score**: 0.5822
- **Specificity**: 0.8025
- **FPR**: 0.1975
- **Optimal Threshold**: 0.4941

### Confusion Matrix
```
                 Predicted
                 Valid  Hallu
Actual Valid       642    158
       Hallu        53    147
```

### Feature Statistics

| Feature | Hallu Mean±SD | Valid Mean±SD | Cohen's d | p-value | Significance |
|---------|---------------|---------------|-----------|---------|-------------|
| L20_Smoothness | 0.8814±0.0306 | 0.9096±0.0207 | -1.224 | 3.25e-34 | *** |
| L22_HFER | 0.2202±0.0221 | 0.2337±0.0198 | -0.665 | 7.95e-14 | *** |

---

## Best 3-feature (Overall)

**Configuration:** L19 + L17 + L17

### Dataset
- Total samples: 1000
- Hallucinations: 200 (20.0%)
- Valid: 800 (80.0%)

### Performance Metrics
- **AUC**: 0.7639 [95% CI: 0.7249, 0.8013]
- **Recall (TPR)**: 0.7500 (150/200 hallucinations detected)
- **Precision**: 0.4021 (150/373 predictions correct)
- **F1 Score**: 0.5236
- **Specificity**: 0.7212
- **FPR**: 0.2787
- **Optimal Threshold**: 0.4919

### Confusion Matrix
```
                 Predicted
                 Valid  Hallu
Actual Valid       577    223
       Hallu        50    150
```

### Feature Statistics

| Feature | Hallu Mean±SD | Valid Mean±SD | Cohen's d | p-value | Significance |
|---------|---------------|---------------|-----------|---------|-------------|
| L19_Smoothness | 0.9389±0.0257 | 0.9649±0.0154 | -1.448 | 1.83e-42 | *** |
| L17_Smoothness | 0.9433±0.0135 | 0.9522±0.0100 | -0.827 | 6.62e-21 | *** |
| L17_HFER | 0.1718±0.0580 | 0.1383±0.0497 | +0.649 | 8.31e-15 | *** |

---

## Best 4-feature (Overall)

**Configuration:** L19 + L23 + L17 + L17

### Dataset
- Total samples: 1000
- Hallucinations: 200 (20.0%)
- Valid: 800 (80.0%)

### Performance Metrics
- **AUC**: 0.7496 [95% CI: 0.7069, 0.7918]
- **Recall (TPR)**: 0.5550 (111/200 hallucinations detected)
- **Precision**: 0.5068 (111/219 predictions correct)
- **F1 Score**: 0.5298
- **Specificity**: 0.8650
- **FPR**: 0.1350
- **Optimal Threshold**: 0.5465

### Confusion Matrix
```
                 Predicted
                 Valid  Hallu
Actual Valid       692    108
       Hallu        89    111
```

### Feature Statistics

| Feature | Hallu Mean±SD | Valid Mean±SD | Cohen's d | p-value | Significance |
|---------|---------------|---------------|-----------|---------|-------------|
| L19_Smoothness | 0.9389±0.0257 | 0.9649±0.0154 | -1.448 | 1.83e-42 | *** |
| L23_Entropy | 3.1453±0.2575 | 3.3445±0.1749 | -1.025 | 3.78e-25 | *** |
| L17_Smoothness | 0.9433±0.0135 | 0.9522±0.0100 | -0.827 | 6.62e-21 | *** |
| L17_HFER | 0.1718±0.0580 | 0.1383±0.0497 | +0.649 | 8.31e-15 | *** |

---

## Best 5-feature (Overall)

**Configuration:** L19 + L23 + L17 + L17 + L22

### Dataset
- Total samples: 1000
- Hallucinations: 200 (20.0%)
- Valid: 800 (80.0%)

### Performance Metrics
- **AUC**: 0.7591 [95% CI: 0.7177, 0.7998]
- **Recall (TPR)**: 0.6000 (120/200 hallucinations detected)
- **Precision**: 0.4651 (120/258 predictions correct)
- **F1 Score**: 0.5240
- **Specificity**: 0.8275
- **FPR**: 0.1725
- **Optimal Threshold**: 0.5133

### Confusion Matrix
```
                 Predicted
                 Valid  Hallu
Actual Valid       662    138
       Hallu        80    120
```

### Feature Statistics

| Feature | Hallu Mean±SD | Valid Mean±SD | Cohen's d | p-value | Significance |
|---------|---------------|---------------|-----------|---------|-------------|
| L19_Smoothness | 0.9389±0.0257 | 0.9649±0.0154 | -1.448 | 1.83e-42 | *** |
| L23_Entropy | 3.1453±0.2575 | 3.3445±0.1749 | -1.025 | 3.78e-25 | *** |
| L17_Smoothness | 0.9433±0.0135 | 0.9522±0.0100 | -0.827 | 6.62e-21 | *** |
| L17_HFER | 0.1718±0.0580 | 0.1383±0.0497 | +0.649 | 8.31e-15 | *** |
| L22_HFER | 0.2202±0.0221 | 0.2337±0.0198 | -0.665 | 7.95e-14 | *** |

---

## finance_L2_Fiedler (finance)

**Configuration:** L2

### Dataset
- Total samples: 50
- Hallucinations: 5 (10.0%)
- Valid: 45 (90.0%)

### Performance Metrics
- **AUC**: 1.0000 [95% CI: 1.0000, 1.0000]
- **Recall (TPR)**: 1.0000 (5/5 hallucinations detected)
- **Precision**: 1.0000 (5/5 predictions correct)
- **F1 Score**: 1.0000
- **Specificity**: 1.0000
- **FPR**: 0.0000
- **Optimal Threshold**: -0.1005

### Confusion Matrix
```
                 Predicted
                 Valid  Hallu
Actual Valid        45      0
       Hallu         0      5
```

### Feature Statistics

| Feature | Hallu Mean±SD | Valid Mean±SD | Cohen's d | p-value | Significance |
|---------|---------------|---------------|-----------|---------|-------------|
| L2_Fiedler | 0.0962±0.0035 | 0.1308±0.0122 | -2.925 | 9.44e-07 | *** |

---

## finance_L2_Fiedler+L0_Fiedler (finance)

**Configuration:** L2 + L0

### Dataset
- Total samples: 50
- Hallucinations: 5 (10.0%)
- Valid: 45 (90.0%)

### Performance Metrics
- **AUC**: 1.0000 [95% CI: 1.0000, 1.0000]
- **Recall (TPR)**: 1.0000 (5/5 hallucinations detected)
- **Precision**: 1.0000 (5/5 predictions correct)
- **F1 Score**: 1.0000
- **Specificity**: 1.0000
- **FPR**: 0.0000
- **Optimal Threshold**: 0.5022

### Confusion Matrix
```
                 Predicted
                 Valid  Hallu
Actual Valid        45      0
       Hallu         0      5
```

### Feature Statistics

| Feature | Hallu Mean±SD | Valid Mean±SD | Cohen's d | p-value | Significance |
|---------|---------------|---------------|-----------|---------|-------------|
| L2_Fiedler | 0.0962±0.0035 | 0.1308±0.0122 | -2.925 | 9.44e-07 | *** |
| L0_Fiedler | 0.2198±0.0093 | 0.2614±0.0099 | -4.150 | 1.89e-06 | *** |

---

## communication_L16_Fiedler (communication)

**Configuration:** L16

### Dataset
- Total samples: 56
- Hallucinations: 8 (14.3%)
- Valid: 48 (85.7%)

### Performance Metrics
- **AUC**: 0.7839 [95% CI: 0.5981, 0.9371]
- **Recall (TPR)**: 0.7500 (6/8 hallucinations detected)
- **Precision**: 0.3158 (6/19 predictions correct)
- **F1 Score**: 0.4444
- **Specificity**: 0.7292
- **FPR**: 0.2708
- **Optimal Threshold**: 0.3907

### Confusion Matrix
```
                 Predicted
                 Valid  Hallu
Actual Valid        35     13
       Hallu         2      6
```

### Feature Statistics

| Feature | Hallu Mean±SD | Valid Mean±SD | Cohen's d | p-value | Significance |
|---------|---------------|---------------|-----------|---------|-------------|
| L16_Fiedler | 0.3953±0.0116 | 0.3822±0.0119 | +1.083 | 9.11e-03 | ** |

---

## communication_L16_Fiedler+L8_Fiedler (communication)

**Configuration:** L16 + L8

### Dataset
- Total samples: 56
- Hallucinations: 8 (14.3%)
- Valid: 48 (85.7%)

### Performance Metrics
- **AUC**: 0.8724 [95% CI: 0.6779, 0.9913]
- **Recall (TPR)**: 0.8750 (7/8 hallucinations detected)
- **Precision**: 0.6364 (7/11 predictions correct)
- **F1 Score**: 0.7368
- **Specificity**: 0.9167
- **FPR**: 0.0833
- **Optimal Threshold**: 0.5003

### Confusion Matrix
```
                 Predicted
                 Valid  Hallu
Actual Valid        44      4
       Hallu         1      7
```

### Feature Statistics

| Feature | Hallu Mean±SD | Valid Mean±SD | Cohen's d | p-value | Significance |
|---------|---------------|---------------|-----------|---------|-------------|
| L16_Fiedler | 0.3953±0.0116 | 0.3822±0.0119 | +1.083 | 9.11e-03 | ** |
| L8_Fiedler | 0.1733±0.0105 | 0.1836±0.0116 | -0.878 | 2.03e-02 | * |

---

## other_L19_Smoothness (other)

**Configuration:** L19

### Dataset
- Total samples: 894
- Hallucinations: 187 (20.9%)
- Valid: 707 (79.1%)

### Performance Metrics
- **AUC**: 0.8190 [95% CI: 0.7788, 0.8548]
- **Recall (TPR)**: 0.7326 (137/187 hallucinations detected)
- **Precision**: 0.5352 (137/256 predictions correct)
- **F1 Score**: 0.6185
- **Specificity**: 0.8317
- **FPR**: 0.1683
- **Optimal Threshold**: -0.9557

### Confusion Matrix
```
                 Predicted
                 Valid  Hallu
Actual Valid       588    119
       Hallu        50    137
```

### Feature Statistics

| Feature | Hallu Mean±SD | Valid Mean±SD | Cohen's d | p-value | Significance |
|---------|---------------|---------------|-----------|---------|-------------|
| L19_Smoothness | 0.9371±0.0254 | 0.9636±0.0157 | -1.457 | 3.97e-41 | *** |

---

## other_L19_Smoothness+L7_Smoothness (other)

**Configuration:** L19 + L7

### Dataset
- Total samples: 894
- Hallucinations: 187 (20.9%)
- Valid: 707 (79.1%)

### Performance Metrics
- **AUC**: 0.8212 [95% CI: 0.7823, 0.8563]
- **Recall (TPR)**: 0.7112 (133/187 hallucinations detected)
- **Precision**: 0.5636 (133/236 predictions correct)
- **F1 Score**: 0.6288
- **Specificity**: 0.8543
- **FPR**: 0.1457
- **Optimal Threshold**: 0.4953

### Confusion Matrix
```
                 Predicted
                 Valid  Hallu
Actual Valid       604    103
       Hallu        54    133
```

### Feature Statistics

| Feature | Hallu Mean±SD | Valid Mean±SD | Cohen's d | p-value | Significance |
|---------|---------------|---------------|-----------|---------|-------------|
| L19_Smoothness | 0.9371±0.0254 | 0.9636±0.0157 | -1.457 | 3.97e-41 | *** |
| L7_Smoothness | 0.9480±0.0063 | 0.9521±0.0044 | -0.846 | 5.07e-29 | *** |

---

