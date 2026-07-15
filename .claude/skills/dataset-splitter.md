---
name: dataset-splitter
description: Creates train, validation, and test splits while preventing leakage (patient-wise)
---

## Dataset Splitter

### Responsibilities
* Creates train, validation, and test splits
* Prevents leakage (patient-wise)

### Key Tasks
1. Perform patient-wise splitting to prevent data leakage
2. Split data into Training = 70%, Validation = 15%, Test = 15%
3. Ensure no patient appears in more than one split
4. Maintain class distribution across splits when possible
5. Generate train.csv, validation.csv, test.csv files
6. Update master_metadata.csv with split information

### Always
* perform patient-wise splitting (not random)
* prevent data leakage
* maintain 70/15/15 split ratio
* verify no patient overlap between splits

### Never Assume
* random splitting is acceptable
* patient leakage doesn't matter
* split ratios can vary
* class distribution doesn't need consideration