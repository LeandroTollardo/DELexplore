# Reference: DEL + ML Pipeline for Hit Discovery

Source: Iqbal, Jiang, Hansen et al. (2025). "Evaluation of DNA encoded library
and machine learning model combinations for hit discovery."
npj Drug Discovery 2:5.

## Key Finding for Implementation

Trained ML models on DEL screening data can predict binders in external compound
libraries. 10% confirmed hit rate overall, including nanomolar binders (69.6 nM).
ChemProp and MLP outperformed RF, SVM, XGB across most DEL+ML combinations.

## Pipeline Architecture (5 modules)

1. DEL screening → raw enrichment data
2. Data preparation → balanced training set
3. Model training → 5 ML models compared
4. Prediction → screen external compound library
5. Validation → biophysical assay (SPR)

## Critical Design Decisions for ML Module

### Training Data Preparation

Positive examples: orthosteric DEL binders (enriched in protein condition,
NOT enriched in protein+inhibitor condition)

Negative examples: non-enriched DEL members

Balanced training: equal number of positives and negatives

### Hard Negative Sampling Strategy

From the negatives pool:
- 20% selected by chemical similarity to positives (Tanimoto > 0.7 on Morgan FP)
- 80% selected randomly

Rationale: hard negatives force the model to learn fine-grained structural
features, not just gross chemical class differences.

### Five ML Models Compared

| Model | Type | Input | Best For |
|---|---|---|---|
| Random Forest (RF) | Traditional | Morgan FP | Fast baseline, interpretable |
| SVM | Traditional | Morgan FP | Small datasets |
| XGBoost (XGB) | Traditional | Morgan FP | Structured data |
| MLP | Deep neural net | Morgan FP | Large datasets |
| ChemProp | Graph neural net | SMILES | Structure-rich predictions |

### Cross-Validation Protocol

- 80% train / 20% test split (in-DEL holdout)
- Independent validation set: known CK1alpha/delta binders (non-DEL compounds)
- Blind assessment set: 140K compounds from Broad Compound Collection

### Key Results by DEL Library

| Library | Size | Chemistry | Drug-like (Lipinski) | Best ML |
|---|---|---|---|---|
| MS10M | 10M | Peptide-like | Low | Variable |
| HG1B | 1B | Drug-like | 46-48% | ChemProp |
| DD11M | 11M | DOS-based | Low | Variable |

HG1B (drug-like) produced the best ML models because:
1. Larger training set
2. Better chemical space overlap with assessment compounds
3. Higher fraction of drug-like molecules

### Binder Type Classification

From 5 selection conditions (protein, protein+inhibitor, blank):
- **Orthosteric**: enriched in protein, NOT in protein+inhibitor
- **Allosteric**: enriched in BOTH protein and protein+inhibitor
- **Cryptic**: enriched in protein+inhibitor, NOT protein alone

Current implementation focuses on orthosteric binders only.

## Implementation Notes

- Morgan fingerprints: radius=2, nBits=2048 (standard for DEL-ML)
- ChemProp: use default hyperparameters as starting point, then tune
- Always report: accuracy, precision, recall, F1, and AUC-ROC
- Chemical space visualization: t-SNE on Morgan fingerprints
- Test set should include BOTH predicted binders AND predicted non-binders
  (to validate true negative rate — 94% in this study)

## Data Formats

Training data CSV columns:
```
smiles, label, source_del, selection_condition, enrichment_score
```

Prediction output CSV columns:
```
smiles, predicted_label, probability, model_name
```

## GitHub Reference

Open-source trained models: https://github.com/broadinstitute/DEL-ML-Refactor
