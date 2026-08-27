# Evaluation Report

## 1. Retrieval Evaluation

The retrieval evaluation contains ten test cases.

| Metric | Result |
|---|---:|
| True Positives | 6 |
| True Negatives | 3 |
| False Positives | 1 |
| False Negatives | 0 |
| Precision | 85.71% |
| Recall | 100.00% |
| F1 Score | 92.31% |

## 2. Interpretation

### Precision — 85.71%

Most retrieved results were relevant to the evaluation questions.

### Recall — 100%

All expected retrievable questions successfully returned relevant information.

### F1 Score — 92.31%

The combined retrieval performance was strong, balancing precision and recall.

## 3. False Positive

The sick-leave question was expected to be unavailable, but retrieval returned document chunks:

```text
Question:
How many sick leave days are employees entitled to?

Expected Retrieval:
False

Retrieved:
True
```

This created one false positive.

This result is documented as a limitation of the current retrieval evaluation.

## 4. RAG Pipeline Evaluation

```text
Passed: 5/5
Success Rate: 100.00%
```

### Test 1 — Known Question

Pass.

### Test 2 — Paraphrased Known Question

Pass.

### Test 3 — Unknown Question

Pass.

The system correctly returned a not-found response for the laptop replacement policy question.

### Test 4 — Partial Information

Pass.

The system returned annual-leave information and reported that maternity-leave information was not found.

### Test 5 — Unrelated Question

Pass.

The system did not answer an unrelated general-knowledge question from outside the supplied enterprise documents.

## 5. Final Assessment

The retrieval evaluation demonstrates strong recall and good overall retrieval performance. The complete RAG pipeline achieved a 100% success rate across the five defined pipeline tests.
