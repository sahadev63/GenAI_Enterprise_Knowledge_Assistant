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


## Exception Handling

The application includes defensive exception handling around LLM generation and the RAG pipeline. Service/runtime failures are converted into clear application-level errors instead of exposing raw stack traces to users. The LLM layer reports a specific recovery message when Ollama or the configured model is unavailable.

## 6. Agent Validation Regression Fix

A validation false-negative was identified during UI testing: the validator could reject an answer that was directly present in the retrieved evidence. The validator has been updated to perform deterministic evidence grounding before invoking the LLM judge. Exact evidence matches and strong paraphrase matches are accepted, while numerical claims must match numbers present in the evidence.

Regression coverage was added for:

- exact evidence match even when the LLM judge returns a false negative;
- close grounded paraphrases with the same numeric value;
- rejection of an unsupported numeric value; and
- partial answers containing an explicit missing-information statement.

The existing fail-closed behavior remains in place for empty evidence, malformed validator responses, and unexpected validation exceptions.


## 7. Agentic RAG reliability fixes

UI regression testing identified two orchestration issues that could cause valid
retrieved evidence to be discarded:

- The planning LLM could replace the user's precise question with vague
  meta-level subtasks. Retrieval subtasks are now deterministic and preserve
  the user's original wording; the LLM is limited to proposing an execution
  strategy.
- For multi-part questions, a missing topic is now treated as a valid
  evidence-based abstention rather than invalidating the supported parts.
  Therefore a question such as "What are the annual leave and maternity leave
  policies?" can return the annual-leave answer and explicitly state that
  maternity-leave information was not found.

The deterministic answerability guard also recognizes direct policy evidence
for timing, numeric, and responsibility questions while filtering generic
terms such as "entitled" that could otherwise create false positives.

Regression tests cover simple-question preservation, multi-part planning,
partial-information validation, direct timing evidence, and unrelated-topic
rejection.
