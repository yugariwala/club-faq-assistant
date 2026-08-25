# Intent Classification Evaluation

Mode: hybrid (rules + LLM fallback)
Eval set: 56 items

## Rule-path vs. LLM-path split

| Path | Count | Fraction of scored traffic | Accuracy |
|---|---|---|---|
| rule | 33 | 58.9% | 100.0% |
| llm | 23 | 41.1% | 95.7% |
| **overall** | 56 | 100.0% | 98.2% |

## Per-class precision / recall / F1

| Class | Precision | Recall | F1 | Support |
|---|---|---|---|---|
| action_request | 1.000 | 1.000 | 1.000 | 11 |
| event_inquiry | 1.000 | 1.000 | 1.000 | 12 |
| faq | 0.923 | 1.000 | 0.960 | 12 |
| greeting | 1.000 | 1.000 | 1.000 | 10 |
| out_of_scope | 1.000 | 0.909 | 0.952 | 11 |
| **macro avg** | 0.985 | 0.982 | 0.982 | 56 |
| **weighted avg** | 0.984 | 0.982 | 0.982 | 56 |

## Confusion matrix (rows = gold, columns = predicted)

| gold \ predicted | action_request | event_inquiry | faq | greeting | out_of_scope |
|---|---|---|---|---|---|
| action_request | 11 | 0 | 0 | 0 | 0 |
| event_inquiry | 0 | 12 | 0 | 0 | 0 |
| faq | 0 | 0 | 12 | 0 | 0 |
| greeting | 0 | 0 | 0 | 10 | 0 |
| out_of_scope | 0 | 0 | 1 | 0 | 10 |

## Misclassifications

| Query | Gold | Predicted | Path |
|---|---|---|---|
| Does the club have a Discord server? | out_of_scope | faq | llm |
