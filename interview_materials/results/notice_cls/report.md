# Notice classification evaluation

Platform categories are the gold labels; they are not relabeled as semantic truth.

| Method | Accuracy | Macro-F1 | Coverage | Mean ms/doc | Mean tokens/doc | Evidence OK |
|---|---:|---:|---:|---:|---:|---:|
| rules | 0.8100 | 0.8213 | 0.8700 | 0.11 | 0.00 | n/a |
| tfidf_char | 0.8800 | 0.8466 | 1.0000 | 1.64 | 0.00 | n/a |
| tfidf_word | 0.8800 | 0.8466 | 1.0000 | 4.15 | 0.00 | n/a |
| llm | 0.8600 | 0.8005 | 1.0000 | 11952.78 | 2145.27 | 1.0000 |

## rules

Per-class precision / recall / F1:

| Label | Precision | Recall | F1 | Support |
|---|---:|---:|---:|---:|
| gkzb (公开招标公告) | 1.0000 | 0.9500 | 0.9744 | 40 |
| zbgg (中标公告) | 0.8750 | 0.7000 | 0.7778 | 20 |
| cjgg (成交公告) | 0.8182 | 0.4500 | 0.5806 | 20 |
| gzgg (更正公告) | 0.9091 | 1.0000 | 0.9524 | 20 |

Confusion matrix (rows=gold; columns=prediction, including no-fire):

| Gold \ Pred | gkzb | zbgg | cjgg | gzgg | <none> |
|---|---:|---:|---:|---:|---:|
| gkzb | 38 | 0 | 0 | 2 | 0 |
| zbgg | 0 | 14 | 2 | 0 | 4 |
| cjgg | 0 | 2 | 9 | 0 | 9 |
| gzgg | 0 | 0 | 0 | 20 | 0 |

## tfidf_char

Per-class precision / recall / F1:

| Label | Precision | Recall | F1 | Support |
|---|---:|---:|---:|---:|
| gkzb (公开招标公告) | 1.0000 | 1.0000 | 1.0000 | 40 |
| zbgg (中标公告) | 0.7857 | 0.5500 | 0.6471 | 20 |
| cjgg (成交公告) | 0.6538 | 0.8500 | 0.7391 | 20 |
| gzgg (更正公告) | 1.0000 | 1.0000 | 1.0000 | 20 |

Confusion matrix (rows=gold; columns=prediction, including no-fire):

| Gold \ Pred | gkzb | zbgg | cjgg | gzgg | <none> |
|---|---:|---:|---:|---:|---:|
| gkzb | 40 | 0 | 0 | 0 | 0 |
| zbgg | 0 | 11 | 9 | 0 | 0 |
| cjgg | 0 | 3 | 17 | 0 | 0 |
| gzgg | 0 | 0 | 0 | 20 | 0 |

## tfidf_word

Per-class precision / recall / F1:

| Label | Precision | Recall | F1 | Support |
|---|---:|---:|---:|---:|
| gkzb (公开招标公告) | 1.0000 | 1.0000 | 1.0000 | 40 |
| zbgg (中标公告) | 0.7857 | 0.5500 | 0.6471 | 20 |
| cjgg (成交公告) | 0.6538 | 0.8500 | 0.7391 | 20 |
| gzgg (更正公告) | 1.0000 | 1.0000 | 1.0000 | 20 |

Confusion matrix (rows=gold; columns=prediction, including no-fire):

| Gold \ Pred | gkzb | zbgg | cjgg | gzgg | <none> |
|---|---:|---:|---:|---:|---:|
| gkzb | 40 | 0 | 0 | 0 | 0 |
| zbgg | 0 | 11 | 9 | 0 | 0 |
| cjgg | 0 | 3 | 17 | 0 | 0 |
| gzgg | 0 | 0 | 0 | 20 | 0 |

## llm

Per-class precision / recall / F1:

| Label | Precision | Recall | F1 | Support |
|---|---:|---:|---:|---:|
| gkzb (公开招标公告) | 1.0000 | 1.0000 | 1.0000 | 40 |
| zbgg (中标公告) | 1.0000 | 0.3000 | 0.4615 | 20 |
| cjgg (成交公告) | 0.5882 | 1.0000 | 0.7407 | 20 |
| gzgg (更正公告) | 1.0000 | 1.0000 | 1.0000 | 20 |

Confusion matrix (rows=gold; columns=prediction, including no-fire):

| Gold \ Pred | gkzb | zbgg | cjgg | gzgg | <none> |
|---|---:|---:|---:|---:|---:|
| gkzb | 40 | 0 | 0 | 0 | 0 |
| zbgg | 0 | 6 | 14 | 0 | 0 |
| cjgg | 0 | 0 | 20 | 0 | 0 |
| gzgg | 0 | 0 | 0 | 20 | 0 |

## Cross-validation (not test results)

- tfidf_char, 5-fold cross-validation on all 150 documents: accuracy 0.8933 ± 0.0389; macro-F1 0.8623 ± 0.0531.
- tfidf_word, 5-fold cross-validation on all 150 documents: accuracy 0.9067 ± 0.0442; macro-F1 0.8811 ± 0.0560.
