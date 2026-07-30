# Routing sweep results

| strategy | n | accuracy | under-route | over-route | GPU-s |
|---|---|---|---|---|---|
| oracle (cheapest adequate) | 30 | 1.0 | 0.0 | 0.0 | 489.4 |
| always qwen3-1.7b | 30 | 0.467 | 0.533 | 0.0 | 147.3 |
| always qwen3-30b-a3b | 30 | 0.867 | 0.133 | 0.467 | 438.7 |
| always gpt-oss-120b | 30 | 0.867 | 0.133 | 0.733 | 1180.2 |
| random | 30 | 0.8 | 0.2 | 0.467 | 687.6 |
| length heuristic | 30 | 0.533 | 0.467 | 0.033 | 244.8 |
| classifier (1,2) | 30 | 0.9 | 0.1 | 0.667 | 1203.3 |
| classifier (1,3) | 30 | 0.9 | 0.1 | 0.667 | 1203.3 |
| classifier (1,4) | 30 | 0.9 | 0.1 | 0.667 | 1203.3 |
| classifier (1,5) | 30 | 0.867 | 0.133 | 0.467 | 438.7 |
| classifier (1,6) | 30 | 0.867 | 0.133 | 0.467 | 438.7 |
| classifier (1,7) | 30 | 0.867 | 0.133 | 0.467 | 438.7 |
| classifier (1,8) | 30 | 0.867 | 0.133 | 0.467 | 438.7 |
| classifier (1,9) | 30 | 0.867 | 0.133 | 0.467 | 438.7 |
| classifier (2,3) | 30 | 0.767 | 0.233 | 0.367 | 1115.9 |
| classifier (2,4) | 30 | 0.767 | 0.233 | 0.367 | 1115.9 |
| classifier (2,5) | 30 | 0.733 | 0.267 | 0.167 | 351.2 |
| classifier (2,6) | 30 | 0.733 | 0.267 | 0.167 | 351.2 |
| classifier (2,7) | 30 | 0.733 | 0.267 | 0.167 | 351.2 |
| classifier (2,8) | 30 | 0.733 | 0.267 | 0.167 | 351.2 |
| classifier (2,9) | 30 | 0.733 | 0.267 | 0.167 | 351.2 |
| classifier (3,4) | 30 | 0.767 | 0.233 | 0.367 | 1115.9 |
| classifier (3,5) | 30 | 0.733 | 0.267 | 0.167 | 351.2 |
| classifier (3,6) | 30 | 0.733 | 0.267 | 0.167 | 351.2 |
| classifier (3,7) | 30 | 0.733 | 0.267 | 0.167 | 351.2 |
| classifier (3,8) | 30 | 0.733 | 0.267 | 0.167 | 351.2 |
| classifier (3,9) | 30 | 0.733 | 0.267 | 0.167 | 351.2 |
| classifier (4,5) | 30 | 0.733 | 0.267 | 0.167 | 351.2 |
| classifier (4,6) | 30 | 0.733 | 0.267 | 0.167 | 351.2 |
| classifier (4,7) | 30 | 0.733 | 0.267 | 0.167 | 351.2 |
| classifier (4,8) | 30 | 0.733 | 0.267 | 0.167 | 351.2 |
| classifier (4,9) | 30 | 0.733 | 0.267 | 0.167 | 351.2 |
| classifier (5,6) | 30 | 0.467 | 0.533 | 0.0 | 147.3 |
| classifier (5,7) | 30 | 0.467 | 0.533 | 0.0 | 147.3 |
| classifier (5,8) | 30 | 0.467 | 0.533 | 0.0 | 147.3 |
| classifier (5,9) | 30 | 0.467 | 0.533 | 0.0 | 147.3 |
| classifier (6,7) | 30 | 0.467 | 0.533 | 0.0 | 147.3 |
| classifier (6,8) | 30 | 0.467 | 0.533 | 0.0 | 147.3 |
| classifier (6,9) | 30 | 0.467 | 0.533 | 0.0 | 147.3 |
| classifier (7,8) | 30 | 0.467 | 0.533 | 0.0 | 147.3 |
| classifier (7,9) | 30 | 0.467 | 0.533 | 0.0 | 147.3 |
| classifier (8,9) | 30 | 0.467 | 0.533 | 0.0 | 147.3 |

**Best classifier thresholds: (1, 2)** with under-route 0.1 and 1203.3 GPU-s.

Classifier returned no parseable score for 0/32 items.

Note: the classifier maps EASY->2, MEDIUM->5, HARD->9 and in practice never emits HARD, so scores are only ever 2 or 5. Threshold pairs whose lower bound (cheap_max) is 5 or greater therefore behave identically to each other, since no score ever falls above 5.
