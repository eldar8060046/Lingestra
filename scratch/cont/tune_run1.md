# Continuous-signing benchmark: continuous, tuning seeds [101, 102, 103], vocabulary 30 signs, 1 enrolled take each (528 s)

| stream | sequences | sequence exact | word accuracy | substituted | deleted | inserted | with clarify/unknown | worst frame ms |
|---|---|---|---|---|---|---|---|---|
| 2 signs, same-take | 90 | 96% | 97% | 0% | 3% | 0% | 10% | 75 |
| 3 signs, same-take | 90 | 91% | 97% | 1% | 2% | 0% | 13% | 104 |
| 2 signs, cross-take | 90 | 3% | 19% | 8% | 72% | 1% | 86% | 112 |
| 3 signs, cross-take | 90 | 1% | 20% | 9% | 70% | 0% | 97% | 171 |
| 2 signs, cross-take, matchable signs | 90 | 79% | 89% | 1% | 8% | 2% | 34% | 84 |
| 3 signs, cross-take, matchable signs | 90 | 58% | 81% | 0% | 17% | 1% | 53% | 111 |

| single sign (no join) | signs | wrongly split into >1 events | exactly the right word | un-enrolled -> a word |
|---|---|---|---|---|
| same-take | 30 per vocab | 0% | 99% | - |
| cross-take | 30 per vocab | 11% | 23% | 14% |

decode_sequence, 200 frames, 150 templates: 148 ms mean / 179 ms max
