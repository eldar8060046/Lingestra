# Continuous-signing benchmark: continuous, tuning seeds [101, 102, 103], vocabulary 30 signs, 1 enrolled take each (524 s)

| stream | sequences | sequence exact | word accuracy | substituted | deleted | inserted | with clarify/unknown | worst frame ms |
|---|---|---|---|---|---|---|---|---|
| 2 signs, same-take | 90 | 93% | 96% | 0% | 3% | 1% | 9% | 84 |
| 3 signs, same-take | 90 | 88% | 96% | 1% | 3% | 1% | 17% | 108 |
| 2 signs, cross-take | 90 | 4% | 20% | 7% | 73% | 1% | 86% | 114 |
| 3 signs, cross-take | 90 | 1% | 20% | 4% | 76% | 0% | 96% | 216 |
| 2 signs, cross-take, matchable signs | 90 | 78% | 88% | 0% | 9% | 3% | 36% | 89 |
| 3 signs, cross-take, matchable signs | 90 | 46% | 77% | 0% | 21% | 1% | 63% | 119 |

| single sign (no join) | signs | wrongly split into >1 events | exactly the right word | un-enrolled -> a word |
|---|---|---|---|---|
| same-take | 30 per vocab | 2% | 97% | - |
| cross-take | 30 per vocab | 11% | 23% | 13% |

decode_sequence, 200 frames, 150 templates: 158 ms mean / 206 ms max
