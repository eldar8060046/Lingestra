# Continuous-signing benchmark: continuous, evaluation seeds [2001, 2002, 2003, 2004, 2005], vocabulary 30 signs, 1 enrolled take each (743 s)

| stream | sequences | sequence exact | word accuracy | substituted | deleted | inserted | with clarify/unknown | worst frame ms |
|---|---|---|---|---|---|---|---|---|
| 2 signs, same-take | 150 | 87% | 94% | 1% | 5% | 1% | 7% | 81 |
| 3 signs, same-take | 150 | 83% | 93% | 0% | 7% | 0% | 14% | 61 |
| 2 signs, cross-take | 150 | 6% | 19% | 4% | 76% | 0% | 84% | 82 |
| 3 signs, cross-take | 150 | 0% | 15% | 3% | 82% | 0% | 90% | 86 |
| 2 signs, cross-take, matchable signs | 150 | 49% | 64% | 2% | 34% | 0% | 57% | 72 |
| 3 signs, cross-take, matchable signs | 150 | 26% | 60% | 1% | 39% | 0% | 77% | 68 |

| single sign (no join) | signs | wrongly split into >1 events | exactly the right word | un-enrolled -> a word |
|---|---|---|---|---|
| same-take | 30 per vocab | 1% | 99% | - |
| cross-take | 30 per vocab | 11% | 24% | 8% |

decode_sequence, 200 frames, 150 templates: 127 ms mean / 217 ms max
