# Continuous-signing benchmark: continuous, evaluation seeds [7101, 7102, 7103, 7104, 7105], vocabulary 30 signs, 1 enrolled take each (626 s)

| stream | sequences | sequence exact | word accuracy | substituted | deleted | inserted | with clarify/unknown | worst frame ms |
|---|---|---|---|---|---|---|---|---|
| 2 signs, same-take | 150 | 91% | 95% | 0% | 4% | 1% | 11% | 49 |
| 3 signs, same-take | 150 | 85% | 95% | 0% | 5% | 0% | 9% | 49 |
| 2 signs, cross-take | 150 | 1% | 15% | 5% | 80% | 0% | 84% | 51 |
| 3 signs, cross-take | 150 | 0% | 14% | 4% | 82% | 0% | 93% | 76 |
| 2 signs, cross-take, matchable signs | 150 | 45% | 68% | 2% | 28% | 2% | 67% | 48 |
| 3 signs, cross-take, matchable signs | 150 | 18% | 58% | 1% | 38% | 2% | 83% | 66 |

| single sign (no join) | signs | wrongly split into >1 events | exactly the right word | un-enrolled -> a word |
|---|---|---|---|---|
| same-take | 30 per vocab | 1% | 99% | - |
| cross-take | 30 per vocab | 7% | 20% | 8% |

decode_sequence, 200 frames, 150 templates: 60 ms mean / 86 ms max
