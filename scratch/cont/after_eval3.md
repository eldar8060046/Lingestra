# Continuous-signing benchmark: continuous, evaluation seeds [2001, 2002, 2003, 2004, 2005], vocabulary 30 signs, 1 enrolled take each (702 s)

| stream | sequences | sequence exact | word accuracy | substituted | deleted | inserted | with clarify/unknown | worst frame ms |
|---|---|---|---|---|---|---|---|---|
| 2 signs, same-take | 150 | 91% | 96% | 0% | 3% | 1% | 3% | 53 |
| 3 signs, same-take | 150 | 89% | 95% | 0% | 5% | 0% | 9% | 51 |
| 2 signs, cross-take | 150 | 6% | 19% | 4% | 77% | 0% | 84% | 68 |
| 3 signs, cross-take | 150 | 0% | 15% | 4% | 81% | 0% | 89% | 67 |
| 2 signs, cross-take, matchable signs | 150 | 49% | 64% | 2% | 34% | 0% | 57% | 58 |
| 3 signs, cross-take, matchable signs | 150 | 25% | 60% | 1% | 40% | 0% | 77% | 63 |

| single sign (no join) | signs | wrongly split into >1 events | exactly the right word | un-enrolled -> a word |
|---|---|---|---|---|
| same-take | 30 per vocab | 1% | 100% | - |
| cross-take | 30 per vocab | 11% | 24% | 8% |

decode_sequence, 200 frames, 150 templates: 73 ms mean / 93 ms max
