# Continuous-signing benchmark: baseline (continuous OFF), evaluation seeds [7101, 7102, 7103, 7104, 7105], vocabulary 30 signs, 1 enrolled take each (85 s)

| stream | sequences | sequence exact | word accuracy | substituted | deleted | inserted | with clarify/unknown | worst frame ms |
|---|---|---|---|---|---|---|---|---|
| 2 signs, same-take | 150 | 2% | 34% | 0% | 65% | 0% | 34% | 17 |
| 3 signs, same-take | 150 | 0% | 14% | 0% | 86% | 0% | 63% | 21 |
| 2 signs, cross-take | 150 | 0% | 3% | 1% | 96% | 0% | 91% | 20 |
| 3 signs, cross-take | 150 | 0% | 1% | 0% | 99% | 0% | 97% | 23 |
| 2 signs, cross-take, matchable signs | 150 | 0% | 18% | 0% | 82% | 0% | 65% | 16 |
| 3 signs, cross-take, matchable signs | 150 | 0% | 3% | 0% | 97% | 0% | 91% | 19 |

| single sign (no join) | signs | wrongly split into >1 events | exactly the right word | un-enrolled -> a word |
|---|---|---|---|---|
| same-take | 30 per vocab | 0% | 97% | - |
| cross-take | 30 per vocab | 0% | 18% | 7% |

decode_sequence, 200 frames, 150 templates: 65 ms mean / 113 ms max
