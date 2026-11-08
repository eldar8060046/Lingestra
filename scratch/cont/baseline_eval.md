# Continuous-signing benchmark: baseline (continuous OFF), evaluation seeds [2001, 2002, 2003, 2004, 2005], vocabulary 30 signs, 1 enrolled take each (100 s)

| stream | sequences | sequence exact | word accuracy | substituted | deleted | inserted | with clarify/unknown | worst frame ms |
|---|---|---|---|---|---|---|---|---|
| 2 signs, same-take | 150 | 0% | 35% | 0% | 65% | 0% | 29% | 55 |
| 3 signs, same-take | 150 | 0% | 17% | 0% | 83% | 0% | 52% | 63 |
| 2 signs, cross-take | 150 | 0% | 4% | 1% | 95% | 0% | 89% | 60 |
| 3 signs, cross-take | 150 | 0% | 1% | 0% | 99% | 0% | 97% | 94 |
| 2 signs, cross-take, matchable signs | 150 | 0% | 11% | 0% | 89% | 0% | 79% | 38 |
| 3 signs, cross-take, matchable signs | 150 | 0% | 2% | 0% | 98% | 0% | 95% | 65 |

| single sign (no join) | signs | wrongly split into >1 events | exactly the right word | un-enrolled -> a word |
|---|---|---|---|---|
| same-take | 30 per vocab | 0% | 99% | - |
| cross-take | 30 per vocab | 0% | 21% | 4% |

decode_sequence, 200 frames, 150 templates: 149 ms mean / 192 ms max
