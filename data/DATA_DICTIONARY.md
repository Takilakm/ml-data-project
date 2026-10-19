# Data dictionary — self-collected daily log

One row per participant per day.

| Column | Type | Units | Notes |
|---|---|---|---|
| Date | date | YYYY-MM-DD | the day described |
| Participant | category | P1 / P2 / P3 | anonymised contributor code |
| Sleep | numeric | hours | sleep the previous night |
| Exercise | numeric | minutes | intentional exercise; 0 if none |
| Screen time | numeric | hours | non-study screen use |
| Study | numeric | hours | focused study or coursework |
| Caffeine | numeric | cups | 0.25 = a few sips |
| Mood | ordinal | 1–10 | 1 worst, 10 best |
| Stress | ordinal | 1–10 | 1 relaxed, 10 overwhelmed |
| Focus | ordinal | 1–10 | 1 unfocused, 10 sharp |
| Notes | text | — | optional free text; excluded from modelling |

## Derived target

`high_stress = 1` where Stress >= 6, else 0.

A per-participant calibrated alternative is also evaluated, where `high_stress = 1` when
that day's stress is above that participant's own mean.

## Known limitations

- Self-reported ratings are not calibrated between participants.
- Missing days are left blank rather than imputed at collection time.
- Three participants only; findings do not generalise beyond this group.
