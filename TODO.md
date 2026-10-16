# TODO

- unit commitment use-case (MAX-LINSAT with DQI)
- finance use-case (tbd)

Goals:
- scaling/crossover analyses
- resource estimation
- fair evaluation (TTS, seeds, confidences, tuning budgets)
- type hints, packaging, reproducibility
- tune D-Wave chain strength: on Advantage2, some job shop runs lose every read to broken chains

Rust integration:
- profile python
- identify the computationally intensive part
- write *that* in Rust behind PyO3, show speedup
