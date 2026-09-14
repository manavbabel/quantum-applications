# README

This is a demo repo showing three applications and a variety of quantum approaches, benchmarked against existing approaches and actual computers.

Applications:

- job shop scheduling
- energy (unit commitment with storage, EV charging?) Elexon BMRS and the National Grid ESO data portal
- finance (not portfolio optimisation! ECR or CVA?)

Formulations:
- QUBO/Ising for annealing/QAOA
- MAX-LINSAT with DQI
- QCMC?

Benchmarks:
- OR-Tools
- MILP (HiGHS or CBC)
- noiseless simulation
- noisy simulation
- real QC

Remember:
- use real public data
- prominently show negative results and crossover analysis
- resource estimation (what's needed, when expected?)
- fair evaluation (TTS, seeds, confidences, tuning budgets)
- tests, CI, type hints, packaged, reproducibility

Rust integration:
- build framework in python
- profile it
- identify the computationally intensive part
- write *that* in Rust behing Py03, show speedup
- be disciplined about scope! do one app thoroughly before touching the others
- write clear short descriptions, communication is important!
