# Linearized HMMs

# Dependencies
* pytorch 1.6.0
* wandb 0.10.30
* torchtext 0.7.0
* genbmm (for the banded model)

# Experiments @ 16k states
* HMM Baseline
```
source scripts/hmm_commands.sh && run_hmm
```
* LHMM
```
source scripts/hmm_commands.sh && run_lhmm
```
* LHMM + Band
```
source scripts/hmm_commands.sh && run_band
```

## Optional
* Sparse Emission HMM (VL-HMM)
```
source scripts/hmm_commands.sh && run_sehmm
```
