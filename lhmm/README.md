# Low-Rank HMMs
This repo provides instructions for reproducing our low-rank HMM (LHMM) results in
``Low-Rank Constraints for Fast Inference in Structured Models''

# Dependencies
* pytorch 1.6.0
* wandb 0.10.30
* torchtext 0.7.0
* genbmm (only needed for running the banded model)

# Experiments @ 16k states
To run each of the models, use the following commands.
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
