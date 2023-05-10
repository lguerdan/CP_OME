## Setup
```
! pip3 install -r requirements.txt
```

## Data
For licensing and privacy reasons, we omit raw data from this repository. The raw data for the semisynthetic evaluations can be downloaded from: 
- JOBS: [train](https://www.fredjo.com/files/jobs_DW_bin.new.10.train.npz), [test](https://www.fredjo.com/files/jobs_DW_bin.new.10.test.npz)
- [OHIE](https://dataverse.harvard.edu/dataset.xhtml?persistentId=doi:10.7910/DVN/SJG1ED)


## Synthetic experiment
```
! python3 drivers.py erm syn_convergence_learned
! python3 drivers.py erm syn_convergence_oracle
```

## Semi-synthetic experiment
```
! python3 drivers.py erm_semisyn semisyn_oracle_exp
! python3 drivers.py erm_semisyn_assumption semisyn_learned_exp
```
