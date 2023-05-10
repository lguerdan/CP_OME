This code is the official implementation of [Counterfactual Prediction Under Outcome Measurement Error](https://arxiv.org/abs/2302.11121), published at FAccT 23'. 

# Requirements
To install dependencies, run
```
! pip3 install -r requirements.txt
```

For licensing and privacy reasons, we omit raw data for semi-synthetic experiments from this repository. The raw data for the semi-synthetic evaluations can be downloaded from: 
- JOBS Dataset: [train](https://www.fredjo.com/files/jobs_DW_bin.new.10.train.npz), [test](https://www.fredjo.com/files/jobs_DW_bin.new.10.test.npz)
- Oregon Health Insurance Experiment (OHIE) Dataset: [https://dataverse.harvard.edu/dataset.xhtml?persistentId=doi:10.7910/DVN/SJG1ED](https://dataverse.harvard.edu/dataset.xhtml?persistentId=doi:10.7910/DVN/SJG1ED)

After accessing the data, place the files in the relevant directory `data/JOBS/` and `data/OHIE/`. The OHIE data can be merged and pre-processed by executing `data/OHIE/preprocess.py` after placing the download package in the folder.  


# Running experiments

To run the synthetic experiment, execute:
```
! python3 drivers.py erm syn_convergence_learned
! python3 drivers.py erm syn_convergence_oracle
```

To run the semi-synthetic experiment, execute:
```
! python3 drivers.py erm_semisyn semisyn_oracle_exp
! python3 drivers.py erm_semisyn_assumption semisyn_learned_exp
```

# Bibtex Citation

```
@article{guerdan2023counterfactual,
  title={Counterfactual Prediction Under Outcome Measurement Error},
  author={Guerdan, Luke and Coston, Amanda and Holstein, Kenneth and Wu, Zhiwei Steven},
  booktitle={Proceedings of the 2023 ACM Conference on Fairness, Accountability, and Transparency},
  year={2023}
}
```