# STORM: A map-reduce framework for Symbolic Time Intervals series classification

## Introduction

This repository is related to the paper "STORM: A map-reduce framework for Symbolic Time Intervals series classification", to be submitted to the *XXXXX* journal.

We introduce STORM - a novel map-reduce framework for the classification of series of Symbolic Time Intervals (STIs). 

The framework operates as follows:

**Setup:** STORM first converts raw input STIs series data into multivariate time series (MTS) representation. 

**Map:** Converted MTS are then discretized into equally-sized blocks. 
Each block is independently transformed into a uniform latent space via a common, desired Rocket (Dempster et al. 2020) variant for MTS used as a base transformation in STORM.
Through Rocket’s transformation, global properties can be extracted from the converted MTS. 
That is, however, limited to the scope of a single block, which also allows for the better capturing of local classification-informative aspects of the input series. 
Balancing local and global information can therefore be achieved by tuning the block size of STORM. 
In the paper, as well as in this repository, STORM is presented and evaluated with two of the more acknowledged variants of Rocket for MTS as base transformations – i.e., MiniRocket (Dempster et al. 2021) and MultiRocket (Tan et al. 2022). 
However, STORM exhibits a generic framework into which any transformation method of arbitrary long MTS into fixed-sized feature vectors can be further integrated.

**Reduce:** The complete sequence of blocks’ transformed feature vectors is then fed into a deep, lightweight, bidirectional LSTM network for classification. 
the network combines the local blocks’ feature vectors into a final global prediction while utilizing the sequential nature of the data. 

We hope that this repository's contents will contribute to future research as well as real-world applications in the field of STIs series classification.

## Repository Contents

The contents of this repository are as follows:
- Real-world benchmark datasets used for the evaluation
- Source code of STORM
- Code for the experiments detailed in the paper
- Experimental results in CSV files
- Jupyter Notebook for running the complete flow
- Running example 
- Extensions for MiniRocket and MultiRocket source codes

## Datasets

- **Location**: All datasets are available under the [data](https://github.com/omerh18/STORM/tree/main/data) directory.
- **Contents**:
    - Real-world benchmark datasets
        - AUSLAN2 (Mörchen and Fradkin 2010)
        - BLOCKS (Mörchen and Fradkin 2010)
        - CONTEXT (Mörchen and Fradkin 2010)
        - HEPATITIS (Patel et al. 2008)
        - MUSEKEY (Bilski and Jastrzębska 2022)
        - PIONEER (Mörchen and Fradkin 2010)
        - SKATING (Mörchen and Fradkin 2010)
        - WEATHER (Bilski and Jastrzębska 2022)
- **Format**:
	- **data.txt** - STIs series data 
		- Each line stands for a single STI in a tab separated format, including the STIs series ID, symbol-type, start-time, and finish-time 
	- **labels.txt** - STis series labels
		- Each line specifies the class label of a single STIs series in a tab separated format, including the STIs series ID and its respective class-label

## Code

### STORM

The code of STORM is implemented under the [src](https://github.com/omerh18/STORM/blob/main/src) directory.

### Experiments

The code for the experiments conducted and presented in the paper is available under the [experiments](https://github.com/omerh18/STORM/tree/main/experiments) directory.

The Jupyter Notebook [storm_flow_notebook.ipynb](https://github.com/omerh18/STORM/blob/main/storm_flow_notebook.ipynb) includes the code for triggering all the experiments, one after the other.

### MiniRocket and MultiRocket Extensions

As described in the paper, we extended the source codes of MiniRocket and MultiRocket with additional functionality.
That is, enabling:
1. Any specified maximal number of channels per kernel (instead of just the default value of 9) 
2. Deterministic selection of any specified number of channels for all kernels (thus, removing one level of randomness in determining the number of channels, while the specific combinations of channels per kernel are still randomized)
3. Optional use of the first order difference series’ transformation, which is used by default in MultiRocket multivariate. 

The extended source codes of MiniRocket and MultiRocket are available under the [extended_minirocket](https://github.com/omerh18/STORM/blob/main/extended_minirocket) and [extended_multirocket](https://github.com/omerh18/STORM/blob/main/extended_multirocket) directories respectively.

Original source codes have been forked from [MiniRocket](https://github.com/angus924/rocket) and [MultiRocket](https://github.com/ChangWeiTan/MultiRocket).

## Experimental Results

The experimental results in CSV files are available under the [results](https://github.com/omerh18/STORM/tree/main/results) directory.

## Dependencies

- Python 3.8.8
- Jupyter Notebook
- Packages
    ```
    numba == 0.53.1
    numpy == 1.23.3
    pandas == 1.2.4
    scipy == 1.9.2 
    scikit_learn == 0.24.1
    sktime == 0.4.3
    tensorflow == 2.10.0
    keras == 2.10.0
    dataclasses == 0.6
    simple_parsing == 0.1.5
    ```

## Running Instructions

To run the experiments, it is recommended to simply run the Jupyter Notebook [storm_flow_notebook.ipynb](https://github.com/omerh18/STORM/blob/main/storm_flow_notebook.ipynb).

To run a specific experiment, it is recommended to simply run the cells corresponding to that specific experiment within [storm_flow_notebook.ipynb](https://github.com/omerh18/STORM/blob/main/storm_flow_notebook.ipynb).

Note that by default results will be automatically saved under the [results](https://github.com/omerh18/STORM/tree/main/results) directory. 

Finally, an example for running STORM on a specific dataset under k-fold CV is provided in the [run_storm_k_fold.py](https://github.com/omerh18/STORM/tree/main/run_storm_k_fold.py) file. 

This file can be also triggered from the command line, for example:

```shell
python run_storm_k_fold.py --k 10 --dataset 'blocks' --use_multirocket False --calc_first_order_diff 0 --num_features 2500 --block_size 1000
```

## References

[1]	Bilski, J. M., & Jastrzębska, A. (2022, October). COSTI: a New Classifier for Sequences of Temporal Intervals. In 2022 IEEE 9th International Conference on Data Science and Advanced Analytics (DSAA) (pp. 1-10). IEEE.

[2] Dempster, A., Petitjean, F., & Webb, G. I. (2020). ROCKET: exceptionally fast and accurate time series classification using random convolutional kernels. Data Mining and Knowledge Discovery, 34(5), 1454-1495.

[3] Dempster, A., Schmidt, D. F., & Webb, G. I. (2021, August). Minirocket: A very fast (almost) deterministic transform for time series classification. In Proceedings of the 27th ACM SIGKDD conference on knowledge discovery & data mining (pp. 248-257).

[4] Mörchen, F., & Fradkin, D. (2010, April). Robust mining of time intervals with semi-interval partial order patterns. In Proceedings of the 2010 SIAM international conference on data mining (pp. 315-326). Society for Industrial and Applied Mathematics.

[5]	Patel, D., Hsu, W., & Lee, M. L. (2008, June). Mining relationships among interval-based events for classification. In Proceedings of the 2008 ACM SIGMOD international conference on Management of data (pp. 393-404).

[6]	Tan, C. W., Dempster, A., Bergmeir, C., & Webb, G. I. (2022). MultiRocket: multiple pooling operators and transformations for fast and effective time series classification. Data Mining and Knowledge Discovery, 36(5), 1623-1646.
