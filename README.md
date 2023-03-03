# SAMS_prediction
Echo state networks to predict South American Monsoon System wet season onset

Based on the work of Takahito Mitsui and Niklas Boers 2021 Environ. Res. Lett. 16 074024 ([https://doi.org/10.1088/1748-9326/ac0acb](https://doi.org/10.1088/1748-9326/ac0acb))

The file `esn_SAMS.jl` optimizes hyperparameters using `Hyperopt.jl`. 

`ensemble_ecn.jl` is running an ensemble simulation using Multithreading with the optimized hyperparamters and saves the 100 best ESNs (output layer (W<sub>out</sub>) and the esn) in two seperate folders.

`ensemble_prediction.jl` is loading the saved ESNs and is predicting the mean proximity function.

`submit_esn_ensemble.sh` is an example script for SLURM submission on a HPC.

The 100 ESNs are available here (if you want access, write me): 
[ESN states & Output layer](https://universitetetitromso-my.sharepoint.com/:f:/r/personal/nbo022_uit_no/Documents/ESN?csf=1&web=1&e=qFaMdy)
