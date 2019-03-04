# A  behavior  driven  approach  for  sampling  rare  event  situations  for autonomous  vehicles.

Performance evaluation of urban autonomous vehicles  requires  a  realistic  model  of  the  behavior  of  other  road users in the environment. Learning such models from data involves collecting naturalistic data of real-world human behavior. In  many  cases,  acquisition  of  this  data  can  be  prohibitively expensive  or  intrusive.  Additionally,  the  available  data  often contain  only  typical  behaviors  and  exclude  behaviors  that  are classified as rare events. To evaluate the performance of AV in such  situations,  we  develop  a  model  of  traffic  behavior  based on the theory of bounded rationality. Based on the experiments performed  on  a  large  naturalistic  driving  data,  we  show  that the  developed  model  can  be  applied  to  estimate  probability  of rare  events,  as  well  as  to  generate  new  traffic  situation.

This repository contains the code from the paper.

### Rare event sampling

Running `run_with_opt_vals()` from `sumo_runner` module runs the simulation with optimal parameters for Bounded Rationality (BR) based sampling, Cross-Entropy (CE) sampling, and crude Monte Carlo (CMC) sampling in sequence. The results are generated in the set of `.list` files under processing_lists folder.

In order to generate the optimal parameters by running the optimization scheme for BR and CE, `run_opt_scheme()`. 

`final_results_plot()` and `final_results_var_plot()` in `data_analysis` module plots the results of the runs.

### Situation generation

`mle_est()` fits the behavior model to the data, and generates the synthetic datapoints.

### Dataset

The SPMD dataset used for the paper is available in [here][dataset]. We use the algorithms developed in [TrafficNet][Zhao2017] to extract the cut-in scenarios. Code can be found in `detect_events` module.

[Zhao2017]: https://arxiv.org/abs/1708.01872
[dataset]: https://catalog.data.gov/dataset/safety-pilot-model-deployment-data
