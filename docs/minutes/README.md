

## Minutes of meetings


This file is a record of discussions, interesting points and suggestions from group meetings around this PTA project


### 20/04


* Is the $\Phi_0$ term correct in the measurement equation? Are all the pulsars really in phase at this time?

    * The measurement equation listed in the manuscript was incorrect. This has now been updated. Some corrections to code implementation of trig form of the measurement equation to account for cross terms. Thanks to Tong, Jackie, Liam. 

* Can we reformulate in terms of phase rather than frequency to make contact with observations better?

* Try for more physical (i.e. smaller!) GW strains

* What is going on with $\iota$? Can we infer this parameter? Try a reparameterisation in terms of h_+ and h_\{times}

* Be clear on how single source PTA analyses are currently done 

    * Typically people talk in terms of the timing residual e.g. [Lee et al. 2011](https://arxiv.org/abs/1103.0115) , [Zhu et al 2014](https://academic.oup.com/mnras/article/444/4/3709/1029897), [Zhu et al 2015](https://academic.oup.com/mnras/article/449/2/1650/1075548), [Hazboun et al 2019](https://arxiv.org/abs/1907.04341) and then fit a model on these residuals. Also generally focus solely on the Earth term, ignoring the pulsar terms. 

* Include the pulsar parameters $d$ and $\sigma_p$ in the inference
    * Use a smaller PTA $\sim$ 20 pulsars
    * Inference on all GW parameters + all pulsar distances
    * Inference on all GW parameters + all $\sigma_p$.

* What are reasonable, astrophysical values for $\sigma_p$ and $\gamma$