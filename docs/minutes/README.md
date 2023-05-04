

## Minutes of meetings


This file is a record of discussions, interesting points and suggestions from group meetings around this PTA project



### 04/05



* Suggestions for reparameterisation of $h, \iota$: 
    * Double check hp, hx results
    * Try a product of $H_{ij} q^i q^j$
    * Check  $ \frac{H_{ij} q^i q^j}{1 + n \cdot q }$  Can we invert to recover?


* Suggestions for small $h$ issues:
    * Work down to smaller $h$ graudally. How do the parameters change at each point?

    * Try zero measurement noise 

    * Try quad precision?

    * Try to subtract off the frequency magnitude i.e. normalise by the ephemeris 

    * integration of S/N? Cadence, 10 years?  



---

### 20/04


* Is the $\Phi_0$ term correct in the measurement equation? Are all the pulsars really in phase at this time?

    * The measurement equation listed in the manuscript was incorrect. This has now been updated. Some corrections to code implementation of trig form of the measurement equation to account for cross terms. See Eq. 39 of [additional notes](https://github.com/tomkimpson/StateSpacePTA.jl/blob/main/docs/measurement_eqn_derivation/LectureNotes_Math571.pdf) Thanks to Tong, Jackie, Liam. 

* Can we reformulate in terms of phase rather than frequency to make contact with observations better?

    * TBD. Seems straightforward, just integrate measurement equation w.r.t time. Did Nicholas see any improvements re parameter estimation using phase over frequency, or it is just a choice to be more 


* What is going on with $\iota$? Can we infer this parameter? Try a reparameterisation in terms of h_+ and h_\{times}
    * We can reparameterise in terms of h_+ and h_\{times} straightforwardly, see branch [reparameterize](https://github.com/tomkimpson/StateSpacePTA.jl/tree/reparameterize). Does not seem to help parameter estimation (seems to settle on finding hp or hx), although this needs some more exploration. Work halted whilst next point was explored.  


* Try for more physical (i.e. smaller!) GW strains
    * Major obstruction over last few weeks. 


* Be clear on how single source PTA analyses are currently done 

    * Typically people talk in terms of the timing residual e.g. [Lee et al. 2011](https://arxiv.org/abs/1103.0115) , [Zhu et al 2014](https://academic.oup.com/mnras/article/444/4/3709/1029897), [Zhu et al 2015](https://academic.oup.com/mnras/article/449/2/1650/1075548), [Hazboun et al 2019](https://arxiv.org/abs/1907.04341) and then fit a model on these residuals. Also generally focus solely on the Earth term, ignoring the pulsar terms. 

* Include the pulsar parameters $d$ and $\sigma_p$ in the inference
    * Use a smaller PTA $\sim$ 20 pulsars
    * Inference on all GW parameters + all pulsar distances
    * Inference on all GW parameters + all $\sigma_p$.

        * TBD. Small strain considerations should be solved first.

* What are reasonable, astrophysical values for $\sigma_p$ and $\gamma$?

    $\gamma$ not too important over our timescales. Just something $\gamma^{-1} >> 10 $ years.
    $\sigma_p$ disussed in notebooks. See discussion relating to numerical float issues.