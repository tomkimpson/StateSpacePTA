Good stuff Tom!

Abstract: I think it should be $h_0 \sim 10^{-15}$, not $10^{15}$ 😉
---> Fixed

Introduction last paragraph: "pulsar pulse frequency" -> "pulsar frequency"; "subject the influence" -> "subject to the influence"; mention Section 5?
--->Fixed

f_em appears in Eq. (1) without being defined. Maybe "the deterministic evolution can" -> "the deterministic evolution $f_{\rm em}(t)$ can"
--->Fixed

"In Eq 2" -> "In Equation~\eqref{eq:2}"; "and $\gamma$ in Eq. 1" -> "and $\gamma$ in Equation~\eqref{eq:1}"
--->Fixed

"$f_{\rm EM}(t)$" -> "$f_{\rm em}(t)$"
--->Great catch!

Very minor, but maybe use \begin{align} \end{align} for Equations (3) and (4)?
--->Fixed. Nice

In Section 2.2 you mention using TOAs, which might be misleading/confusing. 
---> Changed to "Specifically, the pulsar frequency is modulated harmonically at the GW frequency"

Are $\delta$, $\alpha$ the same as $\theta$, $\phi$ in Equations (10) and (11)? (They are not technically defined when they appear in Equation (21))
--->Good catch. Now defined at the start of section 3.

I prefer C1 to Figure 1, however I'd like it even better if the right panels had $f'_p(t)$ not so badly estimated, i.e. not a perfect match, but not so wrong that the true path looks deterministic.
--->Fixed. Now there is less obvious difference in f_m which is a bit annoying, but I think it's a feature of the model that its difficult to show both.

"Equations (17), (18)" -> "Equations (17) and (18)".
--->Fixed

The workflow in Section 3.4 is a little unclear to me. Does (vi) imply repeat from (i)? How do the different live points work (do they each take the same L, \theta?) I know Section 3.2 has the nested sampler explanation, but the summary confused me
--->	Its a good point. Don't want to go too deep into explaining how the nested sampling works here, just an overview of how they interact

"different realisations of the system noise" -> "different realisations of the pulsar and measurement noise"
--->Fixed.

"to other PTAs of the IPTA" -> "to any other PTA"
--->Fixed

"which defines the vector $q^n$" -> "which defines the vector $q^{(n)}$"
--->Fixed

This first paragraph of 4.1.1 has lots of mid-sentence hyphens to deliminate subclauses, which are not grammatical. I recommend replacing with commas. Andrew dislikes em-dashes for subclauses.
--->Fixed to Andrew-ify. I now use brackets. Hopefully these are to taste

Define ATNF up here, as you use it later.
--->Fixed

"$[\gamma^{(n)}]^{-1} > > T_{\rm obs}$" -> ""$[\gamma^{(n)}]^{-1} \gg T_{\rm obs}$" (also in Section 4.2.1)
--->Good eye! Fixed

In Equation (29) is $T$ the observation span $T_{\rm obs}$, or the gap between subsequent TOAs? The paragraph below says it's both. Maybe $T$ -> $T_{\rm gap}$ for clarity, if needed?
--->Updated to give a few more words on why we do this/.

"calculated in this way is $= ...$" -> "calculated in this way is $\sigma^{(n)} = ...$"
--->Fixed

Caption Figure 3 "Equations (29), (30)" -> "Equations (29) and (30)" (and caption of Table 1)
--->Fixed

Relatedly, is Figure 3 really using (29) and (30), or is it the qualitative Andrés approach with baboo? The text at the end of 4.1.1 implies the latter, but the figure caption implies the former. Also applies in caption of Table 1.
--->Its complementary. Use Equations 29/30 to get some \sigma and \gamma values. Then pass values these into Andres+baboo; what do the syntehtic residuals generated look like? To my ear that is clear from the text, but will keep this point in mind.

"We generate $N$ synthetic noisy timesires - one for each pulsar - as follows:" -> ""We generate $N$ synthetic noisy timesires, one for each pulsar, as follows:" (this is Andrew's pedantic preference)
--->Fixed

"Equations (1)- (4)" -> "Equations (1)--(4)" (in multiple places in 4.1.2)
--->Fixed everywhere

I'm personally not a huge fan of $N_{\rm f}^{(n)}$ being the measurement noise, it mixes up the number of pulsars $N$, with a new variable ${\rm f}$. Maybe just $\varepsilon^{(n)}$?
--->Agreed. Nice idea

Equation (32), maybe have the denominator be $T_{\rm gap}$? (see above suggestion)
--->Fixed

Is $\sigma_{\rm m}$ really unitless? Equation (32) seems to say it should have units of frequency.
--->Fixed

I still don't understand the statement about $f_{\rm m}^{(n)}$ being modified by different measurement noise under Equation (32). Are you saying that the distribution $\varepsilon^{(n)}$ will vary pulsar-to-pulsar in reality, but not in the synthetic data? That's a different statement to the one about "noise realisations", which are the particular draws from the distribution of $\varepsilon^{(n)}$ at each measurement.

The text in Table 1 is a bit small. Can you make it span the page width, rather than just one column?
--->Fixed

In Table 1, "$\gamma$" -> "$\gamma^{(n)}$"
--->Fixed

$f_{\rm em}^{(n)}(t_1),\dot{f}_{\rm em}^{(n)}(t_1)$ -> "$f_{\rm em}^{(n)}(t_1)$ and $\dot{f}_{\rm em}^{(n)}(t_1)$" (in multiple places)
--->Fixed eveywhere

A reviewer might not like the phrase "estimate them optimally within" (given your uncertainties are larger than tempo2), maybe just "estimate them within"
--->Sure. I mean "optimal" in the specific sense that the estimates are those which minimise the error within our model/construction, rather than the general sense of "best". Leaving as is for now.

"we still estimate GW parameters" -> "we still accurately estimate GW parameters"
---> Fixed

"as discussed in Section 4.2.2 the" -> "as discussed in Section 4.2.2, the"
--->Fixed

"we do not need to set a prior on $d$" -> ""we do not need to set a prior on $d^{(n)}$"
--->Fixed

"an uninformative prior on $\gamma$ -> "an uninformative prior on $\gamma^{(n)}$"
--->Fixed

Need units in "over e.g. LogUniform(10^{-15}, 10^{-10})"
---> FIxed

$\sigma^{n}$ also needs units, right?
--->fixed

"which has a particularly large $\dot{f}_{\rm p}^{(n)}$ -> "which has a particularly large $\dot{f}_{\rm em}$" (I think?)
--->Great catch. Fixed

"not setting priors on $\gamma$ or $d$" -> "not setting priors on $\gamma^{(n)}$ or $d^{(n)}$"
--->Fixed

"c.f. Equation 20" personally I'd cut this, given you explicitly define in the next line.
---> Personally I like it as it feels like an adjustment to Eq 20 and so want to remind the reader. Leaving as is for now

Summary labels on the top of the one-dimensional posteriors in Figure 4 shouldn't say \pm 0, if they are referring to the 0.16, 0.84 quintiles. 
TO DO

"7 parameters of" -> "seven parameters of" (again, another Andrew pet-peeve)
---Fixed 

Same with "2-dimensional" -> "two-dimensional", "1D-posteriors" -> "one-dimensional posteriors" (this is in many places, especially 4.2.4)
--->Fixed everywhere

"Similar results can be derived for the the 3N parameters" -> "Similar results are derived for the 3N parameters"
--->Fixed

in my opinion you can cut "and the parameters are generally recovered unambiguously" and just end the sentence, starting the next with "We do not display..."
--> Leaving as is, but don't feel strongly

There is a floating \ref to Section 4.3 at the end of first paragraph of 4.2.3
--->fixed

I don't see "evident data gaps" in Figure 5". Maybe rephrase "The sparsity and steep drop-off of $\beta$ for values of $h_0 \lesssim 4\times10{-15} is due to..."
--->Fixed 

Also, is this a "noise artefact" of the sampler? Or just the natural limit of where the signal strength is low enough that the evidence for the model with signal and noise is no longer larger than the evidence for the model with just noise? Z(M1) > Z(M0) shouldn't necessarily hold to arbitrarily low signal strength, as both the process and measurement noise could hide the signal, and M1 has more parameters than M0, naturally down-weighting the evidence if the signal isn't loud enough (i.e. simpler null model may be a "better" explanation for the data, in an Occam sense). I'd also cut mention of the "gaps" from the caption of Fig 5.

---> Gaps removed from figure caption. For the rest, the "gaps" are a noise artefact in the sense that if I run the sampler again with identical data, I might get gaps in different places. Now this is, as you say due to the indistinguishability of M0 and M1. i.e. if it was just due to failure of  Z(M1) > Z(M0) then I would expect it to fail at some cutoff strain. I have rephrased this in the text.

Figure 5: has a mathfont capital B in the axis label, rather than $\beta$.
"strain magnitudes, $h_0$, for" -> "strain magnitude $h_0$ for". 
"detection tolerance cut-off" -> "detection threshold". 
Also missing full-stop at end of last sentence of caption. 
---> All fixed


Section 4.2.4:

"Sections 4.2.2, 4.2.3 is" -> "Sections 4.2.2 and 4.2.3 are"
--->fixed

"9 realisations of the noise" -> "nine realisations of the noise" (everywhere, for numbers less 10, sorry!)
--->fixed. 10 noise would be nicer here

"are highly consistent across the different noise realisations" -> "broadly overlap across the different noise realisations"
---> Fixed

"i.e. each of individual the distributions" -> "i.e. each of the individual distributions"
--->fixed

Cut "That is to say, ", start sentence with "For each"
--->fixed

"are not consistent" -> "do not broadly overlap"
--->fixed

"The large degree of variance in the 1D posteriors of $\iota$ and $h_0$ is due to" -> "The posteriors for $\iota$ and $h_0$ do not overlap for the different noise realisations due to"
--->Fixed

Again with the hyphens, they should be commas. 
--->Replaced with brackets. I hate commas

Add comma after "Regarding $\iota$ and $h_0$"
--->Fixed

"Equations 22, 23" -> "Equations (22) and (23)"
--->Fixed

$h$ -> $h_0$ (in quite a few places!)
--->Fixed

"the variance in the estimates between different noise realisations decreases." -> "the posteriors for the different noise realisations overlap."
--->Fixed

Similar language should replace the "variance" in the rest of the paragraph too, in my opinion.
--->Noted. Not yet fixed


Does Figure 8 show us much more information than Figure 7? I understand it's spelling it out, but an astute reader might be able to grok it from just 7 (and perhaps some of the text around 8). Just cognizant that the paragraph describing Figure 7 and 8 is quite long and detailed. Maybe more appropriate to put in an Appendix?
--->Noted. Will consider. 

Where does Equation (39) come from? I also don't understand what it's computing... I just have lots of questions here (is $\rho$ strictly bounded to [0, 1]? What are the "samples" you refer to? Are they the different noise realisations, or is the 10^3 here a different 10^3? Why compare to the mean of the samples?)
Could the discussion with $\rho$ be summarised with a p-p plot, instead? 

---> Discussed in person. Will use a Wasserstein instead to do this properly

I'd recommend dropping the dashed lines in Figures 6 and 8. (But this goes to our brief chat before group meeting re: these posterior plots having wrong axes/too smoothed).
--->Noted


Section 4.3:

"200 simulated injections" -> "200 injections"
--->Fixed

"5 parameters" -> "five parameters"
---.>Fied

These are technically credible intervals, not confidence intervals (check everywhere).
---> FIxed

"grey contoured regions label the first $3\sigma$ significance ... given 200 simulations." -> "grey contours enclose the $1\sigma$, $2\sigma$, and $3\sigma$ significance levels, given 200 injections."
--->Fixed

"We can see that it only $\Omega$ is well-behaved and falls within the shaded region." -> "We can see that only $\Omega$ falls within the $3\sigma$ shaded region."
--->Fixed

Cut following two sentences in my opinion. 
--->Noted. Left in for now

Spell out the vice-versa for clarity, I think
--->fixed

"confident i.e. narrow" -> "confident, i.e. narrow"
--->Fixed

"which are insufficiently narrow to capture the injected value" -> "which are overly precise (narrow) to contain the injected value"
--->Fixed


Section 4.4:

"(i.e. the median of the 1D marginalised posterior)" -> "(e.g. the median of the 1D marginalised posterior)" (or my preference would be to no longer talk about the medians, sorry)
--->Fixed. Retaining for now, may drop later c.f. new plots

Cut second sentence, first paragraph
--->Noted. Leaving fr now

"We consider 3 separate solutions:" -> "We consider three separate situations:"
--->Fixed

Honestly, this whole section feels a little "loose" in the language. Andrew will tighten it up, but maybe another pass is warranted before he sees it. I'd try and cut out colloquialisms, and reduce repetition. I.e. what are the key things you must communicate from Figure 10, and what is less relevant.
--->Noted. To review

(Noticed a few "grammar-typos" in this section, like no space after a full-stop, so maybe run a grammar checker over it all.)
---->Noted

Section 5:

"Discussion" is a little vague, maybe "Extensions and limitations"? May also want to shorten + roll into conclusions. Although the previous Section already had a deep discussion of limitations, so maybe just "Extensions"
--->Noted

Relatedly, I'd just cut the first paragraph
--->Noted

"Equation 16" -> "Equation (16)"
--->Fixed

Third paragraph, hyphen should be a comma (or an em-dash, which Andrew will promptly delete)

cut "such as PPTA and the EPTA" (political)

cut ", and appropriate for this initial methods work,"

"Equation 6" -> "Equation (6)"
--->Fixed

"phase coherency" -> "phase coherence"

"Equation 16" -> "Equation (16)"
--->Fixed

cut "Evaluating the performance of the method under these conditions would be a further interesting pursuit."
--->Fixed


Section 6:

Should maybe mention $\iota$ next to the $h_0$, or at least point the reader who skipped ahead back to Table 1.
--->Fixed

I didn't check Appendix A, but I see a few \eqref{} missing, haha
----> Appendix TBD


