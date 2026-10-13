> **Original project proposal (2025)**, kept as the design specification that `catbots` implements. The
> physical self-driving-lab work was not carried out, and the LLM-enhanced BO stage was not implemented:
> the pipeline is validated in simulation on published HTE datasets instead (see
> [`docs/results_summary.md`](docs/results_summary.md)).

We propose to develop and validate a fully autonomous two-phase approach that combines statistical modelling with LLM-enhanced Bayesian optimisation (BO) to identify general reaction conditions for a transition-metal-catalysed cyclisation. This methodology addresses the fundamental challenge of navigating vast chemical spaces (>88,000 potential reactions) while maintaining strict experimental feasibility (fewer than 1% of reactions tested).

Using our autonomous synthesis platform, we will explore an initially agnostic reaction design space of 88,000 possibilities, a scale that reflects the limited prior knowledge available for predicting viable conditions.

To overcome this intractable optimisation space, a statistical screening pipeline will be implemented—emulating expert chemical reasoning while preserving algorithmic objectivity—to autonomously reduce this initial space by >97%. This process will transform an unwieldy search space into a (highly) promising subspace defined by a limited set of features strongly correlated with reaction yield. Within this refined subspace, an LLM-enhanced Bayesian optimisation system will be deployed for the final optimisation for generality. To the best of our knowledge, this would represent the first implementation of such an optimisation framework in a live experimental setting and, if successful, would substantially improve on conventional methods by adding dynamic predictive reasoning and adaptive learning. Through this systematic approach, we aim to achieve substrate-general reaction conditions in a fully autonomous fashion using fewer than 800 total experiments (<1% experimental coverage)—thereby pushing the boundaries of what is possible in a self-driving lab and setting a new benchmark for efficient reaction optimisation.

This methodology will be validated on an established pyridone cyclisation, providing benchmark comparisons against conventional optimisation. 


The research is organised into distinct work packages:


WP1: Design and featurisation of the initial reaction space

Objective: Construct a comprehensive chemical space that balances exploratory diversity with experimental feasibility while supporting chemically relevant featurisation

Approach: An initial unconstrained reaction space will be defined with structurally diverse alkynes, metal catalysts, phosphine ligands, and solvents. This agnostic selection ensures broad coverage of key parameters and minimizes the risk of overlooking high-yielding conditions for both cyclization reactions.

i) Metals. Candidate metals will be chosen based on a coarse-grained level by expert chemists across transition metals known to be capable of catalysing this type of cyclisations—namely, Rh, Ir, Pd and Co. For each type of transition metal, two corresponding salts will be chosen based on commercial availability, chemical stability to air and moisture, and solubility in organic solvents. Resulting in a total of 8 metal salts.

ii) Ligands. The Ligand Knowledge Base for Phosphorus(III) ligands (LKB-P) curated by Fey et al. will be used, offering a set of 348 diverse, commercially available or readily accessible monodentate P(III) ligands. The LKB-P also provides 27 chemically meaningful, DFT-derived steric and electronic descriptors enabling effective featurisation and algorithmic exploration (e.g., clustering) of these ligands.

iii) Substrates. An unbiased set of commercially available aryl alkynes will be assembled from SciFinder, filtered by a strict availability threshold (>20 suppliers). Following conformer search (CREST) and DFT optimization of the lowest-energy conformers, the candidates will be featurised using steric and electronic descriptors—including Sterimol parameters, dispersion, solvent-accessible-surface area and buried volume around the alkyne, frontier orbital energies and molecular polarizability of the alkyne at the ωB97X-D3 def2-SVP level of theory and NBO analysis.

iv) Solvents. The high reaction temperature (>110°C) constrains solvent choice to high-boiling, thermally stable options. We selected four solvents spanning different polarities and coordination abilities: toluene (non-polar, non-coordinating), anisole (weakly coordinating), dimethylformamide (polar, coordinating), and 1,4-dioxane (ethereal, chelating). The solvents will be featurised using a comprehensive set of empirical and computational solvent descriptors, developed by Lapkin et al.

Initial design space: With 8 substrates, 4 solvents, 347 ligands and 8 metal salts, giving ≈ 88,000 combinations. This scale exceeds a feasible experimental budget (≈ 800 experiments), even using BO and the SDL (see Section 1.1.1). We therefore introduce an activity mapping and feature-selection pipeline (WP2) to reduce the space rapidly using simple, interpretable models in the spirit of the work of Sigman et al. and Denmark et al.


WP2: Development of the statistical pipeline for rapid reaction-space reduction (pyridone cyclisation) 

Objective: The goal of this initial phase is feature selection and high-activity zone detection at low experimental cost, rather than precise yield prediction. To do so, this work package propose the development of a multi-stage statistical screening pipeline that mimics expert chemical reasoning whilst maintaining algorithmic objectivity. All experiments and decisions will be executed autonomously on the SDL.


WP2A: Metal catalyst screening 

Aim: Autonomously identify active metal centres using minimal experimental resources.

Experimental Design: We will initially consider 123 monophosphine ligands from LKB-P that are either commercially available or held in departmental inventory. These have representatives spanning each of the 12 clusters obtained by dimensionality reduction via principal component analysis (PCA) followed by K-means clustering (elbow method for optimal cluster number) over the full LKB-P (347 ligands), ensuring broad coverage without synthesis at this stage. From these 123 ligands, an initial testing pool of 60 ligands will be selected using Kennard-Stone algorithm (a deterministic sampling method that ensures uniform coverage of chemical space) to maximise uniform coverage of the ligand-feature space (Denmark paper).

The first 12 ligands selected by the Kennard-Stone algorithm will be used a testing set for mapping the catalytic activity of the transition metals. These 12 ligands will be tested against all 8 metal salts using standardised conditions:

•	Fixed substrate: 4-methylphenylacetylene (commercially available, stable, minimal electronic bias)

•	Fixed solvent: Anisole (120°C capability, weak coordination, good substrate solubility)

•	Total experiments: 96 (12 ligands × 8 metals salts)

Data Analysis and Decision Criteria:  The yield of the reaction will be determined by HPLC against an internal standard (e.g., 1,3,5-trimethoxybenzene). Reaction analysis will prioritise “Dead Zone Mapping”: any metal delivering <5 % yield across all 12 tests will be eliminated. We expect this to narrow the space to one or two metal cations (retaining the best salt per metal; it is extremely unlikely that Rh, Ir, Pd and Co all catalyse the same reaction.


WP2B: Statistical feature selection and high-activity zone mapping in phosphine ligand space

Aim: Autonomous mapping the complete ligand space using the optimal metal(s) from WP2A, identifying key ligand features governing the catalytic activity.

Phase 1: Initial Model Training. Once best metals salt(s) have been identified, the remaining 48 ligands from the initial testing pool (see WP2.1) will be tested maintaining the fixed substrate and solvent from WP2.1. here again, the yield of the reaction will be determined by HPLC against an internal standard. Combined with the 12 experiments per metal obtained in WP2.1, this will provide 60 datapoints for model training. Three complementary modelling approaches suitable for low data regime will be employed to (a) highlight high-activity regions and (b) perform feature selection:

•	ElasticNet Regression: The L1 (LASSO) regularisation of this model will perform automatic feature selection whilst handling correlated descriptors through L2 regularisation. 

•	Shallow Neural Network: A flexible non-linear model, SHAP (SHapley Additive exPlanations) analysis will be used for feature importance ranking and model interpretability.

•	Gaussian Process Regression: A probabilistic, non-parametric model that provides uncertainty quantification, with SHAP analysis again used to reveal feature importance.

We anticipate \~5 of the 27 LKB-P descriptors to be retained as most informative at this stage, establishing preliminary structure-activity relationships (SAR) for the ligands.

Phase 2: Model Exploitation for High-Activity Ligand Discovery. To identify superior ligands beyond the initial testing pool, 12 additional ligands will be selected but this time from the full 348-ligand LKB-P database based on model predictions. 

The best performing model from Phase 1 will generate predictions for all 348 ligands. To balance high activity with chemical diversity, we will apply the Kennard-Stone algorithm to the top 15% highest-yielding ligands (52 ligands from 348), selecting 12 that maximise structural diversity within this high-performance subset. While these ligands may require synthesis, the risk is justified by their high predicted activity and their potential to identify high-activity regions previously undersampled. This strategic selection should also help create a more balanced dataset by enriching it with expected higher-yield datapoints, addressing a usual pitfall of experimental datasets that are typically skewed towards low- or zero-yield reactions (Denmark Jensen). With these additional experimental points performed by the SDL, both the ElasticNet regression and neural network will be retrained, refining descriptor selection and high-activity zone mapping.


WP2C: Substrate compatibility mapping, multitask learning for joint substrate-ligand optimisation, and final search space definition

Aim. Autonomous reaction boundaries mapping and establishment of substrate-dependent structure-activity relationships.

Initial Substrate Classification. 36 diverse substrates will be selected from the commercially available aryl alkyne library tabulated in WP1 using the Kennard-Stone algorithm. Each substrate will be tested under the best conditions identified in WP2.2 (fixed metal salt, ligand, and solvent). These outcomes will train a binary logistic regression classifier with ElasticNet regularisation to predict substrate compatibility based on molecular descriptors, using a 5% yield cut-off to define active vs. inactive substrates.

Multi-Substrate Feature Selection. The subset of substrates classified as active will be clustered using K-means (k = 5) to define five chemically diverse groups for detailed investigation. Substrate 1, already extensively characterised in WP2B (72 datapoints), will serve as its cluster representative. The remaining four cluster centroids will be screened using 24 ligands selected via Kennard-Stone from the pool of 60 ligands defined in WP2B, generating 96 experimental points. The resulting curated dataset (168 data points across five substrates) will be used to train a multitask ElasticNet regression model to identify key ligand and substrate descriptors predictive of high yield. By modelling all substrates jointly, the approach will expose high-activity regions and substrate-dependent structure–activity relationships (SARs) within the ligand space.

BO Space Definition. The descriptors selected by the multitask model will serve as the primary featurisation for ligands and substrates in the final Bayesian Optimisation (BO) stage, ensuring chemically meaningful representations and facilitating smoother model convergence. The active substrate set identified earlier in this WP will be reclustered using this refined featurisation. K-means clustering (k = 8) will be applied to identify eight chemically diverse representatives for final optimisation—each cluster centroid serving as cluster representative. The multitask model will generate predictions for all 348 LKB-P ligands across the five tested substrates. The 20 ligands with highest predicted average yields will be selected for Bayesian optimisation. The final search space comprises:

•	8 substrate exemplars

•	20 top-performing ligands

•	All salts for the active metal(s), expecting: 1-2 metals, see WP1 and WP2A

•	4 solvents, see WP1

Following this statistically guided reduction, the search space is narrowed to approximately 1,280–2,560 reaction combinations, representing a 44-fold reduction from the initial 88,000-combination space. This tractable and chemically informed space is now amenable to Bayesian optimisation within realistic experimental budgets (\~25% coverage, 320-640 experiments) for identifying substrate-general reaction conditions.



