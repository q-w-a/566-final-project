# Estimating an Optimal Treatment Regime for the Treatment of Opioid Dependence

Quinn White, Min Jia, and Songran Wang

## Code

* Files in the `scripts` directory:
  * `data_cleaning.Rmd` cleans the phase 2 data from RCT NIDA-CTN-0030 obtained from the [public.ctn0094data R package](https://github.com/CTN-0094/public.ctn0094data) to structure it into the needed format for our analysis.
  * `decision_list_opioid_data.R` runs the decision list approach of Zhang et al. (2018) on this data 
* Files in the `supplementary_code_files` directory is from the supplement of Zhang et al. (2018), which provides an implentation of their approach to estimate decision lists.

## Project Description 

### Background

The randomized clinical trial (identifier NIDA-CTN-0030) sought to test whether additional individualized drug counseling (IDC) is more beneficial compared to only standard medical management (SMM) in the treatment of opioid dependence with buprenorphine / naloxone (BUP / NX). This study contains two phases. The first compares the combination of IDC and BUP/NX treatment to BUP/NX treatment alone with 4 weeks of medication treatment followed by tapering of the medication in both groups. The second again compares the combination of IDC and BUP/NX treatment to BUP/NX treatment alone but with a 12-week treatment period followed by tapering. Those who failed to obtain a successful outcome were offered entry into phase 2. 

We denote IDC+SMM as enhanced medical management (EMM). Both phases include BUP/NX treatment, but in phase 1, patients receive the medication for 4 weeks, including a tapering protocol, while in phase 2 they receive the medication for 12 weeks and then go through a tapering protocol. For phase 1, patients were stratified with respect to their chronic pain and lifetime heroin use and then randomized within groups to receive either SMM or EMM. In the first 4 weeks, patients received BUP/NX treatment combined with SMM/EMM. BUP/NX was then tapered down to 0 at the end of week 4. At this point, patients entered an 8-week stabilization phase where they were no longer provided BUP/NX and only received SMM/EMM. The binary result ("success" or "failure") of phase 1 study was decided on the overall performance of patient during the entire 12 weeks. Success was defined based on an absence of relapse during the stabilization period, which was determined fromfrom patients' self-reports and urine tests. After phase 1, patients who did not attain a success were offered entry into phase 2. Those that opted for further treatment  were stratified to receive either SMM or EMM according to what they received in phase 1. Phase 2 is very similar to phase 1, but patients received treatment for 12-weeks before tapering of BUP/NX. After the 12 weeks treatment period of phase 2, patients were monitored for another 12 weeks to study the longer-term differences in outcomes between the SMM and EMM groups. 

### Analysis Plan 

Our analysis will focus on phase 2 of the study. Due to the complexity and chronic nature of substance use disorders, understanding how treatment can be tailored to the individual to improve outcomes is desirable (Murphy et al., 2007). With this goal in mind, we define our outcome of interest similarly to that in Weiss et al. (2011), where a successful outcome at phase 2 reflects no missed or positive urine screenings for opioids in weeks 22-24 of the study. The data we are using is that cleaned by Balise et al. (2024) and made publicly available [on github](https://github.com/CTN-0094/public.ctn0094data).

Our first aim is to estimate the single-stage optimal dynamic treatment regime (ODTR) at phase 2 using two approaches, using Q-learning where we model the Q-functions using linear regression, as described in detail in Tsiatis et al. (2019), and decision lists, the approach of Zhang et al. (2018).  Our second aim is to compare the expected outcome if all patients were assigned treatment according to the optimal treatment regimes estimated with the two aforementioned approaches to that if all patients were assigned to (1) the static regime where all patients receive SMM alone and (2) the static regime where all patients receive EMM. The history variables we will include in the estimation of the optimal dynamic treatment regime are the proportion of visits from phase 1 where the patient had a positive urine test result, where missing a urine screening was considered positive as recommended by \textcite{weiss2011}; age; sex; and the presence of severe withdrawal symptoms pre-induction.

### Methods

For estimation of the optimal treatment regime, we will implement Q-learning using the [DynTxnRegime](https://cran.r-project.org/web/packages/DynTxRegime/DynTxRegime.pdf) R package and the decision list approach of Zhang et al. (2018) using the code they provided in their supplementary materials. To estimate the value of the static regimes assigning everyone to EMM or SMM respectively, we will use the implementation of the AIPW estimator available in the [DTRBook](https://laber-labs.com/dtr-book/Chapter2/accessCode.html) R package written to accompany the textbook of Tsiatis et al. (2019).

To give some background on the approaches we are considering, augmented inverse probability weighting (AIPW) is a method that estimates the average outcome under a fixed treatment rule by combining information from both the treatment assignment model and the outcome model. It improves estimation accuracy by correcting for confounding and selection bias. A key advantage of AIPW is its double robustness: it gives consistent estimates if either the propensity score model or the outcome model is correctly specified. Also, it has lower variance than methods relying on a single model. 
Q-learning, meanwhile, is a method used to estimate optimal treatment strategies in multiple stages. Q-learning models the outcome associated with each treatment choice and uses a backward recursive approach to find the best treatment strategy at each stage.

## References

Balise, Raymond R. et al. (Nov. 21, 2024). “Data Cleaning and Harmonization of Clinical Trial Data: Medication-
assisted Treatment for Opioid Use Disorder”. In: PLOS ONE 19.11. Ed. by Vinod Kumar Vashistha, e0312695.
issn: 1932-6203. doi: 10.1371/journal.pone.0312695. url: https://dx.plos.org/10.1371/journal.
pone.0312695 (visited on 05/25/2025).

Murphy, Susan A. et al. (May 2007). “Developing Adaptive Treatment Strategies in Substance Abuse Research”.
In: Drug and Alcohol Dependence 88, S24–S30. issn: 03768716. doi: 10.1016/j.drugalcdep.2006.09.008.
(Visited on 05/25/2025).

Tsiatis, Anastasios A. et al. (Dec. 2019). Dynamic Treatment Regimes: Statistical Methods for Precision Medicine.
1st ed. Boca Raton : Chapman, Hall/CRC, 2020. — Series: Chapman & Hall/CRC monographs on statistics,
and applied probability: Chapman and Hall/CRC. isbn: 978-0-429-19269-2. doi: 10 . 1201 / 9780429192692.
(Visited on 04/19/2025).

Weiss, Roger D. (Dec. 2011). “Adjunctive Counseling During Brief and Extended Buprenorphine-Naloxone Treat-
ment for Prescription Opioid Dependence: A 2-Phase Randomized Controlled Trial”. In: Archives of Gen-
eral Psychiatry 68.12, p. 1238. issn: 0003-990X. doi: 10.1001/archgenpsychiatry.2011.121. (Visited on
05/25/2025).

Zhang, Yichi et al. (Oct. 2018). “Interpretable Dynamic Treatment Regimes”. In: Journal of the American Statistical
Association 113.524, pp. 1541–1549. issn: 0162-1459, 1537-274X. doi: 10 . 1080 / 01621459 . 2017 . 1345743.
(Visited on 04/02/2025).
