# Quant Stats Skill Building — A 6-Week Module

A detailed 6-week applied-statistics module for ML interview prep. The core insight is that with a 2-3 hour weekly budget you cannot grind a textbook -- instead you build **8-10 short "quant stories"**: rehearsed, math-anchored explanations you can deliver in 90 seconds and defend under follow-up. This guide turns each week of that plan into a proper study module: formulas with intuition, a hands-on notebook, common interview questions, and a self-test. The notebooks live in `../notebooks/` and use synthetic data shaped like real eval pipelines (LAD-style monitors, SandboxBench-style agent tasks, attack-category leaderboards) so the worked examples match the kinds of stories you actually want to tell.

---

## Table of Contents

1. [Week 1 -- Confidence Intervals That Matter](#week-1--confidence-intervals-that-matter)
2. [Week 2 -- Comparing Two Things Properly](#week-2--comparing-two-things-properly)
3. [Week 3 -- Power and Sample Size](#week-3--power-and-sample-size)
4. [Week 4 -- Multiple Testing](#week-4--multiple-testing)
5. [Week 5 -- The Quant Nose Cheat Code](#week-5--the-quant-nose-cheat-code)
6. [Week 6 -- Rehearse and Pressure-Test](#week-6--rehearse-and-pressure-test)
7. [The 5 Flashcards](#the-5-flashcards)
8. [If You Find Extra Time](#if-you-find-extra-time)

---

## Week 1 -- Confidence Intervals That Matter

**Source plan reading:** Miller, "Adding Error Bars to Evals" (Anthropic, 2024).
**Notebook:** `../notebooks/week1_confidence_intervals.ipynb`

### Learning Outcomes

- Compute a Wilson score interval for a binomial proportion by hand and in code.
- Compute a percentile bootstrap CI for an accuracy / AUROC metric and explain when it is preferred over an analytical CI.
- Explain why the normal-approximation ("Wald") interval is wrong for proportions near 0 or 1, especially at small `n`.
- State the width of an eval CI as a function of sample size and relate it to the cost of running more eval rollouts.
- Deliver the LAD-style flashcard: "My result is X% [low, high] on n samples; the CI is this wide because..."

### Key Concepts

#### 1. The Wald interval (and why it fails)

For a sample proportion `p_hat = k / n`, the textbook normal-approximation interval is

```
p_hat +/- z_{1 - alpha/2} * sqrt( p_hat * (1 - p_hat) / n )
```

Where `z_{1 - alpha/2}` is the standard normal quantile (1.96 for a 95% CI). This is what you usually see in introductory courses. It has two well-known failure modes:

1. **Coverage collapses near the boundary.** If `p_hat = 0` or `p_hat = 1`, the standard error is zero and the interval has zero width, which is obviously wrong.
2. **Coverage is poor for small `n`.** Even at `p_hat = 0.5`, the actual coverage of a "95%" Wald interval drifts well below 95% until `n` is several hundred.

For eval work this matters: you often have `n` in the hundreds, accuracies above 0.9, and stakeholders who treat the interval as if it were exact.

#### 2. Wilson score interval

Wilson inverts a different test: instead of plugging `p_hat` into the standard error, it asks "for which true `p` would the observed `p_hat` not be rejected?" The result is

```
                p_hat + z^2 / (2n)        z         /  p_hat * (1 - p_hat)     z^2     \
CI_Wilson  =  -------------------  +/-  -------- * sqrt | --------------------- + -------- |
                  1 + z^2 / n           1 + z^2/n  \           n                  4 n^2  /
```

Where `z = z_{1 - alpha/2}`. The key features:

- The center is shifted from `p_hat` toward `0.5` by an amount that shrinks as `n` grows. This is the **Wilson "pseudo-count"** behavior -- it acts like adding `z^2 / 2 ~= 1.92` successes and `1.92` failures.
- The interval is always inside `[0, 1]`, even when `k = 0` or `k = n`.
- Coverage is much closer to the nominal level for small and moderate `n`.

**Intuition.** Wilson is what you get when you treat the binomial proportion as a parameter to be inverted from a score test, rather than as a Gaussian random variable. It is the right default for eval accuracies, attack success rates, and any "k out of n" eval metric.

#### 3. Continuity correction

For very small `n` you can sharpen Wilson with a `+/- 1/(2n)` term inside the square root. It rarely matters for `n > 100` and the notebook treats it as a side note. The point worth remembering: the binomial is discrete, the normal is continuous, and the half-step adjustment is what bridges them.

#### 4. Percentile bootstrap CI

For metrics that are not simple proportions -- AUROC, F1, calibration error, mean reward -- there is no clean closed-form CI. The percentile bootstrap is the workhorse:

```
1. For b = 1, ..., B:
     Resample n examples with replacement from the eval set
     Recompute the metric on the resampled set -> theta_hat_b
2. Sort theta_hat_1, ..., theta_hat_B
3. CI = [ percentile(theta_hat, 100 * alpha/2),
          percentile(theta_hat, 100 * (1 - alpha/2)) ]
```

Typical `B` is 1000-10000.

**Why it works.** The bootstrap treats the empirical distribution as a stand-in for the true population. By resampling, you simulate the variability you would see if you reran the eval many times. For smooth statistics with finite variance and reasonable sample sizes, the percentile bootstrap has approximately the right coverage.

**When it breaks.** Boundary metrics (accuracy at 0 or 100), heavy-tailed reward distributions, or very small `n` (say, `n < 30`). For accuracies near the boundary, the bootstrap distribution is degenerate -- you should fall back to Wilson.

#### 5. CI width as a function of `n`

For a proportion near `0.5`, the half-width of a Wald / Wilson interval scales like

```
half-width  ~  z * sqrt( p * (1 - p) / n )  ~  1 / sqrt(n)
```

To halve the interval you need `4x` the data. This is the single most useful "back of envelope" the week gives you. When an interviewer asks "how many more samples to get a tighter result," you should answer in seconds.

### Project Walkthrough

Open `../notebooks/week1_confidence_intervals.ipynb`.

The notebook builds a synthetic LAD-style ("language-agent detector") eval: `n = 200` examples, true monitor accuracy `p = 0.93`. It then:

1. Draws one fresh eval and computes `p_hat`.
2. Computes the Wald, Wilson, and exact Clopper-Pearson intervals side by side. You should notice that Wald is narrower but its center is at `p_hat`, while Wilson is slightly shifted and asymmetric near the boundary.
3. Runs `B = 5000` percentile bootstrap iterations on the same eval. The bootstrap interval will look qualitatively similar to Wilson for accuracy, but the cell that switches the metric to **AUROC over a synthetic score distribution** shows where bootstrap shines -- there is no clean formula.
4. Repeats the whole experiment 1000 times to measure **actual coverage**. Wilson lands near 95%; Wald drops noticeably below, especially when `p` is set to `0.97` and `n = 100`.
5. Plots CI width versus `n` from 25 to 5000 on a log axis and shows the `1/sqrt(n)` scaling.

What to notice and write down:

- The exact width of the Wilson CI for your synthetic LAD result (this is your flashcard number).
- How many additional samples you would need to halve the interval.
- The case where Wald and Wilson disagree most -- it should be high `p`, low `n`.

### Common Interview Questions

**Q1. How confident are you in that 93% accuracy number?**
On `n = 200` examples, the 95% Wilson interval is roughly `[0.886, 0.958]`. The width is driven by the binomial variance and scales like `1/sqrt(n)`, so to get a `+/- 1%` interval at this accuracy I would need around 2500 examples. I would also want to know whether the eval examples are i.i.d. or whether some are correlated -- if so the effective sample size is smaller and the real interval is wider.

**Q2. Why not just use `mean +/- 1.96 * std / sqrt(n)`?**
That is the Wald interval and it has bad coverage near the boundary. At `p_hat = 0.97`, the Wald interval can extend above 1, and at `p_hat = 1` it has zero width even though we obviously do not know the true accuracy is exactly 1. Wilson is the right default for proportions.

**Q3. When would you use bootstrap instead of Wilson?**
For metrics that are not simple proportions. AUROC, F1, expected calibration error, average reward over a reward model -- any statistic where the sampling distribution is not a clean function of `n` and `p`. Bootstrap also handles complex eval pipelines where the metric is computed from multiple subscores.

**Q4. How does the bootstrap actually "know" the variability if you only have one dataset?**
The empirical distribution of the eval set is itself a (consistent) estimate of the population. Resampling with replacement simulates draws from that estimated population. As `n` grows the empirical distribution converges to the true one, and the bootstrap distribution of the statistic converges to its true sampling distribution.

**Q5. What is the difference between a 95% confidence interval and a 95% credible interval?**
A frequentist 95% CI is an interval-valued procedure: across many hypothetical replications, 95% of the constructed intervals contain the true parameter. A Bayesian 95% credible interval is a statement about the posterior: given the data and the prior, there is a 95% posterior probability that the parameter is in the interval. They often look similar numerically but they answer different questions.

**Q6. You ran the eval twice and got 93% and 91%. Are they different?**
Both numbers sit comfortably inside each other's Wilson intervals at `n = 200`, so no -- the difference is well inside the noise floor. The right comparison is a paired test on the per-example outcomes, which is what week 2 covers.

### Self-Test

1. At `n = 150` and `p_hat = 0.92`, compute the Wilson 95% interval to two decimals.
2. Why does the Wald interval undercover near the boundary even when `n` is large? Sketch the failure mode.
3. You bootstrap an AUROC and the resulting distribution is bimodal. What does that suggest about the eval?
4. An interviewer says "use a t-interval." Why is that the wrong default for an accuracy metric?

### Pitfalls

- **Treating `p_hat` as exact.** A point estimate without an interval is not an answer in an interview.
- **Reporting Wald because it is what `numpy` gave you.** The default in many libraries is normal-approximation; verify and prefer Wilson for proportions.
- **Bootstrapping at the wrong granularity.** If your eval has multiple turns per example, you should resample examples (clusters), not turns. Otherwise you understate variance.
- **Forgetting that the CI assumes i.i.d. examples.** Correlated eval items inflate the effective standard error; the nominal 95% interval will undercover.

---

## Week 2 -- Comparing Two Things Properly

**Source plan reading:** Berg-Kirkpatrick et al., "An Empirical Investigation of Statistical Significance in NLP" (2012).
**Notebook:** `../notebooks/week2_paired_comparison.ipynb`

### Learning Outcomes

- Run a paired bootstrap test for the difference of two metrics on the same eval set.
- Run McNemar's test for two binary classifiers on the same eval set, with and without continuity correction.
- Explain why pairing reduces variance and why an unpaired test on paired data is wasteful (and sometimes wrong).
- Choose between paired bootstrap, McNemar, and a permutation test for a given setting.
- Deliver the monitor-comparison flashcard: "To compare A vs B on the same eval set I use paired bootstrap because..."

### Key Concepts

#### 1. Why pairing matters

Suppose monitor A scores 0.91 and monitor B scores 0.93 on the same 500 examples. The naive comparison is a two-sample test on the proportions, which uses the variance

```
Var_unpaired(p_A_hat - p_B_hat)  =  p_A * (1 - p_A) / n  +  p_B * (1 - p_B) / n
```

But A and B saw the **same examples**. If A is right whenever B is right plus a few extras, the per-example outcomes are correlated, and the actual variance of the difference is

```
Var_paired(p_A_hat - p_B_hat)  =  Var_unpaired  -  2 * Cov(A_correct, B_correct) / n
```

When the covariance is positive (the typical case -- both monitors find the easy examples easy), the paired variance is **smaller**, sometimes much smaller. Pairing is a free power boost.

#### 2. Paired bootstrap

```
1. Build per-example records (x_i, A_i, B_i) where A_i, B_i are the per-example metric values.
2. For b = 1, ..., B:
     Resample example indices i_1, ..., i_n with replacement
     Compute metric_A_b and metric_B_b on the resampled examples
     Record delta_b = metric_A_b - metric_B_b
3. CI for the difference: percentile(delta_b, [2.5, 97.5])
4. p-value (two-sided): 2 * min( frac{delta_b <= 0}, frac{delta_b >= 0} )
```

The crucial detail: **resample examples, not predictions**. By resampling the example index, both A's and B's scores move together, preserving the pairing structure.

For a one-sided test the p-value is `frac{delta_b <= 0}` (or its mirror, depending on direction). Berg-Kirkpatrick et al. discuss this in the NLP context; the recipe is the same for eval metrics.

#### 3. McNemar's test (for binary classifiers)

When both monitors output 0/1 decisions, build the 2x2 contingency table

```
                 B correct       B wrong
A correct          n_11            n_10
A wrong            n_01            n_00
```

Only the **discordant cells** `n_10` and `n_01` carry information about the difference. McNemar's statistic is

```
chi^2  =  (n_10 - n_01)^2 / (n_10 + n_01)        ~  chi^2_1   under H0
```

With continuity correction (recommended for `n_10 + n_01 < 25`):

```
chi^2  =  (|n_10 - n_01| - 1)^2 / (n_10 + n_01)
```

For very small discordant counts use the **exact binomial** form: under `H_0`, `n_10` is `Binomial(n_10 + n_01, 0.5)`, and the p-value is the two-sided binomial tail probability.

**Intuition.** When A and B agree, it tells you nothing about which is better. The signal lives entirely in the cases where they disagree.

#### 4. Permutation test as the gold-standard sibling

A paired permutation test randomly swaps `(A_i, B_i)` labels per example and recomputes the difference. The null distribution is built from all (or many random) sign-swaps. It and the paired bootstrap typically agree to within Monte Carlo noise. Use whichever your team uses.

#### 5. When pairing does **not** apply

- Different eval sets for A and B (then pair what you can or report two separate intervals).
- A and B are evaluated by different judges with non-comparable scales (calibrate first).
- The "paired" examples are actually independent samples -- you do not get the variance reduction.

### Project Walkthrough

Open `../notebooks/week2_paired_comparison.ipynb`.

The notebook builds two synthetic LAD-style monitors on `n = 500` shared examples. Monitor A has accuracy 0.90 and monitor B has accuracy 0.92, and crucially their per-example correctness is **highly correlated** (rho around 0.7) so that pairing helps a lot.

The cells walk through:

1. Computing `p_A_hat`, `p_B_hat`, and the raw difference `0.02`.
2. Running an **unpaired** two-proportion z-test. The p-value is uninspiring (around 0.18). This is the wrong test.
3. Running the **paired bootstrap** with `B = 5000`. The CI on the difference is much tighter and the p-value drops well below 0.05. The cell prints the standard deviation of `delta_b` so you can see, numerically, how much pairing shrinks the variance.
4. Computing the **2x2 contingency table** and running **McNemar's test** (with and without continuity correction). It also runs the exact binomial version on a deliberately small subset to show the difference.
5. A side experiment that **breaks the pairing** by independently shuffling B's predictions. The paired bootstrap and McNemar both lose their power, matching the unpaired test. This is the "see why pairing matters" moment.
6. A final cell that runs a paired permutation test and shows agreement with the paired bootstrap.

What to notice:

- The paired bootstrap CI for the difference is roughly half the width of the unpaired CI on the same data.
- McNemar is essentially a special case of the paired story for binary outcomes; the chi-square statistic, the bootstrap p-value, and the permutation p-value all line up.
- When you destroy the pairing structure, the methods correctly stop disagreeing -- this is a sanity check, not a paradox.

### Common Interview Questions

**Q1. How would you decide if monitor A is better than monitor B?**
On the same eval set I would compute the per-example score for each, then run a paired bootstrap on the difference. That gives me a CI and a p-value that respect the pairing. If both are binary I would also report McNemar as a quick sanity check; the two should agree.

**Q2. Why not just compare the two accuracy numbers with a two-proportion z-test?**
Because they were computed on the same examples. The two estimates are positively correlated, so the variance of their difference is smaller than the unpaired formula assumes. The unpaired test is conservative -- you will throw away real differences.

**Q3. What does McNemar's test actually use?**
Only the discordant pairs -- examples where one classifier is right and the other is wrong. The concordant cells are independent of the difference under the null, so they carry no signal. Under `H_0` of equal accuracy, the discordants split 50/50.

**Q4. When would you use a permutation test instead of a bootstrap?**
A permutation test gives an exact p-value under the null hypothesis of exchangeability and does not assume anything about the metric's distribution. The paired bootstrap is more flexible (it gives you a CI for the effect, not just a p-value) but slightly approximate. In practice both are fine for paired eval comparisons.

**Q5. What if A and B were evaluated on different prompts?**
Then there is nothing to pair. I would compute separate Wilson or bootstrap CIs and either accept the wider unpaired comparison or, better, run both monitors on a shared subset to recover pairing.

**Q6. What is the continuity correction for and when do I need it?**
McNemar's chi-square approximation is poor when the number of discordant pairs is small. The continuity correction (subtracting 1 from `|n_10 - n_01|` before squaring) shrinks the statistic and brings the type-I error closer to nominal. For very small discordant counts, switch to the exact binomial test.

### Self-Test

1. Two monitors agree on 480 examples and disagree on 20, of which A is right on 15 and B on 5. Compute the McNemar chi-square and approximate p-value.
2. Why does the paired bootstrap resample example indices instead of predictions?
3. Construct a case where the two-sample z-test and the paired test give opposite conclusions.
4. Sketch the bias of the unpaired test as a function of the per-example correlation between A and B.

### Pitfalls

- **Running an unpaired test on paired data.** Loses power; the most common mistake.
- **Resampling predictions instead of examples in the bootstrap.** Destroys the pairing and you get the unpaired variance back.
- **Forgetting the continuity correction or exact form for tiny discordant counts.**
- **Assuming `H_0` is "no difference in means" when McNemar's `H_0` is actually "the two discordant cells are equally likely."** They are usually equivalent for binary classifiers but be precise about which null you are stating.

---

## Week 3 -- Power and Sample Size

**Source plan reading:** Wasserman, *All of Statistics*, Ch 10.6 (power); or any solid blog post on power for proportions.
**Notebook:** `../notebooks/week3_power_analysis.ipynb`

### Learning Outcomes

- Compute the sample size needed to detect a given absolute difference in proportions at given alpha and power.
- Distinguish between alpha (type-I error), beta (type-II error), and power (1 - beta), and explain the design tradeoffs.
- Construct a power curve for a fixed effect size as `n` varies, and use it to argue for or against running an eval.
- Translate an interviewer's "I want to detect a 5% difference" into a concrete sample-size answer in under a minute.
- Deliver the SandboxBench flashcard: "To detect a 5% ASR difference with 80% power and 5% alpha I need around N tasks per arm because..."

### Key Concepts

#### 1. The four-way tradeoff

Every power calculation balances four numbers; fix any three and the fourth is determined:

| Symbol | Meaning |
|---|---|
| `alpha` | Type-I error rate (reject H0 when H0 is true). Usually 0.05. |
| `beta` | Type-II error rate (fail to reject H0 when H1 is true). Usually 0.2. |
| `1 - beta` | Power. Usually 0.8 or 0.9. |
| `delta` | Effect size you want to detect. E.g. 5% absolute ASR difference. |
| `n` | Sample size per arm. The unknown you usually solve for. |

#### 2. Power formula for two proportions (the one to memorize)

For a two-sided test of `H_0: p_1 = p_2` versus `H_1: |p_1 - p_2| = delta`, with equal sample sizes per arm:

```
                   (z_{1 - alpha/2} + z_{1 - beta})^2  *  ( p_1 (1 - p_1) + p_2 (1 - p_2) )
n_per_arm   ~=   --------------------------------------------------------------------------
                                            delta^2
```

Where:

- `z_{1 - alpha/2}` is the standard normal quantile for two-sided alpha (1.96 at alpha = 0.05).
- `z_{1 - beta}` is the quantile for the desired power (0.84 at 80% power, 1.28 at 90%).
- `p_1`, `p_2` are the two true proportions you are comparing.
- `delta = p_2 - p_1`.

A handy rule-of-thumb plug-in: at `alpha = 0.05`, `power = 0.8`, `(z_{0.975} + z_{0.8})^2 ~= 7.85`.

So for proportions near 0.5, the sample size collapses to roughly `n ~= 7.85 * 0.5 / delta^2 ~= 4 / delta^2` per arm. To detect a 5% difference you need on the order of 1500 per arm; to detect a 10% difference you need around 400.

**Intuition.** The power formula is just the standard error of the difference scaled to the gap between the null and the alternative distributions. The `(z_alpha + z_beta)` term is the number of standard errors you need to put between them for the test to discriminate.

#### 3. Power for a paired binary comparison

If your eval gives paired binary outcomes (per week 2), the right power formula uses the **discordant rate** `pi_d = P(disagree)`:

```
                   (z_{1 - alpha/2} + z_{1 - beta})^2
n_pairs     ~=   ------------------------------------
                            pi_d * delta_d^2
```

Where `delta_d` is the expected difference between the two discordant probabilities. Pairing reduces required sample size when most pairs agree.

#### 4. Power for a single proportion vs a target

To detect that an eval accuracy exceeds a threshold `p_0` by at least `delta`, with one-sided alpha:

```
            (z_{1 - alpha} * sqrt(p_0 (1 - p_0))  +  z_{1 - beta} * sqrt(p_1 (1 - p_1)))^2
n     =    --------------------------------------------------------------------------------
                                        (p_1 - p_0)^2
```

This is the variant you use when an interviewer says "we want to show the model beats 80% with 90% confidence."

#### 5. Effect size, MDE, and the back-translation

In conversations interviewers usually give you one of three things:

- A **fixed `n`** ("we have 200 SandboxBench tasks") -- you solve for the **minimum detectable effect** (MDE).
- A **fixed `delta`** ("we care about a 5% gap") -- you solve for `n`.
- A **fixed budget in dollars** -- you convert dollars to `n` via cost per rollout, then proceed as above.

Practice all three directions. The hardest is the MDE direction because the algebra is asymmetric.

#### 6. Power curves

A power curve plots power as a function of `n` (or `delta`, or both). Two facts to internalize:

- Power is sigmoidal in `n`. For small `n` it is barely above alpha; for large `n` it saturates near 1. The interesting region is the steep middle.
- Going from 80% to 95% power roughly doubles the required `n`. People often forget how expensive the last 15 percentage points of power are.

### Project Walkthrough

Open `../notebooks/week3_power_analysis.ipynb`.

The notebook centers on a SandboxBench-style scenario: you have two agents, each evaluated on the same `N` agentic tasks, and you want to detect a 5 percentage point absolute difference in attack success rate (ASR). Baseline ASR is 0.30, so we are asking whether the second agent achieves 0.35.

Cells:

1. Implement the power formula above as a function `n_for_proportion_diff(p1, p2, alpha, power)` and verify against `statsmodels`.
2. Plug in `p1 = 0.30`, `p2 = 0.35`, `alpha = 0.05`, `power = 0.8`. The answer comes out near 1300 per arm. Repeat for `delta = 0.10` and the answer drops to around 320 per arm.
3. Run a Monte Carlo simulation: for each candidate `n`, draw 5000 simulated evals from `Binomial(n, p1)` and `Binomial(n, p2)`, run the test, and measure empirical power. The simulated curve should track the analytic formula closely.
4. Build a power curve over `n in [100, 5000]` and mark the 80% and 90% power points.
5. Solve the **MDE direction**: given `n = 200` per arm at `p = 0.30`, what is the smallest difference you can detect at 80% power? The answer is around 13 percentage points -- which is a bracing reality check for small benchmarks.
6. A final cell applies the **paired** version on the same data (using a synthetic correlation structure) and shows that pairing brings the required `n` down by roughly the inverse discordant rate.

What to notice:

- The analytic formula and the Monte Carlo curve agree, which builds trust in the formula.
- The MDE for small benchmarks is huge. This is the punchline you should be able to deliver in interviews.
- Pairing is not just a comparison technique, it is also a sample-size strategy.

### Common Interview Questions

**Q1. How would you design an eval to compare two agents on SandboxBench?**
First I would pin down the minimum effect size that matters scientifically -- say 5% absolute ASR. Then with `alpha = 0.05` and 80% power, the two-proportion formula gives me roughly 1300 tasks per arm at baseline ASR 0.30. If we only have 200 tasks per arm, the minimum detectable effect grows to about 13 points, so I would either reduce the ambition, share tasks across arms (paired design), or run more rollouts per task.

**Q2. What is statistical power, and what does 80% power mean concretely?**
Power is the probability of rejecting the null hypothesis when a specified alternative is actually true. 80% power means that if the true effect is the size we care about, we will catch it 4 times out of 5 across hypothetical replications. The other 20% of the time we will fail to reject and say "no significant difference" even though there is one.

**Q3. How does sample size scale with effect size?**
Required `n` scales like `1 / delta^2` for proportions. Halving the effect quadruples the cost. This is why "detect a 1% difference" is dramatically more expensive than "detect a 5% difference."

**Q4. Is 80% power always the right target?**
No. 80% is a convention. For high-stakes decisions, 90% or 95% power is more appropriate, even though it costs more samples. For exploratory work where you mainly want a confidence interval, the framing of "power" is less useful than the framing of "CI half-width."

**Q5. Your eval has 150 examples. The interviewer asks if that is enough to detect a 5% difference.**
At baseline accuracy near 0.5, the variance per example is 0.25, so the standard error of the difference is roughly `sqrt(2 * 0.25 / 150) ~= 0.058`. A 5% effect is less than one standard error wide. Power is barely above alpha. So the honest answer is no, and I would recommend either a paired design, a larger eval, or a more relaxed effect size before committing.

**Q6. What goes wrong if you pick `n` after looking at the data?**
That is "post hoc power" and it is uninformative -- the achieved power is mechanically tied to the observed p-value. Sample size should be set in advance from the effect size you would care about, not from the observed effect.

### Self-Test

1. Compute the per-arm sample size to detect a 3% difference in ASR at baselines `0.20` and `0.23`, with `alpha = 0.05` and 90% power.
2. Why does power increase with `n`? Sketch the two distributions under `H_0` and `H_1`.
3. You have a fixed budget of 1000 rollouts total. Allocate them between two arms to maximize power. (Hint: if `p_1` and `p_2` are similar, equal allocation is optimal.)
4. Translate "we want a 95% CI half-width of 2 percentage points" into an `n`. How does it relate to a power calculation?

### Pitfalls

- **Doing post hoc power.** Use power for design, use confidence intervals for retrospective uncertainty.
- **Confusing absolute and relative effect sizes.** "5% improvement" can mean 0.05 absolute or 5% of the baseline -- nail it down before computing.
- **Using a one-sided alpha by default.** Two-sided is the standard unless you pre-registered a directional hypothesis.
- **Forgetting that effective `n` is smaller than nominal `n` for clustered or correlated examples.** The power formula assumes i.i.d.

---

## Week 4 -- Multiple Testing

**Source plan reading:** Wasserman Ch 10.7 on Bonferroni and BH-FDR; or the Benjamini-Hochberg (1995) original paper.
**Notebook:** `../notebooks/week4_multiple_testing.ipynb`

### Learning Outcomes

- State and apply the Bonferroni correction.
- State and apply the Benjamini-Hochberg (BH) procedure step by step.
- Distinguish family-wise error rate (FWER) from false discovery rate (FDR) and explain when each matters.
- Run BH on a realistic 12-attack-category leaderboard and decide which results survive.
- Deliver the multiple-testing flashcard: "When I have K comparisons I control FDR via BH because Bonferroni is too conservative when..."

### Key Concepts

#### 1. The multiple testing problem

If you run `m` independent tests at `alpha = 0.05`, the probability that at least one comes back "significant" under the global null is

```
P(any false positive)  =  1 - (1 - alpha)^m
```

For `m = 20` this is already 0.64. Run enough tests and you will always find "something." The fix is to control the right family-level error rate.

#### 2. Family-wise error rate (FWER) -- Bonferroni

FWER is the probability of **any** false positive across the family of tests. Bonferroni controls FWER at level `alpha` by testing each hypothesis at `alpha / m`:

```
Reject H_i  iff  p_i  <=  alpha / m
```

**Why it works.** A union bound: `P(any reject) <= sum_i alpha / m = alpha`.

**When it is right.** When even a single false positive is unacceptable -- safety-critical claims, regulatory submissions, or anything where one bad headline blows up the result.

**Why it is too conservative for eval work.** It pays for the worst case (all hypotheses null and worst-case correlation). For 20 attack categories where most are real and modestly correlated, Bonferroni throws away most of your power.

#### 3. False discovery rate (FDR) -- Benjamini-Hochberg

FDR is the **expected proportion of false positives among rejected hypotheses**. If you reject 10 things and on average 1 of them is a false positive, your FDR is 10%. This is usually what you actually care about in screening / exploratory work.

The BH procedure for controlling FDR at level `q`:

```
1. Sort the m p-values in increasing order: p_(1) <= p_(2) <= ... <= p_(m)
2. For each i, compute the BH threshold t_i = (i / m) * q
3. Find the largest i such that p_(i) <= t_i
4. Reject H_(1), ..., H_(i)        (everything up to and including that one)
```

**Intuition.** You are walking down a sorted list of p-values and asking, "if I rejected the first `i`, would the expected fraction of false positives among them be at most `q`?" The expected fraction under independence is `(i * alpha_i) / i = alpha_i`, and BH sets `alpha_i = i * q / m`, which is increasing in `i` -- so weaker p-values get more lenient thresholds, but only when there are stronger p-values "subsidizing" them.

#### 4. BH guarantees and when they hold

The original Benjamini-Hochberg theorem guarantees `FDR <= q` when the test statistics are **independent** or have **positive regression dependence** (PRDS) -- a condition that holds for many common test setups including one-sided tests on multivariate normal data with positive correlations.

For arbitrary dependence structures, the **Benjamini-Yekutieli** variant uses the threshold `(i / (m * H_m)) * q`, where `H_m = sum_{k=1}^m 1/k` is the harmonic number. This is more conservative but valid for any dependence. For 12 tests, `H_12 ~= 3.1`, so BY is roughly 3x stricter than BH.

#### 5. Bonferroni vs BH at a glance

| Aspect | Bonferroni | BH |
|---|---|---|
| Controls | FWER | FDR |
| Threshold | `alpha / m` (constant) | `(i / m) * q` (step) |
| Conservative? | Very | Less so |
| Independence required? | No | Yes (or PRDS) |
| Use when | Any single false positive is fatal | You want a controlled ratio of false discoveries |
| Typical eval setting | Final go/no-go ship decision | Screening attack categories, hyperparameters, monitors |

#### 6. The dependence trap

BH still controls FDR under positive dependence, but two situations break it:

- **Strong negative correlation** between test statistics. Rare in practice but worth knowing.
- **Block-correlated tests where the blocks are the truth.** If 4 of your 12 attacks are essentially the same category measured 4 different ways, you have effectively 9 independent tests, and BH on 12 is slightly too lax. The fix is to deduplicate or use BY.

### Project Walkthrough

Open `../notebooks/week4_multiple_testing.ipynb`.

The notebook simulates a LAD-style sweep across 12 synthetic attack categories. The ground truth is that 4 of the 12 categories really do show a monitor improvement; the other 8 are nulls. For each category the notebook runs a paired bootstrap (week 2) and produces a p-value.

Cells:

1. Generate the 12 p-values and print them sorted. You should see roughly 4 small ones and 8 scattered above 0.1, with some noise.
2. Apply Bonferroni at `alpha = 0.05`. Typically only 2-3 of the 4 true effects survive, depending on the seed; the conservative threshold kills marginal real effects.
3. Apply BH at `q = 0.05`. Usually all 4 real effects survive and one or two of the nulls slip through, which is exactly the FDR tradeoff.
4. Run 1000 simulated experiments and measure the empirical FWER for Bonferroni and the empirical FDR for BH. Bonferroni's FWER lands at or below 0.05; BH's FDR lands at or below 0.05.
5. Repeat with a **block dependence** setting where 4 of the nulls are highly correlated. BH's FDR creeps up slightly. Switching to Benjamini-Yekutieli pulls it back down at the cost of power.
6. A final visualization plots the sorted p-values against the BH thresholds and shows the "largest i below the line" graphically.

What to notice:

- Bonferroni leaves real effects on the table.
- BH catches them at a small, controlled cost in false positives.
- Dependence matters but is usually not catastrophic.

### Common Interview Questions

**Q1. You tested LAD on 12 attack categories and 5 came back significant at p < 0.05. Are they real?**
At `m = 12` and uncorrected `alpha = 0.05`, the expected number of false positives under the global null is about 0.6, so getting 5 hits is suggestive but I would want to apply a correction before claiming anything. With Bonferroni I would test each at `alpha / 12 ~= 0.004`, which probably leaves 2-3. With BH at `q = 0.05` I would expect 4-5 to survive with an expected false discovery rate of about 5%. I would report both and note the dependence assumption.

**Q2. What is the difference between FWER and FDR?**
FWER is the probability of at least one false positive in the family. FDR is the expected proportion of false positives among the rejected hypotheses. FWER protects against any single embarrassing claim; FDR keeps the ratio of bad claims to good ones bounded. For exploratory work FDR is usually the right target; for headline claims FWER is.

**Q3. Walk me through the BH procedure.**
Sort the `m` p-values in ascending order. For each rank `i`, compute the threshold `(i / m) * q`. Find the largest `i` for which `p_(i) <= (i / m) * q`. Reject every hypothesis from rank 1 up to and including that `i`. The procedure is monotone -- if you reject a weaker p-value, you also reject all stronger ones.

**Q4. When does BH fail?**
BH controls FDR under independence or positive regression dependence. Under arbitrary dependence you should use Benjamini-Yekutieli, which divides the threshold by the harmonic number `H_m`. The BH guarantee can also be brittle if the tests are not honestly run -- p-hacking, optional stopping, or rerunning until significance all break it.

**Q5. What is wrong with Bonferroni?**
Nothing technically -- it always controls FWER. But it pays for the worst-case dependence and the worst-case configuration of nulls. For most eval workloads, where you expect several real effects among your `m` tests, Bonferroni leaves substantial power on the table.

**Q6. If I use BH at `q = 0.1`, what does that mean?**
It means that across the set of hypotheses I reject, the expected fraction that are false positives is at most 10%. So if I reject 20 things, I expect on average no more than 2 to be wrong. It is a guarantee on the ratio, not on the absolute number.

### Self-Test

1. You have 8 sorted p-values: 0.001, 0.005, 0.01, 0.02, 0.04, 0.05, 0.07, 0.20. Apply BH at `q = 0.05` and report which are rejected.
2. Why is BH a step procedure rather than a constant threshold like Bonferroni?
3. Construct a case where Bonferroni rejects nothing but BH rejects half the hypotheses.
4. Your 12 tests are highly correlated because they are 12 noisy measurements of the same underlying effect. Is BH still appropriate? What would you do instead?

### Pitfalls

- **Applying BH after data peeking.** Multiple testing corrections do not fix p-hacking.
- **Treating BH as if it controlled FWER.** It does not; some of your "discoveries" are expected to be false.
- **Forgetting the monotonicity step.** If the largest `i` satisfying the inequality is 7, you reject ranks 1 through 7, including any that themselves fail the inequality at their own rank. People often forget this.
- **Using BH on highly dependent tests without thinking.** Use BY or deduplicate first.

---

## Week 5 -- The Quant Nose Cheat Code

**Source plan reading:** Lopez de Prado, *Advances in Financial Machine Learning*, Ch 11 + Ch 14. Skip the finance-specific material; extract the principle.
**Notebook:** `../notebooks/week5_eval_overfitting.ipynb`

### Learning Outcomes

- Explain the deflated Sharpe ratio (DSR) intuition and its analogy to inflated leaderboard metrics.
- Distinguish out-of-sample skill from selection-bias-driven performance gains.
- Quantify the expected best-of-`K` inflation for a noisy metric and apply the correction.
- Argue, in interview-grade prose, why most eval leaderboards are overfitted in the same way most backtests are.
- Deliver the eval-overfitting flashcard: "Why most eval leaderboards lie -- the Deflated Sharpe analogy in 90 seconds."

### Key Concepts

#### 1. The selection problem

Suppose you evaluate `K` candidate models on a held-out eval set. Each model has true skill `mu_k` and an observed score `mu_hat_k` whose noise has standard deviation `sigma`. You pick the best:

```
k_star = argmax_k mu_hat_k
```

Even if all `K` models are equally skilled (`mu_k = mu` for all `k`), the **observed** score of the winner is biased upward. The expected value of the maximum of `K` i.i.d. standard normals is roughly

```
E[max of K standard normals]  ~  sqrt(2 * log(K))   for large K
```

So the winner's score is inflated by approximately `sigma * sqrt(2 * log(K))`. With `K = 50` and `sigma = 0.02` you get an inflation of roughly `0.056` -- enough to manufacture a fake 5 point lead from pure noise. This is the "expected maximum" problem and it is the engine behind essentially every overfitted leaderboard story.

#### 2. The Deflated Sharpe Ratio (DSR), translated

Lopez de Prado's DSR is the financial-econometrics version of the same problem. The Sharpe ratio of the apparently best strategy over `K` trials is biased upward exactly the way the best eval score is. The deflation correction subtracts an expected-maximum term so that the adjusted statistic has roughly the right distribution under the null of "no real skill."

The eval analogue: when you compare `K` models, monitors, or hyperparameter settings on the same eval, the right thing to report for the winner is

```
score_corrected  ~=  score_observed  -  sigma_eval * E[max-of-K under null]
```

Where `sigma_eval` is the standard error of a single eval score (week 1). For interview purposes, the formula matters less than the mental model: **the more things you tried, the more of the winner's lead is selection bias.**

#### 3. The "Probability of Backtest Overfitting" lens

Lopez de Prado's PBO statistic asks: across many random splits of an eval set, how often does the in-sample winner remain best out of sample? If the answer is near 50%, you have learned essentially nothing -- the winner is interchangeable with random under fresh splits. This is a sharper diagnostic than any single corrected score.

In eval terms: take your full test set, split it in half many times, and ask how often the model that wins on split A also wins on split B. If your top-5 are nearly tied, the cross-split rank correlation will be poor and you should stop publishing the order.

#### 4. The leaderboard-overfitting story (the part you actually rehearse)

Map the analogy out loud:

| Backtesting | Eval leaderboards |
|---|---|
| Many strategies on the same price history | Many models on the same eval set |
| Sharpe ratio | Accuracy / win rate / pass@k |
| Best Sharpe across `K` strategies is biased high | Best score across `K` models is biased high |
| Multiple-testing inflation grows with `K` | Same |
| Out-of-sample Sharpe collapses for the winner | Out-of-eval-distribution score collapses for the winner |
| Deflated Sharpe Ratio | Deflated leaderboard score |
| Probability of Backtest Overfitting | Cross-split rank stability |

The story is that the entire research community is repeatedly running an unaccounted multiple-testing experiment on the same benchmarks, and the resulting "progress" is a mixture of real gains, selection bias, and dataset-specific overfitting. You do not need to overstate this -- the math does the work.

#### 5. Concrete corrections you can mention

- Pre-registration of evals (decide what you will report before looking at scores).
- Held-out splits never used for model selection.
- Cross-split rank-correlation diagnostics (PBO-style).
- Reporting CIs on every leaderboard entry (week 1) and pairwise differences (week 2).
- Deflating the top-`K` table by an expected-max correction.

These are the things you bring up when an interviewer asks "what is wrong with current eval methodology?"

### Project Walkthrough

Open `../notebooks/week5_eval_overfitting.ipynb`.

The notebook simulates a synthetic leaderboard. There are `K = 50` candidate models. Some have a small real edge but most have identical true skill. Each model gets a noisy eval score with known noise variance.

Cells:

1. Draw the `K` true skills (mostly equal, a few slightly higher) and the noisy observed scores. Print the leaderboard sorted by observed score.
2. Show the gap between the top-1 observed score and the true skill of the top-1 model. The gap is large and positive even when the top observed model is not actually the best.
3. Compute the expected max-of-K for the noise distribution and subtract it as a deflation. The corrected ranking is much flatter.
4. Repeat the experiment 1000 times and tabulate how often the observed winner is the true winner. For `K = 50` and small effect sizes the answer is depressingly close to chance.
5. Run a PBO-style split: split the eval set in half, rank the models on each half, and compute the rank correlation. With small real effects the cross-split correlation is near zero.
6. Plot the inflation `E[max of K] - 0` as a function of `K` from 1 to 1000 to show the `sqrt(2 log K)` scaling. The growth is slow but steady -- doubling `K` adds a fixed amount of inflation, not a fixed fraction.
7. A final cell takes a real-looking leaderboard (all entries within 2% of each other) and shows that the "rank" is essentially noise.

What to notice:

- A 2-point lead in a leaderboard with `K = 100` candidates is well within selection-bias territory.
- The cross-split correlation is the most damning diagnostic; it is also the simplest to run.
- Deflation does not change the science, it changes how loudly you should claim it.

### Common Interview Questions

**Q1. What is wrong with current eval methodology?**
Most leaderboards run hundreds of model variants against the same fixed test set and report the top one. That is the same procedure as a backtest sweep, and the same Lopez de Prado / DSR argument applies: the winner's score is biased upward by approximately `sigma * sqrt(2 log K)`, where `sigma` is the eval noise level and `K` is the number of variants tried. A 2 point lead between top entries on a noisy benchmark is well within selection-bias territory and should not be treated as scientific evidence of progress without a held-out split or a deflation correction.

**Q2. What is the Deflated Sharpe Ratio?**
It is a correction to the apparent best Sharpe ratio across `K` strategies that subtracts the expected best-of-`K` under the null of zero skill. The point is that picking the maximum out of many noisy estimates is itself a source of bias, and a Sharpe of 1.5 picked from 1000 strategies is not the same as a Sharpe of 1.5 from a single pre-registered strategy. The eval analogue is exactly the same: best-of-`K` accuracy is biased by `~sigma * sqrt(2 log K)`.

**Q3. How would you detect leaderboard overfitting empirically?**
Split the eval set into halves many times; for each split, rank the models on half A and half B; compute the rank correlation. If the correlation is near zero, the leaderboard is mostly noise. Lopez de Prado calls a cleaner version of this the Probability of Backtest Overfitting. It is the single most useful diagnostic for a "are these rankings real" question.

**Q4. Why does the expected best-of-`K` grow like `sqrt(log K)` and not faster?**
Under Gaussian noise, the right tail decays super-exponentially, so the maximum grows only logarithmically with the number of trials. Practically, doubling `K` adds a constant amount of inflation (about `sigma / sqrt(2 log K)` extra), so the inflation increases without bound but slowly.

**Q5. So should we stop reporting leaderboards?**
No -- we should report them with intervals, with a held-out split that was not used for selection, and with an explicit `K`. The complaint is not that leaderboards exist, it is that they are usually presented as point estimates with an implicit `K` of 1.

**Q6. How does this connect to weeks 1-4?**
Week 1 gives you the per-entry CI. Week 2 gives you the right pairwise comparison between two entries. Week 3 lets you estimate whether the eval is even big enough to detect realistic differences. Week 4 corrects the multiple comparisons across attack categories. Week 5 is the meta-level story that ties them together: leaderboard ranks are themselves a multiple-comparison procedure, and the same logic applies.

### Self-Test

1. You ran 100 monitor configurations on a single eval set and the best one scored 0.94. The eval noise standard deviation is 0.01. What is the rough expected-max inflation you should subtract?
2. Explain the analogy between Sharpe ratio inflation in finance and accuracy inflation on eval leaderboards in your own words, in three sentences.
3. Why is cross-split rank correlation a sharper diagnostic than a corrected absolute score?
4. What is the simplest organizational policy that would prevent leaderboard overfitting? Why is it rarely followed?

### Pitfalls

- **Treating the leaderboard top as the truth.** It is the maximum of many noisy estimates; the bias is real and quantifiable.
- **Confusing in-eval noise with out-of-eval-distribution generalization.** Both bite, in different ways.
- **Reading Lopez de Prado as a finance book.** The mechanism (selection bias from many trials on the same data) is universal.
- **Assuming pre-registration is impractical.** The minimum viable version is "freeze a held-out split before you look at scores."

---

## Week 6 -- Rehearse and Pressure-Test

**Notebook:** none -- this week is verbal practice, not code.

### Learning Outcomes

- Deliver each of the five flashcards out loud in under 90 seconds without notes.
- Field follow-up questions on each story without retreating to "it depends."
- Smoothly chain weeks 1-5 into a single end-to-end "how would you design and report an eval" answer.

### How to Practice

Set a 90-second timer. Pull a flashcard. Answer aloud. Stop when the timer fires even if you are mid-sentence -- that is the data point. Then redo it with the constraint "tighten by 10 seconds." Three reps per card per session is enough.

After the cards, run the source plan's mock questions cold:

1. "Your monitor gets 92% AUROC on 150 samples. Ship it?"
2. "How would you design an eval to compare three monitors across five attack categories?"
3. "Your leaderboard shows Model A beats Model B by 2%. Is that real?"
4. "What sample size do you need to detect a 10% change in ASR?"
5. "Walk me through how you'd evaluate a new agentic safety framework."

Each one should chain at least two of the five weeks. Q1 is week 1 + maybe week 2. Q2 is weeks 2-4. Q3 is weeks 1, 2, and 5. Q4 is week 3. Q5 is all of them.

Record one or two attempts on your phone. Listen back at 1.25x speed. The places you cringe are the places to rehearse next.

### Pitfalls

- **Memorizing prose instead of math anchors.** If you can write the formula you can always reconstruct the story; if you only have the story you will freeze under follow-up.
- **Hedging.** "It depends" is fine as a clarifying question, fatal as an answer.
- **Skipping the recording step.** It is the cheapest, most painful, most useful practice.

---

## The 5 Flashcards

The 90-second pitch for each story. These are the things you should be able to deliver cold.

### Flashcard 1 -- Confidence Intervals That Matter

"My LAD result is 93% on 200 examples, which gives a 95% Wilson interval of roughly 88.6% to 95.8%. I use Wilson rather than the normal-approximation interval because Wald collapses near the boundary and undercovers at small `n`; Wilson stays inside [0,1] and has near-nominal coverage. For non-binomial metrics like AUROC I would use a percentile bootstrap with a few thousand resamples. The half-width scales like `1/sqrt(n)`, so to halve the interval I need 4x the data, which is the number I anchor any 'do we need more samples' conversation on."

### Flashcard 2 -- Comparing Two Things Properly

"To compare monitor A and monitor B on the same eval set I run a paired bootstrap on the per-example difference, resampling example indices so the pairing is preserved. Pairing reduces the variance of the difference whenever the two monitors are positively correlated -- which they almost always are, because they find the same easy examples easy. For binary classifiers I cross-check with McNemar, which uses only the discordant pairs and has an exact binomial form for small counts. The unpaired two-proportion z-test is the wrong default here -- it ignores the pairing structure and throws away power."

### Flashcard 3 -- Power and Sample Size

"To detect a 5 percentage point ASR difference between two agents at 80% power and `alpha = 0.05`, with baseline ASR around 30%, the two-proportion power formula gives me roughly 1300 tasks per arm -- the formula is `(z_{1-alpha/2} + z_{1-beta})^2 * (p1 q1 + p2 q2) / delta^2`. The required sample size scales like `1/delta^2`, so halving the effect quadruples the cost. If I only have 200 tasks per arm, the minimum detectable effect is around 13 percentage points, which is the kind of reality check that should be the first thing said in any eval design conversation."

### Flashcard 4 -- Multiple Testing

"When I have `K` comparisons -- say 12 attack categories -- I control FDR with Benjamini-Hochberg rather than FWER with Bonferroni, because Bonferroni pays for the worst-case dependence and configuration and leaves real effects on the table. BH sorts the p-values, finds the largest rank `i` with `p_(i) <= (i/m) * q`, and rejects ranks 1 through `i`. It controls FDR under independence or positive dependence; for arbitrary dependence I would use Benjamini-Yekutieli. I would still use Bonferroni when a single false positive is unacceptable -- safety-critical claims, regulatory submissions, public headlines."

### Flashcard 5 -- Eval Overfitting

"Most eval leaderboards are overfitted in the same way most backtests are. Picking the top of `K` candidates inflates the winner's score by approximately `sigma * sqrt(2 log K)`, where `sigma` is the per-entry standard error -- the same selection bias that motivates Lopez de Prado's Deflated Sharpe Ratio. A 2 point lead between top entries on a noisy benchmark is well within selection-bias territory. The cleanest empirical diagnostic is cross-split rank correlation -- split the eval set, rank on each half, and look at the agreement; if it is near zero, the ordering is mostly noise. The fix is held-out splits, pre-registration, intervals on every entry, and explicit reporting of `K`."

---

## If You Find Extra Time

In the priority order from the source plan:

- **Vershynin Ch 1-2 (high-dimensional probability).** Builds the intuitions you need for probe work and concentration arguments -- "why averaging in high dimensions concentrates so fast" is a recurring theme in monitor / probe research.
- **Wasserman Ch 6-7 (CDF estimation, plug-in principle).** Deepens the bootstrap story by grounding it in the empirical CDF; makes the week 1 and 2 material feel inevitable rather than ad hoc.
- **The METR time-horizon paper.** A clean modern example of careful eval design with explicit attention to sample size and confidence intervals; reading it sharpens your "what does a well-built eval look like" reference point.

---

## Cross-Reference

- For the foundational probability and CI material this guide assumes, see `statistics.md` (especially sections 8, 12, 13).
- For the broader interview prep arc, see `all-in-one-guide.md` and `why-this-matters.md`.
- The companion runnable notebooks live in `../notebooks/week1_*.ipynb` through `week5_*.ipynb` and have Colab badges for one-click running.
