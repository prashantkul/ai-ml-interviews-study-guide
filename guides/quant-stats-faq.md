# Quant Stats Skill-Building -- FAQ

Frequently asked questions about the [6-week Quant Stats Skill-Building module](quant-stats-skill-building.md) and its companion notebooks in `../notebooks/`. Organized from "should I even start this" through "how do I use this in an interview."

```mermaid
graph LR
    A["Getting Started"]:::blue --> B["Statistical Concepts"]:::green
    B --> C["Applying to My Work"]:::orange
    C --> D["Interview Preparation"]:::purple

    classDef blue fill:#4A90D9,stroke:#2E6DB4,color:#fff
    classDef green fill:#50C878,stroke:#3DA35D,color:#fff
    classDef orange fill:#FF8C42,stroke:#E07030,color:#fff
    classDef purple fill:#9B59B6,stroke:#7D3C98,color:#fff
```

---

## Table of Contents

- [Getting Started](#getting-started)
- [Statistical Concepts](#statistical-concepts)
- [Applying to My Work](#applying-to-my-work)
- [Interview Preparation](#interview-preparation)
- [Notebooks and Tooling](#notebooks-and-tooling)

---

## Getting Started

### Who is this series for?

ML engineers and safety researchers preparing for interviews at labs like Anthropic, DeepMind, OpenAI, Far.AI, METR, and similar, where "can you reason about eval uncertainty like a quant" is a real screening signal. It also works as a self-paced module for anyone shipping evals in production and noticing that their uncertainty story is weaker than their modeling story.

### Do I need to know statistics before starting?

You need the level of statistics in the first half of [`statistics.md`](statistics.md): what a random variable is, expectation and variance, the central limit theorem, what a p-value means. If any of those would catch you off guard, spend two hours on that guide first. You do *not* need measure theory, Bayesian hierarchical models, or graduate-level statistics.

### Is this a replacement for `statistics.md`?

No -- the two guides are complementary. `statistics.md` is the encyclopedia (definitions, distributions, theorems). The skill-building guide is the applied module: a narrow slice of statistics deeply internalized through worked examples. If you only have time for one, the skill-building guide will make you look sharper in an interview; `statistics.md` will make you correct on a broader range of questions.

### How much time does this really take?

Two to three hours per week, for six weeks. About 15 hours total. That budget is the whole design constraint -- the module deliberately cuts most of a traditional statistics textbook so the remaining material can be properly internalized. If you try to grind a textbook at this budget you will finish nothing; if you follow this module you will finish six rehearsed interview stories.

### I have two weeks, not six. What do I cut?

In order of value-per-hour for an interview: keep Week 1 (confidence intervals), Week 2 (paired comparison), and Week 5 (eval overfitting). Skip Week 3 (power), Week 4 (multiple testing), and do Week 6's rehearsal on the three flashcards you have. Rationale: CIs and paired comparison are the questions you are almost certain to be asked; the eval-overfitting story is the single piece of content most likely to make you sound senior.

### Do I need to read the full source references (Wasserman, Vershynin, Lopez de Prado)?

No. The guide extracts the pieces you actually need and cites chapter numbers if you want to go deeper. The reason those books are cited is so you can defend yourself in a follow-up ("where did you learn this?"), not because you need to have read them cover to cover.

---

## Statistical Concepts

### Why Wilson over Wald? Isn't the normal approximation "close enough"?

The Wald interval uses `p_hat +/- z * sqrt(p_hat * (1 - p_hat) / n)`. When `p_hat` is near 0 or 1 -- exactly where most safety-relevant eval numbers live -- the standard error estimate itself is tiny, so Wald produces intervals that are too narrow and can extend past 1 (or below 0). Wilson solves this by inverting the test rather than plugging in `p_hat`, so it stays honest as `p_hat` approaches the boundary. It is the default for a reason; when interviewers grade you, they are looking for whether you reach for Wilson automatically or whether they have to prompt you.

### Bootstrap vs Wilson -- which should I use?

Use Wilson when the metric is a simple proportion (accuracy, success rate, ASR). Use bootstrap when the metric is more complex (F1, AUC, BLEU, calibration error, anything that isn't a mean of Bernoullis). The two should agree closely when both apply; if they don't, you either have a small sample, a skewed distribution, or a bug. Running both as a sanity check is a good habit.

### Why is the paired bootstrap so much more powerful than the unpaired z-test?

Because items have shared difficulty. If item 42 is a hard prompt, both monitors will tend to get it wrong; if item 7 is an easy one, both will tend to get it right. The unpaired test treats that shared variance as noise and includes it in the denominator. The paired test conditions on each item and only looks at the difference -- item-level difficulty cancels out. Week 2's notebook shows this empirically: a 40-point power gap at the same sample size on the same synthetic data.

### When should I use McNemar instead of paired bootstrap?

McNemar is exact (given a binary metric and the right small-sample variant) and takes one line of code with only the counts of discordant pairs. Paired bootstrap is general (works for any metric) and gives you a confidence interval for the effect size, not just a p-value. Default to paired bootstrap unless your metric is strictly binary and you only need a p-value; then McNemar is lighter.

### What's the difference between FWER and FDR, in one sentence each?

**Family-wise error rate (FWER)** is the probability of making *any* false positive across the whole family of tests. **False discovery rate (FDR)** is the expected *fraction* of your reported positives that are false. FWER is stricter and usually gives you less power; FDR is more lenient and more powerful, at the cost of accepting that some of your flagged findings will be wrong.

### When do I need Bonferroni and when do I need BH?

Bonferroni when the cost of *any* false positive is catastrophic and you would rather miss real effects than flag a false one (safety-critical triggers, confirmatory clinical decisions). BH when you are triaging many candidate findings and want to catch as many true ones as possible while bounding the junk rate -- which is almost always the right model for ML evals.

### Is BH-FDR valid if my tests are correlated?

BH-FDR is exact under independence and under positive regression dependence (PRDS), which covers most naturally positively correlated tests -- e.g. one-sided comparisons on a multivariate normal with positive correlations. For arbitrary dependence you should use Benjamini-Yekutieli instead, which divides the threshold by the harmonic number `H_m = sum 1/k`. For twelve tests BY is roughly 3x stricter than BH. Week 4's notebook contains the simulation that shows when this matters.

### Why does the Week 5 deflation formula have `sqrt(2 * log K)` in it?

For `K` independent standard-normal draws, the expected maximum grows like `sqrt(2 * log K)` -- this is the classic extreme-value bound. When you run `K` candidate evaluations with noise standard deviation `sigma`, the best observed score is inflated by roughly `sigma * sqrt(2 * log K)` on average, purely from selection. Deflation simply subtracts that expected-max term so the adjusted "best" statistic has roughly the right distribution under the null of "no real skill."

### Isn't the leaderboard-overfitting story just "p-hacking with extra steps"?

Yes, and saying that out loud is good. The difference is that p-hacking is usually framed as a researcher-integrity issue ("someone cheated"), whereas leaderboard overfitting is a structural issue that happens automatically whenever many people evaluate on the same public test set. Nobody has to cheat; the selection pressure does the damage. Lopez de Prado's framing matters because it gives you a clean mathematical way to quantify how much inflation you should expect, rather than just asserting the problem qualitatively.

### Is the Deflated Sharpe formula an upper bound or a real correction?

Treat it as a lower bound on how much you should deflate, not a final correction. The derivation assumes `K` *independent* candidates, which is optimistic -- in reality candidates share architectures and training data, and selection happens iteratively over many months. Real deflation should typically be larger than what the formula gives you. The formula's value is that it is a hard floor: if the inflation exceeds what a simple independent-draws model predicts, the situation is already bad before you even account for real-world dependence.

---

## Applying to My Work

### My eval set has 150 items. Is any of this meaningful?

Short answer: the analysis is still meaningful, the conclusions will just be wider than you want. Week 3's notebook shows that at `n = 150` per arm the minimum detectable effect (MDE) at 80% power is around 13.6 percentage points for a two-proportion comparison near `p = 0.30`. That means your 150-item eval cannot reliably distinguish a 5-point ASR difference; knowing *that* is already useful because it tells you to either (a) report very wide confidence intervals, (b) collect more data, or (c) pivot to paired analysis on the same items if you are comparing two systems.

### My metric is F1, not accuracy. Does any of this still apply?

Yes, with two adjustments. (1) Replace Wilson with bootstrap everywhere -- F1 is not a plain proportion, so Wilson doesn't apply. (2) In Week 2, switch from McNemar (binary) to paired bootstrap on per-item F1 contributions. Everything else (power analysis, multiple testing, leaderboard deflation) transfers unchanged.

### How do I apply this to a held-out benchmark I don't control (e.g., MMLU)?

You can still report CIs on your own scores. You cannot control for leaderboard-level selection bias directly -- but you *can* point out that the benchmark has been seen by `K` models and argue that the top score should be discounted by a DSR-style term. This is exactly the "what is wrong with current eval methodology" interview answer from Week 5.

### Can I reuse these notebooks with my own data?

Yes. Each notebook generates synthetic data in a clearly marked cell. Replace that cell with a cell that loads your own accuracy vector / pair of accuracy vectors / list of per-category p-values and the rest of the notebook runs unchanged. The analytic and bootstrap routines are implemented from scratch specifically so you can copy them into your own codebase without a statsmodels dependency.

### My paper deadline is next month. Should I prioritize this series or the paper?

Do the paper. Then, while the paper is in review, run your own paper's results through the Week 1, Week 2, and Week 4 notebooks and add a paragraph on uncertainty to the camera-ready. That single paragraph, anchored on Wilson / paired bootstrap / BH-FDR with real numbers from your own work, is worth more in an interview than finishing the full 6-week module without applying it.

---

## Interview Preparation

### What kinds of interview questions does this prepare me for?

The guide is built around five specific question patterns that come up in quant-leaning ML interviews:

1. "How confident are you in that number?" (Week 1)
2. "How would you tell if A is better than B?" (Week 2)
3. "How would you design an eval for X?" / "Do you have enough samples?" (Week 3)
4. "You tested K things and some are significant -- are they real?" (Week 4)
5. "What is wrong with current eval methodology?" (Week 5)

If you can handle those five cold, you will clear the statistics bar at almost every lab.

### What if I blank on a formula during an interview?

Reach for the intuition first, then the formula. For example: "The Wilson interval is the set of `p` values for which a normal-approximation z-test at `alpha = 0.05` would fail to reject `H0: true proportion = p`. I can rederive the endpoints by solving the quadratic, but the important thing is it stays inside `[0, 1]` even as `p_hat` approaches the boundary, which Wald doesn't." Interviewers are rarely grading the algebra -- they are grading whether you can talk about what the tool *does*. The flashcards in the guide are deliberately written to the level of detail you can actually remember under pressure.

### What's the best way to practice for the pressure of an interview?

Record yourself answering the Week 6 mock questions on video, then watch the playback. You will catch hedging words ("I think," "maybe"), missing specifics (percentages without sample sizes), and moments where you jump to the conclusion without laying down the argument. Doing this once is worth five passive re-reads of the guide.

### I'm more of an engineer than a statistician. Will interviewers see through me?

Probably not, if you do the hands-on work in the notebooks. Most interview candidates cannot deliver a clean, numbers-backed paired-bootstrap answer; the ones who can are usually mistaken for statisticians even if their formal training is engineering. The point of the notebooks is that you are *actually computing* these quantities on real-feeling (if synthetic) data, which is enough to answer any reasonable follow-up.

### If a question goes beyond the six weeks, how do I handle it?

Say what you know, say what you don't, and say how you would find out. For example: "I haven't worked with Bayesian credible intervals for evals before. Conceptually I know they replace the frequentist coverage guarantee with a posterior probability statement, and if you gave me a day I would start from a Beta-Binomial model with a weakly informative prior. But for the question you are asking me right now, I would default to Wilson because it is what I have tested under pressure." That answer is much stronger than bluffing.

---

## Notebooks and Tooling

### Do I need a GPU / a cluster / a special environment?

No. Every notebook runs in under a minute on a laptop CPU using only `numpy`, `scipy`, and `matplotlib`. The synthetic-data design was chosen specifically so there are no external downloads and no framework dependencies.

### Colab or local?

Either works. The Colab badges at the top of each notebook are the fastest path if you don't want to set up a local Python environment. Local is better if you are going to adapt the notebooks to your own data, because you can edit the data-loading cell and keep iterating. Colab is better if you just want to click through the worked examples once.

### Why are the methods implemented from scratch instead of using `statsmodels`?

Two reasons. First, implementing the formula yourself is the single best way to internalize it -- you can't fake understanding of a 15-line function you wrote five minutes ago. Second, interviewers often ask "what is this function actually doing under the hood?"; having written it once means you can answer without hand-waving. The goal of the notebooks is learning, not running a production pipeline.

### What if I find a bug in a notebook?

Please open an issue or PR on the repo. Each notebook is seeded and the expected numbers are reported in the guide's project-walkthrough section, so any deviation is a real bug and worth tracking down.

---

## Cross-Reference

- [Quant Stats Skill-Building](quant-stats-skill-building.md) -- the main 6-week guide this FAQ supports.
- [Probability & Statistics](statistics.md) -- foundational reference; start here if any of the questions above felt over your head.
- [Why This Matters](why-this-matters.md) -- framing for why these skills show up in real ML pipelines.
- Notebooks: `../notebooks/week1_confidence_intervals.ipynb` ... `week5_eval_overfitting.ipynb`, each with a Colab badge at the top.
