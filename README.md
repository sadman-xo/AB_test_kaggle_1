# A/B Test Significance Analyzer

Statistical significance testing for A/B experiments using both frequentist and Bayesian methods, applied to the Kaggle Marketing A/B Testing dataset (588,101 users).

---

## Running the Notebook

Open `A_B_test_analyzer.ipynb` in Google Colab. The notebook uses `kagglehub` to pull the dataset directly — no manual downloads needed.

```python
import kagglehub
path = kagglehub.dataset_download("faviovaz/marketing-ab-testing")
```

---

## Dataset

**Kaggle — Marketing A/B Testing**
https://www.kaggle.com/datasets/faviovaz/marketing-ab-testing

Users were randomly assigned to see either an advertisement or a public service announcement (PSA). The goal was to measure whether ad exposure caused a statistically significant increase in purchase conversions.

| Column | Description |
|---|---|
| `user_id` | Unique user identifier |
| `test_group` | `ad` (treatment) or `psa` (control) |
| `converted` | Whether the user made a purchase |
| `total_ads` | Number of ads the user was shown |
| `most_ads_day` | Day of the week with highest ad exposure |
| `most_ads_hour` | Hour of the day with highest ad exposure |

**Experiment summary:**

| Group | Users | Conversions | CVR |
|---|---|---|---|
| Control (PSA) | 23,524 | 420 | 1.785% |
| Treatment (Ad) | 564,577 | 14,423 | 2.555% |
| | | Observed Lift | **+43.09%** |

---

## Results

| Method | Metric | Value |
|---|---|---|
| Z-Test | Z-Statistic | 7.3701 |
| Z-Test | p-value | 1.71 × 10⁻¹³ |
| Chi-Square | χ² Statistic | 54.3181 |
| Chi-Square | p-value | ≈ 0 |
| Wilson CI | Control (95%) | (1.62%, 1.96%) |
| Wilson CI | Treatment (95%) | (2.51%, 2.60%) |
| Bayesian | P(Ad beats PSA) | 100.00% |
| Bayesian | Expected Lift | +43.11% |
| Bayesian | 95% Credible Interval | (29.97%, 57.60%) |

**Conclusion:** The ad caused a statistically significant lift in conversion rate. Both methods agree completely. The confidence intervals don't overlap, p ≈ 0, and not a single one of 200,000 Monte Carlo simulations had the PSA outperforming the ad.

---

## Theory

### Why any of this is necessary

The PSA group converted at 1.785% and the ad group at 2.555%. That's a +43% difference. But you can't just look at those two numbers and call it done — you need to rule out the possibility that the difference happened by random chance. If you flipped a coin 588,000 times and split the results into two groups, you'd still see slightly different counts. Statistical testing quantifies exactly how likely it is that the gap you observed is noise.

---

### The Null Hypothesis

Every test starts here. The null hypothesis (H₀) is the default — it says the treatment had no effect and any observed difference is random variation. The alternative (H₁) says the difference is real.

```
H₀: p_PSA = p_Ad     (the ad does nothing)
H₁: p_PSA ≠ p_Ad     (the ad changes conversion rate)
```

You never prove H₁ directly. You just ask: if H₀ were true, how improbable would our data be? If the answer is extremely improbable, you reject H₀.

---

### Frequentist Approach

#### The Z-Test

Each user is a Bernoulli trial — they either convert (1) or don't (0). We're comparing two proportions from two independent groups. The two-proportion z-test is the right tool.

**Step 1 — Pooled proportion**

Under H₀, both groups share the same true conversion rate. We estimate it by pooling:

```
p_pool = (conv_A + conv_B) / (n_A + n_B)
       = (420 + 14423) / (23524 + 564577)
       = 0.02494
```

**Step 2 — Standard error**

How much would the difference in proportions vary between experiments due to sampling randomness alone:

```
SE = sqrt( p_pool × (1 - p_pool) × (1/n_A + 1/n_B) )
```

**Step 3 — Z-statistic**

```
z = (p_B - p_A) / SE
  = (0.02555 - 0.01785) / SE
  = 7.37
```

The z-statistic tells you how many standard errors the observed difference sits away from zero. At α = 0.05, the critical value is ±1.96. Our z = 7.37 is nearly four times past that threshold.

**Step 4 — p-value (two-tailed)**

```
p = 2 × (1 - Φ(|z|))
  = 2 × (1 - Φ(7.37))
  = 1.71 × 10⁻¹³
```

Φ is the standard normal CDF. Under H₀, observing a difference this large has a probability of 0.000000000000171. We reject H₀.

**Why two-tailed:** We test whether the ad is different in either direction, not just better. This is the conservative and correct approach unless you have a strong directional hypothesis established before the experiment starts.

**Why the z-test is valid here:** The Central Limit Theorem says that for large enough samples, the sampling distribution of a proportion approximates a normal distribution. With 23,524 and 564,577 users, this approximation is essentially exact. The rule of thumb is n×p ≥ 5 — we have 420 and 14,423 conversions respectively.

---

#### The Chi-Square Test

The chi-square test approaches the same problem from a different angle. It builds a contingency table and tests whether the rows and columns are independent:

```
              Converted    Not Converted
PSA               420          23,104
Ad             14,423         550,154
```

If the ad had no effect, conversions should distribute proportionally across both groups. The statistic measures how far the observed counts deviate from expectation:

```
χ² = Σ (Observed - Expected)² / Expected  =  54.32
```

For a 2×2 table, χ² = z². Check: 7.37² = 54.3 ✅

Running both tests independently is good practice. They confirm each other without being redundant because they come from different mathematical frameworks.

---

#### Wilson Confidence Intervals

A confidence interval gives a range of plausible values for the true conversion rate. The standard Wald interval is:

```
p ± z × sqrt(p(1-p)/n)
```

This breaks down when p is close to 0 or 1, or when n is small. It can produce impossible intervals that go below 0 or above 1. The Wilson interval corrects for this and has better coverage properties:

```
center = (p + z²/2n) / (1 + z²/n)
margin = z × sqrt(p(1-p)/n + z²/4n²) / (1 + z²/n)
```

Our results:
```
PSA: (1.62%, 1.96%)
Ad:  (2.51%, 2.60%)
```

The two intervals don't overlap. The PSA's upper bound (1.96%) is below the Ad's lower bound (2.51%). You can see the significance without computing a single p-value.

---

#### What a p-value actually is

**What it is:** The probability of observing data as extreme as yours, or more extreme, assuming H₀ is true.

**What it is not:**
- Not the probability that H₀ is true
- Not the probability that your result was a fluke
- Not a measure of effect size or practical importance

A p-value of 0.001 on a 0.001% lift is statistically significant but commercially useless. A p-value of 0.06 on a 50% lift might be worth investigating further. Always look at effect size alongside p-values.

---

#### Statistical Power and Sample Size

Before running an experiment, you need to decide how many users to collect. This depends on three things:

- **α (significance level):** The false positive rate you'll tolerate. We used 0.05.
- **Power (1-β):** The probability of detecting a real effect when one exists. We used 0.80 — an 80% chance of finding the effect if it's real. The flip side is a 20% chance of a false negative (Type II Error).
- **MDE (Minimum Detectable Effect):** The smallest lift worth detecting.

```
n = ( z_α/2 × sqrt(2p̄(1-p̄)) + z_β × sqrt(pA(1-pA) + pB(1-pB)) )²
    ─────────────────────────────────────────────────────────────────
                            (pA - pB)²
```

From our sample size calculator:

| MDE | Required n per variant |
|---|---|
| 1% | 8,677,650 |
| 5% | 353,884 |
| 10% | 90,587 |
| 20% | 23,703 |
| 43% (observed) | ~11,000 |

The baseline CVR of 1.785% is small, which is why detecting tiny effects requires enormous samples. Because the actual lift was 43%, our 23,524-user control group was more than sufficient — the signal was loud enough to hear clearly.

---

### Bayesian Approach

#### A different question

Frequentist testing answers: given the data, how improbable is H₀?

Bayesian testing answers: given the data, what is the probability that the ad actually works?

The second question is what most people actually want answered. Bayesian methods let you make direct probability statements about your hypothesis, which frequentist methods technically don't allow.

---

#### Bayes' Theorem

```
P(θ | data) = P(data | θ) × P(θ) / P(data)
```

- **Prior P(θ):** Your belief about the conversion rate before seeing any data
- **Likelihood P(data|θ):** How probable is the observed data for a given conversion rate
- **Posterior P(θ|data):** Your updated belief after seeing the data

The posterior is what you care about.

---

#### The Beta Distribution

Conversion rates are probabilities — numbers between 0 and 1. The natural distribution for modeling probabilities is the Beta distribution:

```
p ~ Beta(α, β)

Mean = α / (α + β)
```

- α controls how much weight sits toward 1 (successes)
- β controls how much weight sits toward 0 (failures)
- Beta(1, 1) is flat — the uniform distribution — meaning no prior knowledge

---

#### Beta-Binomial Conjugacy

When your likelihood is Binomial (each user either converts or doesn't) and your prior is Beta, the posterior is also Beta. This is called conjugacy.

```
Prior:     Beta(1, 1)
Data:      k conversions out of n trials
Posterior: Beta(1 + k, 1 + n - k)
```

For our data:
```
PSA posterior: Beta(1 + 420,   1 + 23104)  = Beta(421, 23105)
Ad posterior:  Beta(1 + 14423, 1 + 550154) = Beta(14424, 550155)
```

Posterior mean for PSA: 421 / (421 + 23105) = 1.789% — nearly identical to the observed 1.785%. With this much data, the prior is completely irrelevant. The posterior is dominated by the likelihood.

The PSA posterior is wider (more uncertainty) because it has far fewer users. The Ad posterior is extremely narrow because 564k users leave almost no room for ambiguity. This is exactly what the posterior plots show.

---

#### Monte Carlo Simulation

To compute P(Ad > PSA) we sample from both posteriors:

```python
samples_psa = np.random.beta(421,   23105,  200_000)
samples_ad  = np.random.beta(14424, 550155, 200_000)

P(Ad > PSA) = mean(samples_ad > samples_psa)
            = 200,000 / 200,000
            = 100.00%
```

Every single one of 200,000 simulations had the ad outperforming the PSA.

---

#### Credible Interval vs Confidence Interval

**Frequentist 95% CI:** "If we repeated this experiment infinitely many times, 95% of the resulting intervals would contain the true parameter." It makes a statement about the procedure, not about this specific interval.

**Bayesian 95% Credible Interval:** "There is a 95% probability that the true value lies in this range." This is the direct probabilistic statement.

Our 95% credible interval on lift: **(29.97%, 57.60%)**

Even in the worst plausible scenario, the ad still drives a 30% lift.

---

#### Why both methods agree

With 588,000 users and a +43% lift, the evidence is overwhelming. Both approaches are answering the same underlying question with different mathematical machinery. With data this strong, they converge. In closer experiments with less data, the two can diverge — which is exactly why running both is worth doing.

---

## What Each Part of the Code Does

### Data Loading
Downloaded via `kagglehub`. Column names cleaned with `.str.strip().str.lower().str.replace(" ", "_")` to remove spaces that would break attribute access.

### EDA
Four charts: conversion rate by group, by day of week, by hour of day, and distribution of total ads seen (clipped at 100 to handle outliers). Key findings: Monday and Tuesday peak, hour 14-16 peak, ad group is 24x larger than PSA group.

### Aggregation
`groupby("test_group")["converted"].agg(["sum", "count"])` collapses 588k rows into four numbers. Everything downstream uses n_A, conv_A, n_B, conv_B.

### Z-Test
Implemented from scratch. Pooled proportion → standard error → z-statistic → two-tailed p-value via `stats.norm.cdf`. No black box — every step is explicit.

### Chi-Square
`stats.chi2_contingency` on a 2×2 array. Sanity check: χ² = z² for 2×2 tables.

### Wilson CI
Manual implementation using the closed-form formula. Used instead of `proportion_confint` to make the math transparent.

### Bayesian
Beta posteriors computed analytically. `np.random.beta` draws 200,000 samples from each posterior. P(B>A) = `np.mean(samples_b > samples_a)`. Lift distribution = `(samples_b - samples_a) / samples_a * 100`. Credible interval via `np.percentile(lift_dist, [2.5, 97.5])`.

### Sample Size Calculator
Power analysis formula implemented from scratch. Sweeps MDE from 1% to 30% using the observed PSA rate as the baseline.

---

## Project Structure

```
ab_analyzer/
└── A_B_test_analyzer.ipynb   # Full analysis notebook (Google Colab)
```

## Tech Stack

| Library | Purpose |
|---|---|
| SciPy | Z-test, chi-square, normal distribution |
| NumPy | Monte Carlo sampling, array operations |
| Pandas | Data loading and aggregation |
| Matplotlib | Plots and visualizations |
| kagglehub | Dataset download |
