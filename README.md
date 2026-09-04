# Week 3 — Statistical Analysis and Hypothesis Testing

**Question:** Is the average duration of Netflix movies greater than 90 minutes?

H0: μ = 90 minutes  
H1: μ > 90 minutes  
α = 0.05

Dataset: Netflix Movies and TV Shows, standard 8,807-title version.  
Source: https://www.kaggle.com/datasets/shivamb/netflix-shows

## Results
- n = 6,126
- Mean = 99.58 minutes
- Median = 98 minutes
- SD = 28.29 minutes
- t = 26.50
- One-sided p ≈ 7.480e-147
- 95% CI = [98.87, 100.29]
- Decision: Reject H0

A Wilcoxon signed-rank test is included as a robustness check because movie durations are not perfectly normally distributed.
