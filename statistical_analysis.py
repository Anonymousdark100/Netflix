"""
Week 3 — Statistical Analysis and Hypothesis Testing in Python
Research question: Is the average duration of Netflix movies greater than 90 minutes?
"""

from pathlib import Path
import math
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from scipy import stats

DATA = Path("data/netflix_titles.csv")
OUT = Path("visualizations")
OUT.mkdir(exist_ok=True)

df = pd.read_csv(DATA).replace(r"^\s*$", np.nan, regex=True)

# Correct the three known duration values accidentally stored in rating.
bad = df["rating"].astype("string").str.match(r"^\d+\smin$", na=False)
df.loc[bad & df["duration"].isna(), "duration"] = df.loc[
    bad & df["duration"].isna(), "rating"
]
df.loc[bad, "rating"] = np.nan

# Movies have duration in minutes, so they form the statistical sample.
movies = df[df["type"] == "Movie"].copy()
movies["duration_minutes"] = pd.to_numeric(
    movies["duration"].str.extract(r"(\d+)")[0], errors="coerce"
)
movie_duration = movies["duration_minutes"].dropna()

print("Sample size:", len(movie_duration))
print(movie_duration.describe())

# H0: mu = 90
# H1: mu > 90
mu0 = 90
t_stat, p_two = stats.ttest_1samp(movie_duration, popmean=mu0)
p_one = p_two / 2 if t_stat > 0 else 1 - p_two / 2

n = len(movie_duration)
mean = movie_duration.mean()
sd = movie_duration.std(ddof=1)
se = sd / math.sqrt(n)
ci = stats.t.interval(0.95, df=n-1, loc=mean, scale=se)

print("\nOne-sample t-test")
print("t =", t_stat)
print("two-sided p =", p_two)
print("one-sided p =", p_one)
print("95% CI =", ci)

# Non-parametric robustness check.
wilcoxon = stats.wilcoxon(movie_duration - mu0, alternative="greater")
print("\nWilcoxon signed-rank test")
print("statistic =", wilcoxon.statistic)
print("p =", wilcoxon.pvalue)

# Histogram
plt.figure(figsize=(9,5))
plt.hist(movie_duration, bins=30)
plt.axvline(mean, linestyle="--", label=f"Sample mean = {mean:.2f}")
plt.axvline(mu0, linestyle=":", label="Null benchmark = 90")
plt.title("Distribution of Netflix Movie Durations")
plt.xlabel("Duration (minutes)"); plt.ylabel("Number of Movies")
plt.legend(); plt.tight_layout()
plt.savefig(OUT/"01_duration_histogram.png", dpi=180); plt.close()

# Mean + CI
plt.figure(figsize=(8.5,5))
plt.errorbar([0],[mean],yerr=[[mean-ci[0]],[ci[1]-mean]],fmt="o",capsize=8)
plt.axhline(mu0,linestyle="--")
plt.xticks([0],["Netflix movies"]); plt.ylabel("Duration (minutes)")
plt.title("Mean Movie Duration with 95% Confidence Interval")
plt.tight_layout(); plt.savefig(OUT/"02_mean_ci.png",dpi=180); plt.close()

# Box plot
plt.figure(figsize=(8.5,5))
plt.boxplot(movie_duration, vert=False)
plt.axvline(mu0,linestyle="--")
plt.xlabel("Duration (minutes)")
plt.title("Movie Duration Box Plot with 90-Minute Benchmark")
plt.tight_layout(); plt.savefig(OUT/"03_duration_boxplot.png",dpi=180); plt.close()

# Q-Q plot
stats.probplot(movie_duration, dist="norm", plot=plt)
plt.title("Q-Q Plot of Movie Durations")
plt.tight_layout(); plt.savefig(OUT/"04_qq_plot.png",dpi=180); plt.close()

print("\nDecision at alpha=0.05:", "Reject H0" if p_one < 0.05 else "Fail to reject H0")
