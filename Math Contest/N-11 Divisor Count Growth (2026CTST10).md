---
description: Prove a lower bound relating successive counts of divisors of n in intervals [1,n^ell].
tags:
  - MathContest
  - NumberTheory
  - CTST
source: 2026 CTST 10
Statement: false
---
# Problem Statement
Let $n>1$ be an integer. For each positive integer $\ell$, let $d_\ell$ denote the number of divisors of $n$ lying in the interval $[1,n^{\frac{1}{\ell}}]$. Prove that for every integer $k\ge 2$,
$$d_{k+1}\ge \sqrt{2d_k}-k-\frac12.$$

# Solution
Actually, the solution part gives me one feeling: it is not natural, but the people done this says it easily.

We need to find the relations of $d_k\to d_{k+1}$, 
We divide the divisors $s\in[1,n^{\frac{1}{\ell}}]$ into two kinds
- $s = s_1s_2$ which $s_1,s_2\in [1,n^{\frac{1}{\ell+1}}]$. The numbers in this part is $\leq \binom{d_{\ell +1}+1}{2}$ (==why not $\binom{d_{\ell+1}}{2}$==)
- can't be divide into 2 factors all in the interval. For every divisor $s$ in this part, there will be a prime factor of $s$ as $p\in[n^{\frac{1}{\ell+1}},n^{\frac{1}{\ell}}]$, and so, we need to deal with all of the prime factor of $n$.
	- Suppose $p_1<p_2<\dots<p_m$, and we claim that $\frac{n}{p}\in D_{\ell+1}$,(==in the answer, not that natural==)