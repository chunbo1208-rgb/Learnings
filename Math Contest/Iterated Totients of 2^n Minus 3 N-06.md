---
description: Bound the number of distinct prime divisors arising from iterated Euler totients of 2^n-3.
tags:
  - MathContest
  - NumberTheory
source: Yanboyuan Number Theory pp. 43-45
---
# Problem Statement
For any positive integer $m$, let $\varphi(m)$ denote the number of positive integers not exceeding $m$ that are relatively prime to $m$. Define $\varphi_0(m)=m$, and for each positive integer $k$, define $\varphi_k(m)=\varphi(\varphi_{k-1}(m))$. Prove that for every integer $n\ge 3$,
$$\varphi_0(2^n-3)\cdot \varphi_1(2^n-3)\cdots \varphi_n(2^n-3)$$
has at most $n$ distinct prime divisors.
