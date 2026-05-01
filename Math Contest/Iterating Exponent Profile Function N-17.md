---
description: Study the eventual periodicity of an arithmetic function built from prime exponents and maximize its minimal period.
tags:
  - MathContest
  - NumberTheory
source: Yanboyuan Number Theory pp. 43-45
---
# Problem Statement
Define a function $f:\mathbb Z^+\to\mathbb Z^+$ by $f(1)=1$, and for $n\ge 2$, if the standard prime factorization of $n$ is
$$n=\prod_{i=1}^m p_i^{\alpha_i},$$
then define
$$f(n)=\prod_{i=1}^m \alpha_i^{p_i}.$$
For a given positive integer $n$, consider the iteration sequence
$$f(n),\ f(f(n)),\ f(f(f(n))),\ \dots.$$ 
Must this sequence eventually become periodic? If so, what is the largest possible value of its minimal positive period?
