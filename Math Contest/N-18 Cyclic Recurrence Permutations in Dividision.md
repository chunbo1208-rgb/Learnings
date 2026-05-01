---
description: Count the coefficient triples that allow a permutation of residues to satisfy a cyclic linear recurrence modulo n.
tags:
  - MathContest
  - NumberTheory
source: Yanboyuan Number Theory pp. 43-45
Statement:
---
# Problem Statement
Let $n>1$ be an odd integer. Assume that every prime divisor $p$ of $n$ satisfies $\gcd(p-1,n)=1$. Determine the number of ordered triples $(a,b,c)$ such that:

1. $a,b,c\in\{1,2,\dots,n\}$ and $\gcd(a,b,c,n)=1$;
2. There exists a permutation $x_1,x_2,\dots,x_n$ of $1,2,\dots,n$ such that for every $k=1,2,\dots,n$,
$$n\mid ax_{k+2}+bx_{k+1}+cx_k,$$
where $x_{n+1}=x_1$ and $x_{n+2}=x_2$.

# 