---
description: Bound the total sum of a constrained nonnegative integer matrix by its number of nonzero entries.
tags:
  - MathContest
  - Combinatorics
source: 2026042728BeijingA.pdf
---
# Problem Statement
Let $n\ge 2$ be a positive integer, and let $A=[a_{ij}]$ be an $n\times n$ matrix of nonnegative integers satisfying:

1. $a_{ij}=0$ whenever $i+j\le n$;
2. for $1\le i\le n-1$ and $1\le j\le n$, one has $a_{i+1,j}\in\{a_{ij},a_{ij}+1\}$;
3. for $1\le i\le n$ and $1\le j\le n-1$, one has $a_{i,j+1}\in\{a_{ij},a_{ij}+1\}$.

Let $S$ be the sum of all entries of $A$, and let $N$ be the number of nonzero entries of $A$. Prove that
$$S\le \frac{(n+2)N}{3}. $$
