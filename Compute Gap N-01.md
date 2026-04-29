---
tags:
  - MathContest
  - NumberTheory
source: 2026.4.29 Beijing Training
---
# Problem Statement
Given positive real number $C$. We say $a$ is a **good number** if $a=xy$ which $x,y\in \mathbb{N}^*,|x-y|\leq C\cdot\sqrt[4]{a}$ , we order all the good numbers as $a_1<a_2<\cdots$. Find the least real number $\lambda = \lambda(C)$, such that there exist positive number $M$ which for all $n\in \mathbb{N}^*$, we all have $$a_{n+1}\leq a_n+M\cdot a_n^\lambda$$
# Problem Explanation
What we want to consider is $a_{n+1}-a_{n}$(which is the **gap**), will be small.