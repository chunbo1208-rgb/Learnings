---
description: Determine all prime pairs below 2005 satisfying two symmetric divisibility conditions.
tags:
  - MathContest
  - NumberTheory
  - Problem
source: Yanboyuan Number Theory pp. 43-45
Statement: true
---
# Problem Statement
Find all primes $p,q<2005$ such that $p\mid q^2+8$ and $q\mid p^2+8$.

# Solution
This is not a difficult problem, but to combine the 2 conditions into $$pq\mid p^2+q^2+8$$is important and and smart.
Then, it is $$p^2+q^2+8 = kpq\Rightarrow p^2-kq\cdot p+(q^2+8)$$ which is the [[Vieta Jumping]] technique. 
We suppose $(p_0,q_0)(p_0>q_0)$ is a solution of it, then $p_1 = kq_0-p_0\in\mathbb{Z},p_1 = \dfrac{q_0^2+8}{p_0}\Rightarrow p_1\in\mathbb{N}^*$. Then we have $p_1<p_0$ we can make it strictly decrease step by step, until $(1,u)$ is the solution. We have $u$ is odd and $u\mid 9$ which shows that $u = 1,3,9$.

- When $u =1$, $(1,1)\to (1,9)\to (9,89)\to(89,)$ This is the sequence of $\{u_i\}:1,9,89$ which $k = 10$ then we can find later terms and remember with the descipline of $<2005$
- When $u = 3$, $(1,3)\to (17,3)\to(17, 99)$ samely, $k=6$
- $u=9$ already considered in $u=1$

In summary $(17,3),(3,17),(89,881),(881,89),(2,2)$

# Related
1. Try to combine the solutions when to symmetric conditions
2. Vieta jumping