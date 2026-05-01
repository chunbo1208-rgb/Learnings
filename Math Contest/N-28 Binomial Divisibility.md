---
description: Prove that a suitable binomial coefficient is divisible by c under a coprimality hypothesis.
tags:
  - MathContest
  - NumberTheory
source: Yanboyuan Number Theory pp. 43-45
Statement: false
---
# Problem Statement
Let $a,b,c$ be positive integers with $b>a>1$ and $\gcd(ab,c)=1$. Prove that there exists a positive integer $n$ such that
$$c\mid \binom{b^n}{a^n}. $$
# Solving
- First, we will start with prime numbers: the binomial is not easy to deal with, so we consider $c = p_1^{\alpha_1}p_2^{\alpha_2}\cdots p_m^{\alpha_m}$, then we only need to prove that $$\exists n, \forall i\in \{1,2,\cdots,m\},p_i^{\alpha_i}\mid \binom{b^n}{a^n}$$actually, it shows that we need to consider $v_{p_i}(\dbinom{b^n}{a^n})$, that is, [[Kummer's Theorem]].
- By the assumption of [[Kummer's Theorem]], we know that $v_{p_i}(\dbinom{b^n}{a^n})=$ the number of [carries](https://en.wikipedia.org/wiki/Carry_\(arithmetic\) "Carry (arithmetic)") when $b^n$ is added to $b^n− a^n$ in [base](https://en.wikipedia.org/wiki/Radix "Radix") $p$. So we need to consider the number.
- **I am stucked here**, there are some techniques:
	- We can take $a,b$ as $a\equiv b \equiv 1\pmod{c}$  because we can turn $a\to a^{\varphi(c)}$(by [[Euler's Theorem (Number Theory)]]) 
- We only need to prove the seperate part, because we can combine every prime number together easily:
	**Lemma 1**: Find a mass number $M_i$, which $p_i^{M_i}>b^{n_i}>a^{n_i}$, then we know that $b^{n_i+p_i^{M_i}\cdot t}\equiv b^{n_i}\pmod{p_i^{M_i}}$. Because $$v_{p_i}(b^{p^{M_i}\cdot t}-1)\geq M_i$$ by [[Lifting the Exponent Lemma (LTE)]]. Then we have that in base $p_i$ the last $M_i$ digits of $b^{n_i}$ and $b^{n_i+p_i^{M_i}\cdot t}$ are the same, the is a kind of translation, then by [[Chinese Remaining Theorem (CRT)]], we know that $n\equiv n_i\pmod{p_i^{M_i}}$ for $i\in [m]$ have a solution. Then we know that as long as for every $i$, $$\exists n, \forall i\in \{1,2,\cdots,m\},p_i^{\alpha_i}\mid \binom{b^{n_i}}{a^{n_i}}$$ follows, then $c\mid \dbinom{b^n}{a^n}$.
- So we just need to prove that for $a\equiv b\equiv 1\pmod{p}$, we have that $$\forall \alpha\exists n,p^\alpha\mid \binom{b^n}{a^n}$$