[[Sheldon Axler—Linear Algebra Done Right 3e.pdf]]

# Chapter 1
## Exercise 1C
[7]Give an example of a nonempty subset $U$ of $R^2$ such that $U$ is closed under addition and under taking additive inverses (meaning $-u\in U$ whenever $u\in U$), but $U$ is not a subspace of $R^2$.
	What we want is a subspace which follows additive, but not scalar multiplication. Actually, we just need to let $U<\mathbb{Z}^2$   
	Construct $U=\{(x,y)\mid x,y\in \mathbb{Z}\}$ then we have that $U$ is not a subspace of $\mathbb{R}^2$.
[8]Give an example of a nonempty subset $U\in \mathbb{R}^2$ such that $U$ is closed under scalar multiplication, but $U$ is not a subspace of $\mathbb{R}^2$.
	Similarly, we can construct one follow scalar multiplication but not addition. $U=\{(x,0),(0,y)\mid x,y\in \mathbb{R}\}$.
[9]A function $f:\mathbb{R} \to \mathbb{R}$ is called periodic if there exists a positive number $p$ such that $f(x)= f(x+ p)$ for all $x \in \mathbb{R}$. Is the set of periodic functions from $\mathbb{R}$ to $\mathbb{R}$ a subspace of $\mathbb{R}^{\mathbb{R}}$? Explain.
	I think we need to claim how to define the dim $\mathbb{R}$. Maybe the numbers of the length of a period (the $p$) is the dimension. We can test: fix $p\in \mathbb{R}$ and notation $f_p:=\{f\mid f(x)= f(x+ p),f: \mathbb{R}\to \mathbb{R}\}$
	-  Define application: $f_p+g_p:=h\mid \forall x\in \mathbb{R}, h(x) = f(x)+g(x)$ easy to test, $h_p$.
	- Define scalar multiplication: $a\cdot f_p:=h\mid \forall x\in \mathbb{R} ,h(x) = a\cdot f(x)$, also easy to test $\forall a \in \mathbb{R},h_p$.
	- Then test
		- $0_p:= {0}$ 
		- application and scalar multiplication follows.
[12]Prove that the union of two subspaces of $V$ is a subspace of $V$ if and only if one of the subspaces is contained in the other.
	"$\Rightarrow$"
		**Prove by Contradiction**:
		Suppose $U_1,U_2< V$, we have that $U_1\bigcup U_2<V$ but $U_1/U_2, U_2/U_1\not = \emptyset$ then we can suppose $x_1\in U_1/U_2, x_2\in U_2/U_1$ then by the subspace we have $x_1+x_2\in U_1\bigcup U_2$, then we must have $x_1+x_2\in U_1$ or $x_1+x_2\in U_2$, so WLOG(without loss of generality) we suppose $x_1+x_2\in U_1$ then we have $x_2 = (x_1+x_2)-x_1$ which is in $U_1$ Contradiction!
	"$\Leftarrow$"    
		Easy to prove.
[13]Prove that the union of three subspaces of $V$ is a subspace of $V$ if and
only if one of the subspaces contains the other two.
Hint from the book: *This exercise is surprisingly harder than the previous exercise, possibly because this exercise is not true if we replace $\mathbb{F}$ with a field containing only two elements.* And $\mathbb{GL_2}$ which only contains 0 and 1 is such a field. 
	$U_1\bigcup U_2 \bigcup U_3<V$ then we have that from [12] we know that $U_1 \subset (U_2\bigcup U_3)$ or $(U_2\bigcup U_3)\subset U_1$ and similar situations. To prove the rest situations we can write that  $U_1 \subset (U_2\bigcup U_3)$, $U_2 \subset (U_1\bigcup U_3)$ and $U_3 \subset (U_1\bigcup U_2)$.

## 