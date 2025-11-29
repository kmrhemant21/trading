# Lagrange Multipliers

Check out the basics of Lagrange multipliers at the corresponding [Wikipedia page](https://en.wikipedia.org/wiki/Lagrange_multiplier).

## Key Concepts

We can solve a constrained optimization problem of the form:

$$\min_x f(x), \text{ s.t. } g(x) = 0$$

where $g(x)$ is an equality constraint.

The constraints can be absorbed into a single objective function, the **Lagrangian**, which combines the original loss function and the constraints as:

$$L(x,\lambda) = f(x) - \lambda g(x)$$

where $\lambda$ is called a **Lagrange multiplier**.

## Solution Method

We solve the constrained optimization problem by:

1. Computing the partial derivatives $\frac{\partial L}{\partial x}$ and $\frac{\partial L}{\partial \lambda}$
2. Setting them to $0$
3. Solving for $\lambda$ and $x$

## Additional Resources

Lagrange multipliers are covered in more detail in the Mathematics for Machine Learning: Linear Algebra course in this specialisation. For a refresher, see [this quiz](link-to-quiz).
