A **stochastic process** is a mathematical model for a collection of random variables indexed by time or some other parameter.  Formally, it is a family of random variables $\{X_t\}_{t \in T}$ defined on a probability space; each $X_t$ represents the system’s state at time $t$.  Stochastic processes allow us to model uncertain dynamics of systems such as financial markets, queues and physical phenomena.

### Key classifications

* **Discrete‑time vs. continuous‑time:**  If the index set $T$ is countable (e.g., integers), the process is discrete‑time; if $T$ is an interval of the real line, it is continuous‑time.  The choice affects the mathematics: discrete‑time processes often use sums, whereas continuous‑time processes rely on calculus and measure‑theoretic techniques.

* **Stationary vs. non‑stationary:**  A process is strictly stationary if all finite‑dimensional distributions are invariant under time shifts; in other words, the joint distribution of $(X_{t_1}, \dots, X_{t_n})$ is identical to that of $(X_{t_1+h}, \dots, X_{t_n+h})$ for all $h$.  Stationarity implies that the distribution does not change over time, which often simplifies analysis.  Non‑stationary processes exhibit time‑dependent means, variances or other properties (e.g., trending series), so their statistical properties vary with time.

### Important examples in finance

* **Random walk:**  A random walk is a discrete‑time process where each step $S_n$ is the sum of independent increments, often ±1 with equal probability.  Random walks model phenomena such as the fluctuating price of a stock.  In finance, the **random walk hypothesis** states that asset returns are unpredictable and that past price movements have little value in forecasting.  Under this view, price changes behave like a random walk, implying that technical analysis cannot systematically outperform the market.

* **Brownian motion (Wiener process):**  This is the continuous‑time limit of a random walk.  It has stationary, independent Gaussian increments and continuous sample paths.  Brownian motion serves as the driving noise in diffusion models; for example, the **geometric Brownian motion (GBM)** used in the Black–Scholes option‑pricing model assumes that the logarithm of a stock price follows a Brownian motion with drift, ensuring that the stock price remains positive.  Its use in economics and quantitative finance highlights its role in modelling continuous stochastic fluctuations.

* **Markov chain/process:**  A Markov process is a stochastic process with the **Markov property**: conditioned on the present state, future states are independent of the past.  A Markov chain is typically discrete in time and has a finite or countable state space.  In finance, Markov models are used to represent regime switches (e.g., bull vs. bear markets), credit‑rating migrations, and to implement **Markov chain Monte Carlo** simulations for Bayesian inference.

* **Poisson process:**  A Poisson process is a counting process $N(t)$ where $N(0)=0$, increments over disjoint intervals are independent, and the number of events in any interval of length $t$ follows a Poisson distribution with parameter $\lambda t$.  Equivalently, inter‑arrival times are exponentially distributed.  In finance, Poisson processes model **rare events** such as sudden jumps in asset prices or order arrivals.  They underpin **jump‑diffusion** models, where stock prices follow a GBM punctuated by jumps.

### Modelling stock prices

The intuition behind these processes in quantitative finance comes from the efficient‑market view that price changes are largely unpredictable.  A **random walk or Brownian motion** captures continuous, small fluctuations in returns; with a drift term, it yields geometric Brownian motion, where

$$
dS_t = \mu S_t\,dt + \sigma S_t\,dW_t,
$$

so the percentage change $dS_t/S_t$ has a drift $\mu$ and volatility $\sigma$.  This model implies that log‑returns are normally distributed and leads to analytical option pricing.  A **Markov chain** framework allows regime changes—e.g., switching between high‑volatility and low‑volatility states—improving risk management.  **Poisson or jump processes** add discrete jumps to capture sudden news-driven price moves, better reflecting fat tails and skewness in return distributions.  More advanced models combine these elements (e.g., jump‑diffusion, stochastic volatility models) to capture both continuous fluctuations and jumps.

### Application in quantitative trading

Understanding stochastic processes enables the quantitative trader to build realistic models, simulate price paths and estimate risk.  By calibrating stochastic models to market data, traders can:

* **Price derivatives and manage risk:** GBM and jump‑diffusion models provide closed‑form or simulated prices for options, enabling hedging strategies.

* **Develop trading signals:** Markov regime‑switching models can detect shifts between market states (e.g., trending vs. mean‑reverting) and adjust strategies accordingly.

* **Estimate arrival rates:** Poisson processes model order arrivals or extreme events, informing limit‑order placement and stress testing.

* **Backtest strategies:** Simulations of random walks and Brownian motions help test algorithmic trading strategies under realistic randomness.

By combining theoretical understanding with empirical calibration, traders can design strategies that account for randomness and volatility, improving profitability and risk control.
