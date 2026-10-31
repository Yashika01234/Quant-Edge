# QUANT EDGE (Lite)
### Financial Risk, Fraud Detection & Portfolio Analytics in Python

A compact quantitative finance project that answers two questions every bank and wealth manager cares about:

1. **Is something suspicious happening in our transactions?**
2. **How much could our investment portfolio lose?**

Built with Python, Pandas, scikit-learn and Plotly. It runs end to end in one script on synthetic data, then produces an interactive dashboard and a plain-English executive summary.

---

## Features

| Module | What it does |
|---|---|
| **Transaction Intelligence** | Analyses 2,000 simulated transactions (amount, hour, activity in last 24h) |
| **Fraud Detection** | Rule-based flags (large amounts, night hours, bursts) combined with an **Isolation Forest** anomaly model |
| **Portfolio Risk Engine** | Annual return, volatility, **Sharpe ratio**, **maximum drawdown**, **Value at Risk (95%)** |
| **Stress Testing** | Simulates a 2008-style market crash, an interest-rate rise and a crypto collapse |
| **Financial Analytics** | Interactive Plotly dashboard: suspicious transactions, asset allocation, drawdown, stress losses |
| **Executive Insights** | Auto-generated report: *what happened, why it matters, what to do* |

---

## Sample Output

```
WHAT HAPPENED
- 43 of 2000 transactions flagged (2.1%); rules flagged 42, Isolation Forest flagged 40.
- Portfolio: return 6.7%, volatility 11.2%, Sharpe 0.24, max drawdown -18.0%, 1-day 95% VaR 1.08%.
- Worst stress scenario: 'Market crash (-2008 style)' -> -23.5% portfolio loss.

WHY IT MATTERS
- Flagged items cluster at night hours and very large amounts (typical fraud signals).
- Crypto is 10% of weights but a large share of tail risk.

ACTION
- Send flagged transactions to manual review; tune thresholds to cut false positives.
- Consider trimming Crypto/Equity or adding Bonds/Gold to reduce stress losses.
```

Open `quant_edge_report.html` in a browser to see the interactive dashboard.

---

## Getting Started

**Requirements:** Python 3.9+

```bash
git clone https://github.com/Yashika01234/Quant-Edge.git
cd Quant-Edge
python -m pip install numpy pandas scikit-learn plotly
python quant_edge.py
```

The script prints the insights to the terminal and writes `quant_edge_report.html`.

---

## How It Works

### 1. Fraud Detection
- **Rules:** flag amounts above the 99th percentile, transactions between midnight and 5 AM, and more than 10 transactions in 24 hours.
- **Isolation Forest:** an unsupervised model that isolates unusual points quickly. It needs no labels, which suits fraud where labelled data is scarce.
- A transaction is flagged if **either** method catches it.

### 2. Portfolio Risk Metrics
Portfolio: 50% Equity, 30% Bonds, 10% Gold, 10% Crypto (three years of daily returns).

- **Volatility:** annualised standard deviation of daily returns
- **Sharpe ratio:** `(return - risk-free rate) / volatility`
- **Max drawdown:** largest peak-to-trough fall in portfolio value
- **VaR (95%):** the 5th percentile of daily returns, i.e. a loss that should be exceeded only 1 day in 20

### 3. Stress Testing
Applies fixed percentage shocks to each asset and computes the weighted portfolio loss under each scenario.

---

## Tech Stack

- **Python**: core language
- **Pandas / NumPy**: data handling and numerical work
- **scikit-learn**: Isolation Forest
- **Plotly**: interactive dashboard

---

## Limitations

- Uses **synthetic data**, not real transactions or market prices.
- Historical VaR assumes the past resembles the future.
- Stress scenarios are illustrative, not calibrated to actual historical data.

## Roadmap

- [ ] Replace synthetic prices with real market data (`yfinance`)
- [ ] Add Monte Carlo and parametric VaR, plus Expected Shortfall
- [ ] Expose analytics through a **FastAPI** backend
- [ ] Store data in **PostgreSQL**
- [ ] Build a **React + Tailwind** frontend
- [ ] Add supervised fraud models and evaluation (precision/recall)

---

## Author

**Yashika**: [GitHub](https://github.com/Yashika01234)
