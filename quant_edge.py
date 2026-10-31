"""QUANT EDGE (lite): fraud detection + portfolio risk + stress test + insights.
Run: pip install numpy pandas scikit-learn plotly && python quant_edge.py
Outputs: quant_edge_report.html
"""
import numpy as np, pandas as pd
import plotly.graph_objects as go
from plotly.subplots import make_subplots
from sklearn.ensemble import IsolationForest

rng = np.random.default_rng(42)

# ---------- 1. Transaction Intelligence + 2. Fraud Detection ----------
n = 2000
tx = pd.DataFrame({
    "amount": rng.lognormal(4, 0.8, n),
    "hour": rng.integers(6, 23, n),
    "tx_last_24h": rng.poisson(3, n),
})
# inject 40 suspicious transactions: huge amounts, odd hours, bursts
bad = rng.choice(n, 40, replace=False)
tx.loc[bad, "amount"] *= rng.uniform(10, 30, 40)
tx.loc[bad, "hour"] = rng.integers(0, 5, 40)
tx.loc[bad, "tx_last_24h"] += rng.integers(8, 15, 40)

# rule-based flags
tx["rule_flag"] = (tx.amount > tx.amount.quantile(0.99)) | (tx.hour < 5) | (tx.tx_last_24h > 10)
# ML flag (Isolation Forest)
iso = IsolationForest(contamination=0.02, random_state=42)
tx["ml_flag"] = iso.fit_predict(tx[["amount", "hour", "tx_last_24h"]]) == -1
tx["suspicious"] = tx.rule_flag | tx.ml_flag

# ---------- 3. Portfolio Risk Engine ----------
assets = ["Equity", "Bonds", "Gold", "Crypto"]
weights = np.array([0.50, 0.30, 0.10, 0.10])
mu = np.array([0.0006, 0.0002, 0.0003, 0.0010])
vol = np.array([0.012, 0.004, 0.008, 0.035])
rets = pd.DataFrame(rng.normal(mu, vol, (756, 4)), columns=assets)  # ~3 years daily
port = rets @ weights

ann_ret, ann_vol = port.mean() * 252, port.std() * np.sqrt(252)
rf = 0.04
sharpe = (ann_ret - rf) / ann_vol
cum = (1 + port).cumprod()
drawdown = cum / cum.cummax() - 1
max_dd = drawdown.min()
var95 = -np.percentile(port, 5)  # 1-day historical VaR

# ---------- 5. Stress Testing ----------
shocks = {  # asset-level % shocks
    "Market crash (-2008 style)": [-0.40, 0.05, 0.10, -0.60],
    "Interest rate +2%":         [-0.10, -0.12, -0.05, -0.20],
    "Crypto winter":             [-0.03, 0.00, 0.02, -0.70],
}
stress = {k: float(weights @ np.array(v)) for k, v in shocks.items()}

# ---------- 4. Dashboard ----------
fig = make_subplots(
    rows=2, cols=2,
    specs=[[{"type": "scatter"}, {"type": "pie"}],
           [{"type": "scatter"}, {"type": "bar"}]],
    subplot_titles=("Transactions: amount vs hour (red = suspicious)",
                    "Asset allocation", "Portfolio drawdown", "Stress test loss"))
s = tx[tx.suspicious]; ok = tx[~tx.suspicious]
fig.add_trace(go.Scatter(x=ok.hour, y=ok.amount, mode="markers", name="normal",
                         marker=dict(color="steelblue", size=4)), 1, 1)
fig.add_trace(go.Scatter(x=s.hour, y=s.amount, mode="markers", name="suspicious",
                         marker=dict(color="red", size=6)), 1, 1)
fig.add_trace(go.Pie(labels=assets, values=weights, hole=0.4), 1, 2)
fig.add_trace(go.Scatter(y=drawdown, fill="tozeroy", name="drawdown",
                         line=dict(color="crimson")), 2, 1)
fig.add_trace(go.Bar(x=list(stress), y=list(stress.values()), name="loss",
                     marker_color="darkorange"), 2, 2)
fig.update_layout(height=800, title="QUANT EDGE (lite) Dashboard", showlegend=False)

# ---------- 6. Executive Insights ----------
worst = min(stress, key=stress.get)
insights = f"""
WHAT HAPPENED
- {tx.suspicious.sum()} of {n} transactions flagged ({tx.suspicious.mean():.1%});
  rules flagged {tx.rule_flag.sum()}, Isolation Forest flagged {tx.ml_flag.sum()}.
- Portfolio: return {ann_ret:.1%}, volatility {ann_vol:.1%}, Sharpe {sharpe:.2f},
  max drawdown {max_dd:.1%}, 1-day 95% VaR {var95:.2%}.
- Worst stress scenario: '{worst}' -> {stress[worst]:.1%} portfolio loss.

WHY IT MATTERS
- Flagged items cluster at night hours and very large amounts (typical fraud signals).
- Crypto is {weights[3]:.0%} of weights but a large share of tail risk.

ACTION
- Send flagged transactions to manual review; tune thresholds to cut false positives.
- Consider trimming Crypto/Equity or adding Bonds/Gold to reduce stress losses.
"""
print(insights)
with open("quant_edge_report.html", "w") as f:
    f.write(fig.to_html(include_plotlyjs="cdn"))
    f.write(f"<pre style='font-size:14px;padding:20px'>{insights}</pre>")
print("Saved quant_edge_report.html")
