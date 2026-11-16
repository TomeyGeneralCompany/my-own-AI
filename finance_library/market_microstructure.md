# Market structure research and experiment agenda

- Cont, Kukanov, Stoikov, The Price Impact of Order Book Events (2014): https://arxiv.org/abs/1011.6402 — test lagged order-flow imbalance and market depth as short-horizon features. Contemporaneous correlation is not evidence of future predictive profit.
- Alpaca historical stock data: https://docs.alpaca.markets/us/v1.1/docs/historical-stock-data-1 — source of timestamped trades and quotes, with different feed coverage.
- Alpaca market data plans: https://docs.alpaca.markets/us/docs/about-market-data-api — compare coverage and latency before selecting data.
- Alpaca fractional order documentation: https://docs.alpaca.markets/us/docs/fractional-trading — fractional execution specifications and supported order parameters.

The owner's corpus and Austrian foundation remain primary for conceptual instruction. These sources describe empirical mechanisms and instruments.

Experiments: compare passive holding, imbalance thresholds, lagged order flow, and Core-proposed alternatives on chronological quote streams. Separate strategy development windows from final untouched test windows; correction on an evaluation window makes it development data. Measure net cash, residual holdings, costs, depth, latency assumptions and the ideal hindsight benchmark separately. Extend to multi-asset selection and realistic settlement before a live qualification. The baseline is deliberately simple, not an imposed risk preference or claim of profitable performance.

Use the shared web library in PR #42 once merged to acquire and index actual references. This file contains verified links and descriptions, not the full documents. No training or live trading is enabled.
