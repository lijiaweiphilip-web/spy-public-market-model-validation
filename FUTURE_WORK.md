# Future work

This v0.1.0 release keeps the model set intentionally small and the evidence boundary explicit.

- **P1: HAR-RV baseline.** Add a finance-domain heterogeneous autoregressive realised-variance baseline with the same purged walk-forward contract.
- **P2: cross-asset robustness.** Repeat the locked protocol on QQQ, IWM and TLT without changing the evaluation rules after seeing results.
- **P3: optional GARCH(1,1).** Consider only after dependency stability, convergence diagnostics and forecasting semantics are fully tested.

These are next experiments, not claims about the current SPY reference run and not blockers for the public v0.1.0 repository.
