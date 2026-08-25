# References

This short list provides method context for the implemented validation protocol
and the explicitly labelled future work. It is not a related-work survey.

1. J.P. Morgan/Reuters, *RiskMetrics Technical Document* (4th ed., 1996),
   [MSCI research record](https://www.msci.com/research-and-insights/paper/1996-riskmetrics-technical-document).
   The repository uses a fixed RiskMetrics-style daily EWMA decay factor
   `lambda=0.94`; it does not estimate the decay factor on the test data.
2. Andrew J. Patton (2011), “Volatility forecast comparison using imperfect
   volatility proxies,” *Journal of Econometrics*, 160(1), 246–256,
   [doi:10.1016/j.jeconom.2010.03.034](https://doi.org/10.1016/j.jeconom.2010.03.034).
   This is the source context for using QLIKE alongside RMSE when a realised
   volatility proxy is imperfect.
3. Fulvio Corsi (2009), “A Simple Approximate Long-Memory Model of Realized
   Volatility,” *Journal of Financial Econometrics*, 7(2), 174–196,
   [doi:10.1093/jjfinec/nbp001](https://doi.org/10.1093/jjfinec/nbp001).
   HAR-RV is recorded as future work only; no HAR-RV model is implemented in
   v0.1.0.
4. Francis X. Diebold and Roberto S. Mariano (1995), “Comparing Predictive
   Accuracy,” *Journal of Business & Economic Statistics*, 13(3), 253–263,
   [doi:10.1080/07350015.1995.10524599](https://doi.org/10.1080/07350015.1995.10524599).
   The current release reports descriptive fold-bootstrap intervals; a
   dependence-aware forecast-comparison test is future work, not a claim made
   by this release.

