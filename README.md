# Factor Investing in US Commodity Futures

**A 25-year, cost-aware backtest of six commodity factor premia on 26 US futures, built entirely on free public data.**

Momentum · Carry · Value · Skewness · Hedging pressure · Time-series momentum · Multi-factor combination · Long-only tilts

| | |
|---|---|
| **Universe** | 26 US commodity futures across energy, metals, grains, softs and livestock |
| **Sample** | February 2002 to September 2026 (monthly rebalancing) |
| **Data** | Yahoo Finance, U.S. Energy Information Administration (EIA), CFTC Commitments of Traders, FRED, AQR Data Library |
| **Costs** | 5 bps per unit of turnover, applied to rebalancing **and** roll trades |
| **Statistics** | Newey-West t-statistics, spanning regressions, Holm / Bonferroni / Harvey-Liu-Zhu multiple-testing adjustments |
| **Validation** | Independent replication against AQR's published commodity factors (correlation 0.75 to 0.79) |
| **Deliverables** | Six Jupyter notebooks and a nine-page research carousel: [`commodity_factor_investing_carousel.pdf`](commodity_factor_investing_carousel.pdf) |

---

## Headline result

> **Individually, commodity factors were statistically fragile in 2002 to 2026. Combined, they were robustly significant.**
> A pre-specified, equal-risk blend of all six factors earned a **Sharpe ratio of 0.71 (Newey-West t = 3.91)** with a **maximum drawdown of −14%**, against 0.63 and −39% for an equal-weight commodity portfolio, and a correlation of only **0.12** with that market. It is the only result in the study that clears the t > 3 hurdle proposed for new factors.

| Strategy (net of costs) | Sharpe | t-stat (NW) | Max drawdown | Unique alpha t-stat* | Evidence |
|---|---:|---:|---:|---:|---|
| **Multi-factor (equal-risk)** | **0.71** | **3.91** | **−14%** | – | **Strong: passes t > 3** |
| Skewness | 0.46 | 2.13 | −47% | 0.80 | Moderate |
| Value (from Nov 2005) | 0.44 | 2.12 | −40% | 2.87 | Moderate |
| Time-series momentum | 0.40 | 2.03 | −39% | 1.14 | Moderate |
| Hedging pressure | 0.34 | 1.44 | −51% | 2.02 | Weak alone, unique source |
| Carry | 0.17 | 0.97 | −66% | −0.60 | Weak |
| Momentum (12-1) | 0.11 | 0.62 | −66% | 0.47 | Weak |
| *Commodity market (equal-weight, 2004 to 2026)* | *0.63* | *2.73* | *−39%* | – | *Benchmark* |
| **Long-only multi-factor tilt vs market** | **IR 0.53** | **2.88** | – | – | **Significant** |

\*t-statistic of the factor's alpha after controlling for the other five factors and the commodity market (common sample from November 2005). Full results and caveats are in [Section 9](#9-results) and [Section 13](#13-limitations-and-caveats); the machine-learning extension is in [Section 14](#14-machine-learning-extension-notebooks-07-and-08).

---

## Contents

1. [Research question and summary](#1-research-question-and-summary)
2. [Repository contents and how to reproduce](#2-repository-contents-and-how-to-reproduce)
3. [Data](#3-data)
4. [Universe](#4-universe)
5. [From raw prices to investable returns](#5-from-raw-prices-to-investable-returns)
6. [Factor theory and construction](#6-factor-theory-and-construction)
7. [Portfolio construction](#7-portfolio-construction)
8. [Statistical methodology](#8-statistical-methodology)
9. [Results](#9-results)
10. [Notebook guide](#10-notebook-guide)
11. [Code architecture](#11-code-architecture)
12. [Decision log](#12-decision-log)
13. [Limitations and caveats](#13-limitations-and-caveats)
14. [Machine-learning extension (notebooks 07 and 08)](#14-machine-learning-extension-notebooks-07-and-08)
15. [Conclusions](#15-conclusions)
16. [References](#16-references)
17. [Data credits and disclaimer](#17-data-credits-and-disclaimer)

---

## 1. Research question and summary

**Question.** Do the factor premia documented in the commodity futures literature (carry, momentum, value, skewness, hedging pressure and trend) exist in US futures over 2002 to 2026, after realistic costs, careful roll handling and correction for multiple testing? And does combining them help?

**Approach.** The project builds the full pipeline from raw data: downloading and auditing prices, constructing roll-adjusted excess returns from first principles, computing seven factor signals with strict point-in-time timing, forming long-short and long-only portfolios, and evaluating them with methods designed to avoid the most common backtesting errors (look-ahead bias, roll contamination, autocorrelated standard errors, data-mining and cherry-picked variants).

**Summary of findings.**

1. **Single factors are positive but statistically fragile.** Skewness, value and time-series momentum pass a conventional 5% test (t ≈ 2.0 to 2.1), but none survives a Holm adjustment for testing six factors (smallest adjusted p-value 0.20).
2. **The combination is robust.** An equal-risk blend of all six factors, defined before seeing results, earns a Sharpe ratio of 0.71 (t = 3.91), passing the Harvey, Liu & Zhu (2016) t > 3 threshold and a Bonferroni bound across all 30 strategies tested (critical t = 3.14).
3. **Diversification is the mechanism.** Factor returns are close to uncorrelated (mostly |ρ| < 0.2); value is negatively correlated with momentum and trend (−0.38). Combining halves volatility (diversification ratio 2.01) while keeping most of the return.
4. **Value and hedging pressure are the distinctive return sources.** They are the only factors whose alpha stays significant after controlling for all the others.
5. **Momentum and carry are weak in this sample.** AQR's independently built commodity momentum factor confirms the weakness (Sharpe 0.05 over the same period).
6. **The combination is defensive.** It returned between −6% and +6% in the 2008, 2014 to 2016 and 2020 sell-offs, while the commodity market fell 16% to 39%, and its performance barely depends on macro regimes.
7. **Long-only investors benefit too.** A long-only multi-factor tilt beat the equal-weight market by 2.7% a year (information ratio 0.53, t = 2.88) at similar volatility.
9. **Machine learning does not beat the simple blend.** Walk-forward ML models that tilt factor weights by market regime, select the top factors, or choose commodities by their own regime did not improve on the equal-risk blend or a simple signal average after costs (Section 14).
10. **Caveats.** Absolute performance is likely optimistic (front-month data bias), the combination's Sharpe ratio has declined across subperiods (1.21 → 0.74 → 0.48), and carry is limited by the absence of free full-curve data.

---

## 2. Repository contents and how to reproduce

### 2.1 Contents

| File | Purpose |
|---|---|
| `01_data_acquisition.ipynb` | Downloads and audits front-month prices (Yahoo) and NYMEX energy curves (EIA); applies data-quality rules |
| `02_returns.ipynb` | Derives the roll calendar from exchange expiry rules, removes roll gaps and builds clean excess returns |
| `03_signals.ipynb` | Builds all factor signals (momentum, TSMOM, carry, value, skewness, hedging pressure, basis-momentum) |
| `04_portfolios.ipynb` | Backtests long-short and long-only portfolios; costs, robustness, attribution, stability, crises, regimes |
| `05_analysis.ipynb` | Multi-factor combination, spanning regressions, return-cap robustness, AQR validation, multiple testing |
| `06_visualisations.ipynb` | Produces the institutional-style slides and the LinkedIn carousel PDF |
| `07_ml_factor_timing.ipynb` | **ML extension:** walk-forward ML and factor-momentum timing of the six factor portfolios by market regime |
| `08_ml_commodity_selection.ipynb` | **ML extension:** commodity-level ML learning which factor signals work in each commodity's own regime |
| `commodity_factor_investing_carousel.pdf` | Nine-page summary of the research (referenced as **Page 1 to Page 9** below) |

The repository deliberately contains **no data and no separate source files**. Every Python module is written to disk by the notebooks themselves through `%%writefile` cells, so running the six notebooks in order recreates the entire codebase (`src/commodity_factors/`), the cached data (`data/`) and every output (`reports/`).

### 2.2 Requirements

- Python 3.12 (developed on 3.12.15, Anaconda environment)
- Packages (versions used): `numpy 2.5.4`, `pandas 3.0.6`, `pyarrow 26.0.0`, `requests 2.34.2`, `yfinance 1.7.0`, `python-dotenv 1.2.4`, `matplotlib 3.11.2`, plus `openpyxl` (AQR spreadsheets; installed by notebook 01, Cell 1), `pillow` (PDF assembly; installed with matplotlib) and `scikit-learn 1.9.1` (notebooks 07 and 08; installed by notebook 07, Cell 1)
- A free EIA API key: [eia.gov/opendata/register.php](https://www.eia.gov/opendata/register.php)
- Internet access to Yahoo Finance, EIA, CFTC, FRED and AQR

```bash
pip install numpy pandas pyarrow requests yfinance python-dotenv matplotlib openpyxl scikit-learn jupyterlab
```

### 2.3 Running the project

1. Create an empty project folder, place the six notebooks in it, and start `jupyter lab` from that folder.
2. Run the notebooks **in order, top to bottom: 01 → 02 → 03 → 04 → 05 → 06**, then optionally **07 → 08** for the ML extension. Each notebook reads the files produced by the previous ones.
3. In notebook 01, Cell 15 prompts for the EIA key through a hidden input box and stores it in a git-ignored `.env` file. The key never appears in notebook output.
4. Run each notebook **once on a fresh folder**. Several cells append to modules (`%%writefile -a`), and notebook 02 Cell 6 patches `contracts.py` written by Cell 3. Re-running individual cells out of order can duplicate appended code or revert the patch.

Total runtime is roughly 10 to 20 minutes, dominated by the EIA (about 20 requests) and CFTC (11 annual files) downloads. All downloads are cached as Parquet, so later runs are fast.

**Data vintage.** The results in this README use data downloaded on 19 October 2026 (Yahoo prices to 16 October 2026, last complete month September 2026). Re-running later extends the sample automatically: the code always cuts to the last complete month. Numbers will therefore change slightly with new data and any vendor revisions.

### 2.4 Folder structure created by the notebooks

```
project/
├── 01_data_acquisition.ipynb ... 08_ml_commodity_selection.ipynb
├── .env                         # EIA key (git-ignored)
├── .gitignore                   # created by notebook 01
├── src/commodity_factors/       # Python package written by the notebooks
│   ├── config.py  contracts.py  returns.py  signals.py  risk.py  portfolio.py  analytics.py  viz.py
│   ├── ml.py  ml_commodity.py   # ML extension (notebooks 07 and 08)
│   └── data/  base.py  yahoo.py  eia.py  quality.py  cleaning.py  cftc.py  macro.py
├── data/
│   ├── raw/                     # vendor downloads (Yahoo, EIA, CFTC, FRED, AQR); git-ignored
│   └── processed/               # clean prices, returns, signals, factor returns, regimes; git-ignored
└── reports/
    ├── figures/                 # analysis charts
    ├── tables/                  # CSV result tables (16 core, plus ML-extension tables)
    └── linkedin/                # nine slide PNGs and the carousel PDF
```

---

## 3. Data

### 3.1 Sources

| Dataset | Source | Coverage | Used for |
|---|---|---|---|
| Front-month continuous futures prices (`=F` tickers) | Yahoo Finance | Jan 2000 to Oct 2026, daily | Returns for all 26 contracts; momentum, TSMOM, value, skewness, roll-yield carry |
| NYMEX futures prices, contracts 1 to 4 (CL, NG, RB, HO) | EIA API v2 | 1980s to 5 Apr 2024, daily | Validating roll timing; spot carry, basis-momentum; WTI April 2020 fix |
| Commitments of Traders (Legacy, futures only) | CFTC | 1986 to Oct 2026, weekly | Hedging pressure |
| CPI, VIX, NBER recession indicator | FRED | 1995 to 2026 | Regime analysis |
| US Dollar Index (`DX-Y.NYB`) | Yahoo Finance | 1995 to 2026 | Regime analysis |
| Commodity value, momentum and TSMOM factor returns | AQR Data Library | 1972 / 1985 to 2025 / 2026 | Independent validation |

**Why only free data.** The project intentionally avoids proprietary terminals so that anyone can reproduce it. This is a real constraint, not a cosmetic one: free sources provide only **front-month** prices for most contracts, not the full futures curve. Three consequences shape the design and are discussed throughout: roll handling must be inferred (Section 5.3), carry must be measured from realised roll gaps rather than the live curve (Section 6.2), and absolute performance is likely biased upwards (Section 13).

**EIA coverage ends in April 2024.** EIA stopped publishing NYMEX futures prices after April 2024, so EIA-based measures (spot carry, basis-momentum) run to March 2024 only. Nothing in the headline results depends on them.

### 3.2 Data-quality audit and cleaning (notebook 01)

Raw Yahoo data was audited contract by contract (`PriceQualityReport`): coverage, missing days, non-positive prices, daily moves above 20%, and runs of identical consecutive prices. Four problems were found and fixed by explicit, documented rules (`PriceCleaner`):

| Issue | Evidence | Rule applied |
|---|---|---|
| Oats placeholder prices | A repeated price of 134.00 every Sep to Dec from 2001 to 2003 (runs of up to 50 days); 28 to 36% missing days in 2000 to 2001 | Oats start **Jan 2004** |
| Live cattle stale prices | Recurring round-number runs (86.00, 84.50, 90.00) through 2004 to 2005 | Live cattle start **Jun 2005** |
| Platinum gaps | 15 to 58% of days missing each year 2002 to 2009 | Platinum starts **Jan 2010** |
| Palladium gaps | 31 to 59% of days missing 2006 to 2009 | Palladium starts **Jan 2010** |
| Isolated stale runs | One run each in KC wheat (2003) and lean hogs (2001) | Mask runs of **5+ identical prices** (18 observations) |
| Partial final session | The download day's intraday prices were not settlements | Exclude the current day from the Yahoo download |

After cleaning, stale runs fell from 16 to 0 and daily moves above 20% from 72 to 63, all remaining ones being roll gaps or genuine market events. Every change is recorded in `data/processed/cleaning_log.csv`.

---

## 4. Universe

Twenty-six liquid US futures contracts across five sectors:

| Sector | Contracts (symbol, exchange) | Listed delivery months used |
|---|---|---|
| **Energy** (5) | WTI crude (CL, NYMEX), Brent (BZ, NYMEX), Henry Hub natural gas (NG, NYMEX), NY Harbor ULSD / heating oil (HO, NYMEX), RBOB gasoline (RB, NYMEX) | All 12 months |
| **Metals** (5) | Gold (GC, COMEX), Silver (SI, COMEX), Copper (HG, COMEX), Platinum (PL, NYMEX), Palladium (PA, NYMEX) | GC: G J M Q V Z · SI, HG: H K N U Z · PL: F J N V · PA: H M U Z |
| **Grains** (8) | Corn (ZC), Chicago SRW wheat (ZW), KC HRW wheat (KE), Soybeans (ZS), Soybean meal (ZM), Soybean oil (ZL), Oats (ZO), Rough rice (ZR), all CBOT | ZC, ZW, KE, ZO: H K N U Z · ZS: F H K N Q U X · ZM, ZL: F H K N Q U V Z · ZR: F H K N U X |
| **Softs** (5) | Coffee (KC), Sugar No. 11 (SB), Cocoa (CC), Cotton (CT), Orange juice (OJ), all ICE US | KC, CC: H K N U Z · SB: H K N V · CT: H K N V Z · OJ: F H K N U X |
| **Livestock** (3) | Live cattle (LE), Lean hogs (HE), Feeder cattle (GF), all CME | LE: G J M Q V Z · HE: G J K M N Q V Z · GF: F H J K Q U V X |

Month codes: F Jan, G Feb, H Mar, J Apr, K May, M Jun, N Jul, Q Aug, U Sep, V Oct, X Nov, Z Dec.

**Coverage.** About 20 contracts have usable data in 2002, rising to 22 to 24 through the 2000s and all 26 from 2011 (Brent's Yahoo history starts in July 2007; the four cleaned contracts start later as above). All cross-sectional portfolios rank only the contracts available at each date, so the universe grows naturally over time without survivorship assumptions.

---

## 5. From raw prices to investable returns

This is the most important engineering step in the project. A futures "price history" is not an investable return series, and getting this wrong contaminates every factor built on top of it.

### 5.1 Theory: what a futures position earns

A fully collateralised long futures position earns

$$
r^{\text{total}}_t = \underbrace{r^{\text{spot}}_t + r^{\text{roll}}_t}_{\text{excess return}} + r^{\text{collateral}}_t
$$

- **Spot return:** the change in the underlying commodity price.
- **Roll yield:** the gain or loss from the futures price converging towards spot as expiry approaches. In **backwardation** (futures below spot) a long position gains; in **contango** (futures above spot) it loses.
- **Collateral return:** interest on the cash backing the position.

Factor research uses **excess returns** (spot plus roll), because a futures position requires no capital beyond margin. All returns in this project are excess returns; no risk-free rate is subtracted from Sharpe ratios.

The **theory of storage** (Kaldor, 1939; Working, 1949; Brennan, 1958) links the shape of the curve to inventories:

$$
F_{t,T} = S_t \, e^{(r + c - y)(T - t)}
$$

where $r$ is the interest rate, $c$ the storage cost and $y$ the **convenience yield**, the benefit of holding physical inventory. Low inventories mean a high convenience yield, a backwardated curve and volatile spot prices. The **theory of normal backwardation** (Keynes, 1930; Hicks, 1939) adds that producers hedge by selling futures and must pay speculators a premium to take the other side. These two ideas, scarcity and hedging demand, are the economic foundation for carry, hedging pressure and basis-momentum.

### 5.2 Why a stitched front-month series is not a return series

Yahoo's `=F` series holds the nearest contract until it expires, then switches to the next. On the switch day the "return" includes the **price gap between two different contracts**, which is not profit or loss. In contango the new contract is more expensive, so the stitched series jumps up; in backwardation it jumps down. Used naively, these gaps would add natural gas's contango drag back as fake gains, and would distort every return-based signal.

### 5.3 Determining Yahoo's roll convention empirically

Rather than guessing roll days from large returns, the convention was measured. For the four energy contracts, Yahoo's front-month price was matched against EIA's contract-by-contract prices every day (notebook 02, Cell 2):

| | CL | HO | NG | RB |
|---|---:|---:|---:|---:|
| Days Yahoo matches EIA contract 1 (within 0.25%) | 98.9% | 98.7% | 97.9% | 98.9% |
| Days Yahoo matches EIA contract 2 | 0.1% | 0.5% | 0.3% | 0.0% |

The contract-2 matches show no day-of-month pattern (they are days when the curve was flat), so **Yahoo holds the nearest contract until its last trading day and switches on the next trading day.**

### 5.4 A roll calendar from exchange expiry rules

Roll dates are therefore computed exactly from each exchange's last-trading-day rule, implemented as small strategy classes (`ExpiryRule` and four subclasses in `contracts.py`), applied to the **trading days actually present in the data** so exchange holidays are handled automatically:

| Rule class | Definition | Contracts |
|---|---|---|
| `BusinessDaysBeforeCalendarDay(25, −1, 3)` | 3 business days before the last business day on or before the 25th of the prior month | CL |
| `BusinessDaysBeforeMonthEnd(−2, 0)` | Last business day of the second month before delivery | BZ |
| `BusinessDaysBeforeMonthEnd(−1, 2)` | 3 business days before the first calendar day of the delivery month | NG |
| `BusinessDaysBeforeMonthEnd(−1, 0)` | Last business day of the month before delivery | HO, RB, SB |
| `BusinessDaysBeforeCalendarDay(14)` | Business day before the 15th of the delivery month | All grains |
| `BusinessDaysBeforeMonthEnd(0, k)` | k business days before the delivery month's last business day | KC (8), CC (11), CT (16), OJ (14), LE (0) |
| `NthBusinessDayOfMonth(10)` | 10th business day of the delivery month | HE |
| `LastWeekdayOfMonth(Thu)` | Last Thursday of the delivery month | GF |
| `BusinessDaysBeforeMonthEnd(−1, 1)` | Roll on the last business day before the delivery month (Yahoo convention, see below) | GC, SI, HG, PL, PA |

The last-trading-day rules were checked against actual 2024 expiries for CL, NG, HO, BZ, ZC and HE (all matched). The **roll date** is the first trading day after each last trading day: the day whose return spans two contracts.

**Validation.** For every contract, the median absolute return on days −3 to +3 around the computed roll dates was compared with its normal median (notebook 02, Cell 4). A correct calendar produces a spike at day 0 only:

- Grains: 2.0 to 3.6 times normal at day 0 (soybean oil 1.26); softs 1.5 to 3.0 times; livestock 2.4 to 7.7 times; energy 1.2 to 1.7 times, with neighbouring days close to 1.
- Lean hogs and feeder cattle show unusually quiet days just before expiry (0.3 to 0.6 times normal), consistent with expiring contracts converging to their cash-settlement index. This independently confirms that Yahoo holds them to expiry.
- Brent peaks at both day 0 (1.40) and day +1 (1.38): Yahoo sometimes switches a day late.

**The metals exception.** Metals showed no peak anywhere around their last trading days. Because metal curves are only in mild contango, roll gaps are too small for absolute returns to reveal, so a signed-return test was used instead: in contango every roll adds a small positive jump, so averaging signed returns across about 150 rolls gives a t-statistic on the true roll day (Cell 5). Pooling the five metals, the t-statistic peaks at the **first trading day of the delivery month (pooled t = 3.6)** and the day before (2.9), against at most 2.4 anywhere else. **Yahoo rolls metals at the month boundary before delivery**, around first notice day, not at expiry.

**Masking.** The return on each roll date is removed (set to missing). For Brent and metals, two days are masked to cover the observed timing uncertainty (`ROLL_MASK_DAYS`). Removing a roll-day return does not remove roll yield: roll yield is earned gradually through the ordinary daily returns as each contract converges to spot. Only the fake inter-contract jump is removed.

### 5.5 WTI in April 2020

The May 2020 WTI contract settled at **−$37.63** on 20 April 2020, the day before expiry, producing raw returns of −306% and −127%. A rolling investor would already have held the June contract, so between 1 and 21 April 2020 WTI returns are replaced with EIA contract 2 (June) returns (`ReturnOverride`, with the reason recorded in code). The clean series shows −18.4% on 20 April and −43.4% on 21 April, the June contract's actual moves.

### 5.6 Economic validation of the roll adjustment

The annual difference between raw (stitched) and clean returns measures each contract's average curve structure. The signs match known market behaviour in every sector (notebook 02, Cell 8):

| Contract | Raw ann. return | Clean ann. return | Roll gap p.a. | Interpretation |
|---|---:|---:|---:|---|
| Natural gas (NG) | 16.9% | −4.6% | +21.5% | Persistent contango: the drag that hurt natural gas ETFs |
| Chicago wheat (ZW) | 9.0% | −1.0% | +10.0% | Steep grain contango |
| Gold (GC) | 12.1% | 9.7% | +2.4% | Contango close to the financing rate |
| Soybean meal (ZM) | 6.9% | 21.6% | −14.7% | Persistent backwardation (old-crop / new-crop inversions) |
| Live cattle (LE) | 6.2% | 11.9% | −5.6% | Backwardation |
| RBOB gasoline (RB) | 14.3% | 22.9% | −8.6% | Seasonal backwardation |

The metals' consistent +1.8% to +2.4% also confirms the metals roll fix: wrongly placed masks would show roll gaps close to zero.

### 5.7 Outputs and the equal-weight benchmark

`ExcessReturnBuilder` produces clean daily returns, compounded monthly returns (a month needs at least 10 valid days) and a return index. The **equal-weight commodity benchmark** is the monthly average return of all available contracts (7.9% CAGR, Sharpe 0.64 from January 2000; Sharpe 0.74 over the February 2002 backtest period).

**Known limitation kept in the data.** Yahoo holds physically delivered contracts into their **delivery period**, when exchange price limits no longer apply. Some extreme moves (for example natural gas +46.5% on 27 January 2022, an expiry-day squeeze) occurred in contracts an index investor would already have rolled out of. These moves really happened in the contract held, so they were kept and tested separately with a ±20% daily cap (Section 9.5). This effect also inflates absolute returns for some contracts; see Section 13.

---

## 6. Factor theory and construction

### 6.1 Common conventions

- **Signal panel.** Every signal is a table of month-ends × contracts.
- **Point-in-time timing.** The value at month-end *t* uses only information available at the close of *t*; portfolios formed on it earn returns in month *t + 1*. This is enforced in one place (the `Signal` base class) and hand-verified for momentum and hedging pressure.
- **Direction.** For every factor, **a higher signal means go long.** Skewness is therefore stored as minus skewness.
- **Partial months.** The final month is dropped automatically if the source data ends before its last business day (September 2026 for Yahoo; March 2024 for EIA).

**Predictive preview (rank IC).** Before backtesting, each cross-sectional signal was evaluated by its monthly Spearman rank correlation with next-month returns (information coefficient), January 2002 to September 2026:

| Signal | Mean IC | t-stat | % months positive |
|---|---:|---:|---:|
| Skewness | 0.047 | 3.27 | 54.4 |
| Hedging pressure (demeaned) | 0.043 | 3.27 | 58.8 |
| Hedging pressure | 0.043 | 3.07 | 56.4 |
| Value (from Nov 2005) | 0.043 | 2.48 | 54.8 |
| Hedging pressure (speculators) | 0.035 | 2.65 | 59.1 |
| Carry (roll yield) | 0.032 | 2.21 | 55.1 |
| Momentum 12-1 | 0.015 | 0.95 | 50.3 |
| Momentum 12-0 | 0.010 | 0.64 | 53.4 |

### 6.2 Carry: trailing 12-month roll yield

**Theory.** Carry is the return earned if prices stay unchanged: the slope of the futures curve. Backwardation signals scarce inventory and heavy producer hedging; longs are compensated for bearing scarcity and supply-shock risk (Gorton, Hayashi & Rouwenhorst, 2013; Koijen et al., 2018). The textbook measure is spot carry:

$$
\text{Carry}_t = \ln\!\left(\frac{F_{1,t}}{F_{2,t}}\right) \times \frac{12}{\text{months between contracts}}
$$

**The data problem.** Spot carry needs the first two contracts every day. Free data provides them only for four energy contracts (EIA, to March 2024), which is far too few for a cross-sectional factor and has no statistical power (each contract would need a premium of about 1.2% a month to reach t = 2).

**First attempt: single roll gaps (rejected).** On each roll date, the stitched series jumps by ln(new / old contract), which *is* the front spread at that moment. Validated against EIA carry, single-roll estimates correlated only 0.59 (CL), 0.20 (HO), 0.50 (NG) and 0.20 (RB), with 54 to 65% sign agreement. Each gap contains one day's price noise, and it measures the expiring contract's spread rather than the month-end curve.

**Adopted measure: trailing 12-month roll yield.**

$$
\text{RollYield}_t = -\sum_{\text{rolls in } (t-12,\,t]} \ln\!\left(\frac{P^{\text{new}}}{P^{\text{old}}}\right)
$$

Summing all roll gaps over a year averages out the daily noise and, because a full year covers each contract pair once, **cancels seasonal curve shape without any adjustment**. Positive values mean the contract was rolled along a backwardated curve. Validated against EIA's trailing 12-month carry:

| | CL | HO | NG | RB |
|---|---:|---:|---:|---:|
| Correlation | 0.85 | 0.78 | 0.83 | 0.59 |
| Sign agreement | 78% | 78% | 83% | 70% |

This gives a carry signal for **all 26 contracts**, and its cross-section matches known structure: persistent contango in wheat (backwardated only 12% of months), natural gas and metals; persistent backwardation in soybean meal, RBOB, soybeans and cocoa.

**Trade-off.** The measure is slower than spot carry. It reflects the past year's curve, not today's. This explains carry's 2020 loss: energy curves had been backwardated through 2019, so the signal was still long energy into the COVID crash.

**Energy-only spot carry (robustness).** EIA spot carry is kept as a robustness check, both raw and **seasonally adjusted** (the spread minus its average for the same calendar month in prior years, using an expanding window of strictly past data and at least 3 prior years). Seasonal adjustment cut the seasonality of RB and NG carry by about 70% (calendar-month dispersion 0.40 → 0.12 and 0.36 → 0.13), but made HO noisier (0.089 → 0.123). Seasonally adjusted carry was fixed as the primary energy definition in advance, rather than choosing per contract after seeing results.

### 6.3 Cross-sectional momentum

**Theory.** Commodities that outperformed peers over the past year tend to keep outperforming, because investors underreact to slow-moving fundamental news and supply responds slowly (Miffre & Rallis, 2007; Asness, Moskowitz & Pedersen, 2013).

**Construction.** Cumulative excess return from month *t − 12* to *t − 1* (12-1, skipping the latest month), computed on the clean return index so roll gaps cannot contaminate it. A 12-0 variant (no skip) is a robustness check.

### 6.4 Time-series momentum (TSMOM)

**Theory.** Each contract's own past 12-month return predicts its next-month return (Moskowitz, Ooi & Pedersen, 2012). Unlike cross-sectional momentum, TSMOM can be net long or net short the whole asset class, which gives it "crisis alpha" in sustained sell-offs.

**Construction.** Signal = sign of the trailing 12-month excess return. Position sizing (volatility scaling) is applied at the portfolio stage (Section 7.2), keeping signals and sizing separate.

### 6.5 Value

**Theory.** Long-horizon price reversal: high prices attract new supply and destroy demand, low prices do the reverse, so prices mean-revert over three to five years. Behavioural overreaction to long trends reinforces it (Asness, Moskowitz & Pedersen, 2013).

**Construction.**

$$
\text{Value}_t = \ln\!\left(\frac{\overline{P}_{t-66,\,t-54}}{P_t}\right)
$$

the log of the average month-end price 4.5 to 5.5 years ago (at least 10 of 13 months) divided by today's price. It uses **unadjusted front-month price levels**, which approximate spot. Using the roll-adjusted index would be wrong here, because five years of accumulated roll adjustments distort the price level. The 5.5-year lookback means value starts in November 2005 (later for contracts with later data starts).

### 6.6 Skewness

**Theory.** Investors overpay for lottery-like, positively skewed payoffs, so high-skew commodities earn lower future returns; negatively skewed assets carry crash risk and must offer a premium (Fernandez-Perez, Frijns, Fuertes & Miffre, 2018).

**Construction.** Minus the skewness of daily clean excess returns over the trailing 252 trading days (at least 200 valid days). Roll days are already masked, so roll gaps cannot distort the skew.

### 6.7 Hedging pressure

**Theory.** The direct test of Keynes's normal backwardation: when commercial hedgers are heavily net short, they pay speculators an insurance premium through futures priced below the expected future spot price (de Roon, Nijman & Veld, 2000; Basu & Miffre, 2013).

**Data.** CFTC Legacy Commitments of Traders, futures only: the only report going back to 1986 (the Disaggregated report starts in 2006). Contracts are mapped by CFTC Contract Market Code, which stays fixed through exchange mergers and market renames; a market-name audit file confirms every mapping (for example, WTI's 2022 rename to "WTI-PHYSICAL" and the CSCE → NYBOT → ICE US transitions).

**Construction.**

$$
\text{HP}_t = \overline{\left(\frac{\text{Short}^{\text{comm}} - \text{Long}^{\text{comm}}}{\text{Short}^{\text{comm}} + \text{Long}^{\text{comm}}}\right)}_{\text{52 weeks}}
$$

averaged over the trailing 52 weekly reports (at least 40). Each report is dated **7 days after its Tuesday position date**: reports are published on Friday afternoon, after most commodity settlements, so a Friday month-end must not use that day's release, and holiday-delayed releases are also covered. The timing was hand-verified: the signal at 31 May 2024 (a Friday) correctly uses the 21 May report.

**Structural tilt check.** Commercials are net short gold 99.5% of weeks, so hedging-pressure *levels* carry a near-permanent long-precious-metals tilt. A **demeaned** variant (each contract minus its own expanding historical mean, strictly past data) was built to test this. Precious metals' share of the top third falls from 41% to 27%, yet the IC holds (0.043, t = 3.27): the signal reflects time-varying hedging demand, not just a metals bet. Levels remain the primary definition (the published specification, fixed in advance); demeaned and speculator-based versions are robustness checks.

**Known limitation.** In the Legacy report, "commercial" includes swap dealers hedging index investors. This explains why Brent and natural gas commercials are often net long.

### 6.8 Basis-momentum (energy robustness)

**Theory.** The 12-month return of the front contract minus that of the second contract captures changes in the front of the curve that carry and momentum miss (Boons & Prado, 2019).

**Construction.** Requires both contracts' returns, so it covers only the four EIA energy contracts (December 2000 to March 2024). Roll days are masked in both legs, and days where either leg has a non-positive price are dropped from both so the difference stays unbiased. Values before December 2000 were trimmed because the roll calendar starts in 2000. With four contracts the test has almost no power; it is reported only as context.

### 6.9 Signal overlap

Average monthly cross-sectional rank correlation between signals (2002 to 2026):

| | Momentum | Carry | Value | Skewness | Hedging pressure |
|---|---:|---:|---:|---:|---:|
| Momentum | 1.00 | 0.43 | −0.38 | 0.04 | 0.32 |
| Carry | | 1.00 | −0.13 | 0.08 | 0.18 |
| Value | | | 1.00 | −0.17 | −0.30 |
| Skewness | | | | 1.00 | 0.21 |

Momentum and carry overlap (0.43), consistent with storage theory: backwardated, low-inventory commodities tend to be the ones whose prices have been rising. Value is negatively related to momentum and hedging pressure, and skewness is nearly independent of everything.

---

## 7. Portfolio construction

### 7.1 Cross-sectional factors (momentum, carry, value, skewness, hedging pressure)

`QuantileLongShort`: each month, rank the available contracts on the signal; go **long the top third and short the bottom third**, equal-weighted within each leg. Each leg sums to 100%, so portfolios are **dollar-neutral** with gross exposure of 200% (construction check: about 8.5 long and 7.7 short positions on average, gross exposure 2.00, net exposure 0.00). At least 9 contracts must have a signal for a position to be taken.

### 7.2 Time-series momentum

`VolScaledTimeSeries`: each contract is held long or short on the sign of its own signal, with weight

$$
w_{i,t} = \text{sign}(s_{i,t}) \times \frac{40\%}{\sigma_{i,t}} \times \frac{1}{N_t}
$$

where $\sigma_{i,t}$ is ex-ante annualised volatility and $N_t$ the number of contracts with a signal (Moskowitz, Ooi & Pedersen, 2012). Without scaling, natural gas (about 57% volatility) would dominate live cattle (about 15%). Ex-ante volatility (`EWMAVolatility`) is the exponentially weighted standard deviation of daily returns with a 60-day centre of mass, sampled at month-end using data to that date only. TSMOM averages 1.62 gross exposure with a mild net long tilt (0.22).

### 7.3 Backtest mechanics and costs

`Backtester` applies weights set at month-end *t* to returns in month *t + 1*, labelling results by the month the return is earned. It reports gross and net returns plus the **long leg** and **short leg** separately (the short leg is the return of the shorted basket, so the strategy profits when it is negative).

Costs are charged at **5 bps per unit of turnover** on two components:

- **Rebalancing turnover:** sum of absolute weight changes (2 to 9 times gross exposure a year).
- **Roll turnover:** every held contract must be rolled into the next expiry. Estimated as $\sum_i |w_i| \times (\text{rolls per year}_i / 12)$, which is 11 to 14 times gross exposure a year, larger than rebalancing.

The headline results apply 5 bps to both. Liquid US futures typically cost about 1 to 3 bps for a calendar-spread roll and 2 to 5 bps for outright trades, so 5 bps on everything is deliberately conservative.

### 7.4 Multi-factor combination (pre-specified)

The primary combination blends **all six primary factors** with weights inversely proportional to each factor's trailing 36-month volatility (lagged one month, at least 24 months), summing to one. It starts in February 2004 once two years of factor history exist, and value joins when its returns begin. The combination, its weighting and its membership were fixed **before** seeing the factor results, so it counts as a single test. Two comparisons are reported: a simple 1/N blend, and an equal-risk blend of only the four strongest factors, explicitly labelled as **selected in-sample** because it uses hindsight.

### 7.5 Long-only tilts

For investors who cannot short, each cross-sectional factor's long leg (the equal-weighted top third) is compared with the equal-weight benchmark, net of 5 bps rebalancing costs. Roll costs are excluded because the benchmark rolls the same contracts. The **multi-factor tilt** holds equal parts of the five top-third baskets.

---

## 8. Statistical methodology

**Newey-West t-statistics.** Factor returns are autocorrelated, especially for slow-moving signals such as value and hedging pressure, so ordinary t-statistics overstate significance. All t-statistics on mean returns and regression coefficients use Newey-West (1987) heteroskedasticity- and autocorrelation-consistent standard errors with a Bartlett kernel and $\lfloor 4(T/100)^{2/9} \rfloor$ lags (5 lags for about 300 months). The regression implementation (`ols_newey_west`) was verified against statsmodels' HAC estimator to four decimals.

**Spanning regressions.** Two tests (notebook 05):

- **Market exposure:** $r_f = \alpha + \beta \, r_{\text{market}} + \varepsilon$. Is the factor just disguised long commodity exposure?
- **Redundancy:** $r_f = \alpha + \sum_{g \ne f} \beta_g r_g + \beta_m r_{\text{market}} + \varepsilon$. Does the factor earn alpha the other five factors cannot explain? Common sample from November 2005.

**Multiple testing.** Testing many factors and variants guarantees that some look significant by chance. Three corrections are applied:

- **Holm-Bonferroni** across the six primary factors, controlling the probability of any false positive.
- **Bonferroni** across all 30 strategy return series tested in notebooks 04 to 05 (critical |t| = 3.14 at 5%).
- **Harvey, Liu & Zhu (2016):** t > 3.0 as the hurdle for a new factor, reflecting the hundreds of factors already tested in the literature.

**Pre-specification.** To avoid data-mining, the following were fixed before results were seen and never changed afterwards: the primary signal definitions (12-1 momentum, 52-week hedging pressure levels, seasonally adjusted spot carry for energy), the base construction (equal-weighted terciles), and the six-factor equal-risk combination. Every variant is reported as robustness, never promoted to primary because it performed better.

---

## 9. Results

### 9.1 Single factors (Page 3)

Net of 5 bps on rebalancing and rolls, February 2002 to September 2026:

| Factor | Ann. return | Ann. vol | Sharpe | t (NW) | Max DD | Skew | Hit rate |
|---|---:|---:|---:|---:|---:|---:|---:|
| Momentum 12-1 | 2.1% | 18.9% | 0.11 | 0.62 | −66% | −0.11 | 51% |
| Carry | 3.0% | 17.8% | 0.17 | 0.97 | −66% | −0.35 | 52% |
| Value (Nov 2005) | 8.3% | 18.6% | 0.44 | 2.12 | −40% | −0.01 | 59% |
| Skewness | 8.1% | 17.7% | 0.46 | 2.13 | −47% | 0.31 | 53% |
| Hedging pressure | 5.6% | 16.7% | 0.34 | 1.44 | −51% | −0.47 | 58% |
| TSMOM | 5.0% | 12.4% | 0.40 | 2.03 | −39% | 0.08 | 57% |
| *Equal-weight market* | *9.9%* | *13.4%* | *0.74* | *3.33* | *−39%* | *−0.35* | *63%* |

**Short legs rose too.** Every shorted basket gained (+2.9% to +8.4% a year) because commodities trended up over the period and the front-month data bias lifts all contracts. Factors earn their spread because the shorted contracts rose *less* than the longs; value and skewness had the weakest short legs.

**Hedging pressure and carry are weaker as portfolios than as ICs.** Newey-West correctly penalises hedging pressure's slow, autocorrelated returns (2.2 times annual turnover); carry's portfolio is concentrated in a few persistent sector bets (Section 9.6).

**Year-level robustness.** Excluding each factor's two best years, the remaining years still sum to +121% (skewness), +109% (value), +87% (TSMOM) and +82% (hedging pressure). Momentum's result depends on two years (2013 and 2024): without them the remaining years sum to −15%.

### 9.2 Multi-factor combination (Pages 2 and 4)

February 2004 to September 2026, net of costs:

| | Ann. return | Ann. vol | Sharpe | t (NW) | Max DD | Skew |
|---|---:|---:|---:|---:|---:|---:|
| **Equal-risk, six factors (primary)** | 6.0% | 8.4% | **0.71** | **3.91** | **−13.8%** | −0.39 |
| Equal-weight (1/N), six factors | 5.7% | 8.4% | 0.68 | 3.82 | −12.8% | −0.50 |
| Strong four (in-sample, for comparison only) | 7.1% | 8.3% | 0.85 | 4.35 | −10.3% | 0.13 |
| Equal-weight commodity market | 8.5% | 13.6% | 0.63 | 2.73 | −39.1% | −0.29 |

- **Correlations are low.** Monthly factor return correlations are mostly below 0.2; value correlates −0.38 with both momentum and TSMOM, and momentum and TSMOM correlate 0.59. The **diversification ratio is 2.01**: average single-factor volatility of 16.9% falls to 8.4% in the combination.
- **The result does not depend on weighting or on dropping weak factors.** 1/N gives almost the same answer, and hindsight selection adds only modestly.
- **Cumulative return is lower than the market's, but risk-adjusted performance is higher.** The 2002 to 2026 commodity boom favoured simply being long; the combination's advantage is a higher Sharpe ratio with about a third of the drawdown and almost no market correlation (0.12).

### 9.3 Spanning regressions (Page 3, "Unique α" column)

| Factor | Alpha vs market (t) | Market beta | Alpha vs other factors, % p.a. (t) | Main overlaps |
|---|---:|---:|---:|---|
| Momentum | 0.74 | −0.03 | 1.4% (0.47) | TSMOM β = 0.75 (t = 7.7), R² 0.46: **redundant** |
| Carry | 0.56 | 0.11 | −2.4% (−0.60) | Momentum, value, hedging pressure: **redundant** |
| Value | 1.96 | −0.02 | **10.7% (2.87)** | Negative loadings on momentum and TSMOM: **unique, hedges trend** |
| Skewness | 2.12 | −0.02 | 3.7% (0.80) | Hedging pressure (t = 2.6), TSMOM (t = 2.0); also loses its strong 2002 to 2004 years in the common sample |
| Hedging pressure | 1.65 | −0.06 | **7.7% (2.02)** | Mild overlap with skewness: **unique** |
| TSMOM | 1.16 | 0.24 | 2.2% (1.14) | Shares its return with momentum |
| **Combination** | **3.35** | 0.08 | – | – |

Market betas are near zero (R² ≈ 0) for every cross-sectional factor: the premia are not disguised long commodity exposure. Only TSMOM has a meaningful beta (0.24), from its net long tilt.

### 9.4 Statistical evidence after multiple testing (Page 3)

| | t (NW) | p-value | Holm-adjusted p | Passes t > 3 |
|---|---:|---:|---:|---|
| Momentum | 0.62 | 0.53 | 0.66 | No |
| Carry | 0.97 | 0.33 | 0.66 | No |
| Value | 2.12 | 0.03 | 0.20 | No |
| Skewness | 2.13 | 0.03 | 0.20 | No |
| Hedging pressure | 1.44 | 0.15 | 0.45 | No |
| TSMOM | 2.03 | 0.04 | 0.20 | No |
| **Combination (single pre-specified test)** | **3.91** | **< 0.001** | – | **Yes** |

Across all 30 strategies tested, 8 exceed |t| = 1.96 against about 1.5 expected if none were real. The variants are not independent, so this overstates the evidence somewhat, but the clustering around value, skewness and hedging pressure is consistent with real premia that are individually too weak to pin down. Only one strategy besides the combination passes the 30-strategy Bonferroni bound: sector-neutral value (t = 3.28), which is a construction variant and therefore reported as robustness, not as a headline.

### 9.5 Robustness (Page 9)

**Construction variants** (net Sharpe, rebalancing costs only):

| Factor | Base (terciles, equal) | Inverse-vol weights | Sector-neutral | Quintiles |
|---|---:|---:|---:|---:|
| Value | 0.48 | 0.23 | **0.69** | 0.48 |
| Skewness | 0.50 | 0.37 | 0.45 | **0.60** |
| Hedging pressure | 0.38 | **0.50** | 0.44 | 0.27 |
| Carry | 0.21 | 0.27 | 0.35 | 0.18 |
| Momentum | 0.15 | 0.26 | −0.06 | −0.01 |

Skewness, value and hedging pressure are robust to construction. Value works best *within* sectors, and its return comes largely from volatile contracts that inverse-volatility weighting downweights. Momentum is not robust: sector-neutral ranking turns it negative, so what little return it has comes from rotating between sectors.

**Signal-definition variants:** momentum 12-0 has a Sharpe of −0.03 (vs 0.11 for 12-1); demeaned hedging pressure 0.34 (vs 0.38, return correlation 0.70, lower drawdown); speculator-based hedging pressure is much weaker (0.15), consistent with theory, since hedgers pay the premium. On the four energy contracts, trailing roll-yield carry lost money as a time-series signal (Sharpe −0.44, t = −1.85) while spot EIA carry was flat to slightly positive (0.00 adjusted, 0.13 raw): energy curves change shape within weeks, and a 12-month measure keeps positions long after the curve has turned. No energy-only result is statistically meaningful (all |t| < 2).

**±20% daily return cap.** Capping every daily return at ±20% (34 of 157,044 observations) and rebuilding all return-based signals, volatility estimates and portfolios leaves the combination unchanged (Sharpe 0.71 → 0.71, t = 3.91 → 3.85, return correlation 0.99). Skewness moves most (0.46 → 0.42), consistent with its 2020 energy windfall; hedging pressure slightly improves (0.34 → 0.39).

**Transaction costs:**

| Factor | Sharpe at 0 bps | 5 bps | 10 bps | 20 bps | Break-even |
|---|---:|---:|---:|---:|---:|
| Value | 0.50 | 0.44 | 0.39 | 0.28 | 45.8 bps |
| Skewness | 0.52 | 0.46 | 0.40 | 0.28 | 43.0 bps |
| Hedging pressure | 0.38 | 0.34 | 0.29 | 0.20 | 41.0 bps |
| TSMOM | 0.47 | 0.40 | 0.33 | 0.19 | 33.5 bps |
| Carry | 0.23 | 0.17 | 0.11 | 0.00 | 20.0 bps |
| Momentum | 0.17 | 0.11 | 0.05 | −0.07 | 14.2 bps |

The four strong factors have roughly 4 to 10 times headroom over realistic costs; carry and momentum would lose most of their small edge.

### 9.6 Attribution and concentration

Gross return contribution by sector (% per year):

| Factor | Energy | Grains | Livestock | Metals | Softs | Total |
|---|---:|---:|---:|---:|---:|---:|
| Momentum | −2.78 | 2.10 | −0.76 | 2.57 | 2.16 | 3.29 |
| Carry | −1.98 | 4.67 | 0.31 | 1.21 | −0.18 | 4.03 |
| Value | 5.01 | 3.19 | −0.43 | −2.95 | 4.46 | 9.29 |
| Skewness | 5.66 | −1.63 | 0.29 | 2.63 | 2.23 | 9.17 |
| Hedging pressure | −1.72 | 2.69 | −1.30 | 4.12 | 2.62 | 6.41 |
| TSMOM | 0.92 | 2.22 | 1.43 | 1.40 | −0.16 | 5.82 |

| Factor | Top-3 contracts' share of total return | Profitable contracts |
|---|---:|---:|
| TSMOM | 41% | 19 of 26 |
| Value | 52% | 17 |
| Skewness | 60% | 16 |
| Hedging pressure | 65% | 17 |
| Momentum | 100% | 15 |
| Carry | 118% | 16 |

- **Carry is a soybean-complex bet:** soybean meal (+2.2% a year) and soybeans (+1.4%) drive it, while natural gas costs −1.5%; the other 23 contracts lost money in aggregate.
- **Skewness's 2020 windfall** (+67.7%) came almost entirely from shorting energy (RB, CL, BZ and HO contributed 65.6 points) before the COVID oil collapse.
- **Hedging pressure's 2023 to 2025 run** (+117.6% of contributions) was spread across four sectors: natural gas, cocoa, silver, orange juice and platinum.
- **Value loses on metals** (−3.0% a year): it was short gold, silver and copper as expensive while they kept rising.

### 9.7 Stability and drawdowns

| Factor | 3-year windows with positive Sharpe | Worst / best 3-year Sharpe | Max DD | Longest underwater |
|---|---:|---:|---:|---:|
| Value | 90% | −1.07 / 2.29 | −40% | 43 months |
| TSMOM | 77% | −1.25 / 1.62 | −39% | 110 months |
| Carry | 71% | −1.05 / 1.50 | −66% | 89 months |
| Hedging pressure | 70% | −1.01 / 3.06 | −51% | 110 months |
| Skewness | 67% | −1.34 / 1.64 | −47% | 163 months |
| Momentum | 64% | −1.45 / 1.30 | −66% | 144 months |

Every factor had a losing stretch long enough to make a single-factor investor give up. Skewness spent 13.6 years below a previous peak. The factors' good periods rarely overlap (value peaked in 2018 to 2019 when skewness was weak; skewness and hedging pressure surged when value fell), which is exactly what the combination exploits. **The combination's subperiod Sharpe ratios are 1.21 (2004 to 2008), 0.74 (2009 to 2015) and 0.48 (2016 to 2026)**: positive throughout, but declining (Page 9).

### 9.8 Crisis behaviour (Page 5)

Cumulative net return over each episode (%):

| Episode | Market | **Combination** | Value | Hedging pressure | Skewness | TSMOM | Carry | Momentum |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| GFC crash, Jul 2008 to Feb 2009 | −39.1 | **+5.9** | +31.0 | +47.9 | −4.2 | −13.2 | −11.5 | −13.2 |
| Oil collapse, Jul 2014 to Jan 2016 | −26.9 | **−0.1** | +4.3 | −12.9 | −5.8 | +12.6 | −3.2 | −4.4 |
| COVID crash, Jan to Apr 2020 | −16.3 | **−5.5** | −28.2 | −8.9 | +30.4 | +2.4 | −29.0 | +6.4 |
| Avg. in market's worst 10% of months | −6.1 | **−0.4** | +1.0 | +0.4 | +0.2 | −0.8 | −1.5 | −0.8 |

Value and hedging pressure hedged the 2008 crash; TSMOM protects in slow, trending sell-offs (2014 to 2016) but not sudden ones (2008); carry has crash risk, consistent with the carry literature. The combination stayed close to flat in all three episodes.

### 9.9 Regimes (Page 6)

Six regimes, each defined only with information available before the month: rising vs falling commodity market (trailing 12-month benchmark return), high vs low commodity volatility (vs its own expanding median), CPI above vs below 3% (lagged two months for publication), US dollar strengthening vs weakening (trailing 12 months), VIX above vs below 25, and NBER recessions (dated ex post, descriptive only, 20 months).

Difference in annualised net return between the two states (percentage points):

| | Rising mkt | High vol | CPI > 3% | USD up | VIX > 25 | Recession* |
|---|---:|---:|---:|---:|---:|---:|
| **Combination** | −1.2 | +0.5 | +6.3 | +1.1 | −0.6 | −0.4 |
| Value | −5.8 | **+18.2** | +8.0 | −1.1 | +10.6 | −8.5 |
| Hedging pressure | −15.1 | +7.0 | +11.9 | +9.3 | +1.8 | +22.4 |
| Skewness | −8.1 | −3.0 | −0.6 | +6.5 | +3.4 | −12.5 |
| TSMOM | +10.7 | −4.1 | +0.7 | +0.1 | −7.9 | −0.6 |
| Carry | +2.3 | +2.3 | +12.6 | −4.2 | −11.0 | −16.6 |
| Momentum | +11.0 | −7.1 | +7.4 | −3.5 | −11.4 | −2.0 |

Only 1 of 36 factor-regime differences reaches |t| ≥ 1.96 (value in high-volatility regimes, t = 2.3), roughly what chance alone would produce, so these are tendencies. They are economically sensible and complementary: hedging pressure leans towards falling markets (when hedging demand is highest), value towards volatile markets (larger mispricings), and trend towards rising markets. **The combination's differences stay within −1.2 to +6.3 points, none significant: the factors' regime tendencies offset.**

### 9.10 Independent validation against AQR (Page 7)

| Our factor vs AQR | Overlap | Correlation | Our Sharpe (gross) | AQR Sharpe |
|---|---|---:|---:|---:|
| Momentum vs AQR commodity momentum | Feb 2002 to Jan 2025 | 0.75 | 0.18 | 0.05 |
| Value vs AQR commodity value | Nov 2005 to Jan 2025 | 0.79 | 0.55 | 0.33 |
| TSMOM vs AQR commodity TSMOM | Feb 2002 to May 2026 | 0.75 | 0.48 | 0.35 |

Correlations of 0.75 to 0.79 with independently constructed factors (different contract universe and weighting) confirm the pipeline measures the same premia. AQR's own commodity momentum is even weaker than ours, confirming that weak momentum is a feature of the period, not a bug. **Our Sharpe ratios are higher in all three cases**, consistent with the front-month data bias (Section 13): the ranking of factors and the conclusions match AQR, but absolute levels should be treated as optimistic.

### 9.11 Long-only investors (Page 8)

| | Ann. return | Volatility | Sharpe | Max drawdown |
|---|---:|---:|---:|---:|
| **Long-only multi-factor tilt** | 12.5% | 14.2% | 0.89 | −33% |
| Equal-weight commodity market | 9.9% | 13.4% | 0.74 | −39% |

| Tilt vs market | Excess p.a. | Tracking error | Information ratio | t (NW) | Beta |
|---|---:|---:|---:|---:|---:|
| Skewness | 3.5% | 9.8% | 0.36 | 1.70 | 1.02 |
| Value | 3.6% | 10.7% | 0.33 | 1.56 | 0.96 |
| Hedging pressure | 2.8% | 9.6% | 0.30 | 1.33 | 0.89 |
| Carry | 2.4% | 10.0% | 0.24 | 1.29 | 1.07 |
| Momentum | 1.6% | 10.2% | 0.16 | 0.84 | 1.01 |
| **Multi-factor tilt** | **2.7%** | **5.1%** | **0.53** | **2.88** | – |

Every single tilt beat the market with beta close to 1, so the excess came from selection rather than extra market exposure, but none was significant alone. Combining halved tracking error (about 10% to 5%) and made the outperformance statistically significant.

---

## 10. Notebook guide

Each notebook follows the same pattern: modules are written to `src/commodity_factors/` with `%%writefile`, then used in analysis cells that print verifiable diagnostics. Every cell header states its purpose (for example `# 02 · Cell 6 — Apply metals roll fix and roll-mask settings`).

### Notebook 01: Data acquisition (17 cells)

| Cells | Content |
|---|---|
| 1–2 | Install and verify dependencies; create the folder structure, package skeleton and `.gitignore` |
| 3 | `config.py`: project paths, the 26-contract universe (immutable `Commodity` and `Universe` dataclasses with sector lookups), EIA curve series IDs |
| 4 | Import setup (`sys.path`, autoreload) and universe check |
| 5–6 | `CachedDataSource` (abstract base: download once, cache to Parquet, standardise the index) and `YahooFuturesSource` (batched download; excludes the partial current session) |
| 7 | Download front-month prices; per-contract coverage |
| 8–10 | `PriceQualityReport`; quality diagnostics; drill-down into missing days by year and stale runs (oats, cattle, platinum, palladium) |
| 11–13 | Append data-quality rules to the config; `PriceCleaner` (start-date cut-offs and stale-run masking, with an audit log); apply and re-check |
| 14–15 | `EIACurveSource` (API v2 with pagination, retries and backoff); secure key entry via `getpass` into `.env`; download and audit contracts 1 to 4 |
| 16–17 | Carry preview (raw annualised front/next spread; seasonality is visible in RB and NG); summary of outputs |

### Notebook 02: Returns and roll handling (9 analysis cells)

| Cells | Content |
|---|---|
| 1 | Load clean prices and EIA curves |
| 2 | Diagnose Yahoo's roll convention against EIA contracts 1 and 2 |
| 3 | `contracts.py`: `TradingCalendar`, the `ExpiryRule` strategy classes, `ContractSpec` for all 26 contracts, `RollCalendar` |
| 4 | Roll-day profile test for every contract (median absolute return at offsets −3 to +3) |
| 5 | Signed-return scan locating the metals roll day |
| 6 | Patch the metals rule to the month boundary and add `ROLL_MASK_DAYS` (idempotent); verify rolls per year match listed months |
| 7 | `returns.py`: `ReturnOverride`, `ReturnSet`, `ExcessReturnBuilder` |
| 8 | Build clean returns with the WTI April 2020 override; raw vs clean comparison; remaining large moves |
| 9 | Coverage by year, equal-weight overall and sector benchmarks, raw vs clean cumulative chart |

### Notebook 03: Factor signals (19 cells)

| Cells | Content |
|---|---|
| 1 | Load returns, prices, curves; define the backtest start |
| 2–3 | `Signal` base class, `MomentumSignal`, `TimeSeriesMomentumSignal`; trim the partial month; hand-verified timing check; IC preview |
| 4–5 | `CarrySignal` (EIA spot carry, optional seasonal adjustment); raw vs adjusted seasonality and time-series preview |
| 6–9 | Roll-implied carry: single-gap prototype (rejected), trailing 12-month roll yield (validated against EIA), `RollYieldSignal`, consistency check, subperiod ICs, factor correlations |
| 10–11 | `ValueSignal`, `SkewnessSignal`; coverage, latest rankings, ICs, correlations |
| 12–16 | CFTC market codes; `CFTCLegacySource` (history plus annual zip files, market-name audit); download and audit; `HedgingPressureSignal` with release lag; timing check, ICs |
| 17–19 | `BasisMomentumSignal`; demeaned hedging-pressure test; full correlation matrix and IC summary; save `signals.parquet` |

### Notebook 04: Portfolios (18 cells)

| Cells | Content |
|---|---|
| 1 | Load data; define primary and robustness factor sets |
| 2–4 | `risk.py` (`EWMAVolatility`), `portfolio.py` (`QuantileLongShort`, `VolScaledTimeSeries`, `Backtester`), `analytics.py` (`newey_west_tstat`, `PerformanceReport`) |
| 5–6 | Backtest the six primary factors; construction check; gross and net performance; legs; cumulative chart and calendar-year returns |
| 7–8 | Construction robustness (inverse-vol, sector-neutral, quintiles); signal-definition robustness (12-0 momentum, hedging-pressure variants, energy time-series signals) |
| 9–11 | Cost sensitivity with roll turnover and break-even costs; long-only tilts; headline net returns and saved result sets |
| 12–14 | Sector and contract attribution; rolling 3-year Sharpe and drawdowns; crisis event study and tail behaviour |
| 15–18 | `macro.py` (`FREDSource`, `YahooSeriesSource`); regime indicators with strict lags; performance by regime with Welch t-tests and heatmap; save 10 CSV tables |

### Notebook 05: Analysis and validation (8 cells)

| Cells | Content |
|---|---|
| 1 | Load all saved results; check they reproduce notebook 04 |
| 2 | Factor return correlations; equal-risk, 1/N and in-sample combinations; diversification ratio; subperiod Sharpe ratios; long-only multi-factor tilt |
| 3–4 | `ols_newey_west` (verified against statsmodels); spanning regressions vs the market and vs the other factors |
| 5 | ±20% daily return cap, rebuilding return-based signals, volatility and portfolios |
| 6 | Download and parse AQR spreadsheets (robust to text dates and unlabelled headers); correlation and Sharpe comparison |
| 7–8 | Multiple-testing adjustments (Holm, Bonferroni across 30 strategies, Harvey-Liu-Zhu); final summary table |

### Notebook 06: Visualisations

Writes `viz.py` (`SlideStyle` for a single palette, typography and factor colour mapping; `Slide` for the shared header, content area and footer; `build_carousel`), then builds each 1080 × 1350 slide from the saved results, with every number in titles and subtitles computed from the data rather than typed in. The final cell assembles the nine PNGs into `commodity_factor_investing_carousel.pdf`.


### Notebook 07: ML factor timing (9 cells)

| Cell | Content |
|---|---|
| 1 | Install scikit-learn; load factor returns, combination returns, benchmarks, regimes, FRED and DXY data |
| 2 | `ml.py`: `FactorFeatureBuilder` |
| 3 | Build the factor panel; coverage, timing check, cross-factor ICs, market-feature correlations |
| 4 | `WalkForwardPredictor`: expanding-window yearly retraining, factor dummies, optional market × factor interactions |
| 5 | Risk-adjusted relative target; ridge, gradient boosting and factor-momentum predictions; IC and spread evaluation |
| 6 | `FactorTiltPortfolio`: equal-risk base, rank-based tilt, turnover costs |
| 7 | Portfolio comparison, improvement over equal-risk, subperiods, tilt sensitivity |
| 8 | Charts, live weights, saved tables |
| 9 | Top-2 / top-3 factor selection |

### Notebook 08: ML commodity selection (8 cells)

| Cell | Content |
|---|---|
| 1 | Load signals, returns, prices, macro; signal coverage |
| 2 | `ml_commodity.py`: `CommodityFeatureBuilder` (signals, own regime, market, sector, target) |
| 3 | Build the panel; timing check; baseline, conditional and sector ICs |
| 4 | `CommodityWalkForward`: yearly expanding window, signal × regime / market / sector interactions |
| 5 | Ridge, gradient boosting and composite predictions; IC, tercile spreads, prediction correlations |
| 6 | Long-short tercile portfolios with rebalancing and roll costs; head-to-head tests; subperiods |
| 7 | Interpretability: ridge interaction coefficients, gradient-boosting conditional effects and importances |
| 8 | Charts, conclusions, saved tables |

**Running notebooks 07 and 08.** Run notebooks 01 to 05 first (notebook 07 reads their factor returns, regimes and macro data; notebook 08 also reads the signals). Notebook 07 installs scikit-learn (developed on version 1.9.1). As with the core notebooks, run each once, top to bottom: cells marked `%%writefile -a` append to modules. Outputs: `reports/tables/07_*.csv`, `08_*.csv`, `reports/figures/07_factor_timing_oos.png`, `08_commodity_ml_oos.png`, and predictions and returns in `data/processed/`.

---

## 11. Code architecture

The project is organised as an installable-style package written by the notebooks, with the notebooks acting only as orchestration and reporting layers. Design principles: one class per responsibility, immutable configuration, strategy objects for interchangeable rules, and no logic duplicated between notebooks.

| Module | Key components | Responsibility |
|---|---|---|
| `config.py` | `ProjectPaths`, `Sector`, `Commodity`, `Universe`, `CurveSeries`, data-quality rules, `ROLL_MASK_DAYS`, `CFTC_MARKET_CODES` | Single source of truth for paths, universe and settings |
| `data/base.py` | `CachedDataSource` (ABC) | Template method: download once, cache to Parquet, standardise dates |
| `data/yahoo.py`, `eia.py`, `cftc.py`, `macro.py` | `YahooFuturesSource`, `EIACurveSource`, `CFTCLegacySource`, `FREDSource`, `YahooSeriesSource` | One subclass per vendor; retries, pagination, audit files |
| `data/quality.py`, `cleaning.py` | `PriceQualityReport`, `PriceCleaner`, `CleaningResult` | Diagnostics and rule-based, auditable cleaning |
| `contracts.py` | `TradingCalendar`, `ExpiryRule` + 4 subclasses, `ContractSpec`, `RollCalendar` | Exchange expiry rules and roll dates |
| `returns.py` | `ExcessReturnBuilder`, `ReturnOverride`, `ReturnSet` | Roll-masked daily and monthly excess returns |
| `signals.py` | `Signal` (ABC) + 7 signal classes | Point-in-time factor signals with a shared timing contract |
| `risk.py` | `EWMAVolatility` | Ex-ante volatility for sizing |
| `portfolio.py` | `PortfolioConstructor` (ABC), `QuantileLongShort`, `VolScaledTimeSeries`, `Backtester`, `BacktestResult` | Portfolio construction and backtesting |
| `analytics.py` | `newey_west_tstat`, `ols_newey_west`, `PerformanceReport` | Statistics with HAC standard errors |
| `viz.py` | `SlideStyle`, `Slide`, `build_carousel` | Consistent institutional chart template |
| `ml.py` | `FactorFeatureBuilder`, `WalkForwardPredictor`, `FactorTiltPortfolio` | Factor-level features, walk-forward ML and tilted factor portfolios (notebook 07) |
| `ml_commodity.py` | `CommodityFeatureBuilder`, `CommodityWalkForward` | Commodity-level regime features and walk-forward ML (notebook 08) |

Before each module was used, its logic was tested on synthetic data with known answers (for example a deliberately predictive signal to confirm the long leg earns and returns are labelled in the correct month), and key outputs were cross-checked by hand inside the notebooks.

---

## 12. Decision log

| # | Decision | Alternatives considered | Rationale |
|---|---|---|---|
| 1 | US futures only, free data only | Indian MCX, UK LME / ICE Europe, Bloomberg | Largest liquid universe with free positioning data (CFTC) and published benchmarks to validate against; fully reproducible |
| 2 | Long-short as the primary design, long-only as a secondary result | Long-only only | Long-short isolates the factor from commodity beta; shorting futures is costless; the theory is two-sided; matches the literature for validation |
| 3 | Per-contract start dates instead of dropping contracts | Drop oats, cattle, platinum, palladium | Problems were confined to early years; all 26 contracts are clean from 2010 |
| 4 | Roll calendar from exchange rules, validated empirically | Detect rolls from large returns | Threshold detection confuses genuine crashes with rolls; Yahoo's convention was measured against EIA first |
| 5 | Metals roll at the month boundary; two-day masks for metals and Brent | Expiry-based rolls for all | Signed-return evidence (pooled t = 3.6 at the delivery-month boundary); Brent's timing peaks at days 0 and +1 |
| 6 | Replace WTI April 2020 with contract-2 returns | Drop the month, winsorise | Reflects what a rolling investor actually held; documented in code |
| 7 | Keep delivery-period moves; test a ±20% cap separately | Winsorise the base data | The moves happened in the contract held; changing base data would hide a known bias rather than measure it |
| 8 | Carry = trailing 12-month roll yield for all 26 contracts | Energy-only spot carry; single roll gaps | Spot carry has no cross-section in free data; single gaps fail validation; the 12-month sum validates at 0.78 to 0.85 for CL, HO and NG |
| 9 | Seasonally adjusted spot carry fixed in advance for energy robustness | Choose raw or adjusted per contract | Choosing after seeing results would fit to the sample |
| 10 | Hedging pressure from Legacy COT levels, 7-day release lag | Disaggregated report; demeaned levels | Only Legacy covers 2002 onwards; published specification; conservative lag avoids look-ahead; demeaned kept as robustness |
| 11 | Value on unadjusted price levels | Roll-adjusted index | Five years of roll adjustments would distort the level |
| 12 | Equal-weighted terciles as the base construction | Quintiles, inverse-vol, sector-neutral | Standard, transparent and diversified with 26 assets; variants reported, never promoted |
| 13 | Costs on rebalancing and roll turnover | Rebalancing only | Rolling is the larger cost for futures; ignoring it overstates net returns |
| 14 | Six-factor equal-risk combination fixed in advance | Combine only factors that worked | Hindsight selection inflates results; the strong-four version is shown only as a labelled comparison |
| 15 | Newey-West t-statistics throughout; Holm, Bonferroni and t > 3 hurdles | Plain t-statistics | Factor returns are autocorrelated; many factors and variants were tested |
| 16 | Regime indicators lagged to publication; recessions labelled ex post | Contemporaneous indicators | Prevents look-ahead; NBER dates are only known with long delays |
| 17 | Walk-forward with yearly retraining from 2010 (ML extension) | – | Mirrors real-time use; about 7 years of initial training data; no model ever sees its own target |
| 18 | Relative, risk-adjusted targets (ML extension) | – | The decision is ranking (which factors or commodities), and the core portfolios are equal-risk |
| 19 | Simple benchmarks fixed in advance (equal-risk, factor momentum, composite) (ML extension) | – | ML must beat what can be done without it |
| 20 | Market × factor interactions for ridge in notebook 07 (ML extension) | – | Market features are constant across factors in a month and can only rank factors through factor-specific effects |
| 21 | Signal × regime / market / sector interactions in notebook 08 (ML extension) | – | Directly encodes the hypothesis "different signals work in different regimes" for a linear model |
| 22 | One pooled model instead of 26 per-commodity models (ML extension) | – | About 280 months per commodity is too little; pooled regime and sector features still let commodities behave differently |
| 23 | At least 4 of 6 signals, missing set to neutral (ML extension) | – | Keeps 2002 to 2005 and isolated gaps without letting the model read meaning into missing data |
| 24 | Tilt strength 0.5, tilting rather than selecting as the primary method (ML extension) | – | Keeps every factor between 0.5× and 1.5× its base weight, preserving the diversification behind the core result; top-k selection tested separately |
| 25 | No feature dropped after in-sample diagnostics (ML extension) | – | Selecting features on the full sample would leak future information |
| 26 | Ensemble labelled post hoc (ML extension) | – | It was suggested by Cell 5's prediction correlations, so it cannot be presented as a finding |

---

## 13. Limitations and caveats

1. **Absolute performance is likely optimistic.** Yahoo front-month data holds physically delivered contracts into their delivery period, where expiring contracts in tight markets often rally relative to deferred ones (soybean meal's 19% clean CAGR is the extreme case). Index investors roll before this. This lifts every contract's return, explains why our Sharpe ratios exceed AQR's by 0.13 to 0.22, and means the equal-weight benchmark (7.9% CAGR) outperforms investable commodity indices over the period. Long-short comparisons are less affected because the bias applies to both legs, but levels should be read with caution.
2. **Declining performance.** The combination's Sharpe ratio fell from 1.21 to 0.74 to 0.48 across subperiods, consistent with research showing published anomalies weaken after publication (McLean & Pontiff, 2016).
3. **Carry is limited by data.** A trailing roll yield is a slow proxy for the live curve, as the 2020 episode and the energy time-series test show. Full futures curves (for example from Datastream) would allow spot carry for all contracts, likely strengthening the factor.
4. **Momentum and carry are weak in this sample**, confirmed independently by AQR. They contribute diversification in the combination but no unique premium.
5. **Single factors have very long drawdowns** (up to 13.6 years underwater), so they are impractical to hold in isolation.
6. **Legacy COT data mixes swap dealers with physical hedgers.** The Disaggregated report (2006 onwards) would give a cleaner hedging measure.
7. **Regime results are descriptive.** Only 1 of 36 differences is significant, the Welch t-tests ignore autocorrelation, and the recession indicator is ex post.
8. **Simplifications in contract rules.** Exchange expiry exceptions (for example feeder cattle's November and holiday adjustments) are approximated; the roll-day validation shows the calendar is accurate in aggregate.
9. **Backtest, not live trading.** Results are historical simulations with modelled costs; capacity, margin and execution are not modelled.
10. **Short histories (ML extension).** About 200 out-of-sample months, and monthly returns are noisy; differences of 1% to 3% a year are hard to detect, so "no significant gain" does not prove "no gain".
11. **Multiple comparisons (ML extension).** Three models and two notebooks, several portfolio variants and sensitivity checks were tested; the borderline factor-momentum result (t ≈ 1.99) does not survive adjustment for this.
12. **Hyperparameters fixed rather than tuned (ML extension).** Constrained, conventional settings limit overfitting but may leave some predictive power unused; tuning them on the same data would reintroduce selection bias.
13. **Interpretability is in-sample (ML extension).** The learned effects in Section 14.4.7 describe models fitted on the full sample and are not performance evidence.
14. **Same data caveats as the core study (ML extension).** Front-month data makes absolute returns optimistic; the commodity-level portfolios are especially exposed because they hold single contracts in concentrated positions.

---

## 14. Machine-learning extension (notebooks 07 and 08)

The core study found that six commodity factors are individually fragile but jointly robust. Notebooks 07 and 08 ask the natural next question:

> **Can machine learning improve on simply holding all six factors, by learning which factors work in which market regime, or which signals work in each commodity's own regime?**

| | Notebook 07: factor-level timing | Notebook 08: commodity-level selection |
|---|---|---|
| **Decision** | How much weight each of the six factor portfolios gets | Which commodities to hold long and short |
| **Inputs** | Market regime + each factor's own recent behaviour | All six factor signals + each commodity's own regime + market regime + sector |
| **Rows** | 6 factors × ~280 months (1,653 complete) | ~26 commodities × ~290 months (6,751 complete) |
| **Benchmark to beat** | Equal-risk blend of all six factors | Simple composite (average of the six signals) |
| **Result** | No ML gain; a simple factor-momentum rule is borderline | ML does not beat the composite |

### 14.1 Summary of findings

All results are out of sample (January 2010 to September 2026), net of costs.

| Approach | Sharpe | Max DD | vs benchmark | Verdict |
|---|---:|---:|---|---|
| **Equal-risk blend of all six factors** | **0.55** | **−13.8%** | – | **Best risk-adjusted result** |
| Factor tilt, ridge | 0.52 | −14.7% | −0.17% p.a. (t = −0.35) | No gain |
| Factor tilt, gradient boosting | 0.55 | −14.6% | +0.11% p.a. (t = 0.18) | No gain |
| Factor tilt, factor-momentum rule | 0.65 | −15.1% | +1.07% p.a. (t = 1.99) | Borderline |
| Top-3 factors, factor momentum | 0.70 | −18.5% | +2.56% p.a. (t = 1.94) | Borderline, riskier |
| Top-2 / top-3 factors, ML | 0.39–0.46 | −23% to −24% | −0.6% to +0.5% p.a. | Worse |
| Commodity composite (no ML) | 0.43 | −36.2% | – | Benchmark for notebook 08 |
| Commodity ML, gradient boosting | 0.45 | −49.3% | +0.35% p.a. vs composite (t = 0.07) | No gain |
| Commodity ML, ridge | 0.27 | −44.9% | −2.37% p.a. vs composite (t = −0.56) | Worse |

**Key conclusions**

1. **ML does not reliably improve on the equal-risk factor blend.** Across both notebooks, no ML model produced a statistically significant improvement over its simple benchmark after costs.
2. **Regime structure exists but is small.** The one effect that appeared consistently in raw diagnostics, in ridge and in gradient boosting is that **the skewness signal works about twice as well in commodities that are unusually turbulent for themselves**.
3. **Simple rules match or beat ML here.** A no-training factor-momentum rule outperformed both ML models in notebook 07; a plain average of the six signals matched or beat ML in notebook 08.
4. **How factors are combined matters more than which model selects them.** Averaging factor *returns* (the equal-risk blend) diversified far better than averaging *signals* into one concentrated portfolio (Sharpe 0.55 vs 0.43, max drawdown −14% vs −36%), even though the two are 0.87 correlated.
5. **Two candidates merit live tracking, not adoption:** the factor-momentum tilt (borderline, gains concentrated after 2018) and a composite-plus-boosting ensemble (identified post hoc).

### 14.2 Design principles

The extension uses the same safeguards as the core study (Sections 5 to 9), plus several specific to machine learning:

- **Walk-forward testing.** For each test year *Y*, models are trained only on rows dated before 31 December of *Y − 1*, whose outcomes were known by then, and then predict January to December of *Y*. The training window expands and models are retrained every year. The first test year is 2010, so the first model trains on 2002/2003 to 2009.
- **Point-in-time features.** Every feature at month-end *t* uses information available at the close of *t*. CPI is lagged one extra month for its publication delay. Timing was hand-verified in both notebooks (for example, value's 12-month return at December 2024 and the January 2025 target).
- **Simple benchmarks fixed in advance.** Every ML model must beat a no-training rule: the equal-risk blend and a factor-momentum rule in notebook 07, the signal composite in notebook 08.
- **Pre-specified settings.** Tilt strength (0.5), test start (2010), model hyperparameters and feature sets were fixed before results were seen. Features were not dropped after seeing their in-sample diagnostics, which would leak future information into model design.
- **Costs.** Factor-weight changes in notebook 07 are charged at 5 bps × turnover × 2 (typical gross exposure), on top of the factors' own net returns. Commodity portfolios in notebook 08 pay 5 bps on rebalancing and roll turnover, as in the core study (Sections 5 to 9).
- **Honest labelling.** Ideas that came from seeing results (the ensemble in notebook 08) are labelled post hoc and not treated as findings. Interpretability analysis is in-sample and descriptive only.

### 14.3 Notebook 07: timing the factor portfolios

#### 14.3.1 Features

`FactorFeatureBuilder` produces one row per (month-end, factor):

| Group | Feature | Definition |
|---|---|---|
| **Market** (shared by all factors) | `mkt_ret_12m` | Equal-weight commodity market return over the past 12 months |
| | `mkt_vol_12m` | That market's annualised 12-month volatility |
| | `cpi_yoy` | US CPI year-on-year, lagged one month |
| | `dxy_chg_12m` | US Dollar Index 12-month change |
| | `vix` | VIX at month-end |
| **Factor** (each factor's own behaviour) | `ret_1m`, `ret_6m`, `ret_12m` | The factor's own past returns ("factor momentum") |
| | `vol_12m` | The factor's own 12-month volatility |
| | `drawdown` | Distance below the factor's previous peak |

The market features are continuous versions of the regimes used in notebook 04.

**Target:** each factor's next-month return divided by its own monthly volatility, minus the average across the six factors that month: relative, risk-adjusted performance. The model's job is to rank factors, not to forecast whether all factors rise, and on an equal-risk basis so volatile factors do not dominate the errors.

#### 14.3.2 In-sample diagnostics (before modelling)

Cross-factor information coefficient: does ranking the six factors by a feature predict next month's factor ranking?

| Feature | Mean IC | t-stat |
|---|---:|---:|
| `ret_12m` (factor momentum) | 0.060 | 2.09 |
| `ret_6m` | 0.059 | 2.14 |
| `drawdown` | 0.018 | 0.66 |
| `ret_1m` | 0.005 | 0.16 |
| `vol_12m` | −0.011 | −0.40 |

Factor momentum is the only clear signal, consistent with published research that factors with strong recent returns tend to keep outperforming. Market features showed mostly small correlations with next-month factor returns (|ρ| < 0.08), except for hedging pressure, which does better after falling markets (−0.15), in high volatility (+0.17), with high VIX (+0.12) and a strengthening dollar (+0.10), consistent with the core study's regime results.

#### 14.3.3 Models

| Model | Specification |
|---|---|
| **Ridge** | Standardised features, penalty chosen by cross-validation each year (`RidgeCV`, α from 0.1 to 100,000); factor dummies plus every market feature interacted with every factor dummy, because market features are identical across factors in a given month and can only rank factors through factor-specific effects |
| **Gradient boosting** | 300 trees, depth 2, learning rate 0.02, 70% subsampling, at least 40 rows per leaf; raw features plus factor dummies |
| **Factor-momentum rule** | Rank factors by trailing 12-month return per unit of volatility; no training |

#### 14.3.4 Prediction quality (out of sample, 200 months)

| Model | Mean IC | IC t-stat | Top-2 minus bottom-2 factor spread | Spread t (NW) |
|---|---:|---:|---:|---:|
| Ridge | 0.036 | 1.26 | +0.4% p.a. | 0.11 |
| Gradient boosting | 0.042 | 1.32 | +1.6% p.a. | 0.37 |
| **Factor-momentum rule** | **0.089** | **2.65** | **+8.1% p.a.** | **1.90** |

With about 400 to 1,600 noisy training rows (growing each year), both ML models fit noise in the weak market features and dilute the factor-momentum signal among 40 inputs.

#### 14.3.5 From predictions to portfolios

`FactorTiltPortfolio` keeps the equal-risk portfolio as the base (weights proportional to 1 / trailing 36-month volatility) and tilts it:

$$
w_{f,t} \propto w^{\text{base}}_{f,t} \times \left(1 + \lambda \cdot s_{f,t}\right)
$$

where $s_{f,t} \in [-1, 1]$ is the factor's prediction rank scaled from worst to best and $\lambda = 0.5$ (pre-specified), so each factor stays between 0.5× and 1.5× its base weight. Factors are never dropped or shorted, preserving diversification. Weights set at month-end *t* earn returns in *t + 1*.

#### 14.3.6 Results

| Portfolio (net of costs) | Ann. return | Vol | Sharpe | t (NW) | Max DD | Turnover p.a. |
|---|---:|---:|---:|---:|---:|---:|
| Equal-risk | 4.5% | 8.3% | 0.55 | 2.64 | −13.8% | 0.24 |
| Tilt: ridge | 4.3% | 8.3% | 0.52 | 2.43 | −14.7% | 1.06 |
| Tilt: gradient boosting | 4.6% | 8.5% | 0.55 | 2.49 | −14.6% | 3.13 |
| Tilt: factor momentum | 5.6% | 8.6% | 0.65 | 3.05 | −15.1% | 1.25 |

Improvement over equal-risk (monthly return difference):

| Tilt | Excess p.a. | Tracking error | Information ratio | t (NW) | Months better |
|---|---:|---:|---:|---:|---:|
| Ridge | −0.17% | 1.95% | −0.09 | −0.35 | 53.0% |
| Gradient boosting | +0.11% | 2.14% | 0.05 | 0.18 | 50.5% |
| Factor momentum | +1.07% | 2.16% | 0.50 | 1.99 | 57.5% |

| Sharpe by subperiod | 2010–2017 | 2018–2026 |
|---|---:|---:|
| Equal-risk | 0.60 | 0.51 |
| Tilt: ridge | 0.50 | 0.54 |
| Tilt: gradient boosting | 0.52 | 0.58 |
| Tilt: factor momentum | 0.58 | 0.71 |

**Tilt-strength robustness (factor momentum).** Excess return scales almost linearly with tilt strength: +0.54% (λ = 0.25), +1.07% (λ = 0.5), +2.16% (λ = 1.0), with t ≈ 2.0 throughout. The effect is steady rather than a lucky setting, but it is concentrated after 2018: the rolling 36-month excess return was negative in 2013 to 2014, close to zero from 2015 to 2018, and +1.5% to +3.5% a year since.

#### 14.3.7 Top-k factor selection

Holding only the 2 or 3 highest-ranked factors (equal-risk among them) instead of tilting:

| Portfolio | Sharpe | Vol | Max DD | Excess vs equal-risk (t) | Turnover p.a. |
|---|---:|---:|---:|---:|---:|
| Top-3, factor momentum | 0.70 | 10.1% | −18.5% | +2.56% (1.94) | 3.33 |
| Top-2, factor momentum | 0.63 | 11.5% | −18.8% | +2.72% (1.35) | 3.93 |
| Top-3, gradient boosting | 0.46 | 10.1% | −23.6% | +0.08% (0.05) | 7.88 |
| Top-2, gradient boosting | 0.43 | 11.8% | −23.5% | +0.52% (0.25) | 11.08 |
| Top-3, ridge | 0.39 | 9.9% | −24.3% | −0.62% (−0.40) | 2.49 |
| Top-2, ridge | 0.45 | 11.3% | −22.8% | +0.53% (0.28) | 4.13 |
| Equal-risk, all six | 0.55 | 8.3% | −13.8% | – | 0.24 |

Concentrating on ML picks lowers the Sharpe ratio and deepens drawdowns by about 65 to 75%. Top-3 factor momentum is the most consistent concentrated variant (Sharpe 0.61 vs 0.60 in 2010 to 2017; 0.77 vs 0.51 in 2018 to 2026) but remains borderline and adds a third more volatility.

#### 14.3.8 Live weights (set at 30 September 2026, for October 2026)

| Factor | Equal-risk | Factor-momentum tilt | Momentum score |
|---|---:|---:|---:|
| Time-series momentum | 29.2% | 35.1% | +0.20 |
| Value | 18.8% | 26.0% | +0.32 |
| Skewness | 15.0% | 15.2% | −0.13 |
| Momentum | 12.7% | 10.6% | −0.30 |
| Carry | 10.7% | 6.9% | −0.44 |
| Hedging pressure | 13.6% | 6.3% | −1.01 |

These weights start a live record for the factor-momentum tilt; future months provide genuine out-of-sample evidence.

### 14.4 Notebook 08: commodity-level ML

#### 14.4.1 Motivation

Commodities can be in very different situations at the same time: gold may be volatile, backwardated and heavily hedged while corn is calm and in contango. Different factor signals may work in each situation. Notebook 08 lets a model learn this from a panel of commodities, which provides about four times more data than notebook 07.

#### 14.4.2 Features

`CommodityFeatureBuilder` produces one row per (month-end, commodity):

| Group | Feature | Definition |
|---|---|---|
| **Factor signals** (6) | `signal_*` | Momentum, carry, value, skewness and hedging pressure as cross-sectional ranks scaled to [−1, 1]; time-series momentum as its sign (±1). Missing signals set to 0 (neutral); rows need at least 4 of 6 signals |
| **Own regime** (6) | `own_vol_12m` | The commodity's own 12-month volatility |
| | `own_vol_ratio` | 3-month vs 3-year volatility: unusually turbulent for itself |
| | `own_trend_3m` | 3-month return |
| | `own_from_high` | Distance below its 12-month high |
| | `own_backwardated` | Whether its curve has been backwardated (trailing roll yield > 0) |
| | `own_hp_extreme` | Hedging pressure z-score vs its own 3-year history |
| **Market** (5) | as notebook 07 | Shared market environment |
| **Sector** | dummies | Energy, metals, grains, softs, livestock |

**Target:** next-month return per unit of the commodity's own volatility, minus the cross-sectional mean. The plain return is kept for portfolio construction.

**Panel:** 7,451 rows; 6,751 with complete features (June 2002 to August 2026), with a median of 25 commodities per month.

**Engineering note.** Regime features are resampled to month-end column by column. An earlier version resampled the whole table at once, which dropped every day on which any contract was missing data and silently delayed every commodity's features to 2012; it was caught by the coverage check and fixed before any modelling.

#### 14.4.3 In-sample diagnostics: which signals work where

Baseline monthly IC across all commodities: skewness 0.051 (t = 3.62), hedging pressure 0.051 (3.36), value 0.038 (2.32), carry 0.034 (2.35), momentum 0.025 (1.67), time-series momentum 0.015 (1.04).

Conditional IC by the commodity's own regime:

| Signal | Turbulent | Calm | Backwardated | Contango | HP extreme | HP normal |
|---|---:|---:|---:|---:|---:|---:|
| Momentum | 0.015 | 0.039 | −0.009 | 0.045 | 0.004 | 0.040 |
| Carry | 0.038 | 0.014 | 0.018 | 0.038 | 0.033 | 0.034 |
| Value | 0.057 | 0.026 | 0.046 | 0.028 | **0.073** | −0.016 |
| Skewness | **0.091** | 0.013 | 0.064 | 0.038 | 0.040 | 0.046 |
| Hedging pressure | 0.069 | 0.022 | 0.034 | 0.046 | 0.044 | 0.051 |
| Time-series momentum | 0.033 | 0.001 | 0.013 | 0.015 | 0.006 | 0.021 |

Most signals worked better in commodities that were turbulent relative to their own history; momentum was the exception. By sector, value worked in energy and softs (pooled correlation ≈ 0.08) but not livestock (−0.06), and carry worked best in grains (0.07). These are 66 comparisons on in-sample data, so several differences are expected by chance; only the walk-forward test counts.

#### 14.4.4 Models

| Model | Specification |
|---|---|
| **Ridge** | 118 standardised features: 6 signals, 6 regime, 5 market, sector dummies, and every signal interacted with every regime, market and sector column, so a linear model can learn, for example, that skewness matters more when a commodity is turbulent; penalty by cross-validation (α from 1 to 1,000,000) |
| **Gradient boosting** | 400 trees, depth 3, learning rate 0.02, 70% subsampling, at least 100 rows per leaf; base features without interactions (trees find them) |
| **Composite** | Average of the six signals; no training |

#### 14.4.5 Prediction quality (out of sample, 200 months, before costs)

| Model | Mean IC | IC t-stat | Top-third minus bottom-third spread | Spread t (NW) |
|---|---:|---:|---:|---:|
| Ridge | 0.040 | 2.30 | +7.1% p.a. | 1.56 |
| Gradient boosting | 0.047 | 2.89 | +8.3% p.a. | 1.83 |
| **Composite** | **0.059** | **3.32** | **+9.4% p.a.** | **2.61** |

Unlike notebook 07, both ML models have significant out-of-sample ICs: the larger panel helped. But neither beats the composite. Their predictions correlate only 0.47 (ridge) and 0.28 (boosting) with the composite, so they make different bets rather than copying it, and those bets score worse.

#### 14.4.6 Portfolios (long top third, short bottom third, net of rebalancing and roll costs)

| Portfolio | Ann. return | Vol | Sharpe | t (NW) | Max DD | Turnover p.a. |
|---|---:|---:|---:|---:|---:|---:|
| Ridge | 5.1% | 18.9% | 0.27 | 1.19 | −44.9% | 9.0 |
| Gradient boosting | 7.8% | 17.5% | 0.45 | 1.72 | −49.3% | 14.2 |
| Composite | 7.5% | 17.5% | 0.43 | 2.10 | −36.2% | 9.1 |
| Ensemble of composite and boosting (**post hoc**) | 10.3% | 16.7% | 0.62 | 2.68 | −34.4% | 12.1 |
| Equal-risk factor blend (core study) | 4.5% | 8.3% | 0.55 | 2.65 | −13.8% | – |

| Head-to-head (monthly difference) | Diff p.a. | t (NW) | Correlation |
|---|---:|---:|---:|
| Ridge − composite | −2.37% | −0.56 | 0.49 |
| Gradient boosting − composite | +0.35% | 0.07 | 0.23 |
| Ensemble − composite | +2.81% | 0.82 | 0.72 |
| Composite − equal-risk blend | +2.96% | 1.28 | 0.87 |

| Sharpe by subperiod | 2010–2017 | 2018–2026 |
|---|---:|---:|
| Ridge | 0.65 | 0.06 |
| Gradient boosting | 0.68 | 0.33 |
| Composite | 0.70 | 0.27 |
| Ensemble (post hoc) | 0.65 | 0.61 |
| Equal-risk factor blend | 0.61 | 0.52 |

The single long-short portfolios earn more in absolute terms but take far larger drawdowns (30% to 50% in 2020 to 2022), and they faded after 2018 while the equal-risk blend held up. The ensemble was designed after seeing the prediction results and its gain over the composite is not significant, so it is a hypothesis for live testing.

#### 14.4.7 What the models learned (in-sample, descriptive)

**Ridge.** Cross-validation chose a very strong penalty (α = 10,000), shrinking every interaction close to zero (largest ≈ 0.01): the model judged most regime effects to be weak. The interactions it kept agree with the diagnostics: value × livestock (−) and value × softs (+), carry × grains (+), skewness × own volatility ratio (+), hedging pressure × market 12-month return (−), and trend × energy and softs (+).

**Gradient boosting.** Predicted effect of moving a signal from the bottom (−1) to the top (+1) rank:

| Signal | Turbulent | Calm | Backwardated | Contango |
|---|---:|---:|---:|---:|
| Value | 0.238 | 0.243 | 0.228 | 0.250 |
| Skewness | **0.158** | **0.071** | 0.135 | 0.097 |
| Hedging pressure | 0.110 | 0.094 | 0.113 | 0.093 |
| Carry | 0.105 | 0.084 | 0.098 | 0.091 |
| Momentum | 0.013 | 0.014 | 0.021 | 0.007 |
| Time-series momentum | 0.002 | 0.002 | 0.002 | 0.002 |

The model relies most on value, then skewness, hedging pressure and carry, and ignores both momentum signals. Only skewness shows a strong regime dependence (2.2 times larger in turbulent commodities). Its most important features were regime variables used directly (own volatility 0.14, 3-month trend 0.10, volatility ratio 0.09), meaning it partly predicted returns from volatility and short-term moves themselves, a likely source of its noise, 14× turnover and −49% drawdown. Its momentum pattern contradicts the raw diagnostics, further evidence that the weaker regime patterns are noise.

---

## 15. Conclusions

1. **No single commodity factor offers reliable evidence on its own in 2002 to 2026.** Skewness, value and time-series momentum earn Sharpe ratios of 0.40 to 0.46 and pass a conventional 5% test, but none survives correction for testing six factors (Page 3).
2. **Combining them is robust and economically meaningful.** The pre-specified equal-risk combination earns a Sharpe ratio of 0.71 (t = 3.91) with a −14% maximum drawdown and a 0.12 correlation with the commodity market, clearing every multiple-testing hurdle applied (Pages 1 to 3).
3. **The mechanism is diversification.** Near-zero correlations, with value offsetting the trend factors, halve volatility (Page 4).
4. **The combination is defensive and regime-robust.** It stayed between −6% and +6% while the market fell 16% to 39% in three major sell-offs, and its returns barely vary across six macro regimes, because the individual factors' regime tendencies offset (Pages 5 and 6).
5. **The pipeline is externally validated.** Correlations of 0.75 to 0.79 with AQR's published factors confirm the replication; the higher absolute Sharpe ratios are an acknowledged data limitation (Page 7).
6. **Long-only investors can capture part of the premia.** A multi-factor tilt beat the equal-weight market by 2.7% a year (information ratio 0.53, t = 2.88) at similar volatility and lower drawdown (Page 8).
7. **The edge is real but shrinking.** Robustness checks pass, but declining subperiod performance and optimistic data levels argue for conservative expectations (Page 9).
8. **Machine learning does not improve on the blend (Section 14).** Out of sample and after costs, regime-based tilting added nothing, selecting the top factors was worse, and commodity-level models did not beat a simple signal average.
9. **The core study's conclusion strengthens rather than weakens:** in commodity factor investing, robust diversification across factors matters more than predicting which factor will work next.
10. **There is real but small regime structure.** Skewness works about twice as well in turbulent commodities, consistently across raw diagnostics, ridge and gradient boosting. Most other regime effects were weak or inconsistent.
11. **Simple rules deserve monitoring.** A factor-momentum tilt (+1.1% a year, t = 1.99) and a composite-plus-boosting ensemble (post hoc) are candidates for live tracking. Only future months, which no model has seen, can confirm them.

The broader lesson is methodological: with free data and careful engineering it is possible to replicate institutional commodity factor research, but the evidence for any single factor is much weaker than headline backtests suggest once costs, roll handling, autocorrelation and multiple testing are treated properly. Diversification across factors is what turns fragile premia into a statistically defensible strategy, and the machine-learning extension shows that attempts to time or select factors do not improve on it out of sample.

---

## 16. References

- Asness, C., Moskowitz, T. and Pedersen, L. (2013). Value and Momentum Everywhere. *Journal of Finance*, 68(3).
- Bakshi, G., Gao, X. and Rossi, A. (2019). Understanding the Sources of Risk Underlying the Cross Section of Commodity Returns. *Management Science*, 65(2).
- Basu, D. and Miffre, J. (2013). Capturing the Risk Premium of Commodity Futures: The Role of Hedging Pressure. *Journal of Banking & Finance*, 37(7).
- Boons, M. and Prado, M. (2019). Basis-Momentum. *Journal of Finance*, 74(1).
- Brennan, M. (1958). The Supply of Storage. *American Economic Review*, 48(1).
- de Roon, F., Nijman, T. and Veld, C. (2000). Hedging Pressure Effects in Futures Markets. *Journal of Finance*, 55(3).
- Erb, C. and Harvey, C. (2006). The Strategic and Tactical Value of Commodity Futures. *Financial Analysts Journal*, 62(2).
- Fernandez-Perez, A., Frijns, B., Fuertes, A.-M. and Miffre, J. (2018). The Skewness of Commodity Futures Returns. *Journal of Banking & Finance*, 86.
- Gorton, G., Hayashi, F. and Rouwenhorst, K. G. (2013). The Fundamentals of Commodity Futures Returns. *Review of Finance*, 17(1).
- Gorton, G. and Rouwenhorst, K. G. (2006). Facts and Fantasies about Commodity Futures. *Financial Analysts Journal*, 62(2).
- Harvey, C., Liu, Y. and Zhu, H. (2016). … and the Cross-Section of Expected Returns. *Review of Financial Studies*, 29(1).
- Hicks, J. (1939). *Value and Capital*. Oxford University Press.
- Holm, S. (1979). A Simple Sequentially Rejective Multiple Test Procedure. *Scandinavian Journal of Statistics*, 6(2).
- Kaldor, N. (1939). Speculation and Economic Stability. *Review of Economic Studies*, 7(1).
- Keynes, J. M. (1930). *A Treatise on Money*. Macmillan.
- Koijen, R., Moskowitz, T., Pedersen, L. and Vrugt, E. (2018). Carry. *Journal of Financial Economics*, 127(2).
- McLean, R. D. and Pontiff, J. (2016). Does Academic Research Destroy Stock Return Predictability? *Journal of Finance*, 71(1).
- Miffre, J. and Rallis, G. (2007). Momentum Strategies in Commodity Futures Markets. *Journal of Banking & Finance*, 31(6).
- Moskowitz, T., Ooi, Y. H. and Pedersen, L. (2012). Time Series Momentum. *Journal of Financial Economics*, 104(2).
- Newey, W. and West, K. (1987). A Simple, Positive Semi-Definite, Heteroskedasticity and Autocorrelation Consistent Covariance Matrix. *Econometrica*, 55(3).
- Szymanowska, M., de Roon, F., Nijman, T. and van den Goorbergh, R. (2014). An Anatomy of Commodity Futures Risk Premia. *Journal of Finance*, 69(1).
- Working, H. (1949). The Theory of Price of Storage. *American Economic Review*, 39(6).

---

## 17. Data credits and disclaimer

**Data credits.** Energy futures prices: U.S. Energy Information Administration (EIA). Positioning: U.S. Commodity Futures Trading Commission (CFTC), Commitments of Traders. Macro series: Federal Reserve Bank of St. Louis (FRED). Validation factors: AQR Capital Management, LLC, AQR Data Library, used under its terms of use. Futures and index prices: Yahoo Finance. No vendor data is redistributed in this repository; the notebooks download it directly from each source.

**Disclaimer.** This project is independent research for educational purposes. All results are backtested historical simulations, not live trading, and are subject to the data limitations described above. Past performance is not indicative of future results. Nothing in this repository is investment advice or a recommendation to trade any instrument.

**Author.** Navon Shapurkar
