# Futures Arbitrage and Other Techniques

## Futures and Future Options

Exchange: CME Globex. Types: Options, Full, Mini, and Micro sizes.

Classes:
- Agriculture (AG) FUTs/Options.
- Precious Metals
- Crude Oil
- Equities
- Energy

### Target Mircos

Accessable for retail traders.
- Micro E-Mini S&P 500
- Micro E-mini Nasdaq-100
- Micro E-mini Russell 2000
- Micro E-mini Dow
- Micro Bitcoin
- Micro WTI Crude Oil
- Micro 10-Year Yield
- Micro Ether
- Micro Silver
- Micro Henry Hub Natural Gas
- Micro Gold
- Micro EUR/USD
- Micro AUD/USD

## Contango VS Backwardation

Contango:
- Fut > Spot
- High storage costs, high demand/short supply in future. High carry costs.
- "negative roll yield" in roll overs.

Backwardation:
- Fut < Spot
- Immediate demmand/short supply (pay permuim)
- "Positive roll yield"

## Install and Commit Dependencies

Create *python 3.9* env:
`conda create -n fut python=3.11 & conda activate fut`

Install dependencies:
`yes | pip install -r requirements.txt`

When commiting, update the requirements:
`jupyter nbconvert --to script ./notebook.ipynb & pipreqs`

## ENV configurations

A  `.env` file is not provided.
Use your own and add it to `.gitignore`.