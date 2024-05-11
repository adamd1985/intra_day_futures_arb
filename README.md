# Futures Arbitrage and Other Techniques

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