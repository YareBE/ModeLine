# Example dataset

`sample_house_prices.csv` is the dataset behind the **Load example dataset**
button in the app. It exists so that ModeLine can be tried without supplying a
file, and it is **synthetic**, not real market data.

240 rows, five numeric columns and no missing values:

| column | meaning |
|---|---|
| `area_m2` | floor area in square metres |
| `bedrooms` | number of bedrooms |
| `age_years` | age of the property |
| `distance_to_centre_km` | distance to the city centre |
| `price_eur` | target variable |

It was generated from a genuine linear relationship plus Gaussian noise, so that
fitting it demonstrates the tool actually working rather than fitting noise. A
model over all four predictors reaches an R² of about 0.985. The generating
equation was:

    price = 40000
          + 1800 * area_m2
          + 5000 * bedrooms
          -  900 * age_years
          - 3500 * distance_to_centre_km
          + N(0, 12000)

with the predictors drawn uniformly (`numpy` default RNG, seed 20260820) from
area 40–200, bedrooms 1–5, age 0–60 and distance 0.5–25.

Because the underlying coefficients are known, the file doubles as a check on
the app: the fitted formula shown after training should land near them.
