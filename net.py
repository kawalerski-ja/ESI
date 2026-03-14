import pandas as pd
df=pd.read_csv("auction_results_color_svd.csv")

# Initial EDA
print("Head")
print(df.head())

print("\nData types")
df.info()

print("\nStats description")

print(df.describe())

# NaN analysis
print("\nSearching for NaNs")
print(df.isnull().sum())
# no NaNs

# in .xlsx file mean year is 1961 with sd. 88
# prices in given .csv file aren't probably modified

# Trying to decode the artists by counting them and checking for sum of price
# .xlsx file contains count of artists

id_count = df['ARTIST'].value_counts()
print("Top 10 artists by count")
print(id_count.head(10))

# Numbers here don't correspond to numbers in .xlsx file, but we can guess

# Checking for best-selling artists

sum_price = df.groupby('ARTIST')['PRICE'].sum()
suma_price_sort = sum_price.sort_values(ascending=False)

print("Top 10 best selling artists")
print(suma_price_sort.head(10))

# Most possibly:
# 354 = Salvador Dali
# 331 = Pablo Picasso
# 355 = Joan Miro
# 335 = Corneille
# 328 = Schifano
# 367 = Marc Chagall <3