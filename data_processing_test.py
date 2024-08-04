from data_ingest import IngestData
from data_processing import (
    drop_and_fill,
    find_columns_with_few_values,
    find_constant_columns,
)
from feature_engineering import bin_to_num, cat_to_col, one_hot_encoding

ingest_data = IngestData()
df = ingest_data.get_data("ols-regression-challenge-data/data/cancer_reg.csv") 

constant_columns = find_constant_columns(df)
print("Columns that contain a single value: ", constant_columns)
columns_with_few_values = find_columns_with_few_values(df, 10)

df["binnedinc"][0]
df = bin_to_num(df)

df = cat_to_col(df)
df = one_hot_encoding(df)
df = drop_and_fill(df)
print(df.shape)
df.to_csv("ols-regression-challenge-data/cancer_reg_processed.csv", index=False)
