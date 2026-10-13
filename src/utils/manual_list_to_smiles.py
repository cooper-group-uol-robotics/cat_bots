import pandas as pd

manual_list = pd.read_csv("../data/ligands/ks60_updated.csv")

stock_commercial_fey = pd.read_csv("../data/ligands/stock_commercial_fey_match.csv")

final_list = (manual_list[['CAS']].merge(stock_commercial_fey, on='CAS', how='left', indicator=True) .query('_merge == "both"').drop(columns=['_merge']))
final_list = final_list.loc[:, ~final_list.columns.str.contains('unnamed', case=False)]
print(final_list.info())
print(final_list.head())

final_list.to_csv("../data/ligands/final_list.csv", index=False)