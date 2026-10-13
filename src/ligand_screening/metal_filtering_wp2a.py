import pandas as pd

# Load (must have columns: 'catalyst', 'substrate', 'Yield (%)')
df = pd.read_csv("../data/reactions/wp2a_first_96_rxns.csv").copy()

df["catalyst"]  = df["catalyst"].astype(str).str.strip()
df["substrate"] = df["substrate"].astype(str).str.strip()

THRESH = 5

passes = (df.assign(pass_=df["Yield (%)"] >= THRESH)
            .groupby(["catalyst", "substrate"])["pass_"]
            .max()
            .unstack(fill_value=False))

tested = (df.groupby(["catalyst", "substrate"]).size()
            .unstack(fill_value=0)
            .astype(bool))

# 1) STRICT: metal must have >= THRESH for *every* substrate in the dataset
good_strict_mask = passes.all(axis=1)
metals_strict = good_strict_mask[good_strict_mask].index.tolist()

# 2) RELAXED: metal must have >= THRESH on *every substrate it was tested on*
# (Untested substrates are ignored.)
good_relaxed_mask = ((~tested) | passes).all(axis=1)
metals_relaxed = good_relaxed_mask[good_relaxed_mask].index.tolist()

print("Metals meeting strict criterion (>=5% for ALL substrates):")
print(metals_strict)

print("\nMetals meeting relaxed criterion (>=5% on every substrate they were tested on):")
print(metals_relaxed)
