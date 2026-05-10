import pandas as pd
import numpy as np
import os

ROOT_DIR = os.path.abspath(".")
path = os.path.join(ROOT_DIR, "data", "raw", "test.csv")
print(path)
df = pd.read_csv(path)
sample = df.sample(1, random_state=np.random.default_rng().integers(0, 1_000_000)).iloc[0]
print(sample["article"][:100])
