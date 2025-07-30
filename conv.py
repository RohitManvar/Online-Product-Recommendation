import pandas as pd

# Convert Excel to CSV (Run only ONCE)
df = pd.read_excel("OnlineRetail.xlsx")
df.to_csv("OnlineRetail.csv", index=False)