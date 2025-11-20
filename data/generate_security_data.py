import pandas as pd
import numpy as np

np.random.seed(42)

rows = 500

packet_size = np.random.randint(40, 2000, rows)
connection_rate = np.random.randint(1, 200, rows)
login_attempts = np.random.randint(0, 20, rows)

attack = ((packet_size > 1200) & (login_attempts > 5) | (connection_rate > 120)).astype(int)

df = pd.DataFrame({
    "packet_size": packet_size,
    "connection_rate": connection_rate,
    "login_attempts": login_attempts,
    "attack": attack
})

df.to_csv("data/security_dataset.csv", index=False)
print("✅ security_dataset.csv generated")
