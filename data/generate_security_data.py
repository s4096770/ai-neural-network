import pandas as pd
import numpy as np

np.random.seed(42)

def generate_security_data(n=500):
    packet_size = np.random.randint(40, 1500, n)
    connection_rate = np.random.randint(1, 200, n)
    login_attempts = np.random.randint(0, 20, n)

    attack = (
        (packet_size > 1000 & (login_attempts > 10)) |
        (connection_rate > 150)
    ).astype(int)

    df = pd.DataFrame({
        "packet_size": packet_size,
        "connection_rate": connection_rate,
        "login_attempts": login_attempts,
        "attack": attack
    })

    df.to_csv("data/security_dataset.csv", index=False)
    print("✅ security_dataset.csv generated")

if __name__ == "__main__":
    generate_security_data()
