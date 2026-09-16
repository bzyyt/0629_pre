import pandas as pd
from config import cfg


# 保存结果到 CSV
def save_results_to_csv(results):
    df = pd.DataFrame(results)
    df.to_csv(f"{cfg.out_dir}/{cfg.name}_results.csv", index=False)
    print(f"Results saved to {cfg.out_dir}/{cfg.name}_results.csv")
