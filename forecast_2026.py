import pandas as pd
import numpy as np
from sqlalchemy import create_engine
import warnings

warnings.filterwarnings("ignore")

# 1. Database Connection
engine = create_engine("mysql+pymysql://root:samroot@localhost/ods_hyperU")
df = pd.read_sql("SELECT * FROM ventes_ods", con=engine)
df["DATE_VENTE"] = pd.to_datetime(df["DATE_VENTE"])

# 2. Filter Consistent Articles 
sales_per_article = df.groupby("FK_ARTICLE")["DATE_VENTE"].nunique()
articles_consistants = sales_per_article[sales_per_article == 45].index
df = df[df["FK_ARTICLE"].isin(articles_consistants)]

# 3. Process Trends Realistically
trend_results = []

print("Running Robust Macro Trend Forecasting for July 2026...")

for article in df["FK_ARTICLE"].unique():
    art_df = df[df["FK_ARTICLE"] == article].sort_values("DATE_VENTE")
    
    # Calculate historical weekly baseline (what it sells on average per week)
    daily_avg = art_df["QTE_VENTE"].mean()
    historical_weekly_avg = daily_avg * 7
    
    # To determine trend momentum without mathematical explosion:
    # Look at the trajectory over the latest weeks available in your data
    recent_trajectory = art_df["QTE_VENTE"].tail(14).mean() * 7
    
    # Calculate a stable growth multiplier based on momentum
    if historical_weekly_avg > 0:
        momentum_ratio = recent_trajectory / historical_weekly_avg
    else:
        momentum_ratio = 1.0
        
    # Bound the momentum ratio so it doesn't drift to infinity or zero over the horizon
    momentum_ratio = np.clip(momentum_ratio, 0.7, 1.3)
    
    # Forecast July 2026 weekly average based on structural run-rate
    july_2026_weekly_pred = historical_weekly_avg * momentum_ratio
    
    # Absolute safety check: Sales can never be less than zero
    july_2026_weekly_pred = max(0.0, july_2026_weekly_pred)
    
    # Classify the macro trend
    threshold = 0.03  # 3% change threshold
    if july_2026_weekly_pred > historical_weekly_avg * (1 + threshold):
        direction = "UP (Hausse)"
    elif july_2026_weekly_pred < historical_weekly_avg * (1 - threshold):
        direction = "DOWN (Baisse)"
    else:
        direction = "STABLE"
        
    trend_results.append({
        "FK_ARTICLE": article,
        "HISTORICAL_WEEKLY_AVG": round(historical_weekly_avg, 2),
        "JULY_2026_WEEKLY_PRED": round(july_2026_weekly_pred, 2),
        "MACRO_TREND_JULY_2026": direction
    })

# 4. Save to MySQL ODS
trends_df = pd.DataFrame(trend_results)
trends_df.to_sql(name="sales_trends_2026", con=engine, if_exists="replace", index=False)

print("\n[SUCCESS] Stable trend analysis saved to table 'sales_trends_2026'!")
print(trends_df.head(10).to_string(index=False))