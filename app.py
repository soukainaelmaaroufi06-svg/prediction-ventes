import pandas as pd
import numpy as np
import streamlit as st
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import LabelEncoder
from xgboost import XGBRegressor
from statsmodels.tsa.statespace.sarimax import SARIMAX
import datetime
import warnings
warnings.filterwarnings("ignore")


st.set_page_config(page_title="Prévision des ventes - Hyper U Zenata", layout="centered")
st.title("Prévision des ventes - Hyper U Zenata")
password = st.text_input("Mot de passe", type="password")
if password != "hyperu":
    st.stop()

@st.cache_data
def load_and_prepare():
    df = pd.read_csv("QTE VENTE ZENATA 01-06-2025 to 15-07-2025.csv", sep=";", encoding="utf-8")
    df.columns = df.columns.str.strip().str.upper()

    # Support both column name variants
    if "ARTICLE" not in df.columns and "article" in df.columns:
        df.rename(columns={"article": "ARTICLE"}, inplace=True)

    df["DATE_VENTE"] = pd.to_datetime(df["DATE_VENTE"], dayfirst=True)

    sales_per_article = df.groupby("FK_ARTICLE")["DATE_VENTE"].nunique()
    articles_consistants = sales_per_article[sales_per_article == 45].index
    df = df[df["FK_ARTICLE"].isin(articles_consistants)]

    df["jour_semaine"] = df["DATE_VENTE"].dt.dayofweek
    df["mois"] = df["DATE_VENTE"].dt.month
    df["semaine_annee"] = df["DATE_VENTE"].dt.isocalendar().week.astype(int)

    le = LabelEncoder()
    df["article_encoded"] = le.fit_transform(df["FK_ARTICLE"])

    df = df.sort_values(["FK_ARTICLE", "DATE_VENTE"])
    df["lag_1"] = df.groupby("FK_ARTICLE")["QTE_VENTE"].shift(1)
    df["lag_7"] = df.groupby("FK_ARTICLE")["QTE_VENTE"].shift(7)

    temp = df.copy()
    temp["lag_last_saturday"] = temp["QTE_VENTE"].where(temp["jour_semaine"] == 5)
    temp["lag_last_saturday"] = temp.groupby("FK_ARTICLE")["lag_last_saturday"].shift(1).ffill()
    df["lag_last_saturday"] = temp["lag_last_saturday"]

    temp2 = df.copy()
    temp2["lag_last_sunday"] = temp2["QTE_VENTE"].where(temp2["jour_semaine"] == 6)
    temp2["lag_last_sunday"] = temp2.groupby("FK_ARTICLE")["lag_last_sunday"].shift(1).ffill()
    df["lag_last_sunday"] = temp2["lag_last_sunday"]

    temp3 = df.copy()
    temp3["lag_saturday_2w"] = temp3["QTE_VENTE"].where(temp3["jour_semaine"] == 5)
    temp3["lag_saturday_2w"] = temp3.groupby("FK_ARTICLE")["lag_saturday_2w"].shift(2).ffill()
    df["lag_saturday_2w"] = temp3["lag_saturday_2w"]

    temp4 = df.copy()
    temp4["lag_sunday_2w"] = temp4["QTE_VENTE"].where(temp4["jour_semaine"] == 6)
    temp4["lag_sunday_2w"] = temp4.groupby("FK_ARTICLE")["lag_sunday_2w"].shift(2).ffill()
    df["lag_sunday_2w"] = temp4["lag_sunday_2w"]

    df["moyenne_mobile_7"] = df.groupby("FK_ARTICLE")["QTE_VENTE"].transform(
        lambda x: x.shift(1).rolling(window=7).mean()
    )

    df = df.dropna(subset=["lag_1", "lag_7", "moyenne_mobile_7", "lag_last_saturday",
                            "lag_last_sunday", "lag_saturday_2w", "lag_sunday_2w"])

    return df, le

df, le = load_and_prepare()

dates = sorted(df["DATE_VENTE"].unique())
train_dates = dates[:28]
target = "QTE_VENTE"
train_df = df[df["DATE_VENTE"].isin(train_dates)]

features_ml = ["jour_semaine", "mois", "article_encoded", "lag_1", "lag_7",
               "moyenne_mobile_7", "lag_last_saturday", "lag_last_sunday",
               "lag_saturday_2w", "lag_sunday_2w"]

features_gb = ["jour_semaine", "mois", "article_encoded", "lag_1", "lag_7",
               "moyenne_mobile_7", "lag_last_saturday", "lag_last_sunday", "lag_saturday_2w"]

@st.cache_resource
def train_models(_train_df):
    lr = LinearRegression()
    lr.fit(_train_df[features_ml], _train_df[target])

    rf = RandomForestRegressor(n_estimators=100, random_state=42)
    rf.fit(_train_df[features_ml], _train_df[target])

    gb = GradientBoostingRegressor(n_estimators=100, random_state=42)
    gb.fit(_train_df[features_gb], _train_df[target])

    xgb = XGBRegressor(n_estimators=100, random_state=42, verbosity=0)
    xgb.fit(_train_df[features_ml], _train_df[target])

    return lr, rf, gb, xgb

with st.spinner("Entraînement des modèles..."):
    lr_model, rf_model, gb_model, xgb_model = train_models(train_df)

article_names = df[["FK_ARTICLE", "ARTICLE"]].drop_duplicates().sort_values("ARTICLE")
article_dict = dict(zip(article_names["ARTICLE"].str.strip(), article_names["FK_ARTICLE"]))

st.subheader("Paramètres de prédiction")

selected_article_name = st.selectbox("Article", sorted(article_dict.keys()))
selected_model = st.selectbox("Modèle", [
    "Gradient Boosting (MAE 7.14)",
    "Random Forest (MAE 7.17)",
    "XGBoost (MAE 7.55)",
    "Régression Linéaire (MAE 9.28)",
    "SARIMA (MAE 11.57)"
])

# Compute valid date range from data
min_pred_date = (df["DATE_VENTE"].max() + pd.Timedelta(days=1)).date()
max_pred_date = min_pred_date + datetime.timedelta(days=6)

selected_date = st.date_input(
    "Date à prédire",
    min_value=min_pred_date,
    max_value=max_pred_date,
    value=min_pred_date
)

if st.button("Prédire"):
    selected_fk = article_dict[selected_article_name]
    article_df = df[df["FK_ARTICLE"] == selected_fk].sort_values("DATE_VENTE")
    last_row = article_df.iloc[-1]

    target_date = pd.Timestamp(selected_date)
    jour_semaine = target_date.dayofweek
    mois = target_date.month
    article_encoded = last_row["article_encoded"]

    lag_1 = last_row["QTE_VENTE"]
    lag_7_vals = article_df[article_df["DATE_VENTE"] == target_date - pd.Timedelta(days=7)]["QTE_VENTE"].values
    lag_7 = lag_7_vals[0] if len(lag_7_vals) > 0 else last_row["lag_7"]

    lag_last_saturday = article_df[article_df["jour_semaine"] == 5]["QTE_VENTE"].iloc[-1] if len(article_df[article_df["jour_semaine"] == 5]) > 0 else lag_1
    lag_last_sunday = article_df[article_df["jour_semaine"] == 6]["QTE_VENTE"].iloc[-1] if len(article_df[article_df["jour_semaine"] == 6]) > 0 else lag_1
    lag_saturday_2w = article_df[article_df["jour_semaine"] == 5]["QTE_VENTE"].iloc[-2] if len(article_df[article_df["jour_semaine"] == 5]) >= 2 else lag_last_saturday
    lag_sunday_2w = article_df[article_df["jour_semaine"] == 6]["QTE_VENTE"].iloc[-2] if len(article_df[article_df["jour_semaine"] == 6]) >= 2 else lag_last_sunday
    moyenne_mobile_7 = article_df["QTE_VENTE"].iloc[-7:].mean()

    if "SARIMA" in selected_model:
        train_series = article_df[article_df["DATE_VENTE"].isin(train_dates)]["QTE_VENTE"].values
        try:
            sarima = SARIMAX(train_series, order=(1,1,1), seasonal_order=(1,1,1,7))
            result = sarima.fit(disp=False)
            days_ahead = (target_date - pd.Timestamp(train_dates[-1])).days
            forecast = result.forecast(steps=days_ahead)
            prediction = forecast[-1]
        except:
            prediction = article_df["QTE_VENTE"].mean()
    else:
        if "Gradient Boosting" in selected_model:
            input_row = pd.DataFrame([{
                "jour_semaine": jour_semaine, "mois": mois,
                "article_encoded": article_encoded, "lag_1": lag_1,
                "lag_7": lag_7, "moyenne_mobile_7": moyenne_mobile_7,
                "lag_last_saturday": lag_last_saturday,
                "lag_last_sunday": lag_last_sunday,
                "lag_saturday_2w": lag_saturday_2w,
            }])
            prediction = gb_model.predict(input_row)[0]
        else:
            input_row = pd.DataFrame([{
                "jour_semaine": jour_semaine, "mois": mois,
                "article_encoded": article_encoded, "lag_1": lag_1,
                "lag_7": lag_7, "moyenne_mobile_7": moyenne_mobile_7,
                "lag_last_saturday": lag_last_saturday,
                "lag_last_sunday": lag_last_sunday,
                "lag_saturday_2w": lag_saturday_2w,
                "lag_sunday_2w": lag_sunday_2w,
            }])
            if "Random Forest" in selected_model:
                prediction = rf_model.predict(input_row)[0]
            elif "XGBoost" in selected_model:
                prediction = xgb_model.predict(input_row)[0]
            else:
                prediction = lr_model.predict(input_row)[0]

    st.success(f"✅ Quantité prédite : **{round(prediction)} unités**")
    st.write("**Article :**", selected_article_name)
    st.write("**Date :**", selected_date)
    st.write("**Modèle :**", selected_model)