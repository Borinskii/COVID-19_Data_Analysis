from __future__ import annotations

import os
from datetime import datetime, date, timedelta
from typing import Dict, Any, List, Optional
import dash
from dash import dcc, html, Input, Output, State
import dash_bootstrap_components as dbc
import plotly.express as px
import pandas as pd
import requests
from requests.adapters import HTTPAdapter
from urllib3.util.retry import Retry
from dotenv import load_dotenv
import numpy as np
from scipy.stats import pearsonr
import plotly.graph_objects as go
from plotly.subplots import make_subplots

from sklearn.preprocessing import StandardScaler
from sklearn.cluster import KMeans
from sklearn.decomposition import PCA
import warnings
warnings.filterwarnings('ignore')

from scipy.optimize import curve_fit

load_dotenv()

API_BASE_URL = os.getenv("API_BASE_URL", "http://localhost:8003")

PRIMARY  = "#60A5FA"
ACCENT   = "#A78BFA"
OK       = "#34D399"
DANGER   = "#F87171"
MUTED    = "#94A3B8"
FORECAST = "#F59E0B"

CLUSTER_COLORS = ["#F87171","#34D399","#60A5FA","#FBBF24","#A78BFA","#5EEAD4","#F472B6"]

DARK_BG    = "#0B1220"
DARK_EDGE  = "1px solid rgba(255,255,255,0.08)"
GRID_COLOR = "rgba(255,255,255,0.08)"
AXIS_LINE  = "rgba(255,255,255,0.18)"

def create_api_session() -> requests.Session:
    s = requests.Session()
    retry = Retry(total=3, backoff_factor=0.6, status_forcelist=[429, 500, 502, 503, 504])
    adapter = HTTPAdapter(max_retries=retry)
    s.mount("http://", adapter)
    s.mount("https://", adapter)
    return s

api = create_api_session()

def _safe_json_df(resp: requests.Response) -> pd.DataFrame:
    try:
        data = resp.json()
    except Exception:
        return pd.DataFrame()
    if isinstance(data, dict) and data.get("success") and data.get("data"):
        return pd.DataFrame(data["data"])
    return pd.DataFrame()

def api_df(endpoint: str, params: Optional[Dict[str, Any]] = None, timeout: int = 30) -> pd.DataFrame:
    try:
        resp = api.get(f"{API_BASE_URL}{endpoint}", params=params or {}, timeout=timeout)
        if resp.status_code == 200:
            return _safe_json_df(resp)
        return pd.DataFrame()
    except Exception:
        return pd.DataFrame()

def fetch_countries() -> List[str]:
    try:
        resp = api.get(f"{API_BASE_URL}/api/v1/countries", timeout=15)
        data = resp.json()
        if data.get("success"):
            return sorted(data.get("data", []))
    except Exception:
        pass
    return ["Latvia", "United States", "US", "United Kingdom", "France", "Germany", "Italy", "Spain"]

def fetch_cases(country: str, case_type: str, start_date: str, end_date: str, limit: int = 10000) -> pd.DataFrame:
    df = api_df(
        "/api/v1/covid-cases",
        {"country": country, "case_type": case_type, "start_date": start_date, "end_date": end_date, "limit": limit},
    )

    if not df.empty and "DATE" in df.columns:
        df['DATE'] = pd.to_datetime(df['DATE'], errors='coerce')
        start_dt = pd.to_datetime(start_date)
        end_dt = pd.to_datetime(end_date)
        df = df[(df['DATE'] >= start_dt) & (df['DATE'] <= end_dt)]
        df = df.sort_values('DATE')
        print(f"DEBUG: Filtered {country} {case_type} data to {len(df)} rows between {start_date} and {end_date}")

    return df

def fetch_vacc(country: str, start_date: str, end_date: str, limit: int = 5000) -> pd.DataFrame:
    df = api_df(
        "/api/v1/vaccinations",
        {"country": country, "start_date": start_date, "end_date": end_date, "limit": limit},
    )
    if not df.empty and "DATE" in df.columns:
        df['DATE'] = pd.to_datetime(df['DATE'], errors='coerce')
        start_dt = pd.to_datetime(start_date)
        end_dt = pd.to_datetime(end_date)
        df = df[(df['DATE'] >= start_dt) & (df['DATE'] <= end_dt)]
    return df

def fetch_mobility(country: str, transport_type: str, start_date: str, end_date: str, limit: int = 5000) -> pd.DataFrame:
    df = api_df(
        "/api/v1/mobility",
        {"country": country, "transportation_type": transport_type, "start_date": start_date, "end_date": end_date, "limit": limit},
    )
    if not df.empty and "DATE" in df.columns:
        df['DATE'] = pd.to_datetime(df['DATE'], errors='coerce')
        start_dt = pd.to_datetime(start_date)
        end_dt = pd.to_datetime(end_date)
        df = df[(df['DATE'] >= start_dt) & (df['DATE'] <= end_dt)]
    return df

def fetch_enriched(country: str, year: int) -> Dict[str, Any]:
    try:
        resp = api.get(
            f"{API_BASE_URL}/api/v1/enriched-data",
            params={"country": country, "year": year, "include_gdp": True, "include_hdi": True, "include_population": True},
            timeout=25,
        )
        if resp.status_code == 200:
            return resp.json().get("data", {})
    except Exception:
        pass
    return {}

def post_comment(payload: Dict[str, Any]) -> Optional[str]:
    try:
        r = api.post(f"{API_BASE_URL}/api/v1/comments", json=payload, timeout=20)
        if r.ok:
            j = r.json()
            return j.get("id") or j.get("inserted_id") or j.get("_id")
    except Exception as e:
        print(f"Exception posting comment: {e}")
    return None

def get_comments(datapoint_id: Optional[str] = None, author_email: Optional[str] = None, limit: int = 50) -> List[Dict[str, Any]]:
    params: Dict[str, Any] = {"limit": limit}
    if datapoint_id:
        params["datapointId"] = datapoint_id
    if author_email:
        params["authorEmail"] = author_email

    try:
        r = api.get(f"{API_BASE_URL}/api/v1/comments", params=params, timeout=20)
        if r.ok:
            data = r.json().get("data", [])
            for a in data:
                if "_id" in a and not isinstance(a["_id"], str):
                    a["_id"] = str(a["_id"])
            return data
    except Exception as e:
        print(f"Exception getting comments: {e}")
    return []

def exponential_growth_model(t, a, b, c):
    return a * np.exp(b * t) + c

def forecast_time_series(data: pd.Series, periods: int = 30, method: str = "exponential") -> tuple:
    if len(data) < 10:
        return None, None

    data = data.dropna()
    if len(data) < 10:
        return None, None

    try:
        if method == "exponential":
            window = min(14, len(data) // 3)
            recent_avg = data.tail(window).mean()

            trend_window = max(7, len(data) // 3)
            recent_data = data.tail(trend_window)
            if len(recent_data) > 1:
                x = np.arange(len(recent_data))
                y = recent_data.values
                trend = np.polyfit(x, y, 1)[0]
            else:
                trend = 0

            forecast = []
            current_value = data.iloc[-1]

            for i in range(1, periods + 1):
                alpha = 0.3
                next_value = current_value + trend * i
                dampening = 0.95 ** i
                next_value = next_value * dampening + recent_avg * (1 - dampening)
                forecast.append(max(0, next_value))

            forecast = np.array(forecast)

            volatility = data.std()
            confidence_width = volatility * np.sqrt(np.arange(1, periods + 1))
            lower_bound = forecast - 1.96 * confidence_width
            upper_bound = forecast + 1.96 * confidence_width

            return forecast, (lower_bound, upper_bound)

    except Exception as e:
        print(f"Forecasting error: {e}")

    if len(data) < 2:
        return None, None

    x = np.arange(len(data))
    y = data.values

    try:
        z = np.polyfit(x, y, 1)
        p = np.poly1d(z)

        future_x = np.arange(len(data), len(data) + periods)
        forecast = p(future_x)
        forecast = np.maximum(0, forecast)

        residuals = y - p(x)
        std = np.std(residuals)
        lower = forecast - 2 * std
        upper = forecast + 2 * std

        return forecast, (lower, upper)
    except:
        return None, None

def cluster_regions(countries_data: Dict[str, pd.DataFrame], n_clusters: int = 5) -> Dict[str, Any]:
    features_list = []
    country_names = []

    for country, df in countries_data.items():
        if df.empty or len(df) < 10:
            continue

        features = {}

        if "CASES" in df.columns:
            cases = df["CASES"].fillna(0)
            daily_changes = df["DIFFERENCE"].fillna(0) if "DIFFERENCE" in df.columns else pd.Series([0] * len(df))

            features["log_peak_cases"] = np.log1p(cases.max())
            features["log_peak_daily"] = np.log1p(daily_changes.max())

            if len(cases) > 20:
                early_period = cases.iloc[:len(cases)//4].mean()
                late_period = cases.iloc[3*len(cases)//4:].mean()
                features["growth_rate"] = np.log1p(late_period) - np.log1p(early_period)
            else:
                features["growth_rate"] = 0

            features["volatility"] = daily_changes.std() if len(daily_changes) > 1 else 0

            peak_idx = cases.argmax()
            features["time_to_peak"] = peak_idx / len(cases) if len(cases) > 0 else 0

            if len(daily_changes) > 20:
                peak_daily_idx = daily_changes.argmax()
                if peak_daily_idx < len(daily_changes) - 10:
                    post_peak = daily_changes.iloc[peak_daily_idx:peak_daily_idx+10]
                    if len(post_peak) > 1:
                        recovery_slope = np.polyfit(range(len(post_peak)), post_peak.values, 1)[0]
                        features["recovery_rate"] = -recovery_slope
                    else:
                        features["recovery_rate"] = 0
                else:
                    features["recovery_rate"] = 0
            else:
                features["recovery_rate"] = 0

        if len(features) == 6:
            features_list.append(list(features.values()))
            country_names.append(country)

    if len(features_list) < 3:
        return {"clusters": {}, "characteristics": {}}

    n_clusters = min(n_clusters, len(features_list), 7)

    try:
        scaler = StandardScaler()
        features_scaled = scaler.fit_transform(features_list)

        kmeans = KMeans(n_clusters=n_clusters, random_state=42, n_init=10)
        clusters = kmeans.fit_predict(features_scaled)

        pca = PCA(n_components=2)
        features_pca = pca.fit_transform(features_scaled)

        cluster_assignments = {country: int(cluster) for country, cluster in zip(country_names, clusters)}

        characteristics = {}
        for i in range(n_clusters):
            cluster_mask = clusters == i
            cluster_countries = [c for c, cl in zip(country_names, clusters) if cl == i]

            if cluster_mask.any():
                cluster_features = np.array(features_list)[cluster_mask]
                characteristics[i] = {
                    "countries": cluster_countries,
                    "size": len(cluster_countries),
                    "avg_log_peak_cases": np.mean(cluster_features[:, 0]),
                    "avg_growth_rate": np.mean(cluster_features[:, 2]),
                    "avg_volatility": np.mean(cluster_features[:, 3]),
                    "center_pca": features_pca[cluster_mask].mean(axis=0).tolist()
                }

        return {
            "clusters": cluster_assignments,
            "characteristics": characteristics,
            "pca_coordinates": {country: coords.tolist() for country, coords in zip(country_names, features_pca)},
            "explained_variance": pca.explained_variance_ratio_.tolist()
        }

    except Exception as e:
        print(f"Clustering error: {e}")
        return {"clusters": {}, "characteristics": {}}

def build_correlation_charts(country: str) -> Dict[str, go.Figure]:
    figs = {}

    try:
        countries = ["United States", "United Kingdom", "Germany", "France", "Italy", "Spain", "Canada", "Australia", "Japan", "South Korea"]
        correlation_data = []

        for c in countries:
            try:
                vacc_data = fetch_vacc(c, "2021-01-01", "2023-12-31", 1000)
                if not vacc_data.empty:
                    vacc_rate = None
                    for col in ['PEOPLE_FULLY_VACCINATED_PER_HUNDRED', 'PEOPLE_VACCINATED_PER_HUNDRED']:
                        if col in vacc_data.columns and vacc_data[col].notna().any():
                            vacc_rate = float(vacc_data[col].max())
                            break

                    if vacc_rate and vacc_rate > 0:
                        gdp_data = fetch_enriched(c, 2021)
                        if 'gdp' in gdp_data and gdp_data['gdp']:
                            for gdp_item in gdp_data['gdp']:
                                if gdp_item.get('country', '').upper() == c.upper():
                                    gdp_value = gdp_item.get('gdp_per_capita')
                                    if gdp_value and gdp_value > 0:
                                        correlation_data.append({
                                            'country': c,
                                            'vaccination_rate': vacc_rate,
                                            'gdp_per_capita': float(gdp_value)
                                        })
                                    break
            except Exception as e:
                print(f"Error processing {c}: {e}")
                continue

        if len(correlation_data) >= 3:
            df = pd.DataFrame(correlation_data)

            correlation, p_value = pearsonr(df['vaccination_rate'], df['gdp_per_capita'])

            fig_vacc_gdp = go.Figure()
            fig_vacc_gdp.add_trace(go.Scatter(
                x=df['vaccination_rate'],
                y=df['gdp_per_capita'],
                mode='markers+text',
                text=df['country'],
                textposition='top center',
                marker=dict(size=12, color=df['gdp_per_capita'], colorscale='Viridis', showscale=True),
                hovertemplate='<b>%{text}</b><br>Vaccination: %{x:.1f}%<br>GDP: $%{y:,.0f}<extra></extra>'
            ))

            z = np.polyfit(df['vaccination_rate'], df['gdp_per_capita'], 1)
            p = np.poly1d(z)
            x_trend = np.linspace(df['vaccination_rate'].min(), df['vaccination_rate'].max(), 50)

            fig_vacc_gdp.add_trace(go.Scatter(
                x=x_trend, y=p(x_trend), mode='lines', name=f'Trend (r={correlation:.3f})',
                line=dict(color='red', width=2, dash='dash')
            ))

            fig_vacc_gdp.update_layout(
                title=f'💉💰 Vaccination Rate vs GDP per Capita<br><sub>Correlation: {correlation:.3f} (p={p_value:.3f})</sub>',
                xaxis_title='Vaccination Rate (% Fully Vaccinated)',
                yaxis_title='GDP per Capita (USD)',
                showlegend=True
            )

            figs['vacc_gdp'] = fig_defaults(fig_vacc_gdp)

    except Exception as e:
        print(f"Error creating vaccination-GDP correlation: {e}")

    try:
        cases_data = fetch_cases(country, "Confirmed", "2020-03-01", "2022-12-31", 10000)
        if not cases_data.empty:
            cases_agg = _agg_country_simple(_ensure_date(cases_data))
            if not cases_agg.empty and len(cases_agg) > 10:

                mobility_data = []
                for transport in ['driving', 'walking', 'transit']:
                    mob_data = fetch_mobility(country, transport, "2020-03-01", "2022-12-31", 5000)
                    if not mob_data.empty:
                        mob_data = _ensure_date(mob_data)
                        mob_data['transport_type'] = transport
                        mobility_data.append(mob_data)

                if mobility_data:
                    all_mobility = pd.concat(mobility_data, ignore_index=True)

                    daily_mobility = all_mobility.groupby('DATE')['DIFFERENCE'].mean().reset_index()
                    daily_mobility.columns = ['DATE', 'avg_mobility_change']

                    merged_data = pd.merge(
                        cases_agg[['DATE', 'DIFFERENCE']],
                        daily_mobility,
                        on='DATE',
                        how='inner'
                    ).dropna()

                    if len(merged_data) > 10:
                        correlation, p_value = pearsonr(
                            merged_data['avg_mobility_change'],
                            merged_data['DIFFERENCE']
                        )

                        fig_mobility = make_subplots(
                            rows=2, cols=1,
                            subplot_titles=('Daily New Cases', 'Average Mobility Change'),
                            vertical_spacing=0.1, shared_xaxes=True
                        )

                        fig_mobility.add_trace(
                            go.Scatter(x=merged_data['DATE'], y=merged_data['DIFFERENCE'],
                                     mode='lines', name='Daily New Cases', line=dict(color=DANGER, width=2)),
                            row=1, col=1
                        )

                        fig_mobility.add_trace(
                            go.Scatter(x=merged_data['DATE'], y=merged_data['avg_mobility_change'],
                                     mode='lines', name='Avg Mobility Change', line=dict(color=PRIMARY, width=2)),
                            row=2, col=1
                        )

                        fig_mobility.update_layout(
                            title=f'🚗📈 Mobility vs Cases Timeline - {country}<br><sub>Correlation: {correlation:.3f} (p={p_value:.3f})</sub>',
                            height=600
                        )

                        fig_mobility.update_yaxes(title_text="New Cases", row=1, col=1)
                        fig_mobility.update_yaxes(title_text="% Change", row=2, col=1)
                        fig_mobility.update_xaxes(title_text="Date", row=2, col=1)

                        figs['mobility_cases'] = fig_defaults(fig_mobility)

    except Exception as e:
        print(f"Error creating mobility-cases correlation: {e}")

    return figs

external_stylesheets = [dbc.themes.DARKLY, "https://cdnjs.cloudflare.com/ajax/libs/font-awesome/6.5.0/css/all.min.css"]
app = dash.Dash(__name__, external_stylesheets=external_stylesheets, title="COVID-19 Intelligence Dashboard", suppress_callback_exceptions=True)

def kpi_card(title: str, value: str, icon: str, color: str) -> dbc.Card:
    return dbc.Card(
        dbc.CardBody(
            [
                html.Div([html.I(className=f"{icon}"), html.Span(" " + title)], className="text-muted"),
                html.H2(value, className="mb-0", style={"fontWeight": 700, "color": "white"}),
            ]
        ),
        class_name="shadow-sm",
        style={"borderLeft": f"5px solid {color}", "borderRadius": "14px",
               "backgroundColor": DARK_BG, "border": DARK_EDGE},
    )

def base_layout(children):
    return dbc.Card(children, class_name="shadow-sm",
                    style={"borderRadius": 14, "backgroundColor": DARK_BG, "border": DARK_EDGE})

def fig_defaults(fig: go.Figure, title: str = "") -> go.Figure:
    fig.update_layout(
        template="plotly_dark",
        title=dict(text=title, x=0.02, xanchor="left", y=0.96,
                   font=dict(size=18, family="Inter, Segoe UI", color="white")),
        margin=dict(l=40, r=24, t=60, b=50),
        legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="left", x=0.02,
                    bgcolor="rgba(0,0,0,0)"),
        hovermode="x unified",
        paper_bgcolor=DARK_BG,
        plot_bgcolor=DARK_BG,
        font=dict(color="white"),
        xaxis=dict(showgrid=True, gridcolor=GRID_COLOR, linecolor=AXIS_LINE, title="Date"),
        yaxis=dict(showgrid=True, gridcolor=GRID_COLOR, linecolor=AXIS_LINE),
        modebar_remove=["lasso2d", "select2d"],
    )
    return fig

COUNTRIES = fetch_countries()
DEFAULT_COUNTRY = (
    "Latvia" if "Latvia" in COUNTRIES else
    ("LV" if "LV" in COUNTRIES else
     ("US" if "US" in COUNTRIES else
      ("United States" if "United States" in COUNTRIES else
       (COUNTRIES[0] if COUNTRIES else None))))
)

app.layout = dbc.Container(
    [
        dcc.Store(id="store-last-click", data=None),
        dcc.Download(id="download-csv"),
        dcc.Store(id="store-last-data-tab", data="tab-cases"),
        dbc.Row(
            [
                dbc.Col(
                    html.Div([html.H2("🦠 COVID-19 Intelligence Dashboard", className="mb-1", style={"fontWeight": 800}),
                              html.P("Interactive analytics with ML-powered forecasting & clustering", className="text-muted")]),
                    md=8,
                ),
                dbc.Col(
                    dbc.ButtonGroup([dbc.Button("Download CSV", id="btn-download", color="primary", outline=True, class_name="me-2"),
                                     dbc.Button("Refresh", id="btn-refresh", color="secondary", outline=True)]),
                    md=4,
                    class_name="d-flex align-items-center justify-content-md-end mt-3 mt-md-0",
                ),
            ],
            class_name="py-3",
        ),

        base_layout(
            dbc.CardBody(
                dbc.Row(
                    [
                        dbc.Col(
                            [dbc.Label("Country"),
                             dcc.Dropdown(id="dd-country",
                                          options=[{"label": c, "value": c} for c in COUNTRIES],
                                          value=DEFAULT_COUNTRY, clearable=False, placeholder="Select a country",
                                          style={"zIndex": 10000,  "color": "black"})],
                            md=4,
                        ),
                        dbc.Col(
                            [dbc.Label("Date range"),
                             dcc.DatePickerRange(id="dp-range", start_date=date(2020, 3, 1), end_date=date.today(), display_format="YYYY-MM-DD")],
                            md=4,
                        ),
                        dbc.Col(
                            [dbc.Label("Options"),
                             dbc.Checklist(id="chk-options",
                                           options=[{"label": " 7-day average", "value": "avg7"},
                                                    {"label": " Log scale", "value": "logy"}],
                                           value=["avg7"], inline=True)],
                            md=4,
                        ),
                    ]
                )
            )
        ),

        dbc.Row(id="row-kpis", class_name="g-3 my-2"),

        dbc.Tabs(
            [dbc.Tab(label="Cases", tab_id="tab-cases", tab_class_name="fw-semibold"),
             dbc.Tab(label="Vaccinations", tab_id="tab-vacc", tab_class_name="fw-semibold"),
             dbc.Tab(label="Mobility", tab_id="tab-mobility", tab_class_name="fw-semibold"),
             dbc.Tab(label="Economy", tab_id="tab-econ", tab_class_name="fw-semibold"),
             dbc.Tab(label="Correlations", tab_id="tab-correlations", tab_class_name="fw-semibold"),
             dbc.Tab(label="🔮 Forecasting", tab_id="tab-forecast", tab_class_name="fw-semibold"),
             dbc.Tab(label="🎯 Clustering", tab_id="tab-clustering", tab_class_name="fw-semibold"),
             dbc.Tab(label="Comments", tab_id="tab-comments", tab_class_name="fw-semibold")],
            id="tabs", active_tab="tab-cases", class_name="mt-3",
        ),

        html.Div(id="tab-content", className="mt-3"),

        html.Hr(className="my-4"),
        html.Footer(dbc.Row([dbc.Col(html.Small("Built with Dash · Plotly · Bootstrap · Scikit-learn"), md=6),
                             dbc.Col(html.Small("API: /api/v1 — ML Analytics Enabled"), md=6, class_name="text-md-end text-muted")]))
    ],
    fluid=True,
)

def _ensure_date(df: pd.DataFrame) -> pd.DataFrame:
    if "DATE" in df.columns:
        df = df.copy()
        df["DATE"] = pd.to_datetime(df["DATE"], errors="coerce")
        df = df.sort_values("DATE")
    return df

def _moving_avg(s: pd.Series, window: int = 7) -> pd.Series:
    return s.rolling(window, min_periods=1).mean()

def _agg_country_simple(df: pd.DataFrame) -> pd.DataFrame:
    if df.empty or "DATE" not in df.columns:
        return df

    agg_cols = {}
    if "CASES" in df.columns:
        agg_cols["CASES"] = "sum"
    if "DIFFERENCE" in df.columns:
        agg_cols["DIFFERENCE"] = "sum"

    if not agg_cols:
        return pd.DataFrame()

    agg = df.groupby("DATE", as_index=False).agg(agg_cols)
    return agg.sort_values("DATE")

def build_cases_figs(country: str, start: str, end: str, options: List[str]) -> Dict[str, go.Figure]:
    confirmed_df = fetch_cases(country, "Confirmed", start, end)
    deaths_df = fetch_cases(country, "Deaths", start, end)

    figs: Dict[str, go.Figure] = {}

    if confirmed_df.empty:
        figs["cum"] = fig_defaults(go.Figure(), "No data available for selected date range")
        figs["daily"] = fig_defaults(go.Figure(), "No data available for selected date range")
        return figs

    confirmed_df = _ensure_date(confirmed_df)
    confirmed_agg = _agg_country_simple(confirmed_df)

    if confirmed_agg.empty:
        figs["cum"] = fig_defaults(go.Figure(), "No aggregated data")
        figs["daily"] = fig_defaults(go.Figure(), "No aggregated data")
        return figs

    cum_y = confirmed_agg["CASES"].astype(float)
    daily = confirmed_agg["DIFFERENCE"].astype(float)
    daily_sm = _moving_avg(daily) if "avg7" in options else daily

    print(f"DEBUG FIXED: Date range in chart: {confirmed_agg['DATE'].min()} to {confirmed_agg['DATE'].max()}")
    print(f"DEBUG FIXED: Cases range: {cum_y.min():,} to {cum_y.max():,}")

    f1 = go.Figure([go.Scatter(x=confirmed_agg["DATE"], y=cum_y, mode="lines", name="Cumulative confirmed",
                               line=dict(width=3, color=PRIMARY))])
    f1 = fig_defaults(f1, f"Confirmed cases — {country}")
    f1.update_yaxes(title_text="Cases", type=("log" if "logy" in options else "linear"))

    f2 = go.Figure([
        go.Bar(x=confirmed_agg["DATE"], y=daily, name="Daily", marker_color=ACCENT, opacity=0.35),
        go.Scatter(x=confirmed_agg["DATE"], y=daily_sm, mode="lines", name="7-day avg",
                   line=dict(width=3, color=ACCENT)),
    ])
    f2 = fig_defaults(f2, f"Daily new cases — {country}")
    f2.update_yaxes(title_text="Count")

    figs["cum"] = f1
    figs["daily"] = f2

    if not deaths_df.empty:
        deaths_df = _ensure_date(deaths_df)
        deaths_agg = _agg_country_simple(deaths_df)

        if not deaths_agg.empty:
            deaths_y = deaths_agg["CASES"].astype(float)
            f3 = go.Figure()
            f3.add_trace(go.Scatter(x=confirmed_agg["DATE"], y=cum_y, mode="lines", name="Confirmed",
                                    line=dict(width=3, color=PRIMARY)))
            f3.add_trace(go.Scatter(x=deaths_agg["DATE"], y=deaths_y, mode="lines", name="Deaths",
                                    line=dict(width=3, color=DANGER)))
            f3 = fig_defaults(f3, f"Confirmed vs Deaths — {country}")
            f3.update_yaxes(title_text="Count", type=("log" if "logy" in options else "linear"))
            figs["compare"] = f3

    return figs

def build_vacc_figs(country: str, start: str, end: str) -> Dict[str, go.Figure]:
    df = _ensure_date(fetch_vacc(country, start, end))
    figs: Dict[str, go.Figure] = {}
    if df.empty:
        figs["vacc"] = fig_defaults(go.Figure(), "No vaccination data")
        return figs

    f = go.Figure()
    if "PEOPLE_VACCINATED" in df.columns and df["PEOPLE_VACCINATED"].notna().any():
        f.add_trace(go.Scatter(x=df["DATE"], y=df["PEOPLE_VACCINATED"], mode="lines", name="1+ dose", line=dict(width=3, color=OK)))
    if "PEOPLE_FULLY_VACCINATED" in df.columns and df["PEOPLE_FULLY_VACCINATED"].notna().any():
        f.add_trace(go.Scatter(x=df["DATE"], y=df["PEOPLE_FULLY_VACCINATED"], mode="lines", name="Fully vaccinated", line=dict(width=3, color=PRIMARY)))
    f = fig_defaults(f, f"Vaccination progress — {country}")
    f.update_yaxes(title_text="People")

    if "PEOPLE_VACCINATED_PER_HUNDRED" in df.columns:
        r = px.area(df, x="DATE", y="PEOPLE_VACCINATED_PER_HUNDRED")
        r = fig_defaults(r, f"Vaccinated per 100 — {country}")
        r.update_traces(hovertemplate="%{y:.2f} per 100 people")
        figs["rate"] = r

    figs["vacc"] = f
    return figs

def build_mobility_figs(country: str, start: str, end: str) -> Dict[str, go.Figure]:
    figs: Dict[str, go.Figure] = {}
    for mode, color in [("driving", PRIMARY), ("walking", ACCENT), ("transit", DANGER)]:
        df = _ensure_date(fetch_mobility(country, mode, start, end))
        if df.empty:
            continue
        m = go.Figure([go.Scatter(x=df["DATE"], y=df["DIFFERENCE"], mode="lines", name=mode.title(), line=dict(width=3, color=color))])
        m.add_hline(y=0, line_dash="dot", line_color="rgba(255,255,255,0.25)")
        m = fig_defaults(m, f"{mode.title()} mobility vs baseline — {country}")
        m.update_yaxes(title_text="% from baseline")
        figs[mode] = m
    return figs

def build_econ_figs(country: str) -> Dict[str, go.Figure]:
    years = [2019, 2020, 2021, 2022, 2023]
    gdp_rows, hdi_rows = [], []
    for y in years:
        d = fetch_enriched(country, y)
        for it in d.get("gdp", []):
            if it.get("country", "").upper() == country.upper():
                gdp_rows.append({"year": y, "gdp_per_capita": it["gdp_per_capita"]})
        for it in d.get("hdi", []):
            if it.get("country", "").upper() == country.upper():
                hdi_rows.append({"year": y, "hdi": it["hdi"]})

    figs: Dict[str, go.Figure] = {}
    if gdp_rows:
        gdp_df = pd.DataFrame(gdp_rows)
        f = px.bar(gdp_df, x="year", y="gdp_per_capita")
        f = fig_defaults(f, f"GDP per capita — {country}")
        figs["gdp"] = f
    if hdi_rows:
        hdi_df = pd.DataFrame(hdi_rows)
        h = px.line(hdi_df, x="year", y="hdi", markers=True)
        h = fig_defaults(h, f"Human Development Index — {country}")
        figs["hdi"] = h
    return figs

def build_forecast_figs(country: str, start: str, end: str, forecast_days: int = 30) -> Dict[str, go.Figure]:
    figs = {}

    confirmed_df = fetch_cases(country, "Confirmed", start, end)

    if confirmed_df.empty:
        figs["forecast"] = fig_defaults(go.Figure(), "No data available for forecasting")
        return figs

    confirmed_df = _ensure_date(confirmed_df)
    confirmed_agg = _agg_country_simple(confirmed_df)

    if len(confirmed_agg) < 20:
        figs["forecast"] = fig_defaults(go.Figure(), "Insufficient data for forecasting (need at least 20 days)")
        return figs

    daily_cases = confirmed_agg["DIFFERENCE"].fillna(0)

    forecast, confidence = forecast_time_series(daily_cases, forecast_days, "exponential")

    if forecast is None:
        figs["forecast"] = fig_defaults(go.Figure(), "Could not generate forecast")
        return figs

    last_date = confirmed_agg["DATE"].max()
    future_dates = pd.date_range(start=last_date + timedelta(days=1), periods=forecast_days)

    fig = go.Figure()

    fig.add_trace(go.Scatter(
        x=confirmed_agg["DATE"],
        y=daily_cases,
        mode="lines",
        name="Historical Daily Cases",
        line=dict(color=PRIMARY, width=2)
    ))

    fig.add_trace(go.Scatter(
        x=future_dates,
        y=forecast,
        mode="lines",
        name="Forecast",
        line=dict(color=FORECAST, width=2, dash="dash")
    ))

    if confidence:
        lower, upper = confidence
        fig.add_trace(go.Scatter(
            x=np.concatenate([future_dates, future_dates[::-1]]),
            y=np.concatenate([upper, lower[::-1]]),
            fill="toself",
            fillcolor="rgba(255, 107, 107, 0.2)",
            line=dict(color="rgba(255,255,255,0)"),
            showlegend=True,
            name="95% Confidence Interval"
        ))

    fig = fig_defaults(fig, f"📈 {forecast_days}-Day Forecast — {country}")
    fig.update_yaxes(title_text="Daily New Cases")

    fig.add_vline(x=last_date, line_dash="dot", line_color="gray", opacity=0.5)
    fig.add_annotation(x=last_date, y=daily_cases.max(), text="Forecast starts →", showarrow=False)

    figs["forecast"] = fig

    cum_cases = confirmed_agg["CASES"].iloc[-1]
    forecast_cumulative = cum_cases + np.cumsum(forecast)

    fig2 = go.Figure()

    fig2.add_trace(go.Scatter(
        x=confirmed_agg["DATE"],
        y=confirmed_agg["CASES"],
        mode="lines",
        name="Historical Total Cases",
        line=dict(color=PRIMARY, width=3)
    ))

    fig2.add_trace(go.Scatter(
        x=future_dates,
        y=forecast_cumulative,
        mode="lines",
        name="Projected Total Cases",
        line=dict(color=FORECAST, width=3, dash="dash")
    ))

    fig2 = fig_defaults(fig2, f"📊 Cumulative Cases Projection — {country}")
    fig2.update_yaxes(title_text="Total Cases")
    fig2.add_vline(x=last_date, line_dash="dot", line_color="gray", opacity=0.5)

    figs["forecast_cumulative"] = fig2

    return figs

def build_clustering_figs(countries_list: List[str], start: str, end: str) -> Dict[str, go.Figure]:
    figs = {}

    countries_data = {}
    for country in countries_list[:15]:
        df = fetch_cases(country, "Confirmed", start, end)
        if not df.empty:
            df = _ensure_date(df)
            agg = _agg_country_simple(df)
            countries_data[country] = agg

    if len(countries_data) < 3:
        figs["cluster"] = fig_defaults(go.Figure(), "Insufficient data for clustering (need at least 3 countries)")
        return figs

    cluster_results = cluster_regions(countries_data, n_clusters=min(5, len(countries_data)))

    if not cluster_results["clusters"]:
        figs["cluster"] = fig_defaults(go.Figure(), "Could not perform clustering")
        return figs

    fig = go.Figure()

    for cluster_id in cluster_results["characteristics"]:
        cluster_countries = cluster_results["characteristics"][cluster_id]["countries"]

        x_coords = []
        y_coords = []
        texts = []

        for country in cluster_countries:
            if country in cluster_results["pca_coordinates"]:
                coords = cluster_results["pca_coordinates"][country]
                x_coords.append(coords[0])
                y_coords.append(coords[1])
                texts.append(country)

        if x_coords:
            fig.add_trace(go.Scatter(
                x=x_coords,
                y=y_coords,
                mode="markers+text",
                name=f"Cluster {cluster_id + 1}",
                text=texts,
                textposition="top center",
                marker=dict(size=12, color=CLUSTER_COLORS[cluster_id % len(CLUSTER_COLORS)])
            ))

    fig = fig_defaults(fig, "🎯 Country Clustering by COVID-19 Patterns")
    fig.update_xaxes(title_text=f"PC1 ({cluster_results['explained_variance'][0]:.1%} variance)")
    fig.update_yaxes(title_text=f"PC2 ({cluster_results['explained_variance'][1]:.1%} variance)")

    figs["cluster"] = fig

    fig2 = go.Figure()

    cluster_ids = []
    avg_cases = []
    growth_rates = []
    volatilities = []
    cluster_sizes = []

    for cluster_id, chars in cluster_results["characteristics"].items():
        cluster_ids.append(f"Cluster {cluster_id + 1}")
        avg_cases.append(chars["avg_log_peak_cases"])
        growth_rates.append(chars.get("avg_growth_rate", 0))
        volatilities.append(chars.get("avg_volatility", 0))
        cluster_sizes.append(chars["size"])

    fig2 = make_subplots(
        rows=2, cols=2,
        subplot_titles=("Log Peak Cases", "Growth Rate", "Volatility", "Cluster Size"),
        specs=[[{"type": "bar"}, {"type": "bar"}], [{"type": "bar"}, {"type": "bar"}]]
    )

    fig2.add_trace(
        go.Bar(x=cluster_ids, y=avg_cases, marker_color=PRIMARY, name="Log Peak Cases"),
        row=1, col=1
    )

    fig2.add_trace(
        go.Bar(x=cluster_ids, y=growth_rates, marker_color=ACCENT, name="Growth Rate"),
        row=1, col=2
    )

    fig2.add_trace(
        go.Bar(x=cluster_ids, y=volatilities, marker_color=OK, name="Volatility"),
        row=2, col=1
    )

    fig2.add_trace(
        go.Bar(x=cluster_ids, y=cluster_sizes, marker_color=DANGER, name="Size"),
        row=2, col=2
    )

    fig2.update_layout(
        title="📊 Cluster Characteristics Analysis",
        showlegend=False,
        height=600
    )

    figs["cluster_chars"] = fig2

    return figs

def _datapoint_id_with_context(country: str, current_tab: str, last_data_tab: str) -> str:
    current_date = str(date.today())

    tab_for_data = last_data_tab if current_tab in ["tab-comments", "tab-correlations", "tab-forecast", "tab-clustering"] else current_tab

    if tab_for_data == "tab-cases":
        feature = "Confirmed"
        dataset = "JHU_COVID_19"
    elif tab_for_data == "tab-vacc":
        feature = "total_vaccinations_per_hundred"
        dataset = "OWID_VACCINATIONS"
    elif tab_for_data == "tab-mobility":
        feature = "driving"
        dataset = "APPLE_MOBILITY"
    elif tab_for_data == "tab-econ":
        feature = "gdp_per_capita"
        dataset = "RAW_GDP_PER_CAPITA_STG"
    elif tab_for_data == "tab-correlations":
        feature = "correlation_analysis"
        dataset = "CORRELATION_ANALYSIS"
    elif tab_for_data == "tab-forecast":
        feature = "time_series_forecast"
        dataset = "FORECAST_ANALYSIS"
    elif tab_for_data == "tab-clustering":
        feature = "clustering_analysis"
        dataset = "CLUSTERING_ANALYSIS"
    else:
        feature = "general_comment"
        dataset = "GENERAL_COMMENTS"

    return f"{dataset}|{country}|{current_date}|{feature}"

@app.callback(
    Output("row-kpis", "children"),
    Input("dd-country", "value"),
    Input("dp-range", "start_date"),
    Input("dp-range", "end_date"),
)
def update_kpis(country: str, start_date: str, end_date: str):
    confirmed_df = fetch_cases(country, "Confirmed", start_date, end_date)
    deaths_df = fetch_cases(country, "Deaths", start_date, end_date)

    if confirmed_df.empty:
        return [dbc.Alert("No data for selected range", color="warning")]

    confirmed_df = _ensure_date(confirmed_df)
    confirmed_agg = _agg_country_simple(confirmed_df)

    if confirmed_agg.empty:
        return [dbc.Alert("No aggregated data available", color="warning")]

    total_confirmed = int(confirmed_agg["CASES"].iloc[-1])

    last30_daily = confirmed_agg.tail(30)["DIFFERENCE"].fillna(0).sum()
    new_30 = int(last30_daily)

    total_deaths = None
    if not deaths_df.empty:
        deaths_df = _ensure_date(deaths_df)
        deaths_agg = _agg_country_simple(deaths_df)
        if not deaths_agg.empty:
            total_deaths = int(deaths_agg["CASES"].iloc[-1])

    return [
        dbc.Col(kpi_card("Total confirmed", f"{total_confirmed:,}", "fa-solid fa-virus", PRIMARY), md=4),
        dbc.Col(kpi_card("New (30 days)", f"{new_30:,}", "fa-regular fa-chart-bar", ACCENT), md=4),
        dbc.Col(kpi_card("Total deaths", (f"{total_deaths:,}" if total_deaths is not None else "—"),
                         "fa-solid fa-skull-crossbones", DANGER), md=4),
    ]

@app.callback(
    Output("tab-content", "children"),
    Input("tabs", "active_tab"),
    Input("dd-country", "value"),
    Input("dp-range", "start_date"),
    Input("dp-range", "end_date"),
    Input("chk-options", "value"),
)
def render_tab(active_tab: str, country: str, start_date: str, end_date: str, options: List[str]):
    if active_tab == "tab-cases":
        figs = build_cases_figs(country, start_date, end_date, options)
        return dbc.Row([
            dbc.Col(base_layout(dbc.CardBody(dcc.Graph(figure=figs["cum"]))), md=12, class_name="mb-3"),
            dbc.Col(base_layout(dbc.CardBody(dcc.Graph(figure=figs["daily"]))), md=12, class_name="mb-3"),
            (dbc.Col(base_layout(dbc.CardBody(dcc.Graph(figure=figs["compare"]))), md=12) if "compare" in figs else html.Div()),
        ])

    elif active_tab == "tab-vacc":
        figs = build_vacc_figs(country, start_date, end_date)
        children = [dbc.Col(base_layout(dbc.CardBody(dcc.Graph(figure=figs["vacc"]))), md=12)]
        if "rate" in figs:
            children.append(dbc.Col(base_layout(dbc.CardBody(dcc.Graph(figure=figs["rate"]))), md=12))
        return dbc.Row(children, class_name="g-3")

    elif active_tab == "tab-mobility":
        figs = build_mobility_figs(country, start_date, end_date)
        if not figs:
            return dbc.Alert("No mobility data", color="info")
        cols = []
        for key in ["driving", "walking", "transit"]:
            if key in figs:
                cols.append(dbc.Col(base_layout(dbc.CardBody(dcc.Graph(figure=figs[key]))), md=12, class_name="mb-3"))
        return dbc.Row(cols)

    elif active_tab == "tab-econ":
        figs = build_econ_figs(country)
        if not figs:
            return dbc.Alert("No economic indicators", color="info")
        rows = []
        if "gdp" in figs:
            rows.append(dbc.Col(base_layout(dbc.CardBody(dcc.Graph(figure=figs["gdp"]))), md=12, class_name="mb-3"))
        if "hdi" in figs:
            rows.append(dbc.Col(base_layout(dbc.CardBody(dcc.Graph(figure=figs["hdi"]))), md=12))
        return dbc.Row(rows)

    elif active_tab == "tab-correlations":
        figs = build_correlation_charts(country)
        charts = []

        if 'vacc_gdp' in figs:
            charts.append(
                dbc.Col(
                    base_layout(
                        dbc.CardBody([
                            dcc.Graph(figure=figs['vacc_gdp']),
                            html.P("Statistical analysis showing relationship between vaccination rates vs economic indicators across countries.", className="text-muted mt-2")
                        ])
                    ),
                    md=12, className="mb-3"
                )
            )

        if 'mobility_cases' in figs:
            charts.append(
                dbc.Col(
                    base_layout(
                        dbc.CardBody([
                            dcc.Graph(figure=figs['mobility_cases']),
                            html.P(f"Time series analysis of mobility patterns vs COVID cases for {country}.", className="text-muted mt-2")
                        ])
                    ),
                    md=12, className="mb-3"
                )
            )

        if not charts:
            charts.append(
                dbc.Col(dbc.Alert("No correlation data available for the selected parameters.", color="info"), md=12)
            )

        return dbc.Row([
            dbc.Col([
                html.H4("🔬 Correlation Analysis", className="mb-3"),
                html.P("Advanced statistical analysis revealing relationships between COVID-19 metrics, "
                       "economic indicators, and social development factors.", className="text-muted mb-4")
            ], md=12),
            *charts
        ])

    elif active_tab == "tab-forecast":
        figs = build_forecast_figs(country, start_date, end_date)
        charts = []

        if 'forecast' in figs:
            charts.append(
                dbc.Col(
                    base_layout(
                        dbc.CardBody([
                            dcc.Graph(figure=figs['forecast']),
                            html.P("Time series forecasting using exponential smoothing with confidence intervals.", className="text-muted mt-2")
                        ])
                    ),
                    md=12, className="mb-3"
                )
            )

        if 'forecast_cumulative' in figs:
            charts.append(
                dbc.Col(
                    base_layout(
                        dbc.CardBody([
                            dcc.Graph(figure=figs['forecast_cumulative']),
                            html.P("Projected cumulative case growth based on historical trends.", className="text-muted mt-2")
                        ])
                    ),
                    md=12, className="mb-3"
                )
            )

        if not charts:
            charts.append(
                dbc.Col(dbc.Alert("Unable to generate forecast. Need at least 20 days of data.", color="warning"), md=12)
            )

        return dbc.Row([
            dbc.Col([
                html.H4("🔮 Time Series Forecasting", className="mb-3"),
                html.P("Machine learning-powered forecasting to predict future COVID-19 trends based on historical patterns.", className="text-muted mb-4")
            ], md=12),
            *charts
        ])

    elif active_tab == "tab-clustering":
        figs = build_clustering_figs(COUNTRIES, start_date, end_date)
        charts = []

        if 'cluster' in figs:
            charts.append(
                dbc.Col(
                    base_layout(
                        dbc.CardBody([
                            dcc.Graph(figure=figs['cluster']),
                            html.P("Principal Component Analysis (PCA) visualization of country clusters based on COVID-19 patterns.", className="text-muted mt-2")
                        ])
                    ),
                    md=12, className="mb-3"
                )
            )

        if 'cluster_chars' in figs:
            charts.append(
                dbc.Col(
                    base_layout(
                        dbc.CardBody([
                            dcc.Graph(figure=figs['cluster_chars']),
                            html.P("Cluster characteristics showing average patterns across different groups of countries.", className="text-muted mt-2")
                        ])
                    ),
                    md=12, className="mb-3"
                )
            )

        if not charts:
            charts.append(
                dbc.Col(dbc.Alert("Unable to perform clustering analysis. Need data from at least 3 countries.", color="warning"), md=12)
            )

        return dbc.Row([
            dbc.Col([
                html.H4("🎯 Regional Clustering Analysis", className="mb-3"),
                html.P("Machine learning clustering to identify countries with similar COVID-19 spread patterns and outcomes using K-means algorithm.", className="text-muted mb-4")
            ], md=12),
            *charts
        ])

    elif active_tab == "tab-comments":
        return base_layout(
            dbc.CardBody([
                html.H5("Add a comment / annotation", className="mb-3"),
                dbc.Alert([
                    html.I(className="fas fa-info-circle me-2"),
                    html.Span(id="comment-context-info", children="Loading context...")
                ], color="info", className="mb-3"),
                dbc.Row([
                    dbc.Col(dbc.Input(id="inp-email", placeholder="Your email (required)", type="email"), md=3),
                    dbc.Col(dbc.Input(id="inp-user", placeholder="Your name (optional)", type="text"), md=3),
                    dbc.Col(dbc.Textarea(id="inp-comment", placeholder="Share your insight...", rows=2), md=4),
                    dbc.Col(dbc.Button("Save", id="btn-save", color="success", class_name="w-100"), md=2),
                ], class_name="g-2"),
                html.Small("Note: Comments will be saved for the current country and chart type.", className="text-muted"),
                html.Div(id="save-status", className="mt-2"),
                html.Hr(),
                html.H5("Recent comments", className="mb-3"),
                html.Div(id="comments-list"),
            ])
        )
    return html.Div()

@app.callback(
    Output("save-status", "children"),
    Output("inp-email", "value"),
    Output("inp-user", "value"),
    Output("inp-comment", "value"),
    Input("btn-save", "n_clicks"),
    State("dd-country", "value"),
    State("tabs", "active_tab"),
    State("store-last-data-tab", "data"),
    State("inp-email", "value"),
    State("inp-user", "value"),
    State("inp-comment", "value"),
    prevent_initial_call=True,
)
def save_comment(n, country, active_tab, last_data_tab, email, user, comment):
    if not n:
        return dash.no_update, dash.no_update, dash.no_update, dash.no_update

    if not comment or not str(comment).strip():
        return dbc.Alert("Please write a comment.", color="warning"), dash.no_update, dash.no_update, dash.no_update

    if not email or "@" not in str(email):
        return dbc.Alert("Email is required to save a comment.", color="warning"), dash.no_update, dash.no_update, dash.no_update

    context_tab = last_data_tab if active_tab in ["tab-comments", "tab-correlations", "tab-forecast", "tab-clustering"] else active_tab
    context_name = {
        "tab-cases": "Cases Analysis",
        "tab-vacc": "Vaccination Data",
        "tab-mobility": "Mobility Trends",
        "tab-econ": "Economic Indicators",
        "tab-correlations": "Correlation Analysis",
        "tab-forecast": "Time Series Forecasting",
        "tab-clustering": "Clustering Analysis"
    }.get(context_tab, "General")

    payload = {
        "datapointId": _datapoint_id_with_context(country, active_tab, last_data_tab),
        "type": "comment",
        "text": str(comment).strip(),
        "labels": [context_tab.replace("tab-", ""), country, context_name],
        "sourceIds": [],
        "status": "active",
        "attachments": [],
        "authorEmail": str(email).strip()
    }

    comment_id = post_comment(payload)
    if comment_id:
        return dbc.Alert(f"Comment saved for {context_name} ✓", color="success"), dash.no_update, "", ""

    return dbc.Alert("Could not save comment", color="danger"), dash.no_update, dash.no_update, dash.no_update

@app.callback(
    Output("comments-list", "children"),
    Input("tabs", "active_tab"),
    Input("dd-country", "value"),
    Input("btn-save", "n_clicks"),
)
def load_comments(active_tab, country, _):
    items = get_comments()

    if not items:
        return dbc.Alert("No comments yet.", color="light")

    cards = []
    for a in items:
        created_at = a.get("createdAt", "")
        if created_at:
            try:
                dt = datetime.fromisoformat(created_at.replace("Z", "+00:00"))
                time_str = dt.strftime("%Y-%m-%d %H:%M")
            except:
                time_str = str(created_at)
        else:
            time_str = "Unknown time"

        author_id = a.get("authorId", "")
        user_display = f"User {author_id[:8]}..." if author_id else "Anonymous"
        text_display = a.get("text", "")
        labels = a.get("labels", [])

        chips = dbc.Stack([
            dbc.Badge(label, color="secondary", class_name="me-1")
            for label in labels[:3]
        ], direction="horizontal")

        cards.append(
            dbc.Card(
                dbc.CardBody([
                    html.H6(user_display, className="mb-1"),
                    html.P(text_display, className="mb-2"),
                    html.Small(time_str, className="text-muted"),
                    html.Div(chips, className="mt-2") if labels else html.Div()
                ]),
                class_name="mb-2 shadow-sm",
                style={"borderLeft": f"4px solid {OK}"},
            )
        )
    return cards

@app.callback(
    Output("download-csv", "data"),
    Input("btn-download", "n_clicks"),
    State("dd-country", "value"),
    State("dp-range", "start_date"),
    State("dp-range", "end_date"),
    prevent_initial_call=True,
)
def download_csv(n, country, start, end):
    if not n:
        return dash.no_update
    df = fetch_cases(country, "Confirmed", start, end)
    if df.empty:
        return dash.no_update
    return dcc.send_data_frame(df.to_csv, f"cases_{country}_{start}_{end}.csv", index=False)

@app.callback(
    Output("dd-country", "options"),
    Input("btn-refresh", "n_clicks")
)
def refresh_countries(_):
    countries = fetch_countries()
    return [{"label": c, "value": c} for c in countries]

@app.callback(
    Output("store-last-data-tab", "data"),
    Input("tabs", "active_tab"),
    State("store-last-data-tab", "data"),
    prevent_initial_call=True
)
def track_last_data_tab(active_tab, current_last_tab):
    if active_tab in ["tab-cases", "tab-vacc", "tab-mobility", "tab-econ"]:
        return active_tab
    return current_last_tab

@app.callback(
    Output("comment-context-info", "children"),
    Input("store-last-data-tab", "data"),
    Input("dd-country", "value"),
    Input("tabs", "active_tab")
)
def update_comment_context(last_data_tab, country, active_tab):
    context_names = {
        "tab-cases": "📈 Cases Analysis",
        "tab-vacc": "💉 Vaccination Data",
        "tab-mobility": "🚗 Mobility Trends",
        "tab-econ": "📊 Economic Indicators",
        "tab-correlations": "🔬 Correlation Analysis",
        "tab-forecast": "🔮 Time Series Forecasting",
        "tab-clustering": "🎯 Clustering Analysis"
    }

    if active_tab in ["tab-comments", "tab-correlations", "tab-forecast", "tab-clustering"]:
        context_tab = last_data_tab
    else:
        context_tab = active_tab

    context_name = context_names.get(context_tab, "📝 General")
    return f"Your comment will be saved in context: {context_name} for {country}"

if __name__ == "__main__":
    try:
        app.run(debug=True, host="0.0.0.0", port=8050)
    except AttributeError:
        app.run_server(debug=True, host="0.0.0.0", port=8050)