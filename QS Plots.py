from pyxirr import xirr
import streamlit as st
import pandas as pd
import yfinance as yf
import numpy as np
import pandas_datareader as reader
#from scipy.stats import norm
from datetime import datetime
from datetime import timedelta
from datetime import date
import matplotlib.pyplot as plt
from matplotlib.dates import DateFormatter
import matplotlib
import investpy as inv
plt.style.use('default')
import warnings 
warnings.filterwarnings(action='ignore')
import plotly.express as px
import plotly.graph_objects as go
import quantstats as qs
from matplotlib import colors
from scipy import stats

is_dark_mode = st.get_option("theme.base") == "dark"

def color_negativo_positivo_cero(val):
    #val = round(val, 2)  # Redondear el valor a dos decimales
    color = 'black'  # Color predeterminado para valores diferentes de cero
    if val < 0:
        color = '#E57373'  # Rojo para valores negativos
    elif val > 0:
        color = '#81C784'  # Verde para valores positivos
    elif val == 0:
        color = '#9E9E9E'  # Gris para valores iguales a cero
    return 'color: %s' % color

def style_performance(val, is_dark_mode=False):
    light_palette = {
        'positive': '#4CAF50',  # Green
        'negative': '#F44336',  # Red
        'neutral': '#9E9E9E',   # Gray
        'background': '#FFFFFF' # White
    }
    dark_palette = {
        'positive': '#81C784',  # Light Green
        'negative': '#E57373',  # Light Red
        'neutral': '#B0BEC5',   # Light Blue-Gray
        'background': '#424242' # Dark Gray
    }

    palette = dark_palette if is_dark_mode else light_palette

    if pd.isna(val):
        return 'background-color: transparent'
    
    if isinstance(val, str):
        if val.endswith('%'):
            num_val = float(val.strip('%')) / 100
        else:
            return f'background-color: {palette["background"]}; color: {"white" if is_dark_mode else "black"}; font-weight: bold;'
    else:
        num_val = val

    if num_val < 0:
        color = palette['negative']
    elif num_val > 0:
        color = palette['positive']
    else:
        color = palette['neutral']

    intensity = 1 - 1 / (1 + abs(num_val) * 20)
    
    rgb = colors.to_rgb(color)
    rgba = (*rgb, intensity)

    text_color = color
    
    return f'background-color: rgba{rgba}; color: {text_color}; font-weight: bold;'

def apply_styles(df, is_dark_mode=False):
    return df.style.applymap(lambda x: style_performance(x, is_dark_mode))


st.set_page_config(layout="wide")

st.sidebar.title("Dashboard")

st.sidebar.subheader((datetime.utcnow() + timedelta(hours=2)).strftime('%A, %d %B %Y %H:%M'))


# Lista de opciones para el dropdown
opciones = ["Omite monetarios","Incluye monetarios" ]

# Crear el dropdown
opcion_seleccionada = st.selectbox("Selecciona una opción", opciones)


## Procesamos toda la información que vamos a necesitar

@st.cache_data(ttl=15*60)
def process_portfolio_data(opcion_seleccionada):
    # Load and preprocess data
    movimientos = pd.read_excel("records.xlsx", parse_dates=True)
    movimientos = movimientos.set_index("Fecha")

    if opcion_seleccionada == "Omite monetarios":
        movimientos = movimientos[(movimientos["Description"] != "ETF monetario") & (movimientos["Description"] != "Fondo monetario")]
        fecha_inicio = "2023-09-01"
    else:
        fecha_inicio = "2023-01-01"

    fecha_hoy = datetime.now()
    fecha_formateada = fecha_hoy.strftime("%Y-%m-%d")

    rango_fechas = pd.date_range(fecha_inicio, end=fecha_formateada, freq="D")
    rango_fechas = rango_fechas[rango_fechas.dayofweek < 5]

    # Download prices
    precios = pd.DataFrame(index=rango_fechas)
    for i in movimientos["Yahoo Ticker"].dropna().unique():
        precios[i] = yf.download(i, start=fecha_inicio, progress=False)["Adj Close"]

    eurusd = yf.download("EURUSD=X", start=fecha_inicio, progress=False).resample("B").ffill()["Adj Close"]

    precios["WBIT"] = yf.download("BTC-USD", start=fecha_inicio, progress=False).resample("B").ffill()["Adj Close"] / eurusd * 0.0002396
    for ticker in ["JOE", "BN", "BAM"]:
        precios[ticker] = precios[ticker] / eurusd

    precios = precios.fillna(method="ffill")

    # Rename columns
    column_mapping_prices = {
        'CSH2.PA': '0.0 ETF monetario', '0P00002BDB.F': '0.1 Fondo monetario', 'U3O8.DE': '1.6 Uranio',
        'ZPRV.DE': '1.3 USA Small Value', '0P0001AINF.F': '1.1 World', '0P0001AINL.F': '1.4 Emergentes',
        'SMCX.MI': '1.2 Europa Small', 'BN': '2.2 Brookfield Corp', 'JOE': '2.3 St Joe', 'TL0.DE': '2.1 Tesla',
        'WBIT': '1.5 ETF bitcoin', 'BAM': '2.4 Brookfield AM'
    }

    column_mapping_else = {'ETF monetario': '0.0 ETF monetario', 'Fondo monetario': '0.1 Fondo monetario', 'Uranio': '1.6 Uranio', 'USA Small Value': '1.3 USA Small Value', 'World': '1.1 World',
    'Emergentes': '1.4 Emergentes', 'Europa Small': '1.2 Europa Small', 'Brookfield Corp': '2.2 Brookfield Corp', 
    'St Joe': '2.3 St Joe', 'Tesla': '2.1 Tesla', 'ETF bitcoin': '1.5 ETF bitcoin', 'Brookfield AM': '2.4 Brookfield AM'
                        }

    precios.rename(columns=column_mapping_prices, inplace=True)
    precios = precios.sort_index(axis=1)

    # Calculate returns
    rendimientos = precios.pct_change()

    # Download and calculate benchmark returns
    benchmark = yf.download("SPYI.DE", start=fecha_inicio, progress=False).resample("B").ffill()["Adj Close"]
    rendimiento_benchmark = benchmark.pct_change().fillna(0)

    # Calculate positions
    posiciones = pd.DataFrame(index=rango_fechas, columns=movimientos["Description"].unique())
    for i in movimientos["Description"].unique():
        posiciones[i] = movimientos[movimientos["Description"] == i].cumsum()["Flow unidades"]
    posiciones = posiciones.fillna(method="ffill").fillna(0)
    posiciones.rename(columns=column_mapping_else, inplace=True)
    posiciones = posiciones.sort_index(axis=1)

    # Calculate cost
    coste = pd.DataFrame(index=rango_fechas, columns=movimientos["Description"].unique())
    for i in movimientos["Description"].unique():
        coste[i] = movimientos[movimientos["Description"] == i].cumsum()["Flow"]
    coste = coste.fillna(method="ffill").fillna(0)
    coste.rename(columns=column_mapping_else, inplace=True)
    coste = coste[posiciones.columns]

    # Calculate value and weights
    valor = precios * posiciones
    pesos = valor.divide(valor.sum(axis=1), axis=0)

    # Calculate contribution and portfolio return
    contribucion = pesos.shift() * rendimientos
    rendimiento_portfolio = contribucion.sum(axis=1)

    # Calculate P&L
    pl = valor.add(coste)

    # Update movimientos
    movimientos.replace(column_mapping_else, inplace=True)
    movimientos["Ud sim benchmark"] = (-movimientos["Flow"] / benchmark.loc[movimientos.index]).cumsum()

    return rendimientos, rendimiento_benchmark, posiciones, coste, movimientos, pesos, contribucion, pl,valor,benchmark,rendimiento_portfolio,precios

rendimientos, rendimiento_benchmark, posiciones, coste, movimientos, pesos, contribucion, pl,valor,benchmark,rendimiento_portfolio,precios = process_portfolio_data(opcion_seleccionada)

##################################


st.divider()


fig = qs.plots.snapshot(rendimiento_portfolio,show=False,figsize=(8,8))

st.pyplot(fig)

fig = qs.plots.monthly_heatmap(rendimiento_portfolio,benchmark,show=False,figsize=(8,5.5))

st.pyplot(fig)


fig =  qs.plots.drawdowns_periods(rendimiento_portfolio,show=False,figsize=(8,5.5))

st.pyplot(fig)



fig =   qs.plots.rolling_volatility(rendimiento_portfolio,period=20,period_label="21 días",show=False,figsize=(8,5.5))

st.pyplot(fig)
