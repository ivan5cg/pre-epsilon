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

# calculamos el IRR del portfolio

results = []

for i in valor.index:

    date_i = i 

    flows = pd.DataFrame(movimientos).loc[:date_i,:]["Flow"].values  
    value_today = valor.loc[date_i].sum()
    flows_xirr = np.append(flows,value_today)

    flows_dates = pd.DataFrame(movimientos["Flow"])["Flow"].loc[:date_i].index
    value_today_date = valor.index[-1]
    dates_xirr = np.append(flows_dates.date,valor.index[-1].date())
    
    try:
        result = xirr(dates_xirr,flows_xirr,guess=0.2)
    except Exception as e:
        result = 0
    results.append(result)

xirr_portfolio = pd.DataFrame(index=valor.index,data=results)
xirr_portfolio.columns = ["IRR"]
xirr_portfolio = xirr_portfolio * 100


#################################

# calculamos el IRR del benchmark

results = []

valor_benchmark = movimientos["Ud sim benchmark"].groupby(level=0).last().reindex(precios.index, method='ffill') * benchmark

for i in valor.index:

    date_i = i 

    flows = pd.DataFrame(movimientos).loc[:date_i,:]["Flow"].values  

    value_today = valor_benchmark.loc[date_i] 


    flows_xirr = np.append(flows,value_today)

    flows_dates = pd.DataFrame(movimientos["Flow"])["Flow"].loc[:date_i].index
    value_today_date = valor.index[-1]
    dates_xirr = np.append(flows_dates.date,valor.index[-1].date())
    
    try:
        result = xirr(dates_xirr,flows_xirr,guess=0.5)
    except Exception as e:
        result = 0
    results.append(result)

    

xirr_benchmark = pd.DataFrame(index=valor.index,data=results)
xirr_benchmark.columns = ["IRR"]
xirr_benchmark = xirr_benchmark * 100


#################################

### primeros indicadores

col1, col2, col3, col4, col5,col6,col7 = st.columns(7)


with col1:
   st.write("  Valor")
   st.subheader(str(int(valor.sum(axis=1).iloc[-2])))


with col2:
   st.write("P/L hoy")

   pl_today = int(pl.sum(axis=1).diff().iloc[-1])
    
   pl_today_text = '{:,.0f} €'.format(int(pl.sum(axis=1).diff().iloc[-1]))

   if pl_today <= 0:
          
        st.subheader((f":red[{pl_today_text}]"))   

   else:
        st.subheader((f":green[{pl_today_text}]")) 


with col3:

    st.write("Hoy")

    returnhoy = (rendimiento_portfolio.iloc[-1]*100).round(2)

    if returnhoy <= 0:
          
        st.subheader((f":red[{returnhoy} %]"))   

    else:
        st.subheader((f":green[{returnhoy} %]")) 


with col4:

    st.write("P/L %")

    pl_cartera = (((valor.sum(axis=1) / -coste.sum(axis=1)).iloc[-1] - 1)*100).round(2)

    if pl_cartera <= 0:
          
        st.subheader((f":red[{pl_cartera} %]"))   

    else:
        st.subheader((f":green[{pl_cartera} %]")) 


with col5:

    st.write("2024")

    return2024 = (((1+rendimiento_portfolio["2024"]).cumprod()-1).iloc[-1]*100).round(2)

    if return2024 <= 0:
          
        st.subheader((f":red[{return2024} %]"))   

    else:
        st.subheader((f":green[{return2024} %]")) 


with col6:

    st.write("IRR")

    xirr_portfolio_ = xirr_portfolio.iloc[-1].round(2).values[0]

    if xirr_portfolio_ <= 0:
          
        st.subheader((f":red[{xirr_portfolio_} %]"))   

    else:
        st.subheader((f":green[{xirr_portfolio_} %]")) 


with col7:

    st.write("IRR delta")

    gap = xirr_portfolio.iloc[-1].round(2).values[0] - xirr_benchmark.iloc[-1].round(2).values[0]

    gap = gap.round(2)

    if gap <= 0:
          
        st.subheader((f":red[{gap} %]"))   

    else:
        st.subheader((f":green[{gap} %]")) 




st.divider()

######## tabla resumen

st.subheader("Resumen")


from pyxirr import xirr

xirr_df = pd.DataFrame(index=valor.index,columns=movimientos["Description"].unique())


@st.cache_data(ttl=360)
def compute_historic_irr(valor, movimientos):
    assets = movimientos["Description"].unique()
    dates = valor.index
    xirr_df = pd.DataFrame(index=dates, columns=assets)

    # Precompute asset flows and dates
    asset_flows = {asset: movimientos[movimientos["Description"] == asset]["Flow"] for asset in assets}
    asset_dates = {asset: flows.index.date for asset, flows in asset_flows.items()}

    for asset in assets:
        flows = asset_flows[asset]
        flow_dates = asset_dates[asset]
        asset_values = valor[asset]

        results = []
        for date in dates:
            date_i = date.date()
            
            # Use boolean indexing for faster selection
            mask = flow_dates <= date_i
            current_flows = flows[mask].values
            current_dates = flow_dates[mask]

            value_today = asset_values[date]
            
            flows_xirr = np.append(current_flows, value_today)
            dates_xirr = np.append(current_dates, date_i)

            try:
                result = xirr(dates_xirr, flows_xirr, guess=0.1)
            except Exception:
                result = np.nan

            results.append(result)

        xirr_df[asset] = results

    return xirr_df[valor.columns]


xirr_df = compute_historic_irr(valor,movimientos)


tabla_resumen = pd.DataFrame(index=valor.columns)

tabla_resumen["Posiciones"] = posiciones.iloc[-1]
tabla_resumen["Precios"] = precios.iloc[-1]
tabla_resumen["Valor"] = valor.iloc[-1]

tabla_resumen["Peso"] = pesos.iloc[-1] * 100

tabla_resumen["P/L"] = pl.iloc[-1]
tabla_resumen["P/L %"] = (valor/-coste - 1).iloc[-1] * 100
tabla_resumen["IRR"] = xirr_df.iloc[-1] * 100

tabla_resumen = tabla_resumen.style.applymap(color_negativo_positivo_cero, subset=['P/L', 'P/L %', 'IRR']).format({
    'Posiciones': "{:.0f}",
    'Precios': "{:.2f}",
    'Valor': "{:,.0f} €",
    'Peso': "{:,.1f} %",
    'P/L': "{:,.0f} €",
    'P/L %': "{:,.2f} %",
    'IRR': "{:,.2f} %"
})


col1, col2 = st.columns(2)

with col1:



    st.write(tabla_resumen)


with col2:

    import pytz

    # Set time zone to Madrid
    madrid_tz = pytz.timezone("Europe/Madrid")

    ny_tz = pytz.timezone("America/New_York")

    ccy_tz = pytz.timezone("Etc/GMT-1")

    btc_tz = pytz.timezone("Etc/UTC")

    # Define date range
    fecha_inicio_ = datetime.now() - timedelta(2) 
    fecha_hoy_ = datetime.now(madrid_tz)  # Ensure current date is in Madrid timezone
    fecha_formateada = fecha_hoy_.strftime("%Y-%m-%d %H:%M:%S")

    # Generate a range of 15-minute intervals from 08:00 to 23:00, Monday to Sunday
    rango_fechas_ = pd.date_range(start=fecha_inicio_, end=fecha_formateada, freq='15T', tz=madrid_tz)

    # Filter time range to keep only between 08:00 and 23:00
    rango_fechas_ = rango_fechas_[(rango_fechas_.hour >= 8) & (rango_fechas_.hour < 23)]

    # Create an empty DataFrame for prices
    precios_iday = pd.DataFrame(index=rango_fechas_)

    # Crear una copia de los tickers originales
    tickers_iday = movimientos["Yahoo Ticker"].dropna().unique().copy()

    # Reemplazar '0P0001AINL.F' por 'IWDA.AS' y '0P0001AINF.F' por 'EUNM.DE' en la copia
    tickers_iday = [ticker.replace('0P0001AINL.F', 'IWDA.AS').replace('0P0001AINF.F', 'EUNM.DE') for ticker in tickers_iday]


    # Ahora puedes usar la lista `tickers` para el bucle sin afectar al DataFrame original


    # Download intraday data for each ticker and resample it for 15-min intervals
    for i in tickers_iday:


        if i in ['U3O8.DE', 'ZPRV.DE', 'IWDA.AS', 'EUNM.DE', 'SMCX.MI','TL0.DE']:

            data = yf.download(i, start=fecha_inicio_, interval="15m", progress=False)
            data = data.tz_localize(madrid_tz)  # Convert data to Madrid timezone
            precios_iday[i] = data["Adj Close"].reindex(rango_fechas_, method="ffill")  # Resample and forward-fill missing data

        if i in ['BAM', 'BN', 'JOE']:

            data = yf.download(i, start=fecha_inicio_, interval="15m", progress=False)
            data = data.tz_localize(ny_tz) 
            precios_iday[i] = data["Adj Close"].reindex(rango_fechas_, method="ffill")  # Resample and forward-fill missing data # Convert data to Madrid timezone

    # Download EUR/USD exchange rate data (15-min intervals)

    eurusd = yf.download("EURUSD=X", start=fecha_inicio_, interval="15m", progress=False)
    eurusd = eurusd.tz_localize(ccy_tz).reindex(rango_fechas_, method="ffill")["Adj Close"]

    # Download Bitcoin data (BTC-USD) and adjust for EUR/USD conversion
    
    btc_usd = yf.download("BTC-USD", start=fecha_inicio_, interval="15m", progress=False)
    btc_usd = btc_usd.tz_localize(btc_tz).reindex(rango_fechas_, method="ffill")["Adj Close"]
    precios_iday["WBIT"] = btc_usd / eurusd * 0.0002396

    # Adjust other tickers for EUR/USD
    for ticker in ["JOE", "BN", "BAM"]:
        precios_iday[ticker] = precios_iday[ticker] / eurusd

    # Fill forward missing data
    precios_iday = precios_iday.fillna(method="ffill")
    precios_iday = precios_iday.dropna()

    # Optionally, filter out weekends if desired (optional)
    # precios = precios[precios.index.dayofweek < 5]


    # Rename columns
    column_mapping_prices = {
        'CSH2.PA': '0.0 ETF monetario', '0P00002BDB.F': '0.1 Fondo monetario', 'U3O8.DE': '1.6 Uranio',
        'ZPRV.DE': '1.3 USA Small Value', 'IWDA.AS': '1.1 World', 'EUNM.DE': '1.4 Emergentes',
        'SMCX.MI': '1.2 Europa Small', 'BN': '2.2 Brookfield Corp', 'JOE': '2.3 St Joe', 'TL0.DE': '2.1 Tesla',
        'WBIT': '1.5 ETF bitcoin', 'BAM': '2.4 Brookfield AM'
    }

    precios_iday.rename(columns=column_mapping_prices, inplace=True)
    precios_iday = precios_iday.sort_index(axis=1)

    # Calculate returns
    rendimientos_iday = (precios_iday / precios_iday.iloc[0])

    import plotly.graph_objects as go

    # Eje X (asumiendo que rendimientos_iday.index contiene las fechas)
    eje_x = rendimientos_iday.index

    # Serie 1: (rendimientos_iday * pesos).sum(axis=1)
    serie_1 = (rendimientos_iday * pesos.loc[precios_iday.dropna().index[0].normalize().tz_localize(None)]).sum(axis=1)

    # Serie 2: rendimientos_iday["1.1 World"]
    serie_2 = rendimientos_iday["1.1 World"]

    # Crear figura con las dos líneas
    fig = go.Figure()

    # Añadir primera línea
    fig.add_trace(go.Scatter(
        x=eje_x, 
        y=serie_1,
        mode='lines',
        name='Rendimientos Ajustados'
    ))

    # Añadir segunda línea
    fig.add_trace(go.Scatter(
        x=eje_x, 
        y=serie_2,
        mode='lines',
        name='1.1 World'
    ))

    # Configurar eje X con formato dd hh:mm
    fig.update_xaxes(
        tickformat='%d %H:%M',
        title_text='Fecha (dd hh:mm)'
    )

    # Configurar título y leyenda
    fig.update_layout(
        title='Comparación de Rendimientos',
        xaxis_title='Fecha (dd hh:mm)',
        yaxis_title='Rendimiento',
        legend_title='Series'
    )

    # Mostrar gráfico
    st.plotly_chart(fig)




st.divider()



col1, col2,col3,col4,col5,col6 = st.columns(6)


# Initialize session state for start and end dates if not already set
if 'start_date' not in st.session_state or 'end_date' not in st.session_state:
    st.session_state.start_date = datetime.now() - timedelta(days=180)
    st.session_state.end_date = datetime.now()

def update_dates(days):
    st.session_state.end_date = datetime.now()
    st.session_state.start_date = st.session_state.end_date - timedelta(days=days)

def update_year_to_date():
    st.session_state.end_date = datetime.now()
    st.session_state.start_date = datetime(st.session_state.end_date.year, 1, 1)


#with col5:
#    start_date = st.date_input('Start Date', value=st.session_state.start_date, format="DD-MM-YYYY")
 #   st.session_state.start_date = datetime.combine(start_date, datetime.min.time())
#with col6:
 #   end_date = st.date_input('End Date', value=st.session_state.end_date, format="DD-MM-YYYY")
 #   st.session_state.end_date = datetime.combine(end_date, datetime.min.time())



with col1:
    st.text(" ")
    st.text(" ")
    if st.button('Last Month'):
        update_dates(30)
with col2:
    st.text(" ")
    st.text(" ")
    if st.button('Last 3 Months'):
        update_dates(90)
with col3:
    st.text(" ")
    st.text(" ")
    if st.button('Last 6 Months'):
        update_dates(180)
with col4:
    st.text(" ")
    st.text(" ")
    if st.button('Year to Date'):
        update_year_to_date()

if opcion_seleccionada == "Omite monetarios":

    with col5:

        start_date  = col5.date_input("Fecha inicio",value=st.session_state.start_date,format="DD-MM-YYYY")
        st.session_state.start_date = datetime.combine(start_date, datetime.min.time())

else:
    with col5:

        start_date  = col1.date_input("Fecha inicio",st.session_state.start_date,format="DD-MM-YYYY")
        st.session_state.start_date = datetime.combine(start_date, datetime.min.time())



with col6:
    st.session_state.end_date = st.date_input("Fecha final",st.session_state.end_date,format="DD-MM-YYYY")


vsBench = pd.DataFrame(rendimiento_portfolio, columns=["Portfolio"])
vsBench["Benchmark"] = benchmark.pct_change()

growthline_portfolio = (1+vsBench[st.session_state.start_date:st.session_state.end_date]).cumprod()

growthline_portfolio = growthline_portfolio/growthline_portfolio.iloc[0]

fig = px.line(growthline_portfolio, title='Evolución índice cartera')

# Update line colors and widths
fig.update_traces(selector=dict(name="Portfolio"), line=dict(color="#FF8C00", width=3))  # Darker orange
fig.update_traces(selector=dict(name="Benchmark"), line=dict(color="#4FB0C6", width=2))  # Soft sea blue

tickvals = growthline_portfolio.index[::5]
ticktext = growthline_portfolio.index[::5].strftime('%d %b %y')

fig.update_xaxes(ticktext=ticktext, tickvals=tickvals)

fig.update_xaxes(
    rangebreaks=[dict(bounds=["sat", "mon"])])

for date in tickvals:
    fig.add_vline(x=date, line=dict(color='white', width=0.25))

st.plotly_chart(fig,use_container_width=True)


st.divider()


#################################

opciones_valores = ["Total","Desglose"]

opciones_frecuencia = ["Diaria","Semanal","Mensual"]


col1, col2 = st.columns(2)

# Añadir contenido a la primera columna
with col1:
   st.subheader("Valores")
#   st.write("Este es el contenido de la primera columna")
   valor_opcion_seleccionada = st.selectbox("Selecciona una opción", opciones_valores)


# Añadir contenido a la segunda columna
with col2:
    st.subheader("Frecuencia")
    #st.write("Este es el contenido de la segunda columna")
    frecuencia_opcion_seleccionada = st.selectbox("Selecciona una opción", opciones_frecuencia)


if frecuencia_opcion_seleccionada == "Diaria":

    if valor_opcion_seleccionada == "Total":

        fig = px.line(pl.sum(axis=1).rename("Total"),title='P/L de la cartera')

    elif valor_opcion_seleccionada == "Desglose":

        fig = px.line(pl,title='P/L de los valores')

if frecuencia_opcion_seleccionada == "Semanal":

    if valor_opcion_seleccionada == "Total":

        fig = px.line(pl.sum(axis=1).rename("Total").resample("W").last(),title='P/L de la cartera')

    elif valor_opcion_seleccionada == "Desglose":

        fig = px.line(pl.resample("W").last(),title='P/L de los valores')

if frecuencia_opcion_seleccionada == "Mensual":

    if valor_opcion_seleccionada == "Total":

        pl_mensual_total = pl.sum(axis=1).rename("Total").resample("M").last().diff()

        fig = px.bar(pl_mensual_total,title='Gráfico de Línea',text=pl_mensual_total)

        fig.update_traces(texttemplate='%{text:.2f}', textposition='outside')  # Mostrar los valores encima de las barras
        fig.update_layout(title='Gráfico de Barras con Valores Encima',
                        xaxis_title='Fecha',
                        yaxis_title='Valor')

        ticktext = pl_mensual_total.index.strftime('%b %y')
        fig.update_xaxes(ticktext=ticktext, tickvals=pl_mensual_total.index)


        fig.update_traces(texttemplate="<b>%{text:.2f}</b>", textposition='outside', 
                        textfont_color=['green' if val >= 0 else 'red' for val in pl_mensual_total])



    elif valor_opcion_seleccionada == "Desglose":

        #pl_mensual_desglose = pl.resample("M").last().diff()

        fig = px.bar(pl.resample("M").last().diff(),title='P/L de los valores')

        ticktext = pl.resample("M").last().index.strftime('%b %y')
        fig.update_xaxes(ticktext=ticktext, tickvals=pl.resample("M").last().index)



#fig = px.line(pl.sum(axis=1).rename("tr"),title='Gráfico de Línea')

st.plotly_chart(fig,use_container_width=True)



st.divider()

import plotly.graph_objects as go

fig = go.Figure()

# Itera sobre cada columna y agrega un trazo al gráfico
for column in valor.columns:
    fig.add_trace(go.Scatter(x=valor.index, y=valor[column], mode='lines', stackgroup='one', name=column))


suma_total = valor.sum(axis=1).resample("M").last()[-1].round(0)

# Personaliza el diseño del gráfico
fig.update_layout(
    title="Valor de la cartera                                                                                                                                                                             " + str(int(suma_total)) ,
    xaxis_title="Fecha",
    yaxis_title="Valor",
    hovermode='x'
)

# Muestra el gráfico
st.plotly_chart(fig,use_container_width=True)


st.divider()


col1, col2,col3,col4 = st.columns(4)

with col1:


    st.subheader("P/L")

    pl_assets = pl.diff().tail(1).T.round(2)

    pl_assets = pl_assets.sort_values(by=pl_assets.columns[0],ascending=False)

    pl_assets.columns = ["Hoy"]

    pl_assets["Ayer"] = pl.diff().tail(2).T.round(2).iloc[:,0]

    pl_assets = pl_assets.T

    pl_assets["Total"] = pl_assets.sum(axis=1)

    pl_assets = pl_assets.T

    pl_assets = pl_assets.style.pipe(lambda x: x.applymap(color_negativo_positivo_cero)).format({'Hoy': "{:.0f}€", 'Ayer': "{:.0f}€"})

    st.write(pl_assets)


with col2:

    st.subheader("% change")

    chg_assets = ((rendimientos.tail(1).T)).round(6)

    chg_assets = chg_assets.sort_values(by=chg_assets.columns[0],ascending=False)

    chg_assets.columns = ["Hoy"]

    chg_assets["Ayer"] = ((rendimientos.tail(2).T)).round(6).iloc[:,0]

    chg_assets_styled_df = chg_assets.style.pipe(lambda x: x.applymap(color_negativo_positivo_cero)).format({'Hoy': "{:.2%}", 'Ayer': "{:.2%}"})

    st.write(chg_assets_styled_df)


with col3:

    st.subheader("Valor")

    valor_cartera = valor.round(0).iloc[-1]
    valor_cartera = valor_cartera.rename("Valor")

    valor_cartera = pd.DataFrame(valor_cartera).sort_values(by="Valor",ascending=False)

    valor_cartera["Peso"] =  (valor_cartera/valor_cartera.sum())

    #valor_cartera["Valor"] = valor_cartera["Valor"]

    valor_cartera["Peso"] = valor_cartera["Peso"] * 100

    st.write(valor_cartera.style.format({'Valor': "{:,.0f} €", 'Peso':"{:,.1f} %"}))

with col4:
    
    
    st.subheader("Stats")
    
    portfolio_metrics = pd.DataFrame(columns=[["CAGR"]],index=range(1))

    portfolio_metrics["CAGR"] = qs.stats.cagr(rendimiento_portfolio)   

    portfolio_metrics["Vol 6M"] = qs.stats.rolling_volatility(rendimiento_portfolio,rolling_period=21*6).iloc[-1]

    portfolio_metrics["Vol x"] = qs.stats.rolling_volatility(rendimiento_portfolio,rolling_period=21*6).iloc[-1] / qs.stats.rolling_volatility(benchmark,rolling_period=21*6).iloc[-1]

    portfolio_metrics["Alpha"] = qs.stats.greeks(rendimiento_portfolio,benchmark)[1]

    portfolio_metrics["Beta"] = qs.stats.greeks(rendimiento_portfolio,benchmark)[0]

    portfolio_metrics["Best day"] = qs.stats.best(rendimiento_portfolio)

    portfolio_metrics["Worst day"] = qs.stats.worst(rendimiento_portfolio)

    portfolio_metrics["Drawdown"] = qs.stats.to_drawdown_series(rendimiento_portfolio).iloc[-1]

    portfolio_metrics["2 sigma fall"] = qs.stats.cvar(rendimiento_portfolio,sigma=2)

    portfolio_metrics = portfolio_metrics.round(4).T

    portfolio_metrics.columns = ["Stats"]

    st.write(portfolio_metrics)


st.divider()



freq_elegida =  st.selectbox("Frecuencia",["Daily","Weekly","Monthly","Quarterly","Yearly","Total"],index=2)

opcion_seleccionada_freq = freq_elegida


st.subheader("Contribución")

# Mapping options to resample frequencies
frequencies = {
    "Daily": 'B',
    "Weekly": 'W',
    "Monthly": 'M',
    "Quarterly": 'Q',
    "Yearly": 'A',
    "Total": '10A'
}

# Mapping options to date formats
date_formats = {
    "Daily": '%A %d',           # Monday 13
    "Weekly": '%U %B',          # numberofweek month
    "Monthly": '%B %Y',
    "Quarterly": 'Q%q %Y',      # Quarters will be handled separately
    "Yearly": '%Y',
    "Total": 'Total'
}

# Resampling based on selected option
freq = frequencies.get(opcion_seleccionada_freq, 'B')  # Default to 'B' if not found

contribucion_det = contribucion.resample(freq).apply(lambda x: (x + 1).prod() - 1)

# Calculate and reposition the 'Total' column
contribucion_det["Portfolio"] = contribucion_det.sum(axis=1)
contribucion_det.insert(0, "Portfolio", contribucion_det.pop("Portfolio"))

# Format index and sort
contribucion_det.index = pd.to_datetime(contribucion_det.index, dayfirst=True)
contribucion_det = contribucion_det.sort_index(ascending=False)

# Update the date format based on the selected option
date_format = date_formats.get(opcion_seleccionada_freq, '%B %Y')

# Special handling for Quarterly
if opcion_seleccionada_freq == "Quarterly":
    contribucion_det.index = contribucion_det.index.to_period('Q').strftime('Q%q %Y')
else:
    contribucion_det.index = contribucion_det.index.strftime(date_format)

    # Set the formatted date as the index and ensure it's unique

contribucion_det = contribucion_det[~contribucion_det.index.duplicated(keep='first')]

#st.write(contribucion_det.head(15).style.pipe(lambda x: x.applymap(color_negativo_positivo_cero)).format({
 #   '0.0 ETF monetario': "{:.2%}",
  #  '0.1 Fondo monetario': "{:.2%}",
   # '1.1 World': "{:.2%}",
    #'1.2 Europa Small': "{:.2%}",
   # '1.3 USA Small Value': "{:.2%}",
#    '1.4 Emergentes': "{:.2%}",
 #   '1.5 ETF bitcoin': "{:.2%}",
  #  '1.6 Uranio': "{:.2%}",
   # '2.1 Tesla': "{:.2%}",
   # '2.2 Brookfield Corp': "{:.2%}",
   # '2.3 St Joe': "{:.2%}",
   # '2.4 Brookfield AM': "{:.2%}",
   # 'Total': "{:.2%}"
#}))

for col in contribucion_det.columns:
    contribucion_det[col] = contribucion_det[col].apply(lambda x: f"{x*100:.2f}%")

st.write(apply_styles(contribucion_det,is_dark_mode=True))








st.subheader("Asset returns")

#ytdperf = (1 + rendimientos).cumprod() - 1 

# Use the selected frequency for resampling
ytdperf =  rendimientos.resample(freq).apply(lambda x: (x + 1).prod() - 1)
portfolio_returns = rendimiento_portfolio.resample(freq).apply(lambda x: (x + 1).prod() - 1)

# Add the portfolio returns as a new column to ytdperf
ytdperf['Portfolio'] = portfolio_returns

# Move the 'Portfolio' column to the beginning of the dataframe
cols = ytdperf.columns.tolist()
cols = ['Portfolio'] + [col for col in cols if col != 'Portfolio']
ytdperf = ytdperf[cols]

# Sort the index in descending order
ytdperf = ytdperf.sort_index(ascending=False)

# Apply the date format based on the selected frequency
if opcion_seleccionada_freq == "Quarterly":
    ytdperf.index = ytdperf.index.to_period('Q').strftime('Q%q %Y')
else:
    ytdperf.index = ytdperf.index.strftime(date_format)

# Set the formatted date as the index and ensure it's unique

ytdperf = ytdperf[~ytdperf.index.duplicated(keep='first')]



#st.write(ytdperf.style.pipe(lambda x: x.applymap(color_negativo_positivo_cero)).format({
 ##   'Portfolio': "{:.2%}",
  #  '0.0 ETF monetario': "{:.2%}",
  #  '0.1 Fondo monetario': "{:.2%}",
  #  '1.1 World': "{:.2%}",
  #  '1.2 Europa Small': "{:.2%}",
  #  '1.3 USA Small Value': "{:.2%}",
  #  '1.4 Emergentes': "{:.2%}",
  #  '1.5 ETF bitcoin': "{:.2%}",
  #  '1.6 Uranio': "{:.2%}",
  #  '2.1 Tesla': "{:.2%}",
  #  '2.2 Brookfield Corp': "{:.2%}",
  #  '2.3 St Joe': "{:.2%}",
  #  '2.4 Brookfield AM': "{:.2%}"
#}))

for col in ytdperf.columns:
    ytdperf[col] = ytdperf[col].apply(lambda x: f"{x*100:.2f}%")

st.write(apply_styles(ytdperf,is_dark_mode=True))

st.subheader("P/L")


pl_df = pl.diff().resample(freq).sum()

# Calculate and reposition the 'Total' column
pl_df["Portfolio"] = pl_df.sum(axis=1)
pl_df.insert(0, "Portfolio", pl_df.pop("Portfolio"))

# Format index and sort
pl_df.index = pd.to_datetime(pl_df.index, dayfirst=True)
pl_df = pl_df.sort_index(ascending=False)



# Update the date format based on the selected option
date_format = date_formats.get(opcion_seleccionada_freq, '%B %Y')




# Special handling for Quarterly
if opcion_seleccionada_freq == "Quarterly":
    pl_df.index = pl_df.index.to_period('Q').strftime('Q%q %Y')
else:
    pl_df.index = pl_df.index.strftime(date_format)



if freq == "10A":

    pl_df = pd.DataFrame(pl_df.sum()).T

    pl_df.index = ["Total"] 

else:

    pl_df = pl_df[~pl_df.index.duplicated(keep='first')]


st.write(pl_df.style.pipe(lambda x: x.applymap(color_negativo_positivo_cero)).format({
    'Portfolio': "{:.0f}€",
    '0.0 ETF monetario': "{:.0f}€",
 '0.1 Fondo monetario': "{:.0f}€",
   '1.1 World': "{:.0f}€",
    '1.2 Europa Small': "{:.0f}€",
    '1.3 USA Small Value': "{:.0f}€",
   '1.4 Emergentes': "{:.0f}€",
    '1.5 ETF bitcoin': "{:.0f}€",
    '1.6 Uranio': "{:.0f}€",
    '2.1 Tesla': "{:.0f}€",
    '2.2 Brookfield Corp': "{:.0f}€",
    '2.3 St Joe': "{:.0f}€",
    '2.4 Brookfield AM': "{:.0f}€"
}))




################





dim_elegida =  st.selectbox("Gráfico",["Contribución","Returns","P/L"],index=1)

col1, col2, col3, col4, col5,col6 = st.columns(6)

# Initialize session state for start and end dates if not already set
if 'start_date' not in st.session_state or 'end_date' not in st.session_state:
    st.session_state.start_date = datetime.now() - timedelta(days=180)
    st.session_state.end_date = datetime.now()

def update_dates(days):
    st.session_state.end_date = datetime.now()
    st.session_state.start_date = st.session_state.end_date - timedelta(days=days)

def update_year_to_date():
    st.session_state.end_date = datetime.now()
    st.session_state.start_date = datetime(st.session_state.end_date.year, 1, 1)

with col1:
    st.text(" ")
    st.text(" ")
    if st.button('Last Month',key="last month gr"):
        update_dates(30)
with col2:
    st.text(" ")
    st.text(" ")
    if st.button('Last 3 Months',key="last 3 month gr"):
        update_dates(90)
with col3:
    st.text(" ")
    st.text(" ")
    if st.button('Last 6 Months',key="last 6 month gr"):
        update_dates(180)
with col4:
    st.text(" ")
    st.text(" ")
    if st.button('Year to Date',key="last 12 month gr"):
        update_year_to_date()
with col5:
    start_date = st.date_input('Start Date', value=st.session_state.start_date, format="DD-MM-YYYY",key="start date gr")
    st.session_state.start_date = datetime.combine(start_date, datetime.min.time())
with col6:
    end_date = st.date_input('End Date', value=st.session_state.end_date, format="DD-MM-YYYY",key="end   date gr")
    st.session_state.end_date = datetime.combine(end_date, datetime.min.time())



if dim_elegida == "Returns":

    returns_comp = precios.loc[st.session_state.start_date:st.session_state.end_date]
    returns_comp = 100 * returns_comp/returns_comp.iloc[0]


    fig = go.Figure()
    for column in returns_comp.columns:
        fig.add_trace(go.Scatter(x=returns_comp.index, y=returns_comp[column], name=column))

    fig.update_layout(
        xaxis_title='Date',
        yaxis_title='Evolution',
        legend_title='Assets',
        hovermode='x unified'
    )

    st.plotly_chart(fig)

elif dim_elegida=="Contribución":

    returns_comp = contribucion.loc[st.session_state.start_date:st.session_state.end_date]
    returns_comp = returns_comp = ((1+returns_comp).cumprod() - 1) * 100

    fig = go.Figure()
    for column in returns_comp.columns:
        fig.add_trace(go.Scatter(x=returns_comp.index, y=returns_comp[column], name=column))

    fig.update_layout(
        xaxis_title='Date',
        yaxis_title='Evolution',
        legend_title='Assets',
        hovermode='x unified'
    )

    st.plotly_chart(fig)

elif dim_elegida=="P/L":

    pl_diff = pl.diff()

    returns_comp = pl_diff.loc[st.session_state.start_date:st.session_state.end_date].cumsum()  #- pl.loc[st.session_state.start_date]


    fig = go.Figure()
    for column in returns_comp.columns:
        fig.add_trace(go.Scatter(x=returns_comp.index, y=returns_comp[column], name=column))

    fig.update_layout(
        xaxis_title='Date',
        yaxis_title='Evolution',
        legend_title='Assets',
        hovermode='x unified'
    )

    st.plotly_chart(fig)





#####################################################################


st.divider()


asset_elegido =  st.selectbox("Evolución precios",precios.columns)

compras_fecha = movimientos[(movimientos["Description"]==asset_elegido) & (movimientos["Sentido"]=="Compra")].index
compras_precio = movimientos[(movimientos["Description"]==asset_elegido) & (movimientos["Sentido"]=="Compra")]["Precio compra"]

ventas_fecha = movimientos[(movimientos["Description"]==asset_elegido) & (movimientos["Sentido"]=="Venta")].index
ventas_precio = movimientos[(movimientos["Description"]==asset_elegido) & (movimientos["Sentido"]=="Venta")]["Precio compra"]


fig = go.Figure()

# Agregar la línea de evolución del precio
fig.add_trace(go.Scatter(x=precios[asset_elegido].index, y=precios[asset_elegido],
                        mode='lines',
                        name='Precio',
                        line=dict(color='rgba(255, 165, 0, 0.7)', width=3.3)))


fig.add_trace(go.Scatter(x=compras_fecha,
                        y=compras_precio,
                        mode='markers',
                        marker=dict(color='green', size=12, symbol='triangle-up'),
                        name='Compra'))

fig.add_trace(go.Scatter(x=ventas_fecha,
                        y=ventas_precio,
                        mode='markers',
                        marker=dict(color='red', size=12, symbol='triangle-down'),
                        name='Compra'))

fig.update_layout(title='Evolución del precio de {} con compras y ventas'.format(asset_elegido[4:]),
                xaxis_title='Fecha',
                yaxis_title='Precio',
                legend_title='Movimientos')



st.plotly_chart(fig,use_container_width=True)

st.divider()


col1,col2 = st.columns(2)

compare_irr = pd.concat([xirr_portfolio,xirr_benchmark],axis=1)
compare_irr.columns = ["Portfolio","Benchmark"]
compare_irr["Gap"] = compare_irr["Portfolio"] - compare_irr["Benchmark"]
compare_irr["Positive_Gap"] = compare_irr["Gap"].clip(lower=0.00)
compare_irr["Negative_Gap"] = compare_irr["Gap"].clip(upper=-0.00)



with col1:
    irr_opcion = st.selectbox("IRR",["Cartera","Delta","Assets"])

    if irr_opcion == "Cartera":


        fig = px.line(compare_irr[["Portfolio","Benchmark"]],title='IRR')


        fig.update_layout(
            title="IRR",                                                                                                                                                                   
            xaxis_title="Fecha",
            yaxis_title="IRR %",
            hovermode='x'
        )

        fig.update_traces(selector=dict(name="Portfolio"), line=dict(color="#FF8C00", width=4))  # Darker orange
        fig.update_traces(selector=dict(name="Benchmark"), line=dict(color="#4FB0C6", width=3))  # Soft sea blue


        #st.plotly_chart(fig,use_container_width=True)

    elif irr_opcion == "Delta":


        # Create the area plot
        fig = px.area(compare_irr, x=compare_irr.index, y=["Positive_Gap", "Negative_Gap"], title='Difference Area Graph')

        # Customize the plot colors
        fig.data[0].update(mode='lines', line=dict(width=0), fill='tozeroy', fillcolor='rgba(0, 255, 0, 0.5)')
        fig.data[1].update(mode='lines', line=dict(width=0), fill='tozeroy', fillcolor='rgba(255, 0, 0, 0.5)')


    else:

        with col2:

            ##asset_irr = st.selectbox("Activo",xirr_df.columns)

            fig = px.line(100 * xirr_df.loc["2024":],title='IRR')


            fig.update_layout(
                title="IRR",                                                                                                                                                                   
                xaxis_title="Fecha",
                yaxis_title="IRR %",
                hovermode='x'
            )



        #st.plotly_chart(fig,use_container_width=True)


st.plotly_chart(fig,use_container_width=True)


st.divider()


rendimientos_corr = rendimientos
rendimientos_corr["1.1 World"] = rendimiento_benchmark



def compute_rolling_correlations(df, column='1.1 World', window=30, start_date=None, end_date=None):
    if start_date:
        df = df[df.index >= start_date]
    if end_date:
        df = df[df.index <= end_date]
    
    correlations = df.rolling(window=window).corr()
    result = correlations.xs(column, level=1, axis=0)
    result = result.drop(column, axis=1)
    
    return result.dropna()


# Parameters in four columns
col1, col2, col3, col4 = st.columns(4)

with col1:
    start_date = st.date_input('Start Date', rendimientos_corr.index.min(),format="DD-MM-YYYY")

with col2:
    end_date = st.date_input('End Date', rendimientos_corr.index.max(),format="DD-MM-YYYY")

with col3:
    reference_column = st.selectbox('Reference Column', rendimientos_corr.columns, index=rendimientos_corr.columns.get_loc('1.1 World'))

with col4:
    window = st.slider('Rolling Window (days)', min_value=2, max_value=365, value=180)

# Convert dates to string format
start_date = start_date.strftime('%Y-%m-%d')
end_date = end_date.strftime('%Y-%m-%d')

# Compute rolling correlations
correlations = compute_rolling_correlations(rendimientos_corr, column=reference_column, window=window, start_date=start_date, end_date=end_date)

# Plot the correlations
st.subheader(f'Rolling Correlations (Window: {window} days)')

fig = go.Figure()
for column in correlations.columns:
    fig.add_trace(go.Scatter(x=correlations.index, y=correlations[column], name=column))

fig.update_layout(
    xaxis_title='Date',
    yaxis_title='Correlation',
    legend_title='Assets',
    hovermode='x unified'
)

st.plotly_chart(fig, use_container_width=True)


st.divider()


def calculate_volatility(returns):
    return returns.std() * np.sqrt(252)

def create_performance_plot(rendimientos, start_date, end_date):
    # Filter data based on selected date range
    filtered_returns = rendimientos[(rendimientos.index >= start_date) & (rendimientos.index <= end_date)]
    
    returns = (1 + filtered_returns).cumprod().iloc[-1] - 1
    volatility = filtered_returns.apply(calculate_volatility)

    performance_df = pd.DataFrame({'returns': returns, 'volatility': volatility}).reset_index()
    performance_df.columns = ['Asset', 'Returns', 'Volatility']

    # Calculate the Sharpe Ratio with 3.5% risk-free rate
    risk_free_rate = 0.035 * 100  / (end_date-start_date).days # 3.5% annual rate
    performance_df['Sharpe_Ratio'] = (performance_df['Returns'] - risk_free_rate) / performance_df['Volatility']

    # Create a color scale based on Sharpe Ratio
     # Create a color scale based on Sharpe Ratio
    color_scale = [
        [0, 'rgb(165,0,38)'],     # Dark red for very negative Sharpe Ratio
        [0.2, 'rgb(215,48,39)'],  # Red for negative Sharpe Ratio
        [0.4, 'rgb(244,109,67)'], # Orange for slightly negative Sharpe Ratio
        [0.5, 'rgb(253,174,97)'], # Light orange for zero Sharpe Ratio
        [0.6, 'rgb(255,255,191)'],# Yellow for low positive Sharpe Ratio
        [0.7, 'rgb(166,217,106)'],# Light green for moderate Sharpe Ratio
        [0.8, 'rgb(102,189,99)'], # Green for good Sharpe Ratio
        [0.9, 'rgb(26,152,80)'],  # Dark green for very good Sharpe Ratio
        [1, 'rgb(0,104,55)']      # Very dark green for excellent Sharpe Ratio
    ]

 # Calculate min and max Sharpe Ratio for color scaling
    min_sharpe = min(performance_df['Sharpe_Ratio'].min(), -2)  # Use -2 as minimum if no lower values
    max_sharpe = max(performance_df['Sharpe_Ratio'].max(), 3)   # Use 3 as maximum if no higher values


    # Create the scatter plot
    fig = go.Figure()

    fig.add_trace(go.Scatter(
        x=performance_df['Volatility'],
        y=performance_df['Returns'],
        mode='markers+text',
        text=performance_df['Asset'],
        textposition='top center',
        marker=dict(
            size=15,
            color=performance_df['Sharpe_Ratio'],
            colorscale=color_scale,
            colorbar=dict(
                title='Sharpe Ratio',
                tickmode='array',
                tickvals=[min_sharpe, (min_sharpe + max_sharpe) / 2, max_sharpe],
                ticktext=[f'{min_sharpe:.1f}', '0', f'{max_sharpe:.1f}']
            ),
            cmin=min_sharpe,
            cmax=max_sharpe,
            showscale=True
        ),
        hovertemplate='<b>%{text}</b><br>Returns: %{y:.2%}<br>Volatility: %{x:.2%}<br>Sharpe Ratio: %{marker.color:.2f}<extra></extra>'
    ))

    # Calculate mean values for quadrant lines
    mean_volatility = 0.15
    mean_returns = 0

    # Add colored quadrants
    fig.add_shape(type="rect", x0=0, y0=mean_returns, x1=mean_volatility, y1=1, 
                fillcolor="rgba(0,255,0,0.9)", line=dict(width=0))
    fig.add_shape(type="rect", x0=mean_volatility, y0=mean_returns, x1=1, y1=1, 
                fillcolor="rgba(255,255,0,0.9)", line=dict(width=0))
    fig.add_shape(type="rect", x0=0, y0=0, x1=mean_volatility, y1=mean_returns, 
                fillcolor="rgba(255,0,0,0.9)", line=dict(width=0))
    fig.add_shape(type="rect", x0=mean_volatility, y0=0, x1=1, y1=mean_returns, 
                fillcolor="rgba(0,0,255,0.9)", line=dict(width=0))

    # Calculate trendline
    slope, intercept = stats.linregress(performance_df['Volatility'], performance_df['Returns']).slope, stats.linregress(performance_df['Volatility'], performance_df['Returns']).intercept

    # Add trendline as a thin black line without label in legend
    fig.add_trace(go.Scatter(
        x=performance_df['Volatility'],
        y=slope * performance_df['Volatility'] + intercept,
        mode='lines',
        line=dict(color='white', width=0.5),
        showlegend=False
    ))

    # Update layout
    fig.update_layout(
        title={
        'text': f'Asset Performance: Returns vs Volatility from {start_date.date()} to {end_date.date()}<br><br>At 0% volatility, the trendline yields {slope * 0 + intercept:.2%} returns. <br>&nbsp;&nbsp;For every 10% increase in volatility, returns are expected to rise by {(slope*10):.2f}%.',
        'y':0.95,
        'x':0.5,
        'xanchor': 'center',
        'yanchor': 'top'
    },
        xaxis_title='Volatility (Annualized)',
        yaxis_title='Returns',
        height=700,
        width=1000,
        xaxis=dict(tickformat='.2%', range=[0, performance_df['Volatility'].max() * 1.1]),
        yaxis=dict(tickformat='.2%', range=[performance_df['Returns'].min() * 1.1, performance_df['Returns'].max() * 1.1]),
        shapes=[
            dict(type="line", x0=mean_volatility, y0=0, x1=mean_volatility, y1=1, xref="x", yref="paper", line=dict(color="Grey", width=1, dash="dash")),
            dict(type="line", x0=0, y0=0, x1=1, y1=0, xref="paper", yref="y", line=dict(color="Grey", width=1, dash="dash"))
        ]
    )

    return fig


st.subheader("Returns vs. Volatility")

col1, col2, col3, col4, col5,col6 = st.columns(6)

# Initialize session state for start and end dates if not already set
if 'start_date' not in st.session_state or 'end_date' not in st.session_state:
    st.session_state.start_date = datetime.now() - timedelta(days=180)
    st.session_state.end_date = datetime.now()

def update_dates(days):
    st.session_state.end_date = datetime.now()
    st.session_state.start_date = st.session_state.end_date - timedelta(days=days)

def update_year_to_date():
    st.session_state.end_date = datetime.now()
    st.session_state.start_date = datetime(st.session_state.end_date.year, 1, 1)

with col1:
    st.text(" ")
    st.text(" ")
    if st.button('Last Month',key="last month rr"):
        update_dates(30)
with col2:
    st.text(" ")
    st.text(" ")
    if st.button('Last 3 Months',key="last 3 month rr"):
        update_dates(90)
with col3:
    st.text(" ")
    st.text(" ")
    if st.button('Last 6 Months',key="last 6 month rr"):
        update_dates(180)
with col4:
    st.text(" ")
    st.text(" ")
    if st.button('Year to Date',key="last 12 month rr"):
        update_year_to_date()
with col5:
    start_date = st.date_input('Start Date', value=st.session_state.start_date, format="DD-MM-YYYY",key="start date rr")
    st.session_state.start_date = datetime.combine(start_date, datetime.min.time())
with col6:
    end_date = st.date_input('End Date', value=st.session_state.end_date, format="DD-MM-YYYY",key="end   date rr")
    st.session_state.end_date = datetime.combine(end_date, datetime.min.time())

# Create and display the plot
fig = create_performance_plot(rendimientos, st.session_state.start_date, st.session_state.end_date)
st.plotly_chart(fig, use_container_width=True)



st.divider()


valor_df = pd.DataFrame(valor.iloc[-1])
valor_df.columns = ["Hoy"]

valor_stocks = valor_df.loc[["2.3 St Joe","2.4 Brookfield AM","2.2 Brookfield Corp","2.1 Tesla"],:]
#valor_stocks = valor_stocks.divide(valor_stocks.sum())

valor_indices = valor_df.loc[["1.1 World","1.2 Europa Small","1.3 USA Small Value","1.4 Emergentes","1.6 Uranio"],:]
#valor_indices = valor_indices.divide(valor_indices.sum())


if opcion_seleccionada != "Omite monetarios":

    valor_indices_cat = valor_df.loc[["1.1 World","1.2 Europa Small","1.3 USA Small Value","1.4 Emergentes","1.6 Uranio"],:].sum()

    valor_stocks_cat = valor_df.loc[["2.3 St Joe","2.4 Brookfield AM","2.2 Brookfield Corp","2.1 Tesla"],:].sum()

    valor_btc_cat = valor_df.loc[["1.5 ETF bitcoin"],:].sum()

    valor_monetarios_cat = valor_df.loc[["0.0 ETF monetario","0.1 Fondo monetario"],:].sum()

    valor_por_cat = pd.DataFrame(data=[valor_indices_cat,valor_stocks_cat,valor_btc_cat,valor_monetarios_cat],index=["Índices","Stocks","BTC","Monetarios"])
    

else:

    valor_indices_cat = valor_df.loc[["1.1 World","1.2 Europa Small","1.3 USA Small Value","1.4 Emergentes","1.6 Uranio"],:].sum()

    valor_stocks_cat = valor_df.loc[["2.3 St Joe","2.4 Brookfield AM","2.2 Brookfield Corp","2.1 Tesla"],:].sum()

    valor_btc_cat = valor_df.loc[["1.5 ETF bitcoin"],:].sum()

    valor_por_cat = pd.DataFrame(data=[valor_indices_cat,valor_stocks_cat,valor_btc_cat],index=["Índices","Stocks","BTC"])


fig = go.Figure()

# First Pie chart


fig.add_trace(go.Pie(labels=valor_df.index, values=valor_df["Hoy"], name='Pie 1', hole=0.1, domain={'x': [0, 0.45],'y': [0.55, 1]},textinfo='label+percent'))

fig.add_trace(go.Pie(labels=valor_por_cat.index, values=valor_por_cat["Hoy"], name='Pie 2', hole=0.1, domain={'x': [0.55, 1],'y': [0.55, 1]},textinfo='label+percent', textposition='inside'))

fig.add_trace(go.Pie(labels=valor_indices.index, values=valor_indices["Hoy"], name='Pie 2', hole=0.1, domain={'x': [0, 0.45],'y': [0.0, 0.45]},textinfo='label+percent', textposition='inside'))

fig.add_trace(go.Pie(labels=valor_stocks.index, values=valor_stocks["Hoy"], name='Pie 2', hole=0.1, domain={'x': [0.55, 1],'y': [0.0, 0.45]},textinfo='label+percent', textposition='inside'))

fig.update_layout(title_text="Multiple Pie Charts", 
                  # Use 'grid' to arrange in grid, 'horizontal' or 'vertical' for horizontal or vertical arrangement
                  grid=dict(rows=2, columns=2), 
                  # Annotations help to give title to individual pie chart
                  annotations=[dict(text='Cartera', x=-0.10, y=1.0, font_size=17, showarrow=False),
                               dict(text='Categorías', x=0.55, y=1.0, font_size=17, showarrow=False),
                               dict(text='Índices', x=-0.10, y=0.4, font_size=17, showarrow=False),
                               dict(text='Stocks', x=0.55, y=0.4, font_size=17, showarrow=False)], showlegend=True,width=1300,height=1000)



st.plotly_chart(fig,use_container_width=True)


############################

# Assuming 'pesos' is your DataFrame
df = pesos.loc["2024":].dropna()

# Create the stacked area chart
fig = go.Figure()

for column in df.columns:
    fig.add_trace(go.Scatter(
        x=df.index,
        y=df[column],
        mode='lines',
        stackgroup='one',
        name=column,
        hoverinfo='x+y+name'
    ))

# Customize the layout
fig.update_layout(
    title='Portfolio Weight Evolution',
    xaxis_title='Date',
    yaxis_title='Weight',
    yaxis=dict(
        type='linear',
        range=[0, 1],
        tickformat='.1%'
    ),
    legend_title='Assets',
    hovermode='x unified'
)


st.plotly_chart(fig,use_container_width=True)

##################################

# Remove days with zero returns
returns_nonzero = rendimiento_portfolio[rendimiento_portfolio != 0]

# Calculate statistics
mean_return = returns_nonzero.mean()
median_return = returns_nonzero.median()
std_dev = returns_nonzero.std()

# Calculate the range for the bins
max_abs_return = max(abs(returns_nonzero.min()), abs(returns_nonzero.max()))
min_return = -np.ceil(max_abs_return * 1000) / 1000
max_return = np.ceil(max_abs_return * 1000) / 1000

# Create bins for every 10 basis points
bins = np.arange(min_return, max_return + 0.001, 0.001)

# Get the returns from the last 10 days and their dates
last_10_returns = returns_nonzero[-10:]
last_10_dates = rendimiento_portfolio.index[-10:]

# Highlight the bins that contain the last 10 days' returns
highlight_bins = np.digitize(last_10_returns, bins)

# Calculate the histogram manually to get bin heights
hist_values, hist_bins = np.histogram(returns_nonzero, bins=bins)

# Create the Plotly figure
fig = go.Figure()

# Add histogram trace
fig.add_trace(go.Histogram(
    x=returns_nonzero,
    name='Daily Returns',
    opacity=0.75,
    xbins=dict(
        start=min_return,
        end=max_return,
        size=0.001
    ),
    autobinx=False,
    
))

# Highlight bins for the last 10 days with yellow color
for bin_idx in highlight_bins:
    bin_value = bins[bin_idx - 1]  # Get the left edge of the bin
    fig.add_shape(type="rect",
                  x0=bin_value, x1=bin_value + 0.001,  # Bin size is 0.001
                  y0=0, y1=hist_values[bin_idx - 1],  # Respect original bin height
                  fillcolor="yellow", opacity=0.5, line_width=0)

# Add density curve
kde = stats.gaussian_kde(returns_nonzero)
x_range = np.linspace(min_return, max_return, 1000)
y_range = kde(x_range)
fig.add_trace(go.Scatter(x=x_range, y=y_range * len(returns_nonzero) * 0.001, 
                         mode='lines', name='Density', line=dict(color='darkorange')))

# Update layout
fig.update_layout(
    title='Histogram of Portfolio Daily Returns',
    xaxis_title='Return',
    yaxis_title='Frequency',
    bargap=0.05,  # Gap between bars
    showlegend=False,
    xaxis=dict(
        tickformat='.1%',  # Format x-axis labels as percentages
        tickmode='array',
        tickvals=np.arange(min_return, max_return + 0.001, 0.01),  # Major ticks every 1%
        ticktext=[f'{x:.1%}' for x in np.arange(min_return, max_return + 0.001, 0.01)],
        range=[min_return, max_return]  # Center the graph
    )
)

# Add a vertical line at x=0
fig.add_vline(x=0, line_dash="dash", line_color="white")

# Add statistics box
stats_text = (f"Mean: {mean_return:.2%}<br>"
              f"Median: {median_return:.2%}<br>"
              f"Std Dev: {std_dev:.2%}")

fig.add_annotation(
    xref="paper", yref="paper",
    x=0.95, y=0.95,
    text=stats_text,
    showarrow=False,
    font=dict(size=12),
    align="left",
    borderwidth=1
)

# Add a text box with the last 10 days' returns formatted with dates
last_10_text = "<br>".join([f"{datetime.strftime(date, '%b %d')}: {rtn:.2%}" 
                            for date, rtn in zip(last_10_dates, last_10_returns)])
fig.add_annotation(
    xref="paper", yref="paper",
    x=0.95, y=0.75,  # Position under the stats box
    text=f"Last 10 Days:<br>{last_10_text}",
    showarrow=False,
    font=dict(size=10),
    align="left",
    borderwidth=1
)

# Show the plot
st.plotly_chart(fig,use_container_width=True)


################################

st.divider()


fig = qs.plots.snapshot(rendimiento_portfolio,show=False,figsize=(8,8))

st.pyplot(fig)

fig = qs.plots.monthly_heatmap(rendimiento_portfolio,benchmark,show=False,figsize=(8,5.5))

st.pyplot(fig)


fig =  qs.plots.drawdowns_periods(rendimiento_portfolio,show=False,figsize=(8,5.5))

st.pyplot(fig)



fig =   qs.plots.rolling_volatility(rendimiento_portfolio,period=20,period_label="21 días",show=False,figsize=(8,5.5))

st.pyplot(fig)


############################

import plotly.graph_objects as go
from plotly.subplots import make_subplots
from typing import Dict, Tuple
from datetime import datetime, timedelta, date


col1, col2, col3 = st.columns(3)

with col1:
    anos_simulacion = st.slider('Years', min_value=1, max_value=50, value=6)

max_simulations = 30000 - (anos_simulacion - 1) * 2000
if max_simulations < 0:
    max_simulations = 1000


num_simulaciones = col2.slider('Number of Simulations', min_value=0, max_value=max_simulations, value=min(8000, max_simulations))

with col3:
    aportacion_mensual = st.number_input('Monthly Contribution', step=50, value=1250)



def load_asset_expectations() -> Dict[str, Dict[str, float]]:
    """Load and return asset expectations, using session state if available."""
    if 'asset_expectations' not in st.session_state:
        st.session_state.asset_expectations = {
            '0.0 ETF monetario': {'retorno': 0.02, 'volatilidad': 0.005},
            '0.1 Fondo monetario': {'retorno': 0.02, 'volatilidad': 0.005},
            '1.1 World': {'retorno': 0.07, 'volatilidad': 0.15},
            '1.2 Europa Small': {'retorno': 0.085, 'volatilidad': 0.20},
            '1.3 USA Small Value': {'retorno': 0.085, 'volatilidad': 0.20},
            '1.4 Emergentes': {'retorno': 0.08, 'volatilidad': 0.25},
            '1.5 ETF bitcoin': {'retorno': 0.20, 'volatilidad': 0.50},
            '1.6 Uranio': {'retorno': 0.12, 'volatilidad': 0.45},
            '2.1 Tesla': {'retorno': 0.15, 'volatilidad': 0.40},
            '2.2 Brookfield Corp': {'retorno': 0.11, 'volatilidad': 0.22},
            '2.3 St Joe': {'retorno': 0.09, 'volatilidad': 0.20},
            '2.4 Brookfield AM': {'retorno': 0.10, 'volatilidad': 0.21},
        }
    return st.session_state.asset_expectations

def create_editable_df(pesos_actuales: pd.Series, expectativas: Dict[str, Dict[str, float]]) -> pd.DataFrame:
    """Create and return a DataFrame with current weights and expectations for editing."""
    df_cartera = pd.DataFrame({'Activo': pesos_actuales.index, 'Peso': pesos_actuales.values})
    df_cartera['Retorno'] = df_cartera['Activo'].map(lambda x: expectativas.get(x, {}).get('retorno', 0))
    df_cartera['Volatilidad'] = df_cartera['Activo'].map(lambda x: expectativas.get(x, {}).get('volatilidad', 0))
    return df_cartera.query('Peso > 0')


def simular_rendimiento_optimizado(df: pd.DataFrame, dias: int, num_simulaciones: int, degrees_of_freedom: int = 5) -> np.ndarray:
    """Simulate portfolio returns using a t-distribution."""
    retornos_anuales = df['Retorno'].values
    volatilidades_anuales = df['Volatilidad'].values
    pesos = df['Peso'].values

    # Compute portfolio expected return and volatility
    portfolio_return = np.sum(retornos_anuales * pesos)
    portfolio_volatility = np.sqrt(np.sum((volatilidades_anuales * pesos) ** 2))

    # Calculate daily parameters
    daily_return = (portfolio_return - 0.5 * portfolio_volatility ** 2) / 252
    daily_volatility = portfolio_volatility / np.sqrt(252)

    # Generate random returns for the entire portfolio using t-distribution
    random_t = stats.t.rvs(df=degrees_of_freedom, size=(num_simulaciones, dias))
    random_returns = daily_return + daily_volatility * random_t * np.sqrt((degrees_of_freedom - 2) / degrees_of_freedom)

    return np.exp(random_returns) - 1

def generate_monthly_indices(dias: int, start_date: datetime) -> np.ndarray:
    """Generate indices for the first trading day of each month."""
    # Create a range of dates including only weekdays
    dates = pd.bdate_range(start=start_date, periods=dias).to_pydatetime()

    # Find the first trading day of each month
    first_days = []
    current_month = dates[0].month
    for i, d in enumerate(dates):
        if d.month != current_month:
            first_days.append(i)
            current_month = d.month

    return np.array(first_days)

def run_simulation(df_cartera: pd.DataFrame, saldo_inicial: float, num_simulaciones: int, 
                   anos_simulacion: int, aportacion_mensual: float) -> Tuple[np.ndarray, np.ndarray]:
    """Run the Monte Carlo simulation and return results."""
    dias_simulacion = 252 * anos_simulacion
    rendimientos = simular_rendimiento_optimizado(df_cartera, dias_simulacion, num_simulaciones)
    
    # Pre-allocate the simulaciones array
    simulaciones = np.zeros((num_simulaciones, dias_simulacion))
    simulaciones[:, 0] = saldo_inicial

    # Generate monthly contribution indices based on the first trading day of each month
    start_date = datetime.now()
    monthly_indices = generate_monthly_indices(dias_simulacion, start_date)

    # Vectorized simulation
    for dia in range(1, dias_simulacion):
        simulaciones[:, dia] = simulaciones[:, dia - 1] * (1 + rendimientos[:, dia - 1])
        if dia in monthly_indices:
            simulaciones[:, dia] += aportacion_mensual

    return simulaciones, rendimientos

def plot_simulation_results(simulaciones: np.ndarray, anos_simulacion: int):
    """Plot the simulation results using plotly."""
    dias_simulacion = 252 * anos_simulacion
    percentiles = np.percentile(simulaciones, [10, 50, 90], axis=0)
    
    fig = go.Figure()
    x_axis = np.linspace(0, anos_simulacion, dias_simulacion)
    fig.add_trace(go.Scatter(x=x_axis, y=percentiles[1], mode='lines', name='Median'))
    fig.add_trace(go.Scatter(x=x_axis, y=percentiles[2], mode='lines', name='90th percentile'))
    fig.add_trace(go.Scatter(x=x_axis, y=percentiles[0], mode='lines', name='10th percentile'))
    
    fig.update_layout(title='Portfolio Value Simulation',
                      xaxis_title='Years',
                      yaxis_title='Portfolio Value')
    
    return fig

# Load asset expectations
expectativas = load_asset_expectations()


# Get current weights and initial portfolio value
pesos_actuales = pesos.iloc[-1].sort_index()
saldo_inicial = valor.sum(axis=1).iloc[-1]

# Create editable DataFrame
if 'df_editable' not in st.session_state:
    st.session_state.df_editable = create_editable_df(pesos_actuales, expectativas)

# Button to show/hide the editable dataframe
if st.button("Edit Portfolio Weights and Asset Expectations"):
    st.session_state.show_editor = not st.session_state.get('show_editor', False)

# Show the editor if the button has been clicked
if st.session_state.get('show_editor', False):
    st.subheader("Edit Portfolio Weights and Asset Expectations")
    edited_df = st.data_editor(st.session_state.df_editable, key='portfolio_data')

    # Update session state and recreate portfolio if changes were made
    if not edited_df.equals(st.session_state.df_editable):
        # Update expectations in session state
        st.session_state.asset_expectations = {
            row['Activo']: {'retorno': row['Retorno'], 'volatilidad': row['Volatilidad']}
            for _, row in edited_df.iterrows()
        }
        
        # Update weights
        pesos_actuales = pd.Series(edited_df['Peso'].values, index=edited_df['Activo'])
        
        # Update the editable DataFrame in session state
        st.session_state.df_editable = edited_df
        
        st.success("Portfolio weights and asset expectations updated. The simulation will use these new values.")

# Use the most up-to-date data for the simulation
df_cartera = st.session_state.df_editable.query('Peso > 0')  # Remove assets with zero weight

# Run simulation
simulaciones, rendimientos = run_simulation(df_cartera, saldo_inicial, num_simulaciones, 
                                            anos_simulacion, aportacion_mensual)


def format_euro(value: float) -> str:
    """Format a value as euros."""
    return f"{value:,.0f} €".replace(",", ".")


def calculate_kpis(simulaciones: np.ndarray, anos_simulacion: int, saldo_inicial: float, aportacion_mensual: float) -> Dict[str, float]:
    """Calculate KPIs from the simulation results."""
    final_values = simulaciones[:, -1]
    total_invested = saldo_inicial + aportacion_mensual * 12 * anos_simulacion
    
    kpis = {
        "Median Final Value": np.median(final_values),
        "Average Annual Return": (np.median(final_values) / total_invested) ** (1 / anos_simulacion) - 1,
        "Probability of Profit": np.mean(final_values > total_invested),
        "Value at Risk (5%)": np.percentile(final_values, 5),
        "Maximum Drawdown": calculate_max_drawdown(simulaciones),
    }
    return kpis

def calculate_max_drawdown(simulaciones: np.ndarray) -> float:
    """Calculate the maximum drawdown across all simulations."""
    cummax = np.maximum.accumulate(simulaciones, axis=1)
    drawdowns = (simulaciones - cummax) / cummax
    return np.min(drawdowns)

def display_kpis(kpis: Dict[str, float]):
    """Display KPIs in Streamlit."""
    col1, col2, col3 = st.columns(3)
    
    with col1:
        st.metric("Median Final Value", format_euro(kpis['Median Final Value']))
        st.metric("Average Annual Return", f"{kpis['Average Annual Return']:.2%}")
    
    with col2:
        st.metric("Probability of Profit", f"{kpis['Probability of Profit']:.2%}")
        st.metric("Value at Risk (5%)", format_euro(kpis['Value at Risk (5%)']))
    
    with col3:
        st.metric("Maximum Drawdown", f"{kpis['Maximum Drawdown']:.2%}")



# Calculate KPIs
kpis = calculate_kpis(simulaciones, anos_simulacion, saldo_inicial, aportacion_mensual)

# Display KPIs
st.subheader("Key Performance Indicators")
display_kpis(kpis)





def plot_simulation_results(simulaciones: np.ndarray, anos_simulacion: int):
    """Plot the simulation results with date formatting and sorted hover values."""
    dias_simulacion = 252 * anos_simulacion
    percentiles = np.percentile(simulaciones, [1, 10, 25, 50, 75, 90, 99], axis=0)
    final_values = simulaciones[:, -1]  # Final portfolio values for the histogram

    # Generate dates starting from today
    start_date = datetime.now()
    #x_axis = [start_date + timedelta(days=i) for i in range(0, dias_simulacion)]
    x_axis = pd.bdate_range(start=start_date, periods=dias_simulacion).to_pydatetime().tolist()

    # Create a subplot with shared y-axis; the second plot is a histogram
    fig = make_subplots(
        rows=1, cols=2,
        column_widths=[0.8, 0.2],
        shared_yaxes=True,
        horizontal_spacing=0.05,
        specs=[[{"type": "scatter"}, {"type": "bar"}]]
    )

    # Percentile values sorted for hover template
    hover_template = (
        
        '99th Percentile : %{customdata[0]:.0f} €<br>'
        '90th Percentile : %{customdata[1]:.0f} €<br>'
        '75th Percentile : %{customdata[2]:.0f} €<br>'
        '50th Percentile : %{customdata[3]:.0f} €<br>'
        '25th Percentile : %{customdata[4]:.0f} €<br>'
        '10th Percentile : %{customdata[5]:.0f} €<br>'
        '1st Percentile : %{customdata[6]:.0f} €<extra></extra>'
    )

    # Adding all lines
    fig.add_trace(go.Scatter(
        x=x_axis,
        y=percentiles[3],  # Median
        mode='lines',
        name='Median',
        line=dict(color='green', width=2),
        showlegend=False,
        customdata=np.column_stack([percentiles[6], percentiles[5], percentiles[4],
                                    percentiles[3], percentiles[2], percentiles[1], percentiles[0]]),
        hovertemplate=hover_template
    ), row=1, col=1)

    # Add the percentile areas
    fig.add_trace(go.Scatter(
        x=x_axis,
        y=percentiles[6],  # 99th Percentile
        mode='lines',
        line=dict(width=0),
        fill=None,
        showlegend=False,
        hoverinfo='skip'
    ), row=1, col=1)

    fig.add_trace(go.Scatter(
        x=x_axis,
        y=percentiles[0],  # 1st Percentile
        mode='lines',
        fill='tonexty',
        fillcolor='rgba(0, 100, 80, 0.5)',
        line=dict(width=0),
        showlegend=False,
        hoverinfo='skip'
    ), row=1, col=1)



        # Add the percentile areas
    fig.add_trace(go.Scatter(
        x=x_axis,
        y=percentiles[5],  # 99th Percentile
        mode='lines',
        line=dict(width=0),
        fill=None,
        showlegend=False,
        hoverinfo='skip'
    ), row=1, col=1)

    fig.add_trace(go.Scatter(
        x=x_axis,
        y=percentiles[1],  # 1st Percentile
        mode='lines',
        fill='tonexty',
        fillcolor='rgba(0, 150, 80, 0.5)',
        line=dict(width=0),
        showlegend=False,
        hoverinfo='skip'
    ), row=1, col=1)


            # Add the percentile areas
    fig.add_trace(go.Scatter(
        x=x_axis,
        y=percentiles[4],  # 99th Percentile
        mode='lines',
        line=dict(width=0),
        fill=None,
        showlegend=False,
        hoverinfo='skip'
    ), row=1, col=1)

    fig.add_trace(go.Scatter(
        x=x_axis,
        y=percentiles[2],  # 1st Percentile
        mode='lines',
        fill='tonexty',
        fillcolor='rgba(0, 200, 80, 0.5)',
        line=dict(width=0),
        showlegend=False,
        hoverinfo='skip'
    ), row=1, col=1)

    # Add the histogram of final portfolio values
    fig.add_trace(
        go.Histogram(
            y=final_values,
            orientation='h',
            marker=dict(color='rgba(100, 149, 237, 0.8)'),
            showlegend=False,
            nbinsy=100,
            hovertemplate='%{y:.0f} €',
        ), row=1, col=2
    )

    # Update layout for better aesthetics
    fig.update_layout(
        title='Enhanced Portfolio Value Simulation with Distribution Histogram',
        xaxis=dict(
            title='Date', 
            tickformat='%B %Y',  # Display dates as "January 2025"
            tickvals=x_axis[::63]  # Set monthly tick interval
        ),
        xaxis2=dict(title='Number of Simulations'),
        yaxis_title='Portfolio Value',
        hovermode='x unified',
        template='plotly_white',
        legend=dict(orientation="h", yanchor="top", y=0.83, xanchor="center", x=0.5),
        margin=dict(l=50, r=50, b=100, t=100, pad=4),height=650
    )

    # Adjust axis settings
    fig.update_yaxes(showgrid=True, zeroline=True, zerolinewidth=1, zerolinecolor='gray')

    return fig

# Example call to the improved plot function with simulation data


st.plotly_chart(plot_simulation_results(simulaciones, anos_simulacion), use_container_width=True)



# Calcular IRR para cada simulación usando fechas
irrs = []
start_date = datetime.now()
dias_simulacion = anos_simulacion *252
fechas_simulacion = pd.bdate_range(start=start_date, periods=dias_simulacion).to_pydatetime().tolist()
fecha_aportes  = generate_monthly_indices(anos_simulacion*252, datetime.now())


for i in range(num_simulaciones):
    # Crear los flujos de caja: inicial negativo, aportaciones negativas, y valor final positivo
    cash_flows = [-saldo_inicial] + [-aportacion_mensual] * len(fecha_aportes) + [simulaciones[i, -1]]
    
    # Crear las fechas correspondientes para los flujos de caja
    cash_flow_dates = [start_date] + [fechas_simulacion[idx] for idx in fecha_aportes] + [fechas_simulacion[-1]]
    
    # Calcular IRR usando xirr
    try:
        irr = xirr(dict(zip(cash_flow_dates, cash_flows)))
        irrs.append(irr * 100)  # Convertir a porcentaje
    except ValueError:
        irrs.append(np.nan)  # Manejar casos en los que no se pueda calcular



# Limpiar NaN de IRR antes del histograma
irrs = [x for x in irrs if not np.isnan(x)]

# Crear figura de histograma
fig_hist = go.Figure()

# Agregar histograma con normalización a porcentajes
fig_hist.add_trace(go.Histogram(
    x=irrs, 
    nbinsx=50,
    histnorm='percent',  # Normalizar a porcentaje
    marker_color='rgba(46, 204, 113, 0.7)',  # Color verde claro para modo oscuro
    marker_line=dict(color='rgba(255, 255, 255, 0.5)', width=1)
))

# Actualizar diseño del gráfico
fig_hist.update_layout(
    title='Distribución de IRR de las Simulaciones',
    xaxis_title='IRR (%)',
    yaxis_title='Frecuencia (%)',  # Cambiar el título del eje y a porcentaje
    font=dict(color='white'),
    height=600
)

# Mostrar gráfico
st.plotly_chart(fig_hist, use_container_width=True)

################################







