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

@st.cache_data
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

    precios["WBIT"] = yf.download("BTC-USD", start=fecha_inicio, progress=False).resample("B").ffill()["Adj Close"] / eurusd * 0.0002401
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
    benchmark = yf.download("IWDA.AS", start=fecha_inicio, progress=False).resample("B").ffill()["Adj Close"]
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

valor_benchmark = movimientos["Ud sim benchmark"].groupby(level=0).last().reindex(benchmark.index, method='ffill') * benchmark

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

st.write(tabla_resumen)

st.divider()

col1, col2 = st.columns(2)


if opcion_seleccionada == "Omite monetarios":

    with col1:

        start_date  = col1.date_input("Fecha inicio",date(2023,9,1),format="DD-MM-YYYY")

else:
    with col1:

        start_date  = col1.date_input("Fecha inicio",date(2023,1,1),format="DD-MM-YYYY")



with col2:
    end_date = st.date_input("Fecha final",date.today(),format="DD-MM-YYYY")





vsBench = pd.DataFrame(rendimiento_portfolio,columns=["Portfolio"])
vsBench["Benchmark"] = benchmark.pct_change()

growthline_portfolio = (1+vsBench[start_date:end_date]).cumprod()


fig = px.line(growthline_portfolio,title='Evolución índice cartera')

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

        #st.plotly_chart(fig,use_container_width=True)

    elif irr_opcion == "Delta":


        # Create the area plot
        fig = px.area(compare_irr, x=compare_irr.index, y=["Positive_Gap", "Negative_Gap"], title='Difference Area Graph')

        # Customize the plot colors
        fig.data[0].update(mode='lines', line=dict(width=0), fill='tozeroy', fillcolor='rgba(0, 255, 0, 0.5)')
        fig.data[1].update(mode='lines', line=dict(width=0), fill='tozeroy', fillcolor='rgba(255, 0, 0, 0.5)')


    else:

        with col2:

            asset_irr = st.selectbox("Activo",xirr_df.columns)

            fig = px.line(100 * xirr_df[asset_irr],title='IRR')


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
    
    return result


# Parameters in four columns
col1, col2, col3, col4 = st.columns(4)

with col1:
    start_date = st.date_input('Start Date', rendimientos_corr.index.min(),format="DD-MM-YYYY")

with col2:
    end_date = st.date_input('End Date', rendimientos_corr.index.max(),format="DD-MM-YYYY")

with col3:
    reference_column = st.selectbox('Reference Column', rendimientos_corr.columns, index=rendimientos_corr.columns.get_loc('1.1 World'))

with col4:
    window = st.slider('Rolling Window (days)', min_value=2, max_value=180, value=30)

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
    risk_free_rate = 0.035  # 3.5% annual rate
    performance_df['Sharpe_Ratio'] = (performance_df['Returns'] - risk_free_rate) / performance_df['Volatility']

    # Create a color scale based on Sharpe Ratio
    color_scale = [
        [0, 'rgb(165,0,38)'],
        [0.25, 'rgb(215,48,39)'],
        [0.5, 'rgb(244,109,67)'],
        [0.75, 'rgb(253,174,97)'],
        [1, 'rgb(26,152,80)']
    ]

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
            colorbar=dict(title='Sharpe Ratio'),
            showscale=True
        ),
        hovertemplate='<b>%{text}</b><br>Returns: %{y:.2%}<br>Volatility: %{x:.2%}<br>Sharpe Ratio: %{marker.color:.2f}<extra></extra>'
    ))

    # Calculate mean values for quadrant lines
    mean_volatility = performance_df['Volatility'].mean()
    mean_returns = performance_df['Returns'].mean()

    # Add colored quadrants
    fig.add_shape(type="rect", x0=0, y0=mean_returns, x1=mean_volatility, y1=1, 
                  fillcolor="rgba(0,255,0,0.1)", line=dict(width=0))
    fig.add_shape(type="rect", x0=mean_volatility, y0=mean_returns, x1=1, y1=1, 
                  fillcolor="rgba(255,255,0,0.1)", line=dict(width=0))
    fig.add_shape(type="rect", x0=0, y0=0, x1=mean_volatility, y1=mean_returns, 
                  fillcolor="rgba(255,0,0,0.1)", line=dict(width=0))
    fig.add_shape(type="rect", x0=mean_volatility, y0=0, x1=1, y1=mean_returns, 
                  fillcolor="rgba(0,0,255,0.1)", line=dict(width=0))

    # Update layout
    fig.update_layout(
        title={
            'text': f'Asset Performance: Returns vs Volatility ({start_date.date()} to {end_date.date()})',
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
            dict(type="line", x0=0, y0=mean_returns, x1=1, y1=mean_returns, xref="paper", yref="y", line=dict(color="Grey", width=1, dash="dash"))
        ]
    )

    return fig


col1, col2, col3, col4, col5,col6 = st.columns(5)

# Initialize session state for start and end dates if not already set
if 'start_date' not in st.session_state or 'end_date' not in st.session_state:
    st.session_state.start_date = datetime.now() - timedelta(days=180)
    st.session_state.end_date = datetime.now()

def update_dates(days):
    st.session_state.end_date = datetime.now()
    st.session_state.start_date = st.session_state.end_date - timedelta(days=days)

def update_year_to_date():
    st.session_state.end_date = datetime.now()
    st.session_state.start_date = datetime.datetime(st.session_state.end_date.year, 1, 1)

with col1:
    if st.button('Last Month'):
        update_dates(30)
with col2:
    if st.button('Last 3 Months'):
        update_dates(90)
with col3:
    if st.button('Last 6 Months'):
        update_dates(180)
with col4:
    if st.button('Year to Date'):
        update_year_to_date()
with col5:
    start_date = st.date_input('Start Date', value=st.session_state.start_date, format="DD-MM-YYYY")
    st.session_state.start_date = datetime.combine(start_date, datetime.min.time())
with col6:
    end_date = st.date_input('End Date', value=st.session_state.end_date, format="DD-MM-YYYY")
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































































st.divider()


fig = qs.plots.snapshot(rendimiento_portfolio,show=False,figsize=(8,8))

st.pyplot(fig)

fig = qs.plots.monthly_heatmap(rendimiento_portfolio,benchmark,show=False,figsize=(8,5.5))

st.pyplot(fig)


fig =  qs.plots.drawdowns_periods(rendimiento_portfolio,show=False,figsize=(8,5.5))

st.pyplot(fig)



fig =   qs.plots.rolling_volatility(rendimiento_portfolio,period=20,period_label="21 días",show=False,figsize=(8,5.5))

st.pyplot(fig)