# This app is for educational purpose only. Insights gained is not financial advice. Use at your own risk!
import streamlit as st
#from PIL import Image
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from yahoofinancials import YahooFinancials
import yfinance as yf
import datetime, csv, base64
import plotly.express as px
import plotly.figure_factory as ff
import plotly.graph_objects as go
#---------------------------------#
# New feature (make sure to upgrade your streamlit library)
# pip install --upgrade streamlit

#---------------------------------#
# Page layout
## Page expands to full width
st.set_page_config(layout="wide")
#---------------------------------#
# Get the dates for operations
#First date of the full year 4 years ago: 
startdate = datetime.datetime(pd.to_datetime("today").year - 5 , 1, 1).date()
beginningof4yearsgo = datetime.datetime(pd.to_datetime("today").year - 4 , 1, 1).date()
beginningof3yearsgo = datetime.datetime(pd.to_datetime("today").year - 3 , 1, 1).date()
beginningof2yearsgo = datetime.datetime(pd.to_datetime("today").year - 2 , 1, 1).date()
beginningoflastyear = datetime.datetime(pd.to_datetime("today").year - 1 , 1, 1).date()
beginningofthisyear = datetime.datetime(pd.to_datetime("today").year , 1, 1).date()
enddate = (pd.Timestamp.today() - pd.DateOffset(days=1)).date()

st.title('Stock Basic Analysis App')
st.markdown("""
This app retrieves stock prices and add basic analysis from the **YahooFinance**!
""")
#---------------------------------#
# About
expander_bar = st.expander("About")
expander_bar.markdown("""
* **Python libraries:** base64, pandas, streamlit, numpy, matplotlib, seaborn, BeautifulSoup, requests, json, time
* **Data source:** yahoofinancials.
* **Credit:** [Streamlit YouTube examples](https://www.youtube.com/watch?v=JwSS70SZdyM&feature=youtu.be&ab_channel=freeCodeCamp.org) by [Data Professor] (https://data-professor.medium.com/)
""")


#---------------------------------#
# Page layout (continued)
## Divide page to 3 columns (col1 = sidebar, col2 and col3 = page contents)
col1 = st.sidebar
col2, col3 = st.columns((2,1))

#---------------------------------#
# Sidebar + Main panel
col1.header('Input Options')
ticket_input = "ROO.L"
stockYFticker = col1.text_input("Enter Yahoo ticker here. For Example: ROO.L", ticket_input)
## Sidebar - Currency price unit
#currency_price_unit = col1.selectbox('Select currency for price', ('USD', 'BTC', 'ETH'))


def load_data(stockname):
    yahoo_financials = YahooFinancials(stockname)
    data = yahoo_financials.get_historical_price_data(start_date=str(startdate), end_date=str(enddate), time_interval='daily')
    data = pd.DataFrame(data[stockname]['prices']).sort_values('date',ascending = True).reset_index()
    data['formatted_date'] = pd.to_datetime(data['formatted_date'], format='%Y-%m-%d')
    data['month'] = data['formatted_date'].dt.month
    data['year'] = data['formatted_date'].dt.year
    data['returns'] = data['close'].pct_change(1)
    data['Cumulative Return'] = (1 + data['returns']).cumprod()
    data['Normed Return'] = data['adjclose']/data.iloc[0]['adjclose']
    data['DailyRange'] = abs(data['open'] - data['close'])
    # Daily Return
    data['Daily Return'] = data['Normed Return'].pct_change(1)
    data['stock'] = stockname
    data.drop(columns=['index','date'], inplace = True)
    return data
rawdata = load_data(stockYFticker)
#st.dataframe(ftse100.sample(1))
datastart = rawdata['formatted_date'].min()
dataend = rawdata['formatted_date'].max()

inputdtstart = pd.to_datetime(st.sidebar.date_input("StartDate of analysis",datastart))
inputdtend = pd.to_datetime(st.sidebar.date_input("EndDate of analysis",dataend))
def simpleifsloop(Stock):
    comparedates =  [pd.to_datetime(startdate),pd.to_datetime(beginningof4yearsgo),pd.to_datetime(beginningof3yearsgo),pd.to_datetime(beginningof2yearsgo),pd.to_datetime(beginningoflastyear)]
    #,pd.to_datetime(beginningofthisyear)]
    for dt in comparedates: 
        closing = Stock.iloc[abs(Stock['formatted_date'] - dt ).idxmin()+1]['close']   
        if dt >= inputdtstart:           
            st.write('If you bought the stock on ',dt.date(), ', you will buy on the price :    ',np.round(closing , decimals =2),' versus the latest date ',   Stock.iloc[-1]['formatted_date'].date(),'price :    ',np.round(Stock.iloc[-1]['close'], decimals =2)) 
            st.write('Return on investing would be ',np.round((Stock.iloc[-1]['close']/ closing-1)*100, decimals = 2) , '%  ')
        #Stock['Daily Return'].hist(bins=100, title="Distribution of daily return")
simpleifsloop(rawdata)

@st.cache
def getFTSE100():
    #print('Getting data of UK key market from {} to {}'.format(startdate, enddate))
    #print('Index we are getting are')
    ### Getting FTSE 100
    ftse100 = '^FTSE'
    yahoo_financials = YahooFinancials(ftse100)
    data = yahoo_financials.get_historical_price_data(start_date=str(startdate), end_date=str(enddate), time_interval='daily')
    df = pd.DataFrame(data[ftse100]['prices']).sort_values('date',ascending = False).reset_index()
    df['indextype'] = 'ftse100'
    df['formatted_date'] = pd.to_datetime(df['formatted_date'], format='%Y-%m-%d')
    df['month'] = df['formatted_date'].dt.month
    df['year'] = df['formatted_date'].dt.year
    df['returns'] = df['close'].pct_change(1)
    df['Cumulative Return'] = (1 + df['returns']).cumprod()
    df['Normed Return'] = df['adjclose']/df.iloc[0]['adjclose']
    df['DailyRange'] = abs(df['open'] - df['close'])
    # Daily Return
    df['Daily Return'] = df['Normed Return'].pct_change(1)
    return df  
@st.cache 
def getFTSE250():
    #print('Getting data of UK key market from {} to {}'.format(startdate, enddate))
    #print('Index we are getting are')
    ### Getting FTSE 100
    ftse250 = '^FTMC'
    yahoo_financials = YahooFinancials(ftse250)
    data = yahoo_financials.get_historical_price_data(start_date=str(startdate), end_date=str(enddate), time_interval='daily')
    df = pd.DataFrame(data[ftse250]['prices']).sort_values('date',ascending = False).reset_index()
    df['indextype'] = 'ftse250'
    df['formatted_date'] = pd.to_datetime(df['formatted_date'], format='%Y-%m-%d')
    df['month'] = df['formatted_date'].dt.month
    df['year'] = df['formatted_date'].dt.year
    df['returns'] = df['close'].pct_change(1)
    df['Cumulative Return'] = (1 + df['returns']).cumprod()
    df['Normed Return'] = df['adjclose']/df.iloc[0]['adjclose']
    df['DailyRange'] = abs(df['open'] - df['close'])
    # Daily Return
    df['Daily Return'] = df['Normed Return'].pct_change(1)
    return df   
ftse100 = getFTSE100()
ftse100dt = ftse100.loc[(ftse100['formatted_date'] >= inputdtstart) & (ftse100['formatted_date'] <= inputdtend)]
ftse250 = getFTSE250()
ftse250dt = ftse250.loc[(ftse100['formatted_date'] >= inputdtstart) & (ftse250['formatted_date'] <= inputdtend)]
data = rawdata.loc[(rawdata['formatted_date'] >= inputdtstart) & (rawdata['formatted_date'] <= inputdtend)]

st.sidebar.write("On a high level, the FTSE 100 returned ", str(np.round(((ftse100dt.iloc[-1]['close']/ftse100dt.iloc[0]['close'])-1)  * 100,decimals =2)) ,'% and the FTSE 250 returned ', str(np.round(((ftse250dt.iloc[-1]['close']/ftse250dt.iloc[0]['close'])-1)  * 100,decimals =2)), '%')
st.sidebar.write("  and this ", stockYFticker, " has returned ", str(np.round(((data.iloc[-1]['close']/data.iloc[0]['close'])-1)  *100,decimals =2)) , '% between ', inputdtstart , " & ", inputdtend )


# Plot Closing Price of Query Symbol
def price_plot(data):
  #df = pd.DataFrame(data[symbol].close)
  #fig, ax = plt.subplots(figsize=(20,4))
#  data['Date'] = data.formatted_date
#  #plt.fill_between(data.formatted_date, data.close, color='skyblue', alpha=0.3)
#  plt.plot(data.formatted_date, data.close, color='skyblue', alpha=0.8)
#  plt.xticks(rotation=90)
#  plt.title(stockYFticker, fontweight='bold')
#  plt.xlabel('Date', fontweight='bold')
#  plt.ylabel('Closing Price', fontweight='bold')
    fig = px.line(title =  stockYFticker+' daily close interactive trend')  
    fig.add_scatter(x = data['formatted_date'], y = data['close'], name = stockYFticker)
    return st.plotly_chart(fig, use_container_width=True)

st.header("Closing price of " + stockYFticker)
price_plot(data)

ticker = yf.Ticker(stockYFticker)
with st.expander("Company Info"):
    st.write('Business Intro:')
    st.write(ticker.info['longBusinessSummary'])
    st.write('Recommendations by Yahoo advisor:')
    st.write(ticker.recommendations)
    st.write('List of major holders:')
    st.write(ticker.major_holders)
    st.write('List of institutional holders:')
    st.write(ticker.institutional_holders)
    st.write('List of events on calendar:')
    st.write(ticker.calendar)

st.header("Random 10 day's of "+ stockYFticker)
st.dataframe(data.sample(10))

def filedownload(df):
    csv = data.to_csv(index=False)
    b64 = base64.b64encode(csv.encode()).decode()  # strings <-> bytes conversions
    href = f'<a href="data:file/csv;base64,{b64}" download="stockdata.csv">Download CSV File for full history </a>'
    return href

st.markdown(filedownload(data), unsafe_allow_html=True)


## Web scraping of CoinMarketCap data
#@st.cache
#def load_data():
#    cmc = requests.get('https://coinmarketcap.com')
#    soup = BeautifulSoup(cmc.content, 'html.parser')

#    data = soup.find('script', id='__NEXT_DATA__', type='application/json')
#    coins = {}
#    coin_data = json.loads(data.contents[0])
#    listings = coin_data['props']['initialState']['cryptocurrency']['listingLatest']['data']
#    for i in listings:
#      coins[str(i['id'])] = i['slug']

#    coin_name = []
#    coin_symbol = []
#    market_cap = []
#    percent_change_1h = []
#    percent_change_24h = []
#    percent_change_7d = []
#    price = []
#    volume_24h = []

#    for i in listings:
#      coin_name.append(i['slug'])
#      coin_symbol.append(i['symbol'])
#      price.append(i['quote'][currency_price_unit]['price'])
#      percent_change_1h.append(i['quote'][currency_price_unit]['percent_change_1h'])
#      percent_change_24h.append(i['quote'][currency_price_unit]['percent_change_24h'])
#      percent_change_7d.append(i['quote'][currency_price_unit]['percent_change_7d'])
#      market_cap.append(i['quote'][currency_price_unit]['market_cap'])
#      volume_24h.append(i['quote'][currency_price_unit]['volume_24h'])

#    df = pd.DataFrame(columns=['coin_name', 'coin_symbol', 'market_cap', 'percent_change_1h', 'percent_change_24h', 'percent_change_7d', 'price', 'volume_24h'])
#    df['coin_name'] = coin_name
#    df['coin_symbol'] = coin_symbol
#    df['price'] = price
#    df['percent_change_1h'] = percent_change_1h
#    df['percent_change_24h'] = percent_change_24h
#    df['percent_change_7d'] = percent_change_7d
#    df['market_cap'] = market_cap
#    df['volume_24h'] = volume_24h
#    return df

#df = load_data()

## Sidebar - Cryptocurrency selections
#sorted_coin = sorted( df['Symbol'] )
#selected_coin = col1.multiselect('Cryptocurrency', sorted_coin, sorted_coin)#

#df_selected_coin = df[ (df['Symbol'].isin(selected_coin)) ] # Filtering data#

### Sidebar - Number of coins to display
#num_coin = col1.slider('Display Top N Coins', 1, 20, 10)
#df_coins = df_selected_coin[:num_coin]#

### Sidebar - Percent change timeframe
#percent_timeframe = col1.selectbox('Percent change time frame',
#                                    ['7d','24h', '1h'])
##'% 1h','void2', '% 24h', 'void3', '% 7d'
#percent_dict = {"7d":'% 7d',"24h":'% 24h',"1h":'% 1h'}
#selected_percent_timeframe = percent_dict[percent_timeframe]#

### Sidebar - Sorting values
#sort_values = col1.selectbox('Sort values?', ['Yes', 'No'])#

#col2.subheader('Price Data of Selected Cryptocurrency')
#col2.write('Data Dimension: ' + str(df_selected_coin.shape[0]) + ' rows and ' + str(df_selected_coin.shape[1]) + ' columns.')#

#col2.dataframe(df_coins)#

## Download CSV data
## https://discuss.streamlit.io/t/how-to-download-file-in-streamlit/1806
#def filedownload(df,ind):
#    csv = df.to_csv(index=ind)
#    b64 = base64.b64encode(csv.encode()).decode()  # strings <-> bytes conversions
#    href = f'<a href="data:file/csv;base64,{b64}" download="crypto.csv">Download CSV File</a>'
#    return href#

#col2.markdown(filedownload(df_selected_coin,False), unsafe_allow_html=True)#

##---------------------------------#
## Preparing data for Bar plot of % Price change
#col2.subheader('Table of % Price Change')
#df_change = pd.concat([df_coins.Symbol, df_coins['% 1h'], df_coins['% 24h'], df_coins['% 7d']], axis=1)
#df_change = df_change.set_index('Symbol')
#df_change['% 1h'] = df_change['% 1h'].astype(float)
#df_change['% 24h'] = df_change['% 24h'].astype(float)
#df_change['% 7d'] = df_change['% 7d'].astype(float)
#df_change['positive_percent_change_1h'] = df_change['% 1h'].astype(float) > 0
#df_change['positive_percent_change_24h'] = df_change['% 24h'].astype(float) > 0
#df_change['positive_percent_change_7d'] = df_change['% 7d'].astype(float) > 0
#col2.dataframe(df_change)#

## Download CSV data
## https://discuss.streamlit.io/t/how-to-download-file-in-streamlit/1806
#col2.markdown(filedownload(df_change,True), unsafe_allow_html=True)#

## Conditional creation of Bar plot (time frame)
#col3.subheader('Bar plot of % Price Change')#

#if percent_timeframe == '7d':
#    if sort_values == 'Yes':
#        df_change = df_change.sort_values(by=['% 7d'])
#    col3.write('*7 days period*')
#    plt.figure(figsize=(5,10))
#    plt.subplots_adjust(top = 1, bottom = 0)
#    df_change['% 7d'].plot(kind='barh', color=df_change['positive_percent_change_7d'].map({True: 'g', False: 'r'}))
#    col3.pyplot(plt)
#elif percent_timeframe == '24h':
#    if sort_values == 'Yes':
#        df_change = df_change.sort_values(by=['% 24h'])
#    col3.write('*24 hour period*')
#    plt.figure(figsize=(5,10))
#    plt.subplots_adjust(top = 1, bottom = 0)
#    df_change['% 24h'].plot(kind='barh', color=df_change['positive_percent_change_24h'].map({True: 'g', False: 'r'}))
#    col3.pyplot(plt)
#else:
#    if sort_values == 'Yes':
#        df_change = df_change.sort_values(by=['% 1h'])
#    col3.write('*1 hour period*')
#    plt.figure(figsize=(5,10))
#    plt.subplots_adjust(top = 1, bottom = 0)
#    df_change['% 1h'].plot(kind='barh', color=df_change['positive_percent_change_1h'].map({True: 'g', False: 'r'}))
#    col3.pyplot(plt)