# This app is for educational purpose only. Insights gained is not financial advice. Use at your own risk!
import streamlit as st
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
* **Python libraries:** base64, pandas, streamlit, numpy, matplotlib, seaborn, YahooFinancials, plotly, csv, datetime
* **Data source:** yahoofinancials.
* **Credit:** [Streamlit YouTube examples](https://www.youtube.com/watch?v=JwSS70SZdyM&feature=youtu.be&ab_channel=freeCodeCamp.org) by [Data Professor] (https://data-professor.medium.com/)
* **Support:** Please sponsor by visiting my [buymeacoffee] (https://www.buymeacoffee.com/thedataguy) to support my app hosting if you find it useful, ideas welcome! 
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
riskfreevalue = 0 
riskfree_input = col1.number_input("Select the % of risk free rate", value = riskfreevalue)

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
            st.write('If you bought the stock on ',dt.date(), ', you will buy on the price :    ',np.round(closing , decimals =2),' versus the latest date ',   Stock.iloc[-1]['formatted_date'].date(),'price :    ',np.round(Stock.iloc[-1]['close'], decimals =2) , ' | Return on investing would be ',np.round((Stock.iloc[-1]['close']/ closing-1)*100, decimals = 2) , '%  ')
simpleifsloop(rawdata)

@st.cache
def getFTSE100():
    ### Getting FTSE 100
    ftse100 = '^FTSE'
    yahoo_financials = YahooFinancials(ftse100)
    data = yahoo_financials.get_historical_price_data(start_date=str(startdate), end_date=str(enddate), time_interval='daily')
    df = pd.DataFrame(data[ftse100]['prices']).sort_values('date',ascending = True).reset_index()
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
    ### Getting FTSE 100
    ftse250 = '^FTMC'
    yahoo_financials = YahooFinancials(ftse250)
    data = yahoo_financials.get_historical_price_data(start_date=str(startdate), end_date=str(enddate), time_interval='daily')
    df = pd.DataFrame(data[ftse250]['prices']).sort_values('date',ascending = True).reset_index()
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
data = rawdata.loc[(rawdata['formatted_date'] >= inputdtst/art) & (rawdata['formatted_date'] <= inputdtend)]
def CombineData(stockdf,indexdf,dtstart,dtend):
    stockdf = stockdf.loc[(stockdf['formatted_date'] >= dtstart) & (stockdf['formatted_date'] <= dtend)]
    df1 = stockdf[['formatted_date', 'Daily Return']]
    indexdf = indexdf.loc[(indexdf['formatted_date'] >= dtstart) & (indexdf['formatted_date'] <= dtend)]
    df2 = indexdf[['formatted_date', 'Daily Return']]
    df = pd.merge(df1,df2,on='formatted_date', how='inner',suffixes=('_stock', '_index')).dropna()
    return df 
CAPMdf = CombineData(data,ftse250dt,inputdtstart,inputdtend)

# Fit a polynomial between each stock and the S&P500 (Poly with order = 1 is a straight line)
beta, alpha = np.polyfit(CAPMdf['Daily Return_index'], CAPMdf['Daily Return_stock'], 1)
# Assume risk free rate is zero in this case
rf = riskfree_input / 100.00  

# This is the expected return of the market 
rm = CAPMdf['Daily Return_index'].mean() * CAPMdf['Daily Return_index'].shape[0] 
# Calculate return for security using CAPM  
ER = rf + ( beta * (rm-rf) ) 


st.sidebar.write("On a high level, the FTSE 100 returned ", "{:.2%}".format((ftse100dt.iloc[-1]['close']/ftse100dt.iloc[0]['close'])-1) ,' and the FTSE 250 returned ', "{:.2%}".format((ftse250dt.iloc[-1]['close']/ftse250dt.iloc[0]['close'])-1) )
st.sidebar.write("  and this ", stockYFticker, " has returned ", "{:.2%}".format((data.iloc[-1]['close']/data.iloc[0]['close'])-1) , 'between ', inputdtstart , " & ", inputdtend )
st.sidebar.write("***Quick Daily Return Analysis***")
#Avg Daily Return
st.sidebar.write('Avg Daily Return: ',"{:.2%}".format(data['Daily Return'].mean()))
#Std Daily Return
st.sidebar.write('Std Daily Return: ',"{:.2%}".format(data['Daily Return'].std()))

# Plot Closing Price of Query Symbol
def price_plot(data):
    fig = px.line(title =  stockYFticker+' daily close interactive trend')  
    fig.add_scatter(x = data['formatted_date'], y = data['close'], name = stockYFticker)
    return st.plotly_chart(fig, use_container_width=True)

st.header("Closing price of " + stockYFticker)
price_plot(data)

ticker = yf.Ticker(stockYFticker)
with st.expander("Company Info"):
    #st.write('Business Intro:')
    #st.write(ticker.info['longBusinessSummary'])
    #st.write('Recommendations by Yahoo advisor:')
    #st.write(ticker.recommendations)
    #st.write('List of major holders:')
    #st.write(ticker.major_holders)
    #st.write('List of institutional holders:')
    #st.write(ticker.institutional_holders)
    #st.write('List of events on calendar:')
    #st.write(ticker.calendar)
    st.write('This Part of API borken- fixing in progress 09/2024 ')

#CAPM Formula: ri = rf + Bi * (rm - rf)
with st.expander("CAPM explain"):
    st.write("Fit a polynomial between the selected stock and the FTSE100 (Poly with order = 1 is a straight line)") 
fig= plt.figure(figsize = (12,3))
plt.grid()
plt.scatter(CAPMdf['Daily Return_index'] ,CAPMdf['Daily Return_stock'] )
plt.plot(CAPMdf['Daily Return_index'], beta * CAPMdf['Daily Return_index'] + alpha, "r--")
st.pyplot(fig)

#Beta Return: 
with st.expander("Beta explain"):
    st.write("Beta is a measure of the volatility or systematic risk of a security or portfolio compared to the entire market (S&P500) ")
    st.write("Beta is used in the CAPM and describes the relationship between systematic risk and expected return for assets")
    st.write("Beta = 1.0, this indicates that its price activity is strongly correlated with the market.") 
    st.write("Beta < 1, indicates that the security is theoretically less volatile than the market (Ex: Utility stocks). If the stock is included, this will make the portfolio less risky compared to the same portfolio without the stock.")
    st.write("Beta > 1, indicates that the security's price is more volatile than the market. For instance, Tesla stock beta is 1.26 indicating that it's 26% more volatile than the market. ")
if beta > 1:    
    st.write('Beta is', beta, 'meaning it is more volatile than the market by : ',"{:.2%}".format(beta - 1))   
elif beta < 1 :
    st.write('Beta is', beta, 'meaning it is less volatile than the market by : ',"{:.2%}".format(1 - beta))    
else: 
    st.write('Beta is', beta, 'meaning it is strongly correlated with the market.')    
#Alpha Return:
with st.expander("Alpha explain"):
    st.write("Alpha describes the strategy's ability to beat the market (FTSE100)")
    st.write("Alpha indicates the “excess return” or “abnormal rate of return,” ")
    st.write("For example, a positive 0.175 alpha for Tesla means that the portfolio’s return exceeded the benchmark index ftse250 by 17%.")
if alpha > 0:    
    st.write('Alpha is', alpha , 'meaning it is exceeding the benchmark index ftse250 by: ',"{:.2%}".format(alpha))   
elif alpha < 0 :
    st.write('Alpha is',alpha, 'meaning it is under performing the benchmark index ftse250 by : ',"{:.2%}".format(alpha))    
else: 
    st.write('Alpha is',"{:.2%}".format(alpha), 'meaning it is returning about the same as the benchmark index ftse250.')     
#Expected Return 
st.write('Stock CAPM Expected Return is: ',"{:.2%}".format(ER))   

#Sharp Ratio
SR = data['Daily Return'].mean()/data['Daily Return'].std()
ASR = (252**0.5)*SR
with st.expander("Sharp Ratio / Annualised Sharp Ratio"):
    st.write('****Note that the higher the ratios the better the historical return against risk ')
    st.write('and if negative then you are making a historical loss - the risk didn''t pay off!****')
st.write('Sharp Ratio: ',"{:.2%}".format(SR))
st.write('Annualised Sharp Ratio: ',"{:.2%}".format(ASR))


st.header("Random 10 day's of "+ stockYFticker)
st.dataframe(data.sample(10))
st.write('You can download the full daily data between ', inputdtstart ,' to ', inputdtend , 'here')

def filedownload(df):
    csv = data.to_csv(index=False)
    b64 = base64.b64encode(csv.encode()).decode()  # strings <-> bytes conversions
    href = f'<a href="data:file/csv;base64,{b64}" download="stockdata.csv">Download CSV File for full history </a>'
    return href

st.markdown(filedownload(data), unsafe_allow_html=True)

st.write('Many thanks for trying out the tool and please feel free to contact me on LinkedIn for recommendations or spare a coin or two [here](https://www.buymeacoffee.com/thedataguy) for hosting this tool! ')
