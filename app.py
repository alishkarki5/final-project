# This app is for educational purpose only. Insights gained is not financial advice. Use at your own risk!
import streamlit as st
from PIL import Image
import pandas as pd
import random
import base64
import json
import time
import yaml
from dashboard.show_result import get_prediction
#---------------------------------#
# New feature (make sure to upgrade your streamlit library)
# pip install --upgrade streamlit

with open("config.yaml", "r") as stream:
    cl = yaml.safe_load(stream)
train = cl['train']
feature = cl['feature']
train_data = pd.read_csv(train,names=feature).drop("label", axis=1)
st.set_page_config(layout="wide")

#---------------------------------#
# Title

#image = Image.open('./raw_data/detail.png')

#st.image(image, width = 500)

st.title('Attack Detection App')
st.markdown("""
This app is use to detect the **ATTACK**! in Networking
""")
#---------------------------------#
# About
expander_bar = st.expander("INTRODUCTION")
expander_bar.markdown("""
* **ABOUT:** Have you ever wondered how your computer/network is able to avoid being infected with malware and bad traffic inputs from the internet? 
The reason why it can detect it so well is because there are systems in place to protect your valuable information held in your computer or networks. These systems that detect malicious traffic inputs are called Intrusion Detection Systems (IDS) and are trained on internet traffic record data. The most common data set is the NSL-KDD, and is the benchmark for modern-day internet traffic.

* **Data source:** [NSLKDD](https://www.unb.ca/cic/datasets/nsl.html)
* **About Dataset:** This data set is comprised of four sub data sets: KDDTest+, KDDTest-21, KDDTrain+, KDDTrain+_20Percent, although KDDTest-21 and KDDTrain+_20Percent are subsets of the KDDTrain+ and KDDTest+. From now on, KDDTrain+ will be referred to as train and KDDTest+ will be referred to as test. The KDDTest-21 is a subset of test, without the most difficult traffic records (Score of 21), and the KDDTrain+_20Percent is a subset of train, whose record count makes up 20% of the entire train dataset. That being said, the traffic records that exist in the KDDTest-21 and KDDTrain+_20Percent are already in test and train respectively and aren’t new records held out of either dataset.
""")
#---------------------------------#
# Page layout (continued)
## Divide page to 3 columns (col1 = sidebar, col2 and col3 = page contents)
col1 = st.sidebar
col2, col3 = st.columns((2,1))

#---------------------------------#
col1.header('Input Options')

## Sidebar - Currency price unit
choosed_algorithm = col1.selectbox('Choose The Algorithm', ( 
    'SVC', 
    'RandomForestClassifier',
    'DecisionTreeClassifier',
    'VotingClassifier'
    ))
pie_image = Image.open('./static-stremlit/pie_chart_data.png')
col1.text('Data Label and Its Distribution')
col1.image(pie_image)

st.dataframe(train_data.head(random.randint(0,len(train_data))))

if(st.button('Predict')):
    
    st.warning(f'The attrack prediction is **{get_prediction(choosed_algorithm)[0]}**')
    cf_image = Image.open(f'./static-stremlit/{choosed_algorithm}.png')
    st.text(f'This is the Confusion matrix of **{choosed_algorithm}**')
    st.image(cf_image)


# Web scraping of CoinMarketCap data
# @st.cache
# def load_data():
#     cmc = requests.get('https://coinmarketcap.com')
#     soup = BeautifulSoup(cmc.content, 'html.parser')

#     data = soup.find('script', id='__NEXT_DATA__', type='application/json')
#     coins = {}
#     coin_data = json.loads(data.contents[0])
#     listings = coin_data['props']['initialState']['cryptocurrency']['listingLatest']['data']
#     for i in listings:
#       coins[str(i['id'])] = i['slug']

#     coin_name = []
#     coin_symbol = []
#     market_cap = []
#     percent_change_1h = []
#     percent_change_24h = []
#     percent_change_7d = []
#     price = []
#     volume_24h = []

#     for i in listings:
#       coin_name.append(i['slug'])
#       coin_symbol.append(i['symbol'])
#       price.append(i['quote'][currency_price_unit]['price'])
#       percent_change_1h.append(i['quote'][currency_price_unit]['percent_change_1h'])
#       percent_change_24h.append(i['quote'][currency_price_unit]['percent_change_24h'])
#       percent_change_7d.append(i['quote'][currency_price_unit]['percent_change_7d'])
#       market_cap.append(i['quote'][currency_price_unit]['market_cap'])
#       volume_24h.append(i['quote'][currency_price_unit]['volume_24h'])

#     df = pd.DataFrame(columns=['coin_name', 'coin_symbol', 'market_cap', 'percent_change_1h', 'percent_change_24h', 'percent_change_7d', 'price', 'volume_24h'])
#     df['coin_name'] = coin_name
#     df['coin_symbol'] = coin_symbol
#     df['price'] = price
#     df['percent_change_1h'] = percent_change_1h
#     df['percent_change_24h'] = percent_change_24h
#     df['percent_change_7d'] = percent_change_7d
#     df['market_cap'] = market_cap
#     df['volume_24h'] = volume_24h
#     return df

# df = load_data()

# ## Sidebar - Cryptocurrency selections
# sorted_coin = sorted( df['coin_symbol'] )
# selected_coin = col1.multiselect('Cryptocurrency', sorted_coin, sorted_coin)

# df_selected_coin = df[ (df['coin_symbol'].isin(selected_coin)) ] # Filtering data

# ## Sidebar - Number of coins to display
# num_coin = col1.slider('Display Top N Coins', 1, 100, 100)
# df_coins = df_selected_coin[:num_coin]

# ## Sidebar - Percent change timeframe
# percent_timeframe = col1.selectbox('Percent change time frame',
#                                     ['7d','24h', '1h'])
# percent_dict = {"7d":'percent_change_7d',"24h":'percent_change_24h',"1h":'percent_change_1h'}
# selected_percent_timeframe = percent_dict[percent_timeframe]

# ## Sidebar - Sorting values
# sort_values = col1.selectbox('Sort values?', ['Yes', 'No'])

# col2.subheader('Price Data of Selected Cryptocurrency')
# col2.write('Data Dimension: ' + str(df_selected_coin.shape[0]) + ' rows and ' + str(df_selected_coin.shape[1]) + ' columns.')

# col2.dataframe(df_coins)

# # Download CSV data
# # https://discuss.streamlit.io/t/how-to-download-file-in-streamlit/1806
# def filedownload(df):
#     csv = df.to_csv(index=False)
#     b64 = base64.b64encode(csv.encode()).decode()  # strings <-> bytes conversions
#     href = f'<a href="data:file/csv;base64,{b64}" download="crypto.csv">Download CSV File</a>'
#     return href

# col2.markdown(filedownload(df_selected_coin), unsafe_allow_html=True)

# #---------------------------------#
# # Preparing data for Bar plot of % Price change
# col2.subheader('Table of % Price Change')
# df_change = pd.concat([df_coins.coin_symbol, df_coins.percent_change_1h, df_coins.percent_change_24h, df_coins.percent_change_7d], axis=1)
# df_change = df_change.set_index('coin_symbol')
# df_change['positive_percent_change_1h'] = df_change['percent_change_1h'] > 0
# df_change['positive_percent_change_24h'] = df_change['percent_change_24h'] > 0
# df_change['positive_percent_change_7d'] = df_change['percent_change_7d'] > 0
# col2.dataframe(df_change)

# # Conditional creation of Bar plot (time frame)
# col3.subheader('Bar plot of % Price Change')

# if percent_timeframe == '7d':
#     if sort_values == 'Yes':
#         df_change = df_change.sort_values(by=['percent_change_7d'])
#     col3.write('*7 days period*')
#     plt.figure(figsize=(5,25))
#     plt.subplots_adjust(top = 1, bottom = 0)
#     df_change['percent_change_7d'].plot(kind='barh', color=df_change.positive_percent_change_7d.map({True: 'g', False: 'r'}))
#     col3.pyplot(plt)
# elif percent_timeframe == '24h':
#     if sort_values == 'Yes':
#         df_change = df_change.sort_values(by=['percent_change_24h'])
#     col3.write('*24 hour period*')
#     plt.figure(figsize=(5,25))
#     plt.subplots_adjust(top = 1, bottom = 0)
#     df_change['percent_change_24h'].plot(kind='barh', color=df_change.positive_percent_change_24h.map({True: 'g', False: 'r'}))
#     col3.pyplot(plt)
# else:
#     if sort_values == 'Yes':
#         df_change = df_change.sort_values(by=['percent_change_1h'])
#     col3.write('*1 hour period*')
#     plt.figure(figsize=(5,25))
#     plt.subplots_adjust(top = 1, bottom = 0)
#     df_change['percent_change_1h'].plot(kind='barh', color=df_change.positive_percent_change_1h.map({True: 'g', False: 'r'}))
#     col3.pyplot(plt)