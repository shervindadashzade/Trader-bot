import requests
import json
import datetime
import time
import telegram
import pytz
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from tensorflow.keras.models import load_model

def get_csv(symbol, start_date,end_date, res):

  start_date_stamp  = int(time.mktime(datetime.datetime.strptime(start_date,'%Y-%m-%d').timetuple()))
  end_date_stamp  = int(time.mktime(datetime.datetime.strptime(f'{end_date} 23:59','%Y-%m-%d %H:%M').timetuple()))

  url = f"https://api.nobitex.ir/market/udf/history?symbol={symbol}&resolution={res}&from={start_date_stamp}&to={end_date_stamp}"

  payload={}
  headers = {}

  response = requests.request("GET", url, headers=headers, data=payload)

  res = response.text
  y = json.loads(res)
  df = pd.DataFrame({'Date':y['t'][::-1],'Close':y['c'][::-1], 'Open':y['o'], 'High':y['h'], 'Low':y['o']})
  df['Date'] = df['Date'].apply(lambda x: datetime.datetime.fromtimestamp(x, tz=pytz.timezone('Asia/Tehran')))
  return df

def get_csv_datetime(symbol, start_date,end_date, res):

  start_date_stamp  = int(datetime.datetime.timestamp(start_date) )
  end_date_stamp  = int(datetime.datetime.timestamp(end_date))

  url = f"https://api.nobitex.ir/market/udf/history?symbol={symbol}&resolution={res}&from={start_date_stamp}&to={end_date_stamp}"
  payload={}
  headers = {}

  response = requests.request("GET", url, headers=headers, data=payload)

  res = response.text
  y = json.loads(res)
  df = pd.DataFrame({'Date':y['t'][::-1],'Close':y['c'][::-1], 'Open':y['o'], 'High':y['h'], 'Low':y['o']})
  df['Date'] = df['Date'].apply(lambda x: datetime.datetime.fromtimestamp(x, tz=pytz.timezone('Asia/Tehran')))
  return df


def window_dataframe(df, start_date,end_date,num_of_features):
  
  #windowed_df = df[ (start_date <= df['Date']) & (df['Date'] <= end_date)]
  windowed_df = df[['Date','Close']]
  windowed_df = windowed_df.rename({'Close':'target'}, axis=1)

  for n in range(num_of_features):
    feature = []
    for index in range(len(windowed_df)):
      if index + num_of_features >= len(windowed_df):
        feature.append(0)
      else:
        feature.append(windowed_df.loc[index+n+1].target)
    windowed_df[f'target-{n+1}'] = feature

  windowed_df.drop(windowed_df.tail(num_of_features).index, inplace=True)
  return windowed_df

def windowed_df_to_date_X_y(windowed_df):
  
  windowed_df = windowed_df.sort_values(by='Date')

  df_as_np = windowed_df.to_numpy()

  dates = df_as_np[:,0]

  X = df_as_np[:,2:]

  X = X.reshape((len(dates), X.shape[1],1))

  Y = df_as_np[:,1]

  return dates, X.astype(np.float32), Y.astype(np.float32)




cryptoes = [
    {'symbol':'TRXIRT'},
    {'symbol':'ADAIRT'},
    {'symbol':'LINKIRT'},
    {'symbol':'EOSIRT'},
    {'symbol':'ETHIRT'},
    {'symbol':'XRPIRT'},
    {'symbol':'DOTIRT'},
    {'symbol':'UNIIRT'},
    {'symbol':'SANDIRT'}
]



TELEGRAM_BOT_TOKEN = '5474558689:AAEqdTKZdqLdw10P7BZhlo9lTN89AxJLQv4'
TELEGRAM_CHAT_ID = '-1001639881360'

bot = telegram.Bot(token=TELEGRAM_BOT_TOKEN)




for crypto in cryptoes:
  model = load_model(f'models/{crypto["symbol"]}.h5')
  crypto['model'] = model



# get the start time
start_time = datetime.datetime.now(tz=pytz.timezone('Asia/Tehran'))
start_time = start_time - datetime.timedelta(hours=1,minutes=30)

DAYS_TO_RUN = 1
## Consider that we are running this for a day.
end_time = start_time + datetime.timedelta(days = DAYS_TO_RUN+1)

times = 0



start_date = '2022-07-27'
end_date = '2022-08-12'
for index, crypto in enumerate(cryptoes):
  m_df = get_csv(crypto['symbol'], start_date, end_date,'60')
  crypto['min'] = m_df.Close.min()
  crypto['max'] = m_df.Close.max()
  print(f'{crypto["symbol"]} : {crypto["min"]} - {crypto["max"]}')

  


while(datetime.datetime.now(tz=pytz.timezone('Asia/Tehran')) < end_time):
  for index, crypto in enumerate(cryptoes):
    #for crypto in cryptoes:
    symbol = crypto['symbol']
    print(symbol)
    ## Getting the data
    now_time = datetime.datetime.now(tz=pytz.timezone('Asia/Tehran'))
    if times == 0:
      new_df = get_csv_datetime(symbol,start_time, now_time ,'5')
    else:
      new_df = get_csv_datetime(symbol,now_time - datetime.timedelta(minutes=11), now_time ,'5')
    # Normalizing the data
    new_df['Date'] = pd.to_datetime(new_df['Date'])
    new_df['Close'] = new_df['Close'].apply(lambda x: str(x).replace(',',''))
    new_df['Close'] = pd.to_numeric(new_df['Close'],errors='coerce')
    min = crypto['min']
    max = crypto['max']
    new_df.Close = (new_df.Close - min) / (max - min)

    if len(new_df) < 4:

      crypto['df'].loc[-1] = new_df.loc[0]
      crypto['df'].index = crypto['df'].index + 1
      crypto['df'] = crypto['df'].sort_index()
    else:
      crypto['df'] = new_df
      print(new_df)
    ## Windowing the df
    windowed_df = window_dataframe(crypto['df'],'1999-1-1','2090-1-1',4)
    dates, X,y = windowed_df_to_date_X_y(windowed_df)
    crypto['dates'] = dates
    crypto['X'] = X
    crypto['y'] = y

  
  fig = plt.figure(figsize=(26,12))
  fig.suptitle(f'{now_time.hour}:{now_time.minute}')
  axes = fig.subplots(nrows=3,ncols=3)
  for index, crypto in enumerate(cryptoes):
      ax = axes[int(index/3)][index%3]
      y_preds = crypto['model'].predict(crypto['X']).flatten()
      idx = np.argwhere(np.diff(np.sign(crypto['y'] - y_preds))).flatten()
      #idx = list(map(lambda x:x+1, idx))
      ax.set_title(crypto['symbol'])

      min = crypto['min']
      max = crypto['max']
      
      signals = []
      for id in idx:
        price = crypto['y'][id] * (max-min) + min
        time_str = f'{crypto["dates"][id].hour}:{crypto["dates"][id].minute}' 
        if id+1 < len(crypto['y']):
          if crypto['y'][id] < y_preds[id]:
            signals.append({'type':'sell','time':time_str, 'price':price,'idx':id})
          elif crypto['y'][id] > y_preds[id]:
            signals.append({'type':'buy','time':time_str, 'price':price,'idx':id})
          else:
            if crypto['y'][id] < crypto['y'][id+1]:
              signals.append({'type':'sell','time':time_str, 'price':price,'idx':id})
            elif crypto['y'][id] > crypto['y'][id+1]:
              signals.append({'type':'buy','time':time_str, 'price':price, 'idx':id})
      
      if times == 0:
        crypto['signals'] = signals
        message = f'lost signals {crypto["symbol"]}: {signals}'  
        print(message)
        bot.send_message(chat_id=TELEGRAM_CHAT_ID, text=message)
      else:
        len_new =  len(signals) - len(crypto['signals'])
        if len_new > 0:
          new_signals = signals[(-1) * len_new:]
          for signal in new_signals:
            signal['time'] = f'{crypto["dates"][-1].hour}:{crypto["dates"][-1].minute}' 
            signal['price'] = crypto['y'][-1] * (max-min) + min
          message = f'new signals {crypto["symbol"]}: {new_signals}'
          print(message)
          ##TODO:: send message telegram
          bot.send_message(chat_id=TELEGRAM_CHAT_ID, text=message)
        crypto['signals'] = signals
      

      ax.plot(crypto['dates'], y_preds)

      ax.plot(crypto['dates'],crypto['y'])
      sells = [signal['idx']+1 for signal in list(filter(lambda x:x['type'] == 'sell', signals))]
      buys = [signal['idx']+1 for signal in list(filter(lambda x:x['type'] == 'buy', signals))]

      ax.plot(dates[sells], crypto['y'][sells], 'ro' )
      ax.plot(dates[buys], crypto['y'][buys], 'go' )
      ax.legend(['Predicted Price','Actual Price','Sell Points','Buy Points'])
  
  fig.savefig('chart-5.jpg')
  bot.send_photo(chat_id=TELEGRAM_CHAT_ID, photo=open('chart-5.jpg', 'rb'))

  times+=1
  plt.show()
  print('----------------------------------------------------------------------------------------------------------------------------------------------')
  for crypto in cryptoes:
    print(f'signals {crypto["symbol"]}: {crypto["signals"]}')
  time.sleep(300)



