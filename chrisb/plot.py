import pandas as pd
import matplotlib.pyplot as plt
from datetime import datetime

fld = pd.read_csv('experiments/dmc_result/coup_dmc_prob/fields.csv')
df = pd.read_csv('experiments/dmc_result/coup_dmc_prob/logs.csv', comment='#')
df.set_axis(fld.columns, axis=1, inplace=True)
group_size = 10000
roll = pd.DataFrame({
  'return': df.mean_episode_return_0.rolling(group_size).mean(),
  'loss': df.loss_0.rolling(group_size, min_periods=1).mean(),
  'time': [datetime.fromtimestamp(t) for t in df._time.rolling(group_size, min_periods=1).min()]
})
# Focus on most recent part of the data
roll = roll.tail(30000)
plt.plot(roll.time, roll.loss)
plt.show()
