# %%
import pandas as pd
import matplotlib.pyplot as plt
from statsmodels.tsa.seasonal import seasonal_decompose



df = pd.read_excel('./datasets/mecaniqa_dataset.xlsx')


df['Data'] = pd.to_datetime(df['Data'])
df = df.sort_values('Data').set_index('Data')


series = df.select_dtypes('number').iloc[:, 0].astype(float)


model = 'additive'
periodo = 7

if series.isna().any():
	series = series.interpolate(method='time')
	series = series.ffill().bfill()
	series = series.dropna()

result = seasonal_decompose(series, model=model, period=periodo)


fig, axes = plt.subplots(4, 1, figsize=(10, 8), sharex=True)
axes[0].plot(series)
axes[0].set_title('Observed')
axes[1].plot(result.trend)
axes[1].set_title('Tendência')
axes[2].plot(result.seasonal)
axes[2].set_title('Sazonalidade')
axes[3].plot(result.resid)
axes[3].set_title('Ruído')
plt.tight_layout()
plt.show()