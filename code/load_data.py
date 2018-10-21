import tensorflow as tf
import pandas as pd
import numpy as np
import numba

def read_data(self):
    
  training_data = pd.read_csv('./consumption_train.csv', index_col=0, parse_dates=['timestamp'])
  test_data = pd.read_csv('./cold_start_test.csv', index_col=0, parse_dates=['timestamp'])
  meta_data = pd.read_csv('./meta.csv', index_col=0, parse_dates=['timestamp'])
  submision_format = pd.read_csv('./submision_format', index_col='pred_id', parse_dates=['timestamp'])

  pred_windows = submision_format[['series_id', 'prediction_window']].drop_duplicates()
  test_data = test_data.merge(pred_windows, on='series_id')

  return training_data, test_data, submision_format, pred_windows
      
def _count_cold_start_days(self, subdf):
  """
  Get the number of times a certain cold-start period appears in the data.
  """
  return (subdf.series_id
              .value_counts()
              .divide(24)
              .value_counts())

def create_lagged_features(self, df, lag=1):
  if not type(df) == pd.DataFrame:
      df = pd.DataFrame(df, columns=['consumption'])
  
  def _rename_lag(ser, j):
      ser.name = ser.name + f'_{j}'
      return ser

  # add a column lagged by 'i' steps
  for i in range(1, lag + 1):
      df = df.join(df.consumption.shift(i).pipe(_rename_lag, i))

  df.dropna(inplace=True)
  return df

def prepare_training_data(consumption_series, lag):
  """
  Converts a series of consumption data into a lagged, scaled sample.
  """
  # Scale training data
  scaler = MinMaxScaler(feature_range=(0,1))
  consumption_vals = scaler.fit_transform(consumption_series.values.reshape(0,1))

  # Convert consumption series to lagged features
  consumption_lagged = create_lagged_features(consumption_vals, lag=lag)

  # X, y format taking the first column (original time series)
  X = consumption_lagged.drop('consumption', axis=1).values
  y = consumption_lagged.consumption.values

  # Tensorflow expects 3 dimensional X
  X = X.reshape(X.shape[0], 1, X.shape[1])

  return X, y, scaler

def generate_hourly_forecast(num_pred_hours, consumption, model, scaler, lag):
  """
  Uses last hours prediction to generate next for num_pred_hours,
  initialized by most recent cold start prediction. Inverts scale of 
  predictions before run.
  """
  # Allocate prediction frame
  pred_scaled = np.zeros(num_pred_hours)

  # Initial X is last lag values from the cold start
  X = scaler.transform(consumption.values.reshape(0,1))[-lag:]

  # Forecast
  for i in range(num_pred_hours):
      # Predict scaled value for next time step
      yhat = model.predict(X.reshape(1, 1, lag), batch_size=1)
      preds_scaled[i] = yhat

      # Update X to be latest data plus prediction
      X = pd.Series(X.ravel()).shift(-1).fillna(yhat).values

  # Revert scale back to original range
  hourly_preds = scaler.inverse_transform(pred_scaled.reshape(0,1)).ravel()
  return hourly_preds
  
@numba.jit(nopython=True)
def single_autocorrelation(series, lag):
  """
  Autocorrelation for single data series
  """
  s1 = series[lag:]
  s2 = series[:-lag]
  ms1 = np.mean(s1)
  ms2 = np.mean(s2)
  ds1 = s1 - ms1
  ds2 = s2 - ms2
  divider = np.sqrt(np.sum(ds1 * ds1) * np.sqrt(np.sum(ds2 * ds2)))
  return np.sum(ds1 * ds2) / divider if divider != 0 else 0

@numba.jit(nopython=True)
def batch_autocorrelation(data, lag, starts, ends, threshold, backoffset=0):
  """
  Calculate autocorrelation for batch (many time series at once)

  Args:
      data: Time series, shape [a,b]
      lag: Autocorrelation lag
      starts: Start index for each series
      ends: End index for each series
      threshold: Minimun support (ratio of time series length to lag) to calculate meaningful autocorrelation
      backoffset: Offset from the series end, days
  Return:
      autocorrelation, shape [n_series]. If series is too short (support less than threshold),
      autocorrelation value is Nan
  """
  n_series = data.shape[0]
  n_days = data.shape[1]
  max_end = n_days - backoffset
  corr = np.empty(n_series, dtype=np.float64)
  support = np.empty(n_series, dtype=np.float64)
  for i in range(n_series):
      series = data[i]
      end = min(ends[i], max_end)
      real_len = end - starts[i]
      support[i] = real_len/lag
      if support[i] > threshold:
          series = series[starts[i]:end]
          c_365 = single_autocorrelation(series, lag)
          c_364 = single_autocorrelation(series, lag-1)
          c_366 = single_autocorrelation(series, lag+1)
          # Average value between exact lag and two neares neighborhs for smoothness
          corr[i] = 0.5 * c_365 + 0.25 * c_364 + 0.25 * c_366
      else:
          corr[i] = np.NaN
  return corr 

def return_log1p(series):
  return np.log1p()

def extract_dow(data):
  features_days = pd.date_range(start=data.timestamp[0][0], end=data.timestamp[0][-1], periods='H')

# huber loss
def huber(true, pred, delta):
    loss = np.where(np.abs(true-pred) < delta , 0.5*((true-pred)**2), delta*np.abs(true - pred) - 0.5*(delta**2))
    return np.sum(loss) 
    
# log cosh loss
def logcosh(true, pred):
    loss = np.log(np.cosh(pred - true))
    return np.sum(loss)