# @title Define all functions you will use
def initialisation(df):
    df["time_step"] = pd.to_datetime(df["time_step"])
    # Segmenting periods
    df["month"] = df.apply(lambda x: x["time_step"].month, axis=1)
    df["day"] = df.apply(lambda x: x["time_step"].day, axis=1)
    df["hour"] = df.apply(lambda x: x["time_step"].hour, axis=1)

    # Minuted could be especially valuable
    df["minute"] = df.apply(lambda x: x["time_step"].minute, axis=1)
    df = df.drop(columns={"Unnamed: 9"})
    df.head()
    return df


def fill_weather(data):
    data_minute_0 = data[data["minute"] == 0]
    ###
    # Identifying the NaNs we have to fix
    humidity_nans = data_minute_0[data_minute_0["humidity"].isna()]
    humidex_nans = data_minute_0[data_minute_0["humidex"].isna()]
    windchill_nans = data_minute_0[data_minute_0["windchill"].isna()]
    wind_nans = data_minute_0[data_minute_0["wind"].isna()]
    visibility_nans = data_minute_0[data_minute_0["visibility"].isna()]
    pressure_nans = data_minute_0[data_minute_0["pressure"].isna()]
    temperature_nans = data_minute_0[data_minute_0["temperature"].isna()]
    # Introducing a table with the mean of each weather variable within a given month
    monthly_weather = data.groupby("month").mean().loc[:, "consumption":"pressure"].reset_index()

    # Filling the nans using monthly mean for minute 0 by mapping dictionaries within a loop
    weather_nans = [humidity_nans, humidex_nans, windchill_nans, wind_nans, visibility_nans, pressure_nans,
                    temperature_nans]
    weather_columns = ["humidity", "humidex", "windchill", "wind", "visibility", "pressure", "temperature"]
    keys = monthly_weather["month"].tolist()  # need to pass keys and values to list to make a dictionary
    for i in np.arange(len(weather_nans)):
        values = monthly_weather[weather_columns[i]].tolist()
        my_dict = dict(zip(keys, values))
        weather_nans[i][weather_columns[i]] = weather_nans[i]["month"].map(my_dict)
        data_minute_0[weather_columns[i]][data_minute_0[weather_columns[i]].isna()] = weather_nans[i][
            weather_columns[i]]

    data[data["minute"] == 0] = data_minute_0
    ###

    ###
    # Taking care of minute 1 for the first month given that there is no minute 0
    if np.isnan(data.iloc[0, 1]):
        data.iloc[0, 1:8] = monthly_weather.iloc[0, 1:9]

    # Creating a weather dataframe for forward fill
    data_weather = data.loc[:, "visibility":"pressure"]
    data_weather = data_weather.fillna(method='ffill')
    # Finally replacing the incomplete weather data by our filled table
    data.loc[:, "visibility":"pressure"] = data_weather

    ###

    ###
    # Creating a daytime feature
    day_boundaries = [7, 20]  # defining daytime hours, could segment more precisely
    data["daytime"] = data.apply(lambda x: (x["hour"] in range(day_boundaries[0], day_boundaries[1])) * 1, axis=1)
    ###
    return data


def forward_fill_consumption(data):
    data['consumption'] = data['consumption'].fillna(method='ffill')
    # data['washing_machine'] = data['washing_machine'].fillna(method='ffill')
    # data['fridge_freezer'] = data['fridge_freezer'].fillna(method='ffill')
    # ata['TV'] = data['TV'].fillna(method='ffill')
    # data['kettle'] = data['kettle'].fillna(method='ffill')
    return data


def metric_nilm(dataframe_y_true, dataframe_y_pred):
    score = 0.0
    test = np.array(dataframe_y_true['washing_machine'])
    pred = np.array(dataframe_y_pred['washing_machine'])
    score += mt.sqrt(sum((pred - test) ** 2) / len(test)) * 5.55
    test = np.array(dataframe_y_true['fridge_freezer'])
    pred = np.array(dataframe_y_pred['fridge_freezer'])
    score += mt.sqrt(sum((pred - test) ** 2) / len(test)) * 49.79
    test = np.array(dataframe_y_true['TV'])
    pred = np.array(dataframe_y_pred['TV'])
    score += mt.sqrt(sum((pred - test) ** 2) / len(test)) * 14.57
    test = np.array(dataframe_y_true['kettle'])
    pred = np.array(dataframe_y_pred['kettle'])
    score += mt.sqrt(sum((pred - test) ** 2) / len(test)) * 4.95
    score /= 74.86
    return score


def create_y_predictions(x_out, best_model):
    Y_test_temp = best_model.predict(x_out)
    df = pd.DataFrame(Y_test_temp)
    Y_test = pd.DataFrame(x_out_temp["time_step"])
    Y_test["washing_machine"] = df.iloc[:, 0]
    Y_test["fridge_freezer"] = df.iloc[:, 1]
    Y_test["TV"] = df.iloc[:, 2]
    Y_test["kettle"] = df.iloc[:, 3]
    Y_test.to_csv('y_test_no_activation.csv', header=True)
    print(Y_test)
    files.download("y_test_no_activation_500est.csv")
    return Y_test

# Number of days gone by since 1970, catches long term trends in consumption
def add_ndays(data):
  data['n_days'] = data.loc[:,'time_step'].apply(lambda date: (date - pd.to_datetime("1970-01-01")).days)
  return data

# Express hours with sine and cosine
def add_sine_cosine_hour(data):
  hours_in_day = 24
  data['sin_hour'] = np.sin(2*np.pi*data.hour/hours_in_day)
  data['cos_hour'] = np.cos(2*np.pi*data.hour/hours_in_day)
  return data

# Create a table with average hourly consumption per hour
def periodical_avg_consumption_table(data, equipment,period): # period is a string
  avg_hourly_consumption = pd.DataFrame(data.groupby(period).mean().loc[:,equipment])
  avg_hourly_consumption = avg_hourly_consumption.reset_index()
  avg_hourly_consumption.columns = [period,"avg_"+period+"ly_"+equipment]
  return avg_hourly_consumption
# Add average consumption per hour as a feature to your dataframe
def add_periodical_avg_consumption_feature(data, periodical_avg_consumption_table):
  data = data.reset_index().merge(periodical_avg_consumption_table, how="left").set_index('index')
  return data

# Indicate whether the observation occurs during a holiday or not
def add_holiday(data):
  france_holidays = holidays.France()
  data["is_holiday"] = data["time_step"].apply(lambda x: x in france_holidays)*1
  return data

#  Washing machine works more during off-peak. Chose 2 am as lower boundary because consumption is low from 2 to 7 am
def add_peak_hours(data):
  peak_boundaries = [2,21]
  data["peak_hours"] = data.apply(lambda x: (x["hour"]in range(peak_boundaries[0],peak_boundaries[1]))*1, axis=1)
  return data

## A few examples of features based on consumption.
# Past and future lags. We also experimented with log and exponential lags, but they proved less useful.
def add_lag(data, lag):
  data["conso - "+str(lag)] = data["consumption"].shift(lag).fillna(method="bfill")
  data["conso + "+str(lag)] = data["consumption"].shift(-lag).fillna(method="ffill")
  return data

# Moving consumption based on the previous n observations (including the one of the current row)
def conso_moving(data, type, n_minutes): # type is a string, in ''
  data['moving_conso_'+type+'_'+str(n_minutes)]= 0 # initialization
  last_row = data.shape[0] # to get until the last row
  # Choose computation type
  if type == 'max':
    data["moving_conso_"+type+'_'+str(n_minutes)] = data["consumption"].rolling(n_minutes).max()
    computation = max
  if type == 'mean':
    data["moving_conso_"+type+'_'+str(n_minutes)] = data["consumption"].rolling(n_minutes).mean()
    computation = mean
  if type == 'min':
    data["moving_conso_"+type+'_'+str(n_minutes)] = data["consumption"].rolling(n_minutes).min()
    computation = min
  if type == "median":
    data["moving_conso_"+type+'_'+str(n_minutes)] = data["consumption"].rolling(n_minutes).median()
    computation = median
  if type == 'std':
    data["moving_conso_"+type+'_'+str(n_minutes)] = data["consumption"].rolling(n_minutes).std()
    computation = std
  # Compute separately the n_minutes first cells. For other functions we used fill backward and forward instead.
  for i in range(n_minutes):
    data['moving_conso_'+type+'_'+str(n_minutes)][i] = computation(data['consumption'][:n_minutes])

# Adding the moving mean, but with a window in both past and future:
def add_rolling_mean(data,window):
  n = int(np.floor(int(window)/2))
  data["rolling_mean_"+str(window)] = ((data["consumption"].rolling(n).sum() + data["consumption"].rolling(n).sum().shift(-n))/window).fillna(method="bfill").fillna(method="ffill")
  return data

# Consumption normalised for the moving mean
def add_conso_minus_mvg_conso(data,window):
  data["conso - mvg_avg_"+str(window)] = data["consumption"] - data["moving_conso_mean_"+str(window)]
  return data

# Rolling std in consumption for both past and future
def add_rolling_std(data,window):
  data["rolling_std_past_"+str(window)] = data["consumption"].rolling(window).std().fillna(method="bfill")
  data["rolling_std_future_"+str(window)] = data["consumption"].rolling(window).std().shift(-window).fillna(method="ffill")
  return data

# Expanding consumption: based on the previous n observations (including the one of the current row)
def conso_expanding(data, n_minutes):
  data["expanding_conso_mean_"+str(n_minutes)] = data["consumption"].expanding(n_minutes).mean()
  data["expanding_conso_max_"+str(n_minutes)] = data["consumption"].expanding(n_minutes).max()
  data["expanding_conso_min_"+str(n_minutes)] = data["consumption"].expanding(n_minutes).min()
  data["expanding_conso_median_"+str(n_minutes)] = data["consumption"].expanding(n_minutes).median()
  data["expanding_conso_std_"+str(n_minutes)] = data["consumption"].expanding(n_minutes).std()
  data.fillna(method="bfill").fillna(method="ffill") # we forward and backward fill the rest of the values to save computing time

# OneHotEncoding of hour, month, day
def one_hot_encode_time_variables(data):
  encoded_variables=["minute", "hour", "day", "day_of_week","month"]
  data_encoded = data.copy()
  for variable in encoded_variables:
    data_encoded = data_encoded.join(pd.get_dummies(data[variable], prefix=variable))
  return data_encoded