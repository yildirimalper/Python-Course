# SCRAPING TWEETS

# In order to avoid ClientPayloadError or RuntimeError, tweets were scraped through dividing them based on time range and hashtags.
# Below, there is a sample code only for one hashtag with a specific time range.

import twint

c = twint.Config()
c.Search = "#BogaziciDireniyor"
c.Limit = 100000
c.Store_csv = True
c.Output = "02BogaziciDireniyor.csv"
c.Since = "2021-02-01"
c.Until = "2021-02-28"
twint.run.Search(c)

# Alternatively, "config.Username" can be used to scrape tweets based on username.
# Storing options of Twint are csv, json, and SQLite.

# ----------------------------------------------------------------------------

# CONCATENATION AND PROCESSING OF TWEETS

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from datetime import datetime as dt
import re

# Tweeted in January

# importing .csv files
j_diren = pd.read_csv("../dataset/01raw_BogaziciDireniyor.csv")
j_serbest = pd.read_csv("../dataset/01raw_ArkadaslarimiziSerbestBirakin.csv")
j_susma = pd.read_csv("../dataset/01raw_BogaziciSusmayacak.csv")
j_lgbti = pd.read_csv("../dataset/01raw_LGBTIHaklariInsanHaklari.csv")
j_kabul = pd.read_csv("../dataset/01raw_KabulEtmiyoruzVazgecmiyoruz.csv")

# concatenating separate files into a dataframe while adjusting index
january_list = [j_diren, j_serbest, j_susma, j_lgbti, j_kabul]
january = pd.concat(january_list).reset_index(drop=True)
january.info()

january = january[["id", "date", "user_id", "username", "tweet", "language", "hashtags", "retweet"]]
january = january[january['language']=="tr"] # to exclude non-Turkish tweets from dataframe
january = january.drop_duplicates(subset=["tweet"]) # to exclude duplicates from dataframe
january['date'] = pd.to_datetime(january['date'], errors='coerce') # to convert types of "date" column to datetime object
january.info() # exploration of dataframe

# Tweeted in February

serbest = pd.read_csv("../dataset/02raw_ArkadaslarimiziSerbestBirakin.csv", dtype='unicode', low_memory=False)
asagi1 = pd.read_csv("../dataset/02raw_AsagiBakmayacagiz.csv", dtype='unicode', low_memory=False)
asagi2 = pd.read_csv("../dataset/02raw_AsagiBakmayacagiz_2.csv", dtype='unicode', low_memory=False)
diren = pd.read_csv("../dataset/02raw_BogaziciDireniyor.csv", dtype='unicode', low_memory=False)
susma1 = pd.read_csv("../dataset/02raw_BogaziciSusmayacak.csv", dtype='unicode', low_memory=False)
susma2 = pd.read_csv("../dataset/02raw_BogaziciSusmayacak_2.csv", dtype='unicode', low_memory=False)
susma3 = pd.read_csv("../dataset/02raw_BogaziciSusmayacak_3.csv", dtype='unicode', low_memory=False)
sonra = pd.read_csv("../dataset/02raw_BundanSonrasiHepimizde.csv", dtype='unicode', low_memory=False)
kabul = pd.read_csv("../dataset/02raw_KabulEtmiyoruzVazgecmiyoruz.csv", dtype='unicode', low_memory=False)
lgbti1 = pd.read_csv("../dataset/02raw_LGBTIHaklariInsanHaklari.csv", dtype='unicode', low_memory=False)
lgbti2 = pd.read_csv("../dataset/02raw_LGBTIHaklariInsanHaklari_2.csv", dtype='unicode', low_memory=False)
lgbti3 = pd.read_csv("../dataset/02raw_LGBTIHaklariInsanHaklari_3.csv", dtype='unicode', low_memory=False)

february_list = [serbest, asagi1, asagi2, diren, susma1, susma2, susma3, sonra, kabul, lgbti1, lgbti2, lgbti3]
february = pd.concat(february_list).reset_index(drop=True)
february.info()

february = february[["id", "date", "user_id", "username", "tweet", "language", "hashtags", "retweet"]]
february = february[february['language']=="tr"]
february = february.drop_duplicates(subset=["tweet"])
february['date'] = pd.to_datetime(february['date'], errors='coerce')
february.info()

# Tweeted in March

serbest = pd.read_csv("../dataset/03raw_ArkadaslarimiziSerbestBirakin.csv")
asagi = pd.read_csv("../dataset/03raw_AsagiBakmayacagiz.csv")
diren = pd.read_csv("../dataset/03raw_BogaziciDireniyor.csv")
susma = pd.read_csv("../dataset/03raw_BogaziciSusmayacak.csv")
sonra = pd.read_csv("../dataset/03raw_BundanSonrasiHepimizde.csv")
kabul = pd.read_csv("../dataset/03raw_KabulEtmiyoruzVazgecmiyoruz.csv")
lgbti = pd.read_csv("../dataset/03raw_LGBTIHaklariInsanHaklari.csv")

march_list = [serbest, asagi, diren, susma, sonra, kabul, lgbti]
march = pd.concat(march_list).reset_index(drop=True)
march.info()

march = march[["id", "date", "user_id", "username", "tweet", "language", "hashtags", "retweet"]]
march = march[march['language']=="tr"]
march = march.drop_duplicates(subset=["tweet"])
march['date'] = pd.to_datetime(march['date'], errors='coerce')
march.info()

# Concatenation
# After completing those steps for each month, the tweets will merged into a dataframe and exported as csv file.

dataframes = [january, february, march]
all_tweets = pd.concat(dataframes).reset_index(drop=True)
all_tweets.info()

tweets_by_time = all_tweets.groupby(['date'])['tweet'].count()
type(tweets_by_time)

all_tweets.to_csv("../dataset/all_tweets.csv")

# Exploratory Visualization of Tweets in a Time-Series

fig, ax = plt.subplots(figsize=(10,6))
plt.plot(tweets_by_time, marker="o", linestyle=":", color="royalblue", alpha=0.7)
plt.style.use("ggplot")
ax.text(dt.date(2021, 2, 10), 75000, "The frequency distribution of 221,740 tweets.", fontsize=13, bbox={'facecolor': 'grey', 'alpha': 0.5, 'pad': 10})
ax.set_ylabel("Number of Tweets", fontsize=15)
ax.set_title("Time-Series Graph of Tweets", fontsize=20)
plt.setp(ax.get_xticklabels(), rotation = 45)
plt.savefig("../dataset/tweets_by_time.png")
plt.show()