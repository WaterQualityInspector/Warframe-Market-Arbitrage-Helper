from datetime import datetime, timezone

from tqdm import tqdm

import numpy as np
import time
import pandas as pd
import requests
import re

def_wfm_url = "https://api.warframe.market/v1/items"  # default access link
rq_min_delay = 0.05
accounted_days = 30
item_exclusion_tags=['barrel','receiver','stock','chassis','neuroptics','systems','relic', 'emote','pouch', 'hilt','guard',
                     'string','lower_limb','upper_limb','key','handle','ornament','capsule',
                     'casing','weapon_','engine','handle','blade','grip','scene','carapace','cerebrum','head',
                     'motor','blade','barrels','link','receivers','prime_blueprint','chassis_blueprint', 'neuroptics_blueprint',
                     'systems_blueprint', 'prime_harness','prime_wings','prime_chain', 'vandal_blueprint', 'wraith_blueprint',
                     'prime_disc','left_gauntlet','right_gauntlet']


# 0 for stuff without mod ranks (e.g. frame chassis)
# Whole row being 0 if the trade history is not available (no trades are available. e.g. adept_surge)


def GetAllItemsList(wfm_url=def_wfm_url):  # Gets the list of ALL items that are available on the WFM
    r = requests.get(wfm_url, timeout=2)
    #print(f"Items List HTML Response: {r.status_code}\n")

    exclusion_pattern = r'(?:'+'|'.join(item_exclusion_tags) + r')\b'

    items_list_json = r.json()['payload']['items']
    items_list_df = pd.DataFrame(items_list_json)
    items_list_df = items_list_df[~items_list_df['url_name'].str.contains(exclusion_pattern, case=False, regex=True)]
    items_list_df.index = items_list_df.url_name
    items_list_df = items_list_df.drop(['url_name'], axis=1)
    items_list_df = items_list_df.fillna({'vaulted': False})
    items_list_df = items_list_df.sort_index()
    return items_list_df


def GetItemInformation(item_url_name, wfm_url=def_wfm_url):  # Gets all the information of a prompted item
    r = requests.get(f"{wfm_url}/{item_url_name}")
    #print(f"Item Info HTML Response: {r.status_code}\n")

    item_info_df = pd.DataFrame(r.json()['payload']['item']['items_in_set'])
    item_info_df.index = item_info_df.url_name
    item_info_df = item_info_df.drop(['url_name'], axis=1)
    item_info_df = item_info_df.iloc[:, :-13]
    new_order = ['id', 'rarity', 'tags', 'mod_max_rank', 'trading_tax', 'icon_format', 'icon', 'thumb', 'sub_icon']
    item_info_df = item_info_df[new_order]

    # print(item_info_df)

    return item_info_df


def GetItemOrderInformation(item_url_name, row_limit=8,
                            wfm_url=def_wfm_url):  # Gets all the order information of a prompted item
    r = requests.get(f"{wfm_url}/{item_url_name}/orders")

    #print(f"Item Order HTML Response: {r.status_code}\n")

    item_orders_df = pd.DataFrame(r.json()['payload']['orders'])

    # extract userdata
    normalized_user_data = pd.json_normalize(item_orders_df['user'])

    item_orders_df = pd.concat([item_orders_df, normalized_user_data], axis=1)
    item_orders_df = item_orders_df.drop(
        ['visible', 'creation_date', 'user', 'avatar', 'locale', 'region', 'platform', 'id', 'quantity'], axis=1)

    item_orders_df['status'] = pd.Categorical(item_orders_df['status'], categories=['ingame', 'online', 'offline'],
                                              ordered=True)
    # convert last update/seen dates to days since so.
    item_orders_df['last_update'] = pd.to_datetime(item_orders_df['last_update'])
    item_orders_df['last_update'] = (datetime.now(timezone.utc) - item_orders_df['last_update']).dt.days

    item_orders_df['last_seen'] = pd.to_datetime(item_orders_df['last_seen'])
    item_orders_df['last_seen'] = (datetime.now(timezone.utc) - item_orders_df['last_seen']).dt.days

    item_orders_df = item_orders_df.rename(
        columns={'last_update': 'updated_days_ago', 'last_seen': 'seen_days_ago'})

    # process mod rank to only select top values
    item_orders_df = item_orders_df[(item_orders_df['mod_rank'] == item_orders_df['mod_rank'].max())]

    # process sell data
    item_sells_df = item_orders_df[(item_orders_df['order_type'] == "sell")].sort_values(['status', 'platinum'],
                                                                                         ascending=[True, True])
    item_sells_df = item_sells_df.reset_index(drop=True)
    item_sells_df = item_sells_df.iloc[:row_limit]

    # process buy data
    item_buys_df = item_orders_df[(item_orders_df['order_type'] == "buy")].sort_values(['status', 'platinum'],
                                                                                       ascending=[True, False])
    item_buys_df = item_buys_df.reset_index(drop=True)
    item_buys_df = item_buys_df.iloc[:row_limit]

    # process listed spread
    current_spread = item_sells_df['platinum'].min() - item_buys_df['platinum'].max()

    return item_orders_df, item_sells_df, item_buys_df, current_spread


def CreateEmptyHistory(): # this is used in case the item has invalid data, stuff like no transaction within date range, and etc.
    # please match the backup_columns with whatever the columns should be if the item is valid.
    error_columns = (
        'datetime', 'volume', 'min_price', 'max_price', 'open_price', 'closed_price', 'avg_price', 'wa_price',
        'median',
        'moving_avg', 'donch_top', 'donch_bot', 'id')

    error_datetime = datetime(1900, 1, 1, 0, 0)

    # create dataframe with 0 as the error code

    broken_item_df = pd.DataFrame(np.zeros((1, len(error_columns))), columns=error_columns)

    broken_item_df['datetime'] = error_datetime
    broken_item_df['datetime'] = pd.to_datetime(broken_item_df['datetime'], utc=True)

    return broken_item_df


def CheckTradeHistory(item_url_name, wfm_url=def_wfm_url):
    r = requests.get(f"{wfm_url}/{item_url_name}/statistics")
    item_trade_hist_df = pd.DataFrame(r.json()['payload']['statistics_closed']['90days'])
    #print(f"\nItem Market Statistics HTML Response: {r.status_code}\n")

    if item_trade_hist_df.empty:
        print(f"\n{item_url_name}'s market data is not found!")

        return CreateEmptyHistory()
    else:
        return item_trade_hist_df


def GetItemMarketStatistics(item_url_name, wfm_url=def_wfm_url,
                            days=accounted_days):  # Gets all the market statistics of a prompted item

    item_statistics_closed_df = CheckTradeHistory(item_url_name, wfm_url)

    # sort and trim by days wanted, take out nulls

    item_statistics_closed_df = item_statistics_closed_df.sort_values(['datetime'], ascending=False).reset_index(
        drop=True)
    item_statistics_closed_df = item_statistics_closed_df.fillna(method="ffill")

    # clean data
    item_statistics_closed_df = item_statistics_closed_df.drop(['id'], axis=1)

    # convert datettime to time format so we can process days since
    item_statistics_closed_df['datetime'] = pd.to_datetime(item_statistics_closed_df['datetime'])

    # this is here for tz-naive and tz-aware -- i'm not sure what this means but it fixes it..
    current_utc_datetime = datetime.now(timezone.utc)
    item_statistics_closed_df['datetime'] = (current_utc_datetime - item_statistics_closed_df['datetime']).dt.days

    # rename datetime to days ago
    item_statistics_closed_df = item_statistics_closed_df.rename(columns={'datetime': 'days_ago'})

    # cutoff day threshold
    if ~(item_statistics_closed_df['days_ago'] <= days).any(): # check if there are no entries any less than the days
        # of interest
        print(f"\nNo trade data were found for {item_url_name} within {days} days.")
        item_statistics_closed_df = CreateEmptyHistory()
    else:
        item_statistics_closed_df = item_statistics_closed_df.loc[item_statistics_closed_df['days_ago']<= days]

    # filter out top mod rank only.
    if 'mod_rank' in item_statistics_closed_df.columns:
        # if there is a mod rank variable
        item_statistics_closed_df = item_statistics_closed_df[
            (item_statistics_closed_df['mod_rank'] == item_statistics_closed_df['mod_rank'].max())]
    else:
        # if it's not available, then set to 0
        item_statistics_closed_df['mod_rank'] = 0

    item_statistics_closed_df['spread'] = item_statistics_closed_df['donch_top'] - item_statistics_closed_df[
        'donch_bot']

    return item_statistics_closed_df


def GetSummarizedItemMarketStatistics(item_url_name, wfm_url=def_wfm_url,
                                      days=accounted_days):  # Gets all the summarized stats of a prompted item

    item_statistics_closed_df = GetItemMarketStatistics(item_url_name, wfm_url, days)

    avg_volume = item_statistics_closed_df['volume'].mean().round(2)
    avg_spread = item_statistics_closed_df['spread'].mean().round(2)
    avg_median_price = item_statistics_closed_df['median'].mean().round(2)
    avg_min_price = item_statistics_closed_df['min_price'].min().round(2)
    avg_typical_price = item_statistics_closed_df.eval(
        "(min_price + max_price + closed_price)/3").mean().round(2)

    # mod_rank number is filtered out anyway, so just get the first one. I just need *a* number.
    mod_rank = item_statistics_closed_df['mod_rank'].iloc[0]

    return avg_volume, avg_spread, avg_median_price, avg_min_price, avg_typical_price, mod_rank


def GetItemMarketStatisticsWithDelay(item_url_name, wfm_url=def_wfm_url, days=accounted_days,
                                     request_delay=rq_min_delay):
    # fetch market statistics for all listed items that are available
    item_statistics_closed_df = GetItemMarketStatistics(item_url_name, wfm_url, days)

    # introduce delay so we don't overwhelm the api
    time.sleep(request_delay)

    return item_statistics_closed_df


def GetSummarizedItemMarketStatisticsWithDelay(item_url_name, wfm_url=def_wfm_url, days=accounted_days,
                                               request_delay=rq_min_delay):
    # fetch market statistics for all listed items that are available
    result = GetSummarizedItemMarketStatistics(
        item_url_name,
        wfm_url, days)

    # introduce delay so we don't overwhelm the api
    time.sleep(request_delay)

    return result


def GetAllSummarizedMarketStatistics(wfm_url=def_wfm_url, days=accounted_days):
    tqdm.pandas()

    item_list_df = GetAllItemsList(wfm_url)

    numeric_columns = ['avg_volume', 'avg_spread', 'avg_median_price', 'avg_min_price', 'avg_typical_price', 'mod_rank']

    summarized_data_df = pd.DataFrame(columns=['url_name', 'result'])
    summarized_data_df['url_name'] = item_list_df.index

    #summarized_data_df = summarized_data_df[:100] #if you want to do this piece wise :)

    # for some reason that I don't understand, the apply function returns a tuple no matter what I do, so I will just
    # take it as it is and just unpack it promptly.
    summarized_data_df['result'] = summarized_data_df['url_name'].progress_apply(
        GetSummarizedItemMarketStatisticsWithDelay)
    summarized_data_df[numeric_columns] = summarized_data_df['result'].apply(pd.Series)
    summarized_data_df.index = summarized_data_df['url_name']
    summarized_data_df = summarized_data_df.drop(['result', 'url_name'], axis=1)

    print(summarized_data_df)

    summarized_data_df.to_csv("market_summary.csv", mode="w")

    return summarized_data_df
