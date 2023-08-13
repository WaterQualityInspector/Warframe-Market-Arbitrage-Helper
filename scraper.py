from datetime import datetime, timezone, timedelta
from tqdm import tqdm

import asyncio
import numpy as np
import pandas as pd
import requests
import os

jwt_key = "AZ"
key_header = {
    'Authorization': f"JWT {jwt_key}"
}

csv_filename = "market_summary.csv"
def_wfm_url = "https://api.warframe.market/v1/items"  # default access link
rq_min_delay = 1
update_threshold_days = 7  # threshold days for considering a full scan outdated
accounted_days = 30

item_exclusion_tags = ['barrel', 'receiver', 'stock', 'chassis', 'neuroptics', 'systems', 'relic', 'emote', 'pouch',
                       'hilt', 'guard',
                       'string', 'lower_limb', 'upper_limb', 'key', 'handle', 'ornament', 'capsule',
                       'casing', 'weapon_', 'engine', 'handle', 'blade', 'grip', 'scene', 'carapace', 'cerebrum',
                       'head',
                       'motor', 'blade', 'barrels', 'link', 'receivers', 'prime_blueprint', 'chassis_blueprint',
                       'neuroptics_blueprint',
                       'systems_blueprint', 'prime_harness', 'prime_wings', 'prime_chain', 'vandal_blueprint',
                       'wraith_blueprint',
                       'prime_disc', 'left_gauntlet', 'right_gauntlet']

mod_rank_error_code = 0

# used in GetItemMarketStatistics to cache column data from functioning items so that items with no
# trade data won't return an empty df.
columns_cache = None


# 0 for stuff without mod ranks (e.g. frame chassis)
# Whole row being 0 if the trade history is not available (no trades are available. e.g. adept_surge)
def CheckForMissingColumns(df, check_for_mod_rank=False, check_for_datetime=False, check_for_id=False):
    # process mod rank to only select top values
    if check_for_mod_rank:
        if 'mod_rank' in df.columns:
            df = df[(df['mod_rank'] == df['mod_rank'].max())]
        else:
            df['mod_rank'] = mod_rank_error_code

    # process datetime if it is missing
    if check_for_datetime:
        if 'datetime' in df.columns:
            df = df.sort_values(['datetime'], ascending=False).reset_index(drop=True)
        else:
            df['datetime'] = datetime(1900, 1, 1, 0, 0)
            df['datetime'] = pd.to_datetime(df['datetime'], utc=True)

    if check_for_id:
        if 'id' in df.columns:
            df = df.drop(['id'], axis=1)

    return df


def GetAllItemsList(wfm_url=def_wfm_url):  # Gets the list of ALL items that are available on the WFM
    r = requests.get(wfm_url, timeout=5)
    # print(f"Items List HTML Response: {r.status_code}\n")

    exclusion_pattern = r'(?:' + '|'.join(item_exclusion_tags) + r')\b'

    items_list_json = r.json()['payload']['items']
    items_list_df = pd.DataFrame(items_list_json)
    items_list_df = items_list_df[~items_list_df['url_name'].str.contains(exclusion_pattern, case=False, regex=True)]
    items_list_df.index = items_list_df.url_name
    items_list_df = items_list_df.drop(['url_name'], axis=1)
    items_list_df = items_list_df.fillna({'vaulted': False})
    items_list_df = items_list_df.sort_index()
    return items_list_df


def GetItemInformation(item_url_name, wfm_url=def_wfm_url):  # Gets all the information of a prompted item
    r = requests.get(f"{wfm_url}/{item_url_name}", headers=key_header)
    # print(f"Item Info HTML Response: {r.status_code}\n")

    item_info_df = pd.DataFrame(r.json()['payload']['item']['items_in_set'])
    if 'mod_max_rank' in item_info_df.columns:
        item_info_df = item_info_df.rename(columns={'mod_max_rank': 'mod_rank'})

    item_info_df = CheckForMissingColumns(item_info_df, check_for_mod_rank=True)
    item_info_df.index = item_info_df['url_name']

    columns_to_concat = ['mod_rank', 'icon', 'sub_icon', 'trading_tax', 'tags', 'thumb']
    clean_item_info_df = pd.DataFrame(index=item_info_df['url_name'])
    clean_item_info_df = pd.concat([clean_item_info_df, item_info_df[columns_to_concat]], axis=1)

    return clean_item_info_df


def GetItemOrderInformation(item_url_name, row_limit=8, wfm_url=def_wfm_url):
    # Gets all the order information of a prompted item

    r = requests.get(f"{wfm_url}/{item_url_name}/orders")

    # print(f"Item Order HTML Response: {r.status_code}\n")

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
    if 'mod_rank' in item_orders_df.columns:
        item_orders_df = item_orders_df[(item_orders_df['mod_rank'] == item_orders_df['mod_rank'].max())]
    else:
        item_orders_df['mod_rank'] = mod_rank_error_code

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

    # 3x weighted spread for bid/ask
    # avg_lowest_sell = item_sells_df['platinum']

    return item_orders_df, item_sells_df, item_buys_df, current_spread


def CreateEmptyHistory():  # this is used in case the item has invalid data, stuff like no transaction within date
    # range, etc.
    # please match the backup_columns with whatever the columns should be if the item is valid.
    error_columns = (
        'datetime', 'volume', 'min_price', 'max_price', 'open_price', 'closed_price', 'avg_price', 'wa_price',
        'median',
        'moving_avg', 'donch_top', 'donch_bot', 'id')

    # create dataframe with the date time in
    broken_item_df = CheckForMissingColumns(pd.DataFrame(np.zeros((1, len(error_columns))), columns=error_columns),
                                            check_for_datetime=True)

    return broken_item_df


def CheckTradeHistory(item_url_name, wfm_url=def_wfm_url):
    # some items do not have a single trade record, so it will return an empty JSON.
    # this is to process those and spit out something that can be commonly processed.
    # CreateEmptyHistory() is the function that generates "empty" df.

    r = requests.get(f"{wfm_url}/{item_url_name}/statistics", headers=key_header)
    if r.status_code == 200:
        print(f"\n{item_url_name} [{r.status_code}]")
        item_trade_hist_df = pd.DataFrame(r.json()['payload']['statistics_closed']['90days'])
        return item_trade_hist_df
    else:
        print(f"\n{item_url_name} is unreachable. [{r.status_code}]")
        return CreateEmptyHistory()


def GetItemMarketStatistics(item_url_name, wfm_url=def_wfm_url,
                            days=accounted_days):  # Gets all the market statistics of a prompted item
    global columns_cache

    item_statistics_closed_df = CheckTradeHistory(item_url_name, wfm_url)

    # clean and sort some columns since sometimes the returned json aren't quite consistent
    # checks for mod_ranks, datetime columns, and id.
    item_statistics_closed_df = CheckForMissingColumns(item_statistics_closed_df, True, True, True)

    # fill nas
    item_statistics_closed_df = item_statistics_closed_df.fillna(method="ffill")

    # this is here for tz-naive and tz-aware -- i'm not sure what this means, but it fixes it..
    current_utc_datetime = datetime.now(timezone.utc)
    item_statistics_closed_df['datetime'] = pd.to_datetime(item_statistics_closed_df['datetime'])
    item_statistics_closed_df['datetime'] = (current_utc_datetime - item_statistics_closed_df['datetime']).dt.days

    # rename datetime to days ago
    item_statistics_closed_df = item_statistics_closed_df.rename(columns={'datetime': 'days_ago'})

    # drop moving avg
    if 'moving_avg' in item_statistics_closed_df.columns:
        item_statistics_closed_df = item_statistics_closed_df.drop('moving_avg', axis=1)

    # cutoff day threshold
    item_statistics_closed_df = item_statistics_closed_df.loc[item_statistics_closed_df['days_ago'] <= days]
    if (~item_statistics_closed_df.empty) and (columns_cache is None):
        columns_cache = item_statistics_closed_df.columns
    elif item_statistics_closed_df.empty:
        item_statistics_closed_df = pd.DataFrame(data={**{col: [0] for col in columns_cache}})

    # calculate spread
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


def CheckIfCSVNeedsUpdate(filename=csv_filename, days_until_outdated=update_threshold_days):
    # last modified time date
    file_modified_time = datetime.fromtimestamp(os.path.getmtime(filename))
    # allowed threshold date for considering it outdated
    threshold_time = datetime.now() - timedelta(days=days_until_outdated)

    if file_modified_time <= threshold_time:
        # notify user
        print(
            f"The CSV file is outdated by {(datetime.now() - file_modified_time).days} day(s).\nRecommend a full "
            f"market scan.\n")
        # ask if you want to update
        response = input("Would you like to perform the update? [y/n]: ").lower()
        if 'y' in response:
            print("Commencing CSV Update. This will take a while - Estimated 15 Min")
            return True
        elif 'n' in response:
            print("Ignoring update.")
            return False
        else:
            print("Invalid response, proceeding with outdated data.")
            return False
    else:
        return False


async def GetSummarizedItemMarketStatisticsAsync(item_url_name, wfm_url=def_wfm_url, days=accounted_days):
    avg_volume, avg_spread, avg_median_price, avg_min_price, avg_typical_price, mod_rank = await asyncio.to_thread(
        GetSummarizedItemMarketStatistics, item_url_name, wfm_url, days)
    return avg_volume, avg_spread, avg_median_price, avg_min_price, avg_typical_price, mod_rank


async def GetAllSummarizedMarketStatistics(wfm_url=def_wfm_url, days=accounted_days, chunk_size=5, delay=rq_min_delay):
    item_list_df = GetAllItemsList(wfm_url)

    numeric_columns = ['avg_volume', 'avg_spread', 'avg_median_price', 'avg_min_price', 'avg_typical_price', 'mod_rank']

    summarized_data_df = pd.DataFrame()
    summarized_data_df.index = item_list_df.index

    # summarized_data_df = summarized_data_df[0:100]  # if you want to do this piece wise :)

    # the official rate limit is 3 requests per second. chunk_size of 5 and rq delay of 1 works well.
    for start in tqdm(range(0, len(summarized_data_df), chunk_size)):
        chunk = summarized_data_df[start:start + chunk_size]

        tasks = [GetSummarizedItemMarketStatisticsAsync(item_name, wfm_url, days) for item_name, _ in chunk.iterrows()]
        results = await asyncio.gather(*tasks)
        await asyncio.sleep(delay)

        # Convert results to a DataFrame
        chunk_results_df = pd.DataFrame(results, columns=numeric_columns, index=chunk.index)

        # Merge chunk_results_df into summarized_data_df using .loc
        summarized_data_df.loc[chunk_results_df.index, numeric_columns] = chunk_results_df

    print(summarized_data_df)

    summarized_data_df.to_csv(csv_filename, mode="w")

    return summarized_data_df

"""
async def GetAllLiveListingsData(wfm_url=def_wfm_url, days=accounted_days, chunk_size=5, delay=rq_min_delay):
    item_list_df = GetAllItemsList(wfm_url)

    numeric_columns = ['avg_volume', 'avg_spread', 'avg_median_price', 'avg_min_price', 'avg_typical_price', 'mod_rank']

    summarized_data_df = pd.DataFrame()
    summarized_data_df.index = item_list_df.index

    # summarized_data_df = summarized_data_df[0:100]  # if you want to do this piece wise :)

    # the official rate limit is 3 requests per second. chunk_size of 5 and rq delay of 1 works well.
    for start in tqdm(range(0, len(summarized_data_df), chunk_size)):
        chunk = summarized_data_df[start:start + chunk_size]

        tasks = [GetSummarizedItemMarketStatisticsAsync(item_name, wfm_url, days) for item_name, _ in chunk.iterrows()]
        results = await asyncio.gather(*tasks)
        await asyncio.sleep(delay)

        # Convert results to a DataFrame
        chunk_results_df = pd.DataFrame(results, columns=numeric_columns, index=chunk.index)

        # Merge chunk_results_df into summarized_data_df using .loc
        summarized_data_df.loc[chunk_results_df.index, numeric_columns] = chunk_results_df

    print(summarized_data_df)

    summarized_data_df.to_csv(csv_filename, mode="w")

    return summarized_data_df
"""