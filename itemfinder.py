from datetime import datetime, timezone, timedelta
from tqdm import tqdm

import asyncio
import numpy as np
import pandas as pd
import requests
import os

import scraper


async def GetListOfInterest(csv_filename, min_volume, min_spread, max_buy, min_buy=0):
    # max spread is here, so it filters out bogus transactions, e.g. proof fragment
    # at 100000 plat sold with spread at 99k plat
    max_spread = 8000

    with open(csv_filename, "r") as file:
        item_list = pd.read_csv(file, index_col='url_name')

    item_list = item_list.query(f"(avg_volume >= {min_volume})"
                                f"&(abs(avg_spread) >= {min_spread})"
                                f"&(abs(avg_spread) <= {max_spread})"
                                f"&(avg_min_price >= {min_buy})")

    return item_list


async def UnpackItemOrderInformation(url_name):
    # last element is the dictionary containing numerical values
    result = scraper.GetItemOrderInformation(url_name)[-1]
    print(url_name, result)
    return result


async def GetAllOrdersOfInterest(csv_filename, orders_of_interest_filename, min_volume, min_spread, max_buy, min_buy=0):
    chunk_size = 5
    rq_delay = 1.1

    item_list = await GetListOfInterest(csv_filename, min_volume, min_spread, max_buy, min_buy)

    if ~item_list.empty:
        print("Gathering current order data for items of interest...\n")
        orders_of_interest_df = pd.DataFrame(index=item_list.index)

        # the official rate limit is 3 requests per second. chunk_size of 5 and rq delay of 1.1 works well.
        for start in tqdm(range(0, len(orders_of_interest_df), chunk_size)):
            print() # just adding a little space
            chunk = orders_of_interest_df[start:start + chunk_size]

            # Assuming the GetItemOrderInformation returns a list of dictionaries
            tasks = [UnpackItemOrderInformation(url_name) for url_name in chunk.index]
            results = await asyncio.gather(*tasks)
            await asyncio.sleep(rq_delay)

            # Convert results to a DataFrame
            chunk_results_df = pd.DataFrame(results, index=chunk.index)

            # Merge chunk_results_df using .loc
            orders_of_interest_df.loc[chunk_results_df.index, chunk_results_df.columns] = chunk_results_df

        # process reverse trade possibilities
        # sometimes the bid and ask is reversed, and this area can be exploited with easy money.
        # this factor determines the multiplier to consider the minimum magnitude of the reversed spread to exploit.
        # essentially min_spread * reversed_spread_factor = minimum in the negative direction.

        orders_of_interest_df['short_trade_viable'] = orders_of_interest_df['current_spread'] < 0
        orders_of_interest_df=orders_of_interest_df.sort_values(['avg_spread'], ascending=False).reset_index(drop=False)


        print(orders_of_interest_df)

        orders_of_interest_df.to_csv(orders_of_interest_filename, mode="w")

        return orders_of_interest_df

    return item_list
