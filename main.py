import pandas as pd
import asyncio

import scraper
import itemfinder

pd.set_option('display.max_columns', None)
pd.set_option('display.max_colwidth', 20)  # Set maximum column width
pd.set_option('display.expand_frame_repr', False)  # Do not wrap to multiple lines

wfm_url = "https://api.warframe.market/v1/items"
market_summary_filename = "market_summary.csv"
orders_of_interest_filename = "IOI_orders.csv"


scraper.ScrapeAll(market_summary_filename, days_until_outdated=1)

asyncio.run(itemfinder.GetAllOrdersOfInterest(market_summary_filename, orders_of_interest_filename, 10, 20, 500, 20))

print("Done")
