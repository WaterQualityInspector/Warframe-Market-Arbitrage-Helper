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

# print(scraper.GetItemInformation("primed_cryo_rounds"))
# print(scraper.GetItemInformation("saryn_prime_set"))

"""
_, sells, buys, avg_sell, avg_buy, avg_spread, spread = scraper.GetItemOrderInformation("primed_cryo_rounds")
print(f"{sells}\n{buys}\n{spread}\n{avg_sell}, {avg_buy}, {avg_spread}\n")
_, sells, buys, avg_sell, avg_buy, avg_spread, spread = scraper.GetItemOrderInformation("saryn_prime_set")
print(f"{sells}\n{buys}\n{spread}\n{avg_sell}, {avg_buy}, {avg_spread}\n")
_, sells, buys, avg_sell, avg_buy, avg_spread, spread = scraper.GetItemOrderInformation("legendary_fusion_core")
print(f"{sells}\n{buys}\n{spread}\n{avg_sell}, {avg_buy}, {avg_spread}\n")
"""

# print(scraper.GetItemMarketStatistics("saryn_prime_set"))

#print(scraper.GetItemOrderInformation("galvanized_savvy")[-1])

#scraper.ScrapeAll(market_summary_filename, days_until_outdated=0)

asyncio.run(itemfinder.GetAllOrdersOfInterest(market_summary_filename,orders_of_interest_filename, 10, 20, 500, 20))

print("Done")
