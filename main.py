import pandas as pd
import scraper
import asyncio
import aiohttp

pd.set_option('display.max_columns', None)
pd.set_option('display.max_colwidth', 30)  # Set maximum column width
pd.set_option('display.expand_frame_repr', False)  # Do not wrap to multiple lines

wfm_url = "https://api.warframe.market/v1/items"

#print(scraper.GetItemMarketStatistics("ammo_drum"))
print(asyncio.run(scraper.GetAllSummarizedMarketStatistics()))


#items_df = scraper.GetAllItemsList(wfm_url)
#items_df.to_csv('item_list.csv', index=True, mode='w')
