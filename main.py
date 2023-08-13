import pandas as pd
import scraper
import asyncio

pd.set_option('display.max_columns', None)
pd.set_option('display.max_colwidth', 20)  # Set maximum column width
pd.set_option('display.expand_frame_repr', False)  # Do not wrap to multiple lines

wfm_url = "https://api.warframe.market/v1/items"

scraper.GetItemInformation("primed_cryo_rounds")
scraper.GetItemInformation("saryn_prime_set")


if scraper.CheckIfCSVNeedsUpdate(days_until_outdated=0):
    asyncio.run(scraper.GetAllSummarizedMarketStatistics())

print("Done")
