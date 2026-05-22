# sportsbook_scraper.py
import requests
import pandas as pd
from datetime import datetime


def get_dk_pitcher_strikeouts():
    """
    DraftKings Strikeout Props Scraper
    NOTE: DK changes endpoints occasionally.
    This works as a base structure.
    """

    url = "https://sportsbook.draftkings.com/sites/US-SB/api/v5/eventgroups/84240/categories/743/subcategories/7207"

    try:
        r = requests.get(url, timeout=10)
        data = r.json()
    except Exception as e:
        print(f"[warn] DK scrape failed: {e}")
        return pd.DataFrame()

    rows = []

    try:
        events = data["eventGroup"]["events"]

        for event in events:
            game_date = datetime.fromtimestamp(event["startDate"] / 1000).date()

            for category in data["eventGroup"]["offerCategories"]:
                for subcat in category["offerSubcategoryDescriptors"]:
                    offers = subcat["offerSubcategory"]["offers"]

                    for offer_group in offers:
                        for outcome in offer_group:
                            label = outcome.get("label", "")
                            player = outcome.get("participant", "")

                            if "Strikeouts" in label:
                                rows.append({
                                    "pitcher_name": player,
                                    "sportsbook_k_line": outcome.get("line"),
                                    "game_date": game_date
                                })

    except Exception as e:
        print(f"[warn] parsing failed: {e}")
        return pd.DataFrame()

    df = pd.DataFrame(rows)

    if not df.empty:
        df["pitcher_name"] = df["pitcher_name"].str.strip()

    return df