"""
Parses every saved wantlist-marketplace JSON page in pages/ into a
single CSV.

Save pages by navigating the browser directly to the JSON API (not the
rendered UI -- that list is virtualized and drops items on scroll):

  https://www.discogs.com/api/shop-page-api/sell_item?sellerRatingMin=90
  &sort=listedDate&sortOrder=descending&count=250&offset=N&facets=true

bump offset by 250 each time (0, 250, 500, ... up to totalCount) and
Cmd+S the raw response into pages/ (either .html or .json extension
is fine).

Run:  /Users/benjamincaulfield/.pyenv/versions/3.11.5/bin/python3 parse_pages.py
"""
import csv
import json
import pathlib

from ..models import DiscogsListing

HERE = pathlib.Path(__file__).parent
PAGES_DIR = HERE / "pages"
OUT_PATH = HERE.parent / "discogs_scrapes" / "wantlist_listings.csv"

FIELDNAMES = [
    "listing_id", "release_id", "artist", "title", "label", "catno",
    "year", "country", "format", "genres", "styles", "release_rating",
    "media_condition", "sleeve_condition", "comments", "allows_offers",
    "is_deal", "listed_date", "price_amount", "price_amount_usd",
    "price_currency", "previous_price_amount", "shipping_price",
    "seller", "seller_rating", "seller_rating_count", "ships_from",
    "listing_url", "release_url", "image_url", "source_file",
]


def clean_text(value: str) -> str:
    return " ".join(value.split())


def parse_item(item: dict, source_file: str) -> dict:
    release = item.get("release") or {}
    price = item.get("price") or {}
    prev_price = item.get("previousPrice") or {}
    shipping = item.get("shipping") or {}
    seller = item.get("seller") or {}
    labels = release.get("labels") or []

    listing_id = item.get("itemId")
    release_id = release.get("releaseId")

    return {
        "listing_id": listing_id or "",
        "release_id": release_id or "",
        "artist": "; ".join(a.get("name", "") for a in release.get("artists") or []),
        "title": release.get("title", ""),
        "label": "; ".join(l.get("name", "") for l in labels),
        "catno": "; ".join(l.get("catno", "") for l in labels),
        "year": release.get("year", "") or "",
        "country": release.get("country", "") or "",
        "format": "; ".join(release.get("formatNames") or []),
        "genres": "; ".join(g.get("name", "") for g in release.get("genres") or []),
        "styles": "; ".join(s.get("name", "") for s in release.get("styles") or []),
        "release_rating": release.get("rating", "") or "",
        "media_condition": item.get("mediaCondition", "") or "",
        "sleeve_condition": item.get("sleeveCondition", "") or "",
        "comments": clean_text(item.get("comments", "") or ""),
        "allows_offers": item.get("allowsOffers", ""),
        "is_deal": item.get("isDeal", ""),
        "listed_date": item.get("listedDate", "") or "",
        "price_amount": price.get("amount", "") or "",
        "price_amount_usd": price.get("amountUsd", "") or "",
        "price_currency": price.get("currencyCode", "") or "",
        "previous_price_amount": prev_price.get("amount", "") or "",
        "shipping_price": shipping.get("buyerShippingPrice", "") or "",
        "seller": seller.get("name", "") or "",
        "seller_rating": seller.get("rating", "") or "",
        "seller_rating_count": seller.get("ratingCount", "") or "",
        "ships_from": seller.get("shipsFrom", "") or "",
        "listing_url": f"https://www.discogs.com/shop/item/{listing_id}" if listing_id else "",
        "release_url": f"https://www.discogs.com/release/{release_id}" if release_id else "",
        "image_url": item.get("imageUrl", "") or "",
        "source_file": source_file,
    }


def parse_file(path: pathlib.Path) -> list:
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except json.JSONDecodeError:
        print(f"{path.name}: not JSON, skipping (old rendered-HTML page?)")
        return []
    items = payload.get("items")
    if not isinstance(items, list):
        print(f"{path.name}: no 'items' list found -- keys: {list(payload.keys())}")
        return []
    return [parse_item(item, path.name) for item in items]


def main():
    files = sorted(PAGES_DIR.glob("*.html")) + sorted(PAGES_DIR.glob("*.json"))
    if not files:
        raise SystemExit(f"No .html/.json files found in {PAGES_DIR} -- save your pages there first.")

    seen_ids = set()
    rows = []

    for path in files:
        file_rows = parse_file(path)
        new_count = 0
        for row in file_rows:
            key = row["listing_id"] or f"{row['title']}|{row['artist']}|{row['seller']}"
            if key in seen_ids:
                continue
            seen_ids.add(key)
            rows.append(row)
            new_count += 1
        print(f"{path.name}: {len(file_rows)} items found, {new_count} new")

    OUT_PATH.parent.mkdir(parents=True, exist_ok=True)
    with OUT_PATH.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=FIELDNAMES)
        writer.writeheader()
        writer.writerows(rows)

    print(f"\n{len(rows)} unique listings from {len(files)} files -> {OUT_PATH}")


if __name__ == "__main__":
    main()
