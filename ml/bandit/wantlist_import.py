"""
Shared logic for loading wantlist_listings.csv-shaped rows into
DiscogsRecord/DiscogsListing. Used by both the import_wantlist_listings
management command (reads from disk) and the wantlist_new_arrivals API
view (reads uploaded files) so the two never drift apart.
"""
import json

from django.db.models import Q
from django.utils.dateparse import parse_datetime

from .models import DiscogsRecord, DiscogsListing, DiscogsSeller
from .discogs_client import authenticate_client


def clean_text(value: str) -> str:
    return " ".join(value.split())


def parse_json_item(item: dict, source_file: str = "") -> dict:
    """Flattens one item from the raw Discogs shop-page-api JSON response
    into the same row shape wantlist_listings.csv uses. Mirrors
    wantlist_scrape/parse_pages.py's parse_item -- keep the two in sync
    if the Discogs API response shape ever changes."""
    release = item.get("release") or {}
    price = item.get("price") or {}
    seller = item.get("seller") or {}
    labels = release.get("labels") or []

    listing_id = item.get("itemId")
    release_id = release.get("releaseId")

    return {
        "listing_id": str(listing_id) if listing_id else "",
        "release_id": str(release_id) if release_id else "",
        "artist": "; ".join(a.get("name", "") for a in release.get("artists") or []),
        "title": release.get("title", ""),
        "label": "; ".join(l.get("name", "") for l in labels),
        "catno": "; ".join(l.get("catno", "") for l in labels),
        "year": release.get("year", "") or "",
        "country": release.get("country", "") or "",
        "format": "; ".join(release.get("formatNames") or []),
        "genres": "; ".join(g.get("name", "") for g in release.get("genres") or []),
        "styles": "; ".join(s.get("name", "") for s in release.get("styles") or []),
        "media_condition": item.get("mediaCondition", "") or "",
        "sleeve_condition": item.get("sleeveCondition", "") or "",
        "comments": clean_text(item.get("comments", "") or ""),
        "listed_date": item.get("listedDate", "") or "",
        "price_amount": price.get("amount", "") or "",
        "price_currency": price.get("currencyCode", "") or "",
        "seller": seller.get("name", "") or "",
        "ships_from": seller.get("shipsFrom", "") or "",
        "source_file": source_file,
    }


def rows_from_json_payload(payload: dict, source_file: str = "") -> list:
    items = payload.get("items")
    if not isinstance(items, list):
        return []
    return [parse_json_item(item, source_file) for item in items]


def rows_from_upload(name: str, text: str) -> list:
    """Auto-detects a raw Discogs API JSON dump vs. a wantlist_listings.csv
    -shaped CSV and returns row dicts either way."""
    stripped = text.lstrip()
    if stripped.startswith("{"):
        try:
            payload = json.loads(stripped)
        except json.JSONDecodeError:
            payload = None
        if payload is not None:
            return rows_from_json_payload(payload, name)

    import csv
    import io
    return list(csv.DictReader(io.StringIO(text)))


def split_list(value: str) -> list:
    return [v.strip() for v in value.split(";") if v.strip()] if value else []


def parse_year(value: str):
    try:
        return int(float(value))
    except (ValueError, TypeError):
        return None


def upsert_records(rows, log=print):
    created = 0
    by_release = {}
    for row in rows:
        rid = row.get('release_id')
        if not rid or rid in by_release:
            continue
        by_release[rid] = row

    for rid, row in by_release.items():
        record, was_created = DiscogsRecord.objects.get_or_create(
            discogs_id=rid,
            defaults={
                'artist': row.get('artist', ''),
                'title': row.get('title', ''),
                'label': row.get('label', ''),
                'catno': row.get('catno', ''),
                'genres': split_list(row.get('genres', '')),
                'styles': split_list(row.get('styles', '')),
                'format': split_list(row.get('format', '')),
                'year': parse_year(row.get('year')),
                'country': row.get('country', ''),
            }
        )
        if was_created:
            created += 1

    return created, set(by_release.keys())


def enrich_records(release_ids, limit=None, log=print):
    needs_enrichment = list(
        DiscogsRecord.objects.filter(discogs_id__in=release_ids).filter(
            Q(wants=0, haves=0) | Q(suggested_price='')
        )
    )

    if limit:
        needs_enrichment = needs_enrichment[:limit]

    total = len(needs_enrichment)
    if total == 0:
        log("No releases need API enrichment.")
        return 0, 0

    log(f"Enriching {total} releases from the Discogs API (~{total/60:.1f} min at 60/min)...")
    client = authenticate_client()

    updated = 0
    errors = 0
    for i, record in enumerate(needs_enrichment, 1):
        try:
            release = client.release(int(record.discogs_id))
            # release.data is just a stub {'id', 'resource_url'} until
            # something forces the real fetch -- .community does that.
            wants = release.community.want or 0
            haves = release.community.have or 0

            update_fields = []
            if wants or haves:
                record.wants = wants
                record.haves = haves
                update_fields += ['wants', 'haves']

            if not record.suggested_price:
                try:
                    vg_plus = release.price_suggestions.very_good_plus
                    if vg_plus is not None and vg_plus.value is not None:
                        record.suggested_price = str(vg_plus.value)
                        update_fields.append('suggested_price')
                except AttributeError:
                    pass

            if update_fields:
                record.save(update_fields=update_fields)
                updated += 1

        except Exception as e:
            log(f"  Error enriching {record.discogs_id}: {e}")
            errors += 1

        if i % 25 == 0:
            log(f"  [{i}/{total}] updated={updated} errors={errors}")

    log(f"Enrichment done: {updated} updated, {errors} errors")
    return updated, errors


def upsert_listings(rows, log=print):
    """Returns (created_count, updated_count, created_listing_ids)."""
    created = 0
    updated = 0
    created_listing_ids = set()

    records_by_release = {
        r.discogs_id: r
        for r in DiscogsRecord.objects.filter(
            discogs_id__in={row['release_id'] for row in rows if row.get('release_id')}
        )
    }
    sellers_by_name = {}

    for row in rows:
        if not row.get('listing_id') or not row.get('release_id'):
            continue
        record = records_by_release.get(row['release_id'])
        if record is None:
            continue

        seller_name = row.get('seller') or 'unknown'
        seller = sellers_by_name.get(seller_name)
        if seller is None:
            seller, _ = DiscogsSeller.objects.get_or_create(
                name=seller_name, defaults={'currency': row.get('price_currency') or ''}
            )
            sellers_by_name[seller_name] = seller

        ships_from = row.get('ships_from') or ''
        if ships_from and seller.ships_from != ships_from:
            seller.ships_from = ships_from
            seller.save(update_fields=['ships_from'])

        try:
            price = float(row['price_amount']) if row.get('price_amount') else 0.0
        except ValueError:
            price = 0.0

        listed_date = parse_datetime(row['listed_date']) if row.get('listed_date') else None

        _, was_created = DiscogsListing.objects.update_or_create(
            discogs_listing_id=row['listing_id'],
            defaults={
                'seller': seller,
                'record': record,
                'record_price': price,
                'currency': row.get('price_currency') or '',
                'media_condition': row.get('media_condition') or '',
                'sleeve_condition': row.get('sleeve_condition') or '',
                'listed_date': listed_date,
                'ships_from': ships_from or None,
                'free_shipping_min_amount': seller.free_shipping_min_amount,
                'free_shipping_min_currency': seller.free_shipping_min_currency,
                'free_shipping_min_usd': seller.free_shipping_min_usd,
                'flat_rate_amount': seller.flat_rate_amount,
                'flat_rate_currency': seller.flat_rate_currency,
                'flat_rate_usd': seller.flat_rate_usd,
            }
        )
        if was_created:
            created += 1
            created_listing_ids.add(row['listing_id'])
        else:
            updated += 1

    return created, updated, created_listing_ids


def import_rows(rows, skip_enrich=False, limit_enrich=None, log=print):
    """Runs the full pipeline for a batch of CSV-shaped rows. Returns a
    dict summary plus the set of discogs_listing_id values that were
    newly created (didn't already exist before this call) -- the basis
    for a "new arrivals" report."""
    records_created, release_ids = upsert_records(rows, log=log)
    log(f"DiscogsRecord: {records_created} created, {len(release_ids)} unique releases in this batch")

    if not skip_enrich:
        enrich_records(release_ids, limit_enrich, log=log)
    else:
        log("Skipping API enrichment (skip_enrich=True)")

    listings_created, listings_updated, new_listing_ids = upsert_listings(rows, log=log)
    log(f"DiscogsListing: {listings_created} created, {listings_updated} updated")

    return {
        'rows_parsed': len(rows),
        'records_created': records_created,
        'unique_releases': len(release_ids),
        'listings_created': listings_created,
        'listings_updated': listings_updated,
        'new_listing_ids': new_listing_ids,
    }
