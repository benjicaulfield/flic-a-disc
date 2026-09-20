export interface DiscogsRecord {
  id: number;
  discogs_id: string;
  artist: string;
  title: string;
  label: string;
  wants: number;
  haves: number;
  catno: string;
  format: string[];
  genres: string[];
  styles: string[];
  suggested_price: number;
  year: number | null;
  country: string;
  record_image?: string,
  wanted: boolean;
  evaluated: boolean;
  description?: string;
  heldout: boolean;
}

export interface DiscogsListing {
  listing_id: number;
  discogs_id: string;
  media_condition: string;
  record_price: string;
  seller: string;
  artist: string;
  title: string;
  label: string;
  catno?: string | null;
  wants: number;
  haves: number;
  genres: string[];
  styles: string[];
  year: number | null;
  suggested_price: number | null;
  format: string[];
  score?: number;
  price?: number;
  wanted?: boolean;
  wantlist?: boolean;
  evaluated?: boolean;
  wantlist_evaluated?: boolean;
  probability?: number;
  uncertainty?: number;
}

export interface DiscogsSeller {
  id: number;
  name: string;
  currency: string;
}

export interface WantlistListing extends DiscogsListing {
  sleeve_condition?: string;
  embedding_score?: number;
  price_delta_pct?: number | null;
}

export interface WantlistSellerSummary {
  seller: string;
  listing_count: number;
  priced_listing_count: number;
  avg_price_delta_pct: number | null;
}

export interface NewArrivalRow {
  listing_id: string;
  artist: string;
  title: string;
  seller: string;
  media_condition: string;
  sleeve_condition: string;
  price: number | null;
  suggested_price: number | null;
  price_delta_pct: number | null;
  listed_date: string | null;
  listing_url: string;
}

export interface NewArrivalsResponse {
  files_processed: number;
  rows_parsed: number;
  new_arrivals_count: number;
  results: NewArrivalRow[];
  log?: string[];
}

export interface DiscogsKeepersAPIResponse {
  listings: DiscogsListing[];
  count: number;
}
