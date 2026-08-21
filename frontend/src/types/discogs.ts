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
}

export interface DiscogsSeller {
  id: number;
  name: string;
  currency: string;
}

export interface DiscogsKeepersAPIResponse {
  listings: DiscogsListing[];
  count: number;
}
