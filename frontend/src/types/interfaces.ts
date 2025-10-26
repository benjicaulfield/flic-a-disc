export interface DiscogsRecord {
  id: number;
  discogs_id: string;
  artist: string;
  title: string;
  label: string;
  wants: number;
  haves: number;
  genres: string[];
  styles: string[];
  suggested_price: string;
  year: number | null;
  record_image?: string,
  wanted: boolean;
  evaluated: boolean;
  description?: string;
}

export interface DiscogsListing {
  id: number;
  record_price: string;
  media_condition: string;
  record: DiscogsRecord;
}

export interface EbayListing {
  id: number;
  ebay_id: number;
  title: string;
  price: string;
  currency: string;
  current_bid: string;
  bid_count: string;
  end_date: string;
  creation_date: string;
  seller_name: string;
  artist: string;
  album_title: string;
  format: string;
  year: string;
  record_condition: string;
  genres: string;
  styles: string
  saved: boolean;
  created_at: string;
  updated_at: string;
}

export interface DiscogsKeepersAPIResponse {
  listings: DiscogsListing[];
  count: number;
}

export interface MLData {
  predictions: number[];
  mean_predictions: number[];
  threshold?: number;
  uncertainties: number[];
  model_version: string;
}

export interface PerformanceStats {
  batch_accuracy: number;
  cumulative_accuracy: number;
  total_batches: number;
  total_records: number;
}

export interface User {
  id: number;
  username: string;
}

export interface LoginProps {
  onLogin: (user: User) => void;
}

export interface LoginResponse {
  onLogin: (user: User) => void;  
  onLogout: () => void;
}

export interface LandingPageProps {
  onLogin: (user: User) => void;
  onLogout: () => void;
}
