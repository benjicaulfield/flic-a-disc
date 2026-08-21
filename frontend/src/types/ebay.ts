export interface EbayListing {
  id: number;
  ebay_id: string;
  ebay_title: string;
  price: number;
  artist: string;
  title: string;
  format: string;
  year: number;
  media_condition: string;
  genres: string;
  styles: string;
  source: string;
  keeper_score: number;
}

export interface EbayScoredListing {
  id: number;
  ebay_id: string;
  ebay_title: string;
  price: number;
  tfidf_score: number;
}