export interface User {
  id: number;
  username: string;
}

export interface Record {
  id: number;
  artist: string;
  title: string;
  format: string[];
  genres: string[];
  styles: string[];
  suggested_price: string;
  year: number | null;
  record_image?: string,
  wanted: boolean;
  evaluated: boolean;
  description?: string;
}

export interface TodoItem {
  id: number;
  user_id: number;
  text: string;
  status: 'in-progress' | 'backlog' | 'done';
  order: number;
  updated_at: string;
  created_at: string;
}

export interface UserDashboardProps {
  onLogout: () => void;
  tourMode?: boolean;
}

export interface PerformanceData {
  batch_number: number;
  accuracy: number;
  correct: number;
  total: number;
}

export interface StatsData {
  total_records: number;
  evaluated_records: number;
  keeper_count: number;
  keeper_rate: number;
  discogs_accuracy: number;
  ebay_accuracy: number;
  model_version: string;
  total_batches: number;
  ebay_evaluated?: number;
  ebay_total?: number;
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

export type Annotation = { keeper?: boolean; wantlist?: boolean };
