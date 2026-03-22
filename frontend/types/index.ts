// ---- Recommendations ----

export interface ContentRecommendation {
  StockCode: string;
  Description: string;
  UnitPrice: number;
  PopularityScore: number;
  PriceScore: number;
  SimilarityScore: number;
  Reason: string;
}

export interface CollaborativeRecommendation {
  StockCode: string;
  Description: string;
  UnitPrice: number;
  Score: number;
  Reason: string;
}

export interface HybridRecommendation {
  StockCode: string;
  Description: string;
  UnitPrice: number;
  HybridScore: number;
  SimilarityScore?: number;
  Score?: number;
  Reason: string;
}

export type RecommendationType = "content" | "collaborative" | "hybrid";

export interface ContentRecommendResponse {
  product: string;
  type: "content";
  recommendations: ContentRecommendation[];
}

export interface CollaborativeRecommendResponse {
  customerId: string;
  type: "collaborative";
  recommendations: CollaborativeRecommendation[];
}

export interface HybridRecommendResponse {
  product: string;
  customerId: string | null;
  type: "hybrid";
  weights: { content: number; collaborative: number };
  recommendations: HybridRecommendation[];
}

// ---- Analytics ----

export interface KpiMetrics {
  totalRevenue: number;
  totalOrders: number;
  totalProducts: number;
  totalCustomers: number;
  avgOrderValue: number;
  totalCountries: number;
}

export interface ProductStat {
  StockCode: string;
  Description: string;
  Quantity: number;
  TotalPrice: number;
}

export interface TopProductsResponse {
  byRevenue: ProductStat[];
  byQuantity: ProductStat[];
}

export interface GeoSaleEntry {
  country: string;
  orders: number;
  itemsSold: number;
  revenue: number;
}

export interface TimelineEntry {
  date: string;
  orders: number;
  revenue: number;
}

export interface TopCustomer {
  customerId: string;
  orders: number;
  totalSpend: number;
}

export interface CustomerStats {
  totalCustomers: number;
  totalCountries: number;
  avgOrderValue: number;
  topCustomers: TopCustomer[];
  customersByCountry: { country: string; customers: number }[];
}
