import type {
  ContentRecommendResponse,
  CollaborativeRecommendResponse,
  HybridRecommendResponse,
  KpiMetrics,
  TopProductsResponse,
  GeoSaleEntry,
  TimelineEntry,
  CustomerStats,
} from "@/types";

const BASE = "/api";

async function fetchJSON<T>(url: string, options?: RequestInit): Promise<T> {
  const res = await fetch(url, options);
  if (!res.ok) {
    const text = await res.text();
    throw new Error(`API error ${res.status}: ${text}`);
  }
  return res.json() as Promise<T>;
}

// ---- Products & Customers ----

export const getProducts = (category?: string) => {
  const url = category
    ? `${BASE}/products?category=${encodeURIComponent(category)}`
    : `${BASE}/products`;
  return fetchJSON<{ products: string[] }>(url).then((r) => r.products);
};

export const getCategories = () =>
  fetchJSON<{ categories: { name: string; count: number }[] }>(`${BASE}/categories`)
    .then((r) => r.categories);

export const getCustomers = () =>
  fetchJSON<{ customers: string[] }>(`${BASE}/customers`).then((r) => r.customers);

// ---- Recommendations ----

export const getContentRecommendations = (productName: string, n = 5) =>
  fetchJSON<ContentRecommendResponse>(`${BASE}/recommend/content`, {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify({ product_name: productName, num_recommendations: n }),
  });

export const getCollaborativeRecommendations = (customerId: string, n = 5) =>
  fetchJSON<CollaborativeRecommendResponse>(`${BASE}/recommend/collaborative`, {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify({ customer_id: customerId, num_recommendations: n }),
  });

export const getHybridRecommendations = (
  productName: string,
  customerId: string | null,
  n = 5,
  contentWeight = 0.6,
  collabWeight = 0.4
) =>
  fetchJSON<HybridRecommendResponse>(`${BASE}/recommend/hybrid`, {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify({
      product_name: productName,
      customer_id: customerId,
      num_recommendations: n,
      content_weight: contentWeight,
      collab_weight: collabWeight,
    }),
  });

// ---- Analytics ----

export const getKpiMetrics = () =>
  fetchJSON<KpiMetrics>(`${BASE}/analytics/kpi`);

export const getTopProducts = (n = 10) =>
  fetchJSON<TopProductsResponse>(`${BASE}/analytics/top-products?n=${n}`);

export const getGeoSales = (product?: string) => {
  const url = product
    ? `${BASE}/analytics/geo-sales?product=${encodeURIComponent(product)}`
    : `${BASE}/analytics/geo-sales`;
  return fetchJSON<{ data: GeoSaleEntry[] }>(url).then((r) => r.data);
};

export const getTimeline = (product?: string) => {
  const url = product
    ? `${BASE}/analytics/timeline?product=${encodeURIComponent(product)}`
    : `${BASE}/analytics/timeline`;
  return fetchJSON<{ data: TimelineEntry[] }>(url).then((r) => r.data);
};

export const getCustomerStats = () =>
  fetchJSON<CustomerStats>(`${BASE}/analytics/customers`);
