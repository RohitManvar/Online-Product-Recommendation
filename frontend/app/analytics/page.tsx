"use client";

import { useState, useMemo, useRef, useEffect } from "react";
import { useQuery } from "@tanstack/react-query";
import {
  getKpiMetrics,
  getTopProducts,
  getGeoSales,
  getCustomerStats,
  getProducts,
} from "@/lib/api";
import { KpiCard } from "@/components/ui/KpiCard";
import { TopProductsChart } from "@/components/charts/TopProductsChart";
import { CountryBarChart } from "@/components/charts/CountryBarChart";
import { Skeleton } from "@/components/ui/Skeleton";
import {
  DollarSign,
  ShoppingBag,
  Package,
  Users,
  Globe,
  CreditCard,
  AlertCircle,
  BarChart3,
  MapPin,
  Star,
  Search,
  X,
} from "lucide-react";
import { formatCurrency, formatNumber } from "@/lib/utils";
import { cn } from "@/lib/utils";

// ── Shared product filter dropdown ──────────────────────────────────────────

function ProductFilter({
  products,
  loading,
  value,
  onChange,
}: {
  products: string[];
  loading: boolean;
  value: string;
  onChange: (v: string) => void;
}) {
  const [search, setSearch] = useState(value);
  const [open, setOpen] = useState(false);
  const ref = useRef<HTMLDivElement>(null);

  // keep search input in sync when parent clears the value
  useEffect(() => {
    if (!value) setSearch("");
  }, [value]);

  useEffect(() => {
    function handleClickOutside(e: MouseEvent) {
      if (ref.current && !ref.current.contains(e.target as Node)) setOpen(false);
    }
    document.addEventListener("mousedown", handleClickOutside);
    return () => document.removeEventListener("mousedown", handleClickOutside);
  }, []);

  const filtered = useMemo(() => {
    if (!search) return products.slice(0, 50);
    const q = search.toLowerCase();
    return products.filter((p) => p.toLowerCase().includes(q)).slice(0, 50);
  }, [products, search]);

  return (
    <div className="relative" ref={ref}>
      <div className="relative">
        <Search className="absolute left-2.5 top-1/2 -translate-y-1/2 h-3.5 w-3.5 text-gray-400 pointer-events-none" />
        <input
          type="text"
          placeholder="Filter by product…"
          value={search}
          onFocus={() => setOpen(true)}
          onChange={(e) => {
            setSearch(e.target.value);
            onChange("");
            setOpen(true);
          }}
          className="w-52 pl-8 pr-7 py-1.5 text-xs rounded-lg border bg-gray-50 dark:bg-gray-800 focus:outline-none focus:ring-2 focus:ring-brand-500"
        />
        {(search || value) && (
          <button
            type="button"
            onClick={() => { setSearch(""); onChange(""); setOpen(false); }}
            className="absolute right-2 top-1/2 -translate-y-1/2 text-gray-400 hover:text-gray-600"
          >
            <X className="h-3.5 w-3.5" />
          </button>
        )}
      </div>

      {open && (
        <div className="absolute right-0 z-20 mt-1 w-72 border rounded-xl bg-white dark:bg-gray-800 shadow-xl max-h-52 overflow-y-auto">
          {loading ? (
            <p className="p-3 text-xs text-gray-400">Loading products…</p>
          ) : filtered.length === 0 ? (
            <p className="p-3 text-xs text-gray-400">No matches found</p>
          ) : (
            filtered.map((p) => (
              <button
                key={p}
                type="button"
                onClick={() => { setSearch(p); onChange(p); setOpen(false); }}
                className={cn(
                  "w-full text-left px-3 py-2 text-xs truncate transition-colors",
                  p === value
                    ? "bg-brand-50 dark:bg-brand-900/30 text-brand-700 dark:text-brand-300 font-medium"
                    : "hover:bg-gray-50 dark:hover:bg-gray-700 text-gray-700 dark:text-gray-300"
                )}
              >
                {p}
              </button>
            ))
          )}
        </div>
      )}
    </div>
  );
}

// ── Chart card wrapper ───────────────────────────────────────────────────────

function ChartCard({
  title,
  icon: Icon,
  children,
  className,
  action,
}: {
  title: string;
  icon?: React.ElementType;
  children: React.ReactNode;
  className?: string;
  action?: React.ReactNode;
}) {
  return (
    <div className={cn("bg-white dark:bg-gray-900 rounded-2xl border p-6 shadow-sm", className)}>
      <div className="flex items-center justify-between mb-5 gap-3 flex-wrap">
        <div className="flex items-center gap-2">
          {Icon && (
            <div className="p-1.5 bg-brand-50 dark:bg-brand-900/30 rounded-lg">
              <Icon className="h-4 w-4 text-brand-600" />
            </div>
          )}
          <h3 className="font-semibold text-gray-900 dark:text-white">{title}</h3>
        </div>
        {action}
      </div>
      {children}
    </div>
  );
}

// ── Page ─────────────────────────────────────────────────────────────────────

export default function AnalyticsPage() {
  const [productMode, setProductMode] = useState<"revenue" | "quantity">("revenue");
  const [geoProduct, setGeoProduct] = useState("");

  const kpi         = useQuery({ queryKey: ["kpi"],           queryFn: getKpiMetrics });
  const topProducts = useQuery({ queryKey: ["topProducts"],   queryFn: () => getTopProducts() });
  const customers   = useQuery({ queryKey: ["customerStats"], queryFn: getCustomerStats });
  const products    = useQuery({ queryKey: ["products"],      queryFn: getProducts });

  const geoSales = useQuery({
    queryKey: ["geoSales", geoProduct],
    queryFn: () => getGeoSales(geoProduct || undefined),
  });

  const anyError = [kpi, topProducts, geoSales, customers].some((q) => q.isError);

  return (
    <div className="max-w-7xl mx-auto px-4 sm:px-6 lg:px-8 py-10 space-y-8">
      {/* Header */}
      <div>
        <h1 className="text-3xl font-bold text-gray-900 dark:text-white">Sales Analytics</h1>
        <p className="text-gray-500 dark:text-gray-400 mt-1">
          Overview of revenue, products, customers, and geographic performance
        </p>
      </div>

      {anyError && (
        <div className="flex items-center gap-3 p-4 bg-red-50 dark:bg-red-900/20 border border-red-200 dark:border-red-800 rounded-2xl">
          <AlertCircle className="h-5 w-5 text-red-500 flex-shrink-0" />
          <p className="text-sm text-red-700 dark:text-red-300">
            Some data failed to load. Make sure the backend is running on port 8000.
          </p>
        </div>
      )}

      {/* KPI cards */}
      {kpi.isLoading ? (
        <div className="grid grid-cols-2 sm:grid-cols-3 lg:grid-cols-6 gap-4">
          {Array.from({ length: 6 }).map((_, i) => <Skeleton key={i} className="h-24 rounded-2xl" />)}
        </div>
      ) : kpi.data ? (
        <div className="grid grid-cols-2 sm:grid-cols-3 xl:grid-cols-6 gap-4">
          <KpiCard title="Total Revenue"  value={formatCurrency(kpi.data.totalRevenue)}   icon={DollarSign}  iconColor="text-emerald-600" iconBg="bg-emerald-50 dark:bg-emerald-900/30" />
          <KpiCard title="Total Orders"   value={formatNumber(kpi.data.totalOrders)}       icon={ShoppingBag} iconColor="text-blue-600"    iconBg="bg-blue-50 dark:bg-blue-900/30" />
          <KpiCard title="Products"       value={formatNumber(kpi.data.totalProducts)}     icon={Package}     iconColor="text-violet-600"  iconBg="bg-violet-50 dark:bg-violet-900/30" />
          <KpiCard title="Customers"      value={formatNumber(kpi.data.totalCustomers)}    icon={Users}       iconColor="text-orange-600"  iconBg="bg-orange-50 dark:bg-orange-900/30" />
          <KpiCard title="Countries"      value={String(kpi.data.totalCountries)}          icon={Globe}       iconColor="text-cyan-600"    iconBg="bg-cyan-50 dark:bg-cyan-900/30" />
          <KpiCard title="Avg. Order"     value={formatCurrency(kpi.data.avgOrderValue)}   icon={CreditCard}  iconColor="text-pink-600"    iconBg="bg-pink-50 dark:bg-pink-900/30" />
        </div>
      ) : null}

      {/* Top Products + Revenue by Country */}
      <div className="grid grid-cols-1 lg:grid-cols-2 gap-6">
        <ChartCard
          title="Top 10 Products"
          icon={BarChart3}
          action={
            <div className="flex gap-1">
              {(["revenue", "quantity"] as const).map((m) => (
                <button
                  key={m}
                  onClick={() => setProductMode(m)}
                  className={cn(
                    "px-3 py-1 rounded-lg text-xs font-medium transition-all",
                    productMode === m
                      ? "bg-brand-600 text-white shadow-sm"
                      : "bg-gray-100 dark:bg-gray-800 text-gray-600 dark:text-gray-300 hover:bg-gray-200 dark:hover:bg-gray-700"
                  )}
                >
                  {m === "revenue" ? "By Revenue" : "By Qty"}
                </button>
              ))}
            </div>
          }
        >
          {topProducts.isLoading ? (
            <Skeleton className="h-72 w-full" />
          ) : topProducts.data ? (
            <TopProductsChart
              data={productMode === "revenue" ? topProducts.data.byRevenue : topProducts.data.byQuantity}
              mode={productMode}
            />
          ) : null}
        </ChartCard>

        <ChartCard
          title="Revenue by Country"
          icon={MapPin}
          action={
            <div className="flex items-center gap-2">
              {geoProduct && (
                <span className="text-xs text-brand-600 font-medium bg-brand-50 dark:bg-brand-900/20 px-2 py-1 rounded-full truncate max-w-[120px]">
                  {geoProduct}
                </span>
              )}
              <ProductFilter
                products={products.data ?? []}
                loading={products.isLoading}
                value={geoProduct}
                onChange={setGeoProduct}
              />
            </div>
          }
        >
          {geoSales.isLoading ? (
            <Skeleton className="h-72 w-full" />
          ) : geoSales.data && geoSales.data.length > 0 ? (
            <CountryBarChart data={geoSales.data} />
          ) : geoSales.data?.length === 0 ? (
            <div className="h-64 flex items-center justify-center text-gray-400 text-sm">
              No data found for this product
            </div>
          ) : null}
        </ChartCard>
      </div>

      {/* Customer stats */}
      <div className="grid grid-cols-1 lg:grid-cols-2 gap-6">
        <ChartCard title="Top 10 Customers by Spend" icon={Star}>
          {customers.isLoading ? (
            <Skeleton className="h-64 w-full" />
          ) : customers.data ? (
            <div className="overflow-x-auto">
              <table className="w-full text-sm">
                <thead>
                  <tr className="border-b border-gray-100 dark:border-gray-800 text-left">
                    <th className="pb-3 font-semibold text-xs uppercase tracking-wide text-gray-400">#</th>
                    <th className="pb-3 font-semibold text-xs uppercase tracking-wide text-gray-400">Customer</th>
                    <th className="pb-3 font-semibold text-xs uppercase tracking-wide text-gray-400 text-right">Orders</th>
                    <th className="pb-3 font-semibold text-xs uppercase tracking-wide text-gray-400 text-right">Spend</th>
                  </tr>
                </thead>
                <tbody className="divide-y divide-gray-50 dark:divide-gray-800/50">
                  {customers.data.topCustomers.map((c, i) => (
                    <tr key={c.customerId} className="hover:bg-gray-50 dark:hover:bg-gray-800/30 transition-colors">
                      <td className="py-2.5 text-gray-400 text-xs font-medium">{i + 1}</td>
                      <td className="py-2.5"><span className="font-semibold text-gray-900 dark:text-white">#{c.customerId}</span></td>
                      <td className="py-2.5 text-right text-gray-500 text-sm">{c.orders}</td>
                      <td className="py-2.5 text-right"><span className="font-bold text-brand-600">{formatCurrency(c.totalSpend)}</span></td>
                    </tr>
                  ))}
                </tbody>
              </table>
            </div>
          ) : null}
        </ChartCard>

        <ChartCard title="Top Countries by Customers" icon={Globe}>
          {customers.isLoading ? (
            <Skeleton className="h-64 w-full" />
          ) : customers.data ? (
            <div className="space-y-3">
              {customers.data.customersByCountry.map((c, i) => {
                const max = customers.data!.customersByCountry[0].customers;
                const pct = (c.customers / max) * 100;
                const colors = ["bg-brand-600","bg-brand-500","bg-brand-400","bg-brand-300","bg-violet-500","bg-violet-400","bg-cyan-500","bg-cyan-400","bg-emerald-500","bg-emerald-400"];
                return (
                  <div key={c.country}>
                    <div className="flex justify-between text-sm mb-1">
                      <span className="font-medium text-gray-800 dark:text-gray-200">{c.country}</span>
                      <span className="text-gray-500 text-xs font-semibold">{formatNumber(c.customers)}</span>
                    </div>
                    <div className="h-2 bg-gray-100 dark:bg-gray-800 rounded-full overflow-hidden">
                      <div className={cn("h-full rounded-full transition-all duration-500", colors[i % colors.length])} style={{ width: `${pct}%` }} />
                    </div>
                  </div>
                );
              })}
            </div>
          ) : null}
        </ChartCard>
      </div>
    </div>
  );
}
