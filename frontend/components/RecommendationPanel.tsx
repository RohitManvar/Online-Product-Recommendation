"use client";

import { useState, useMemo, useRef, useEffect, useCallback } from "react";
import { useQuery } from "@tanstack/react-query";
import {
  getProducts,
  getCustomers,
  getCategories,
  getContentRecommendations,
  getCollaborativeRecommendations,
  getHybridRecommendations,
} from "@/lib/api";
import { ProductCard } from "@/components/ui/ProductCard";
import { ProductCardSkeleton } from "@/components/ui/Skeleton";
import { ProductDrawer } from "@/components/ProductDrawer";
import { Search, Users, Layers, AlertCircle, Tag, X } from "lucide-react";
import { cn } from "@/lib/utils";

type Mode = "content" | "collaborative" | "hybrid";

const MODES: { id: Mode; label: string; icon: React.ElementType; desc: string }[] = [
  { id: "content",       label: "Content-Based",  icon: Search, desc: "Find products similar to a selected item" },
  { id: "collaborative", label: "Collaborative",   icon: Users,  desc: "Personalized for a specific customer" },
  { id: "hybrid",        label: "Hybrid",          icon: Layers, desc: "Blend of both methods for best results" },
];

// Category chip colors cycling through a palette
const CHIP_COLORS = [
  "bg-blue-50 text-blue-700 border-blue-200 hover:bg-blue-100 dark:bg-blue-900/20 dark:text-blue-300 dark:border-blue-800",
  "bg-violet-50 text-violet-700 border-violet-200 hover:bg-violet-100 dark:bg-violet-900/20 dark:text-violet-300 dark:border-violet-800",
  "bg-emerald-50 text-emerald-700 border-emerald-200 hover:bg-emerald-100 dark:bg-emerald-900/20 dark:text-emerald-300 dark:border-emerald-800",
  "bg-orange-50 text-orange-700 border-orange-200 hover:bg-orange-100 dark:bg-orange-900/20 dark:text-orange-300 dark:border-orange-800",
  "bg-pink-50 text-pink-700 border-pink-200 hover:bg-pink-100 dark:bg-pink-900/20 dark:text-pink-300 dark:border-pink-800",
  "bg-cyan-50 text-cyan-700 border-cyan-200 hover:bg-cyan-100 dark:bg-cyan-900/20 dark:text-cyan-300 dark:border-cyan-800",
  "bg-amber-50 text-amber-700 border-amber-200 hover:bg-amber-100 dark:bg-amber-900/20 dark:text-amber-300 dark:border-amber-800",
  "bg-teal-50 text-teal-700 border-teal-200 hover:bg-teal-100 dark:bg-teal-900/20 dark:text-teal-300 dark:border-teal-800",
  "bg-red-50 text-red-700 border-red-200 hover:bg-red-100 dark:bg-red-900/20 dark:text-red-300 dark:border-red-800",
  "bg-indigo-50 text-indigo-700 border-indigo-200 hover:bg-indigo-100 dark:bg-indigo-900/20 dark:text-indigo-300 dark:border-indigo-800",
  "bg-lime-50 text-lime-700 border-lime-200 hover:bg-lime-100 dark:bg-lime-900/20 dark:text-lime-300 dark:border-lime-800",
];

export function RecommendationPanel() {
  const [mode, setMode] = useState<Mode>("content");
  const [selectedProduct, setSelectedProduct] = useState("");
  const [selectedCustomer, setSelectedCustomer] = useState("");
  const [productSearch, setProductSearch] = useState("");
  const [submitted, setSubmitted] = useState(false);
  const [dropdownOpen, setDropdownOpen] = useState(false);
  const [selectedCategory, setSelectedCategory] = useState<string | null>(null);
  const [drawerProduct, setDrawerProduct] = useState<{
    stockCode: string; description: string; unitPrice: number;
    score?: number; scoreLabel?: string; reason?: string;
  } | null>(null);
  const dropdownRef = useRef<HTMLDivElement>(null);

  // Recently searched — persisted in localStorage
  const [recentSearches, setRecentSearches] = useState<string[]>(() => {
    if (typeof window === "undefined") return [];
    try { return JSON.parse(localStorage.getItem("recentSearches") ?? "[]"); } catch { return []; }
  });

  const addRecentSearch = useCallback((product: string) => {
    setRecentSearches((prev) => {
      const next = [product, ...prev.filter((p) => p !== product)].slice(0, 5);
      localStorage.setItem("recentSearches", JSON.stringify(next));
      return next;
    });
  }, []);

  // Sort & result-count state
  const [sortBy, setSortBy] = useState<"score" | "price_asc" | "price_desc">("score");
  const [numResults, setNumResults] = useState(5);

  useEffect(() => {
    function handleClickOutside(e: MouseEvent) {
      if (dropdownRef.current && !dropdownRef.current.contains(e.target as Node)) {
        setDropdownOpen(false);
      }
    }
    document.addEventListener("mousedown", handleClickOutside);
    return () => document.removeEventListener("mousedown", handleClickOutside);
  }, []);

  // Fetch categories
  const { data: categories = [] } = useQuery({
    queryKey: ["categories"],
    queryFn: getCategories,
  });

  // Fetch products — re-fetches when category changes
  const { data: allProducts = [], isLoading: productsLoading } = useQuery({
    queryKey: ["products", selectedCategory],
    queryFn: () => getProducts(selectedCategory ?? undefined),
  });

  const { data: customers = [] } = useQuery({
    queryKey: ["customers"],
    queryFn: getCustomers,
  });

  // Client-side search filter on top of the category-filtered list
  const filteredProducts = useMemo(() => {
    if (!productSearch) return allProducts.slice(0, 50);
    const q = productSearch.toLowerCase();
    return allProducts.filter((p) => p.toLowerCase().includes(q)).slice(0, 50);
  }, [allProducts, productSearch]);

  // Recommendation queries — only run when submitted
  const contentQuery = useQuery({
    queryKey: ["recommend", "content", selectedProduct, numResults],
    queryFn: () => getContentRecommendations(selectedProduct, numResults),
    enabled: submitted && (mode === "content" || mode === "hybrid") && !!selectedProduct,
  });

  const collabQuery = useQuery({
    queryKey: ["recommend", "collaborative", selectedCustomer, numResults],
    queryFn: () => getCollaborativeRecommendations(selectedCustomer, numResults),
    enabled: submitted && mode === "collaborative" && !!selectedCustomer,
  });

  const hybridQuery = useQuery({
    queryKey: ["recommend", "hybrid", selectedProduct, selectedCustomer, numResults],
    queryFn: () => getHybridRecommendations(selectedProduct, selectedCustomer || null, numResults),
    enabled: submitted && mode === "hybrid" && !!selectedProduct,
  });

  const activeQuery =
    mode === "content" ? contentQuery
    : mode === "collaborative" ? collabQuery
    : hybridQuery;

  const recs =
    mode === "content"      ? contentQuery.data?.recommendations ?? []
    : mode === "collaborative" ? collabQuery.data?.recommendations ?? []
    : hybridQuery.data?.recommendations ?? [];

  function handleSubmit(e: React.FormEvent) {
    e.preventDefault();
    if (selectedProduct) addRecentSearch(selectedProduct);
    setSubmitted(true);
  }

  const sortedRecs = useMemo(() => {
    if (!recs.length) return recs;
    const copy = [...recs];
    if (sortBy === "price_asc") return copy.sort((a, b) => a.UnitPrice - b.UnitPrice);
    if (sortBy === "price_desc") return copy.sort((a, b) => b.UnitPrice - a.UnitPrice);
    return copy; // score order (default from API)
  }, [recs, sortBy]);

  function handleCategorySelect(cat: string) {
    if (selectedCategory === cat) {
      setSelectedCategory(null);
    } else {
      setSelectedCategory(cat);
      // Clear current selection if it won't be in the new category
      setSelectedProduct("");
      setProductSearch("");
      setSubmitted(false);
    }
    setDropdownOpen(true);
  }

  const needsProduct  = mode === "content" || mode === "hybrid";
  const needsCustomer = mode === "collaborative" || mode === "hybrid";

  return (
    <div className="max-w-7xl mx-auto px-4 sm:px-6 lg:px-8 py-10 space-y-8">
      {/* Header */}
      <div>
        <h1 className="text-3xl font-bold text-gray-900 dark:text-white">
          Product <span className="gradient-text">Recommendations</span>
        </h1>
        <p className="text-gray-500 dark:text-gray-400 mt-1">
          ML-powered suggestions using content similarity and collaborative filtering
        </p>
      </div>

      {/* Mode selector */}
      <div className="grid grid-cols-1 sm:grid-cols-3 gap-3">
        {MODES.map((m) => (
          <button
            key={m.id}
            onClick={() => { setMode(m.id); setSubmitted(false); }}
            className={cn(
              "flex items-start gap-3 p-4 rounded-2xl border text-left transition-all",
              mode === m.id
                ? "border-brand-600 bg-brand-50 dark:bg-brand-900/20 shadow-sm"
                : "border-gray-200 dark:border-gray-700 hover:border-gray-300 dark:hover:border-gray-600"
            )}
          >
            <m.icon className={cn("h-5 w-5 mt-0.5", mode === m.id ? "text-brand-600" : "text-gray-400")} />
            <div>
              <p className={cn("font-semibold text-sm", mode === m.id ? "text-brand-700 dark:text-brand-300" : "text-gray-700 dark:text-gray-300")}>
                {m.label}
              </p>
              <p className="text-xs text-gray-500 dark:text-gray-400">{m.desc}</p>
            </div>
          </button>
        ))}
      </div>

      {/* Input form */}
      <form
        onSubmit={handleSubmit}
        className="bg-white dark:bg-gray-900 rounded-2xl border p-6 shadow-sm space-y-5"
      >
        {/* Category filter — shown only when product field is needed */}
        {needsProduct && categories.length > 0 && (
          <div className="space-y-2">
            <div className="flex items-center gap-2">
              <Tag className="h-3.5 w-3.5 text-gray-400" />
              <span className="text-xs font-semibold text-gray-500 dark:text-gray-400 uppercase tracking-wide">
                Filter by Category
              </span>
              {selectedCategory && (
                <button
                  type="button"
                  onClick={() => { setSelectedCategory(null); setSelectedProduct(""); setProductSearch(""); setSubmitted(false); }}
                  className="ml-auto flex items-center gap-1 text-xs text-gray-400 hover:text-gray-600 dark:hover:text-gray-200 transition-colors"
                >
                  <X className="h-3 w-3" /> Clear
                </button>
              )}
            </div>
            <div className="flex flex-wrap gap-2">
              {categories.map((cat, i) => (
                <button
                  key={cat.name}
                  type="button"
                  onClick={() => handleCategorySelect(cat.name)}
                  className={cn(
                    "px-3 py-1 rounded-full border text-xs font-medium transition-all",
                    selectedCategory === cat.name
                      ? "bg-brand-600 text-white border-brand-600 shadow-sm"
                      : CHIP_COLORS[i % CHIP_COLORS.length]
                  )}
                >
                  {cat.name}
                  <span className={cn(
                    "ml-1.5 opacity-60 text-[10px]",
                    selectedCategory === cat.name ? "opacity-80" : ""
                  )}>
                    {cat.count}
                  </span>
                </button>
              ))}
            </div>
          </div>
        )}

        <div className="grid grid-cols-1 sm:grid-cols-2 gap-4">
          {/* Product selector */}
          {needsProduct && (
            <div className="space-y-1 relative" ref={dropdownRef}>
              <label className="text-sm font-medium text-gray-700 dark:text-gray-300">
                Product
                {selectedCategory && (
                  <span className="ml-2 text-xs text-brand-600 font-normal">
                    in {selectedCategory}
                  </span>
                )}
              </label>
              <div className="relative">
                <Search className="absolute left-3 top-1/2 -translate-y-1/2 h-4 w-4 text-gray-400 pointer-events-none" />
                <input
                  type="text"
                  placeholder={selectedCategory ? `Search in ${selectedCategory}…` : "Search products…"}
                  value={productSearch}
                  onFocus={() => setDropdownOpen(true)}
                  onChange={(e) => {
                    setProductSearch(e.target.value);
                    setSelectedProduct("");
                    setSubmitted(false);
                    setDropdownOpen(true);
                  }}
                  className="w-full pl-9 pr-8 py-2 rounded-lg border bg-gray-50 dark:bg-gray-800 text-sm focus:outline-none focus:ring-2 focus:ring-brand-500"
                />
                {(selectedProduct || productSearch) && (
                  <button
                    type="button"
                    onClick={() => { setSelectedProduct(""); setProductSearch(""); setSubmitted(false); setDropdownOpen(true); }}
                    className="absolute right-3 top-1/2 -translate-y-1/2 text-gray-400 hover:text-gray-600"
                  >
                    <X className="h-3.5 w-3.5" />
                  </button>
                )}
              </div>

              {dropdownOpen && (
                <div className="absolute z-10 mt-1 w-full border rounded-xl bg-white dark:bg-gray-800 shadow-xl max-h-64 overflow-y-auto">
                  {/* Recent searches shown when input is empty */}
                  {!productSearch && recentSearches.length > 0 && (
                    <div className="border-b border-gray-100 dark:border-gray-700 pb-1">
                      <p className="px-3 pt-2 pb-1 text-[10px] font-semibold uppercase tracking-wide text-gray-400">Recent</p>
                      {recentSearches.map((p) => (
                        <button
                          key={p}
                          type="button"
                          onClick={() => { setSelectedProduct(p); setProductSearch(p); setSubmitted(false); setDropdownOpen(false); }}
                          className="w-full text-left px-3 py-2 text-sm truncate transition-colors hover:bg-gray-50 dark:hover:bg-gray-700 flex items-center gap-2 text-gray-600 dark:text-gray-300"
                        >
                          <span className="text-gray-300 dark:text-gray-600 text-xs">↩</span>
                          {p}
                        </button>
                      ))}
                    </div>
                  )}
                  {productsLoading ? (
                    <p className="p-3 text-sm text-gray-400">Loading…</p>
                  ) : filteredProducts.length === 0 ? (
                    <p className="p-3 text-sm text-gray-400">No matches found</p>
                  ) : (
                    filteredProducts.map((p) => (
                      <button
                        key={p}
                        type="button"
                        onClick={() => { setSelectedProduct(p); setProductSearch(p); setSubmitted(false); setDropdownOpen(false); }}
                        className={cn(
                          "w-full text-left px-3 py-2 text-sm truncate transition-colors",
                          p === selectedProduct
                            ? "bg-brand-50 dark:bg-brand-900/30 text-brand-700 dark:text-brand-300 font-medium"
                            : "hover:bg-gray-50 dark:hover:bg-gray-700"
                        )}
                      >
                        {p}
                      </button>
                    ))
                  )}
                </div>
              )}

              {selectedProduct && (
                <p className="text-xs text-green-600 dark:text-green-400 flex items-center gap-1">
                  <span>✓</span> {selectedProduct}
                </p>
              )}
            </div>
          )}

          {/* Customer selector */}
          {needsCustomer && (
            <div className="space-y-1">
              <label className="text-sm font-medium text-gray-700 dark:text-gray-300">
                Customer ID
              </label>
              <select
                value={selectedCustomer}
                onChange={(e) => { setSelectedCustomer(e.target.value); setSubmitted(false); }}
                className="w-full px-3 py-2 rounded-lg border bg-gray-50 dark:bg-gray-800 text-sm focus:outline-none focus:ring-2 focus:ring-brand-500"
              >
                <option value="">Select a customer...</option>
                {customers.map((c) => (
                  <option key={c} value={c}>Customer #{c}</option>
                ))}
              </select>
            </div>
          )}
        </div>

        <div className="flex flex-wrap items-center justify-between gap-4">
          {/* Number of results */}
          <div className="flex items-center gap-2">
            <span className="text-xs text-gray-500 dark:text-gray-400 font-medium">Results:</span>
            {[5, 10, 15].map((n) => (
              <button
                key={n}
                type="button"
                onClick={() => { setNumResults(n); setSubmitted(false); }}
                className={cn(
                  "w-9 h-8 rounded-lg text-xs font-semibold transition-all",
                  numResults === n
                    ? "bg-brand-600 text-white shadow-sm"
                    : "bg-gray-100 dark:bg-gray-800 text-gray-600 dark:text-gray-300 hover:bg-gray-200 dark:hover:bg-gray-700"
                )}
              >
                {n}
              </button>
            ))}
          </div>

          <button
            type="submit"
            disabled={
              (needsProduct && !selectedProduct) ||
              (mode === "collaborative" && !selectedCustomer)
            }
            className="px-6 py-2.5 bg-brand-600 hover:bg-brand-700 disabled:bg-gray-200 disabled:text-gray-400 disabled:cursor-not-allowed text-white rounded-xl text-sm font-semibold transition-all shadow-sm hover:shadow-md"
          >
            Get Recommendations
          </button>
        </div>
      </form>

      {/* Results */}
      {submitted && (
        <div>
          <div className="flex flex-wrap items-center justify-between gap-3 mb-4">
            <h2 className="text-lg font-semibold text-gray-900 dark:text-white flex items-center gap-2">
              <span className="w-1 h-5 bg-brand-600 rounded-full inline-block" />
              Results
              {recs.length > 0 && (
                <span className="text-xs font-medium bg-brand-50 dark:bg-brand-900/20 text-brand-600 px-2 py-0.5 rounded-full">
                  {recs.length}
                </span>
              )}
            </h2>
            {recs.length > 0 && (
              <div className="flex items-center gap-2">
                <span className="text-xs text-gray-400">Sort:</span>
                {(["score", "price_asc", "price_desc"] as const).map((s) => (
                  <button
                    key={s}
                    onClick={() => setSortBy(s)}
                    className={cn(
                      "px-2.5 py-1 rounded-lg text-xs font-medium transition-all",
                      sortBy === s
                        ? "bg-brand-600 text-white"
                        : "bg-gray-100 dark:bg-gray-800 text-gray-600 dark:text-gray-300 hover:bg-gray-200 dark:hover:bg-gray-700"
                    )}
                  >
                    {s === "score" ? "Best Match" : s === "price_asc" ? "Price ↑" : "Price ↓"}
                  </button>
                ))}
              </div>
            )}
          </div>

          {activeQuery.isLoading && (
            <div className="grid grid-cols-1 sm:grid-cols-2 lg:grid-cols-3 gap-4">
              {Array.from({ length: 5 }).map((_, i) => <ProductCardSkeleton key={i} />)}
            </div>
          )}

          {activeQuery.isError && (
            <div className="flex items-center justify-between gap-3 p-4 bg-red-50 dark:bg-red-900/20 border border-red-200 dark:border-red-800 rounded-xl">
              <div className="flex items-center gap-3">
                <AlertCircle className="h-5 w-5 text-red-500 flex-shrink-0" />
                <p className="text-sm text-red-700 dark:text-red-300">
                  Failed to load recommendations. Make sure the backend is running.
                </p>
              </div>
              <button
                onClick={() => activeQuery.refetch()}
                className="text-xs font-semibold text-red-600 dark:text-red-400 hover:underline flex-shrink-0"
              >
                Retry
              </button>
            </div>
          )}

          {!activeQuery.isLoading && !activeQuery.isError && recs.length === 0 && (
            <p className="text-gray-500 dark:text-gray-400 text-sm">
              No recommendations found. Try a different product or customer.
            </p>
          )}

          {sortedRecs.length > 0 && (
            <div className="grid grid-cols-1 sm:grid-cols-2 lg:grid-cols-3 gap-4">
              {sortedRecs.map((rec, i) => {
                const score =
                  "SimilarityScore" in rec ? (rec as { SimilarityScore: number }).SimilarityScore
                  : "HybridScore" in rec    ? (rec as { HybridScore: number }).HybridScore
                  : "Score" in rec          ? (rec as { Score: number }).Score
                  : undefined;

                const scoreLabel =
                  mode === "content" ? "Similarity"
                  : mode === "collaborative" ? "Match"
                  : "Hybrid Score";

                return (
                  <ProductCard
                    key={rec.StockCode + i}
                    rank={i + 1}
                    stockCode={rec.StockCode}
                    description={rec.Description}
                    unitPrice={rec.UnitPrice}
                    score={score}
                    scoreLabel={scoreLabel}
                    reason={rec.Reason}
                    onClick={() => setDrawerProduct({
                      stockCode: rec.StockCode,
                      description: rec.Description,
                      unitPrice: rec.UnitPrice,
                      score,
                      scoreLabel,
                      reason: rec.Reason,
                    })}
                  />
                );
              })}
            </div>
          )}
        </div>
      )}

      <ProductDrawer
        product={drawerProduct}
        onClose={() => setDrawerProduct(null)}
      />
    </div>
  );
}
