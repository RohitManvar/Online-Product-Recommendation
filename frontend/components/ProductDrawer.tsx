"use client";

import { useEffect } from "react";
import { useQuery } from "@tanstack/react-query";
import { getContentRecommendations } from "@/lib/api";
import { formatCurrency, formatPercent, cn } from "@/lib/utils";
import { X, Tag, DollarSign, TrendingUp, Sparkles, Package } from "lucide-react";
import { ProductCardSkeleton } from "@/components/ui/Skeleton";

interface DrawerProduct {
  stockCode: string;
  description: string;
  unitPrice: number;
  score?: number;
  scoreLabel?: string;
  reason?: string;
}

interface ProductDrawerProps {
  product: DrawerProduct | null;
  onClose: () => void;
}

export function ProductDrawer({ product, onClose }: ProductDrawerProps) {
  // Close on Escape key
  useEffect(() => {
    function onKey(e: KeyboardEvent) {
      if (e.key === "Escape") onClose();
    }
    document.addEventListener("keydown", onKey);
    return () => document.removeEventListener("keydown", onKey);
  }, [onClose]);

  // Lock body scroll while open
  useEffect(() => {
    if (product) document.body.style.overflow = "hidden";
    else document.body.style.overflow = "";
    return () => { document.body.style.overflow = ""; };
  }, [product]);

  const { data, isLoading, isError } = useQuery({
    queryKey: ["recommend", "content", product?.description],
    queryFn: () => getContentRecommendations(product!.description),
    enabled: !!product,
  });

  const recs = data?.recommendations ?? [];

  return (
    <>
      {/* Backdrop */}
      <div
        onClick={onClose}
        className={cn(
          "fixed inset-0 bg-black/40 backdrop-blur-sm z-40 transition-opacity duration-300",
          product ? "opacity-100" : "opacity-0 pointer-events-none"
        )}
      />

      {/* Drawer panel */}
      <div
        className={cn(
          "fixed top-0 right-0 h-full w-full sm:w-[480px] bg-white dark:bg-gray-900 z-50 shadow-2xl flex flex-col transition-transform duration-300 ease-in-out",
          product ? "translate-x-0" : "translate-x-full"
        )}
      >
        {/* Header */}
        <div className="flex items-center justify-between px-6 py-4 border-b">
          <div className="flex items-center gap-2">
            <div className="p-1.5 bg-brand-50 dark:bg-brand-900/30 rounded-lg">
              <Package className="h-4 w-4 text-brand-600" />
            </div>
            <span className="font-semibold text-gray-900 dark:text-white">Product Details</span>
          </div>
          <button
            onClick={onClose}
            className="p-2 rounded-lg hover:bg-gray-100 dark:hover:bg-gray-800 transition-colors"
          >
            <X className="h-4 w-4 text-gray-500" />
          </button>
        </div>

        {/* Scrollable content */}
        <div className="flex-1 overflow-y-auto">
          {product && (
            <>
              {/* Product info card */}
              <div className="p-6 border-b bg-gray-50 dark:bg-gray-800/50">
                <h2 className="text-lg font-bold text-gray-900 dark:text-white leading-snug">
                  {product.description}
                </h2>

                <div className="mt-4 flex flex-wrap gap-3">
                  {/* Stock code */}
                  <div className="flex items-center gap-1.5 bg-white dark:bg-gray-800 border rounded-lg px-3 py-2">
                    <Tag className="h-3.5 w-3.5 text-gray-400" />
                    <span className="text-xs font-mono text-gray-600 dark:text-gray-300">{product.stockCode}</span>
                  </div>

                  {/* Price */}
                  <div className="flex items-center gap-1.5 bg-white dark:bg-gray-800 border rounded-lg px-3 py-2">
                    <DollarSign className="h-3.5 w-3.5 text-emerald-500" />
                    <span className="text-xs font-bold text-emerald-600 dark:text-emerald-400">
                      {formatCurrency(product.unitPrice)}
                    </span>
                  </div>

                  {/* Score */}
                  {product.score !== undefined && (
                    <div className="flex items-center gap-1.5 bg-white dark:bg-gray-800 border rounded-lg px-3 py-2">
                      <TrendingUp className="h-3.5 w-3.5 text-green-500" />
                      <span className="text-xs font-semibold text-green-600 dark:text-green-400">
                        {product.scoreLabel ?? "Score"}: {formatPercent(product.score)}
                      </span>
                    </div>
                  )}
                </div>

                {/* Reason */}
                {product.reason && (
                  <div className="mt-3 flex items-start gap-2 text-xs text-gray-500 dark:text-gray-400 italic">
                    <Sparkles className="h-3.5 w-3.5 text-brand-400 flex-shrink-0 mt-0.5" />
                    {product.reason}
                  </div>
                )}
              </div>

              {/* Similar products */}
              <div className="p-6 space-y-3">
                <h3 className="text-sm font-semibold text-gray-900 dark:text-white flex items-center gap-2">
                  <span className="w-1 h-4 bg-brand-600 rounded-full inline-block" />
                  Similar Products
                </h3>

                {isLoading && (
                  <div className="space-y-3">
                    {Array.from({ length: 4 }).map((_, i) => (
                      <ProductCardSkeleton key={i} />
                    ))}
                  </div>
                )}

                {isError && (
                  <p className="text-sm text-red-500">Failed to load recommendations.</p>
                )}

                {!isLoading && !isError && recs.length === 0 && (
                  <p className="text-sm text-gray-400">No similar products found.</p>
                )}

                {recs.map((rec, i) => (
                  <div
                    key={rec.StockCode + i}
                    className="bg-gray-50 dark:bg-gray-800 rounded-xl border p-4 space-y-2"
                  >
                    <div className="flex items-start justify-between gap-3">
                      <div className="flex items-start gap-2.5 flex-1 min-w-0">
                        <span className="flex-shrink-0 w-6 h-6 rounded-full bg-gradient-to-br from-brand-500 to-brand-700 text-white text-xs font-bold flex items-center justify-center">
                          {i + 1}
                        </span>
                        <div className="flex-1 min-w-0">
                          <p className="text-sm font-semibold text-gray-900 dark:text-white leading-snug line-clamp-2">
                            {rec.Description}
                          </p>
                          <div className="flex items-center gap-1 mt-1">
                            <Tag className="h-3 w-3 text-gray-400" />
                            <span className="text-xs text-gray-400 font-mono">{rec.StockCode}</span>
                          </div>
                        </div>
                      </div>
                      <span className="flex-shrink-0 text-sm font-bold text-gray-900 dark:text-white bg-white dark:bg-gray-700 px-2.5 py-1 rounded-lg border">
                        {formatCurrency(rec.UnitPrice)}
                      </span>
                    </div>

                    <div className="flex items-center gap-1.5 bg-green-50 dark:bg-green-900/20 px-2.5 py-1 rounded-full w-fit">
                      <TrendingUp className="h-3 w-3 text-green-500" />
                      <span className="text-xs text-green-700 dark:text-green-400 font-semibold">
                        Similarity: {formatPercent(rec.SimilarityScore)}
                      </span>
                    </div>
                  </div>
                ))}
              </div>
            </>
          )}
        </div>
      </div>
    </>
  );
}
