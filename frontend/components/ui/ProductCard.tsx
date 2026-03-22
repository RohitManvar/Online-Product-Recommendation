"use client";

import { useState } from "react";
import { cn, formatCurrency, formatPercent } from "@/lib/utils";
import { Tag, TrendingUp, Sparkles, Check } from "lucide-react";

interface ProductCardProps {
  rank: number;
  stockCode: string;
  description: string;
  unitPrice: number;
  score?: number;
  scoreLabel?: string;
  reason?: string;
  className?: string;
  onClick?: () => void;
}

export function ProductCard({
  rank,
  stockCode,
  description,
  unitPrice,
  score,
  scoreLabel = "Score",
  reason,
  className,
  onClick,
}: ProductCardProps) {
  const [copied, setCopied] = useState(false);

  function handleCopy(e: React.MouseEvent) {
    e.stopPropagation();
    navigator.clipboard.writeText(stockCode).then(() => {
      setCopied(true);
      setTimeout(() => setCopied(false), 1500);
    });
  }

  return (
    <div
      onClick={onClick}
      className={cn(
        "bg-white dark:bg-gray-900 rounded-2xl border p-4 shadow-sm hover:shadow-lg hover:-translate-y-0.5 transition-all duration-200",
        onClick && "cursor-pointer",
        className
      )}
    >
      <div className="flex items-start justify-between gap-3">
        <div className="flex items-start gap-3 flex-1 min-w-0">
          {/* Rank badge */}
          <span className="flex-shrink-0 w-8 h-8 rounded-full bg-gradient-to-br from-brand-500 to-brand-700 text-white text-sm font-bold flex items-center justify-center shadow-sm">
            {rank}
          </span>

          <div className="flex-1 min-w-0">
            <p className="font-semibold text-gray-900 dark:text-white text-sm leading-snug line-clamp-2">
              {description}
            </p>
            <button
              onClick={handleCopy}
              title="Copy stock code"
              className="flex items-center gap-1 mt-1.5 group"
            >
              {copied ? (
                <Check className="h-3 w-3 text-green-500" />
              ) : (
                <Tag className="h-3 w-3 text-gray-400 group-hover:text-brand-500 transition-colors" />
              )}
              <span className={cn(
                "text-xs font-mono transition-colors",
                copied ? "text-green-500" : "text-gray-400 group-hover:text-brand-500"
              )}>
                {copied ? "Copied!" : stockCode}
              </span>
            </button>
          </div>
        </div>

        {/* Price */}
        <div className="flex-shrink-0">
          <span className="text-sm font-bold text-gray-900 dark:text-white bg-gray-50 dark:bg-gray-800 px-2.5 py-1 rounded-lg">
            {formatCurrency(unitPrice)}
          </span>
        </div>
      </div>

      {/* Score + reason */}
      {(score !== undefined || reason) && (
        <div className="mt-3 pt-3 border-t border-gray-100 dark:border-gray-800 flex items-center justify-between gap-2">
          {score !== undefined && (
            <div className="flex items-center gap-1.5 bg-green-50 dark:bg-green-900/20 px-2.5 py-1 rounded-full">
              <TrendingUp className="h-3 w-3 text-green-500" />
              <span className="text-xs text-green-700 dark:text-green-400 font-semibold">
                {scoreLabel}: {formatPercent(score)}
              </span>
            </div>
          )}

          {reason && (
            <div className="flex items-center gap-1 ml-auto">
              <Sparkles className="h-3 w-3 text-brand-400 flex-shrink-0" />
              <span className="text-xs text-gray-400 dark:text-gray-500 italic truncate max-w-[160px]">
                {reason}
              </span>
            </div>
          )}
        </div>
      )}
    </div>
  );
}
