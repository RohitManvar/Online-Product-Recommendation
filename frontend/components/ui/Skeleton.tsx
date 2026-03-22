import { cn } from "@/lib/utils";

export function Skeleton({ className }: { className?: string }) {
  return (
    <div
      className={cn(
        "animate-pulse rounded-md bg-gray-200 dark:bg-gray-700",
        className
      )}
    />
  );
}

export function ProductCardSkeleton() {
  return (
    <div className="bg-white dark:bg-gray-900 rounded-xl border p-4 shadow-sm">
      <div className="flex items-start gap-3">
        <Skeleton className="w-8 h-8 rounded-full flex-shrink-0" />
        <div className="flex-1 space-y-2">
          <Skeleton className="h-4 w-full" />
          <Skeleton className="h-3 w-1/3" />
        </div>
        <Skeleton className="h-5 w-14 flex-shrink-0" />
      </div>
      <div className="mt-3 flex gap-2">
        <Skeleton className="h-3 w-24" />
        <Skeleton className="h-3 w-32 ml-auto" />
      </div>
    </div>
  );
}
