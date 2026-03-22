import { cn } from "@/lib/utils";
import type { LucideIcon } from "lucide-react";

interface KpiCardProps {
  title: string;
  value: string;
  icon: LucideIcon;
  iconColor?: string;
  iconBg?: string;
  className?: string;
}

export function KpiCard({ title, value, icon: Icon, iconColor = "text-brand-600", iconBg = "bg-brand-50 dark:bg-brand-900/30", className }: KpiCardProps) {
  return (
    <div
      className={cn(
        "bg-white dark:bg-gray-900 rounded-2xl border p-5 flex items-center gap-4 shadow-sm hover:shadow-md transition-shadow",
        className
      )}
    >
      <div className={cn("p-3 rounded-xl flex-shrink-0", iconBg)}>
        <Icon className={cn("h-5 w-5", iconColor)} />
      </div>
      <div className="min-w-0">
        <p className="text-xs font-medium text-gray-500 dark:text-gray-400 uppercase tracking-wide">{title}</p>
        <p className="text-xl font-bold text-gray-900 dark:text-white mt-0.5 truncate">{value}</p>
      </div>
    </div>
  );
}
