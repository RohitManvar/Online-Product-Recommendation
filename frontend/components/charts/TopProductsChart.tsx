"use client";

import {
  BarChart,
  Bar,
  XAxis,
  YAxis,
  CartesianGrid,
  Tooltip,
  ResponsiveContainer,
  Cell,
} from "recharts";
import type { ProductStat } from "@/types";
import { formatCurrency, formatNumber } from "@/lib/utils";

interface Props {
  data: ProductStat[];
  mode: "revenue" | "quantity";
}

const COLORS = [
  "#2563eb", "#3b82f6", "#60a5fa", "#93c5fd",
  "#1d4ed8", "#1e40af", "#7c3aed", "#8b5cf6",
  "#4f46e5", "#6366f1",
];

export function TopProductsChart({ data, mode }: Props) {
  const formatted = data.map((d) => ({
    ...d,
    label: d.Description.length > 22 ? d.Description.slice(0, 22) + "…" : d.Description,
  }));

  return (
    <ResponsiveContainer width="100%" height={320}>
      <BarChart
        data={formatted}
        layout="vertical"
        margin={{ top: 5, right: 30, left: 10, bottom: 5 }}
      >
        <CartesianGrid strokeDasharray="3 3" stroke="#f1f5f9" horizontal={false} />
        <XAxis
          type="number"
          tickFormatter={(v) =>
            mode === "revenue" ? `$${(v / 1000).toFixed(0)}k` : formatNumber(v)
          }
          tick={{ fontSize: 11, fill: "#94a3b8" }}
          axisLine={false}
          tickLine={false}
        />
        <YAxis
          dataKey="label"
          type="category"
          width={175}
          tick={{ fontSize: 11, fill: "#64748b" }}
          axisLine={false}
          tickLine={false}
        />
        <Tooltip
          formatter={(value: number) =>
            mode === "revenue"
              ? [formatCurrency(value), "Revenue"]
              : [formatNumber(value), "Units Sold"]
          }
          labelStyle={{ fontWeight: 600, color: "#1e293b" }}
          contentStyle={{ borderRadius: "12px", border: "1px solid #e2e8f0", boxShadow: "0 4px 6px -1px rgb(0 0 0 / 0.1)" }}
        />
        <Bar
          dataKey={mode === "revenue" ? "TotalPrice" : "Quantity"}
          radius={[0, 6, 6, 0]}
        >
          {formatted.map((_, index) => (
            <Cell key={`cell-${index}`} fill={COLORS[index % COLORS.length]} />
          ))}
        </Bar>
      </BarChart>
    </ResponsiveContainer>
  );
}
