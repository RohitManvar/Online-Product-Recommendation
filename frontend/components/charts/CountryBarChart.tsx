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
import { formatCurrency } from "@/lib/utils";
import type { GeoSaleEntry } from "@/types";

const COLORS = [
  "#2563eb", "#3b82f6", "#60a5fa", "#93c5fd", "#bfdbfe",
  "#1d4ed8", "#4f46e5", "#7c3aed", "#8b5cf6", "#a78bfa",
];

export function CountryBarChart({ data }: { data: GeoSaleEntry[] }) {
  const top10 = data.slice(0, 10);

  return (
    <ResponsiveContainer width="100%" height={280}>
      <BarChart data={top10} margin={{ top: 5, right: 20, left: 10, bottom: 50 }}>
        <CartesianGrid strokeDasharray="3 3" stroke="#f1f5f9" vertical={false} />
        <XAxis
          dataKey="country"
          tick={{ fontSize: 10, fill: "#64748b" }}
          angle={-40}
          textAnchor="end"
          axisLine={false}
          tickLine={false}
        />
        <YAxis
          tickFormatter={(v) => `$${(v / 1000).toFixed(0)}k`}
          tick={{ fontSize: 11, fill: "#94a3b8" }}
          axisLine={false}
          tickLine={false}
        />
        <Tooltip
          formatter={(value: number) => [formatCurrency(value), "Revenue"]}
          labelStyle={{ fontWeight: 600, color: "#1e293b" }}
          contentStyle={{ borderRadius: "12px", border: "1px solid #e2e8f0", boxShadow: "0 4px 6px -1px rgb(0 0 0 / 0.1)" }}
        />
        <Bar dataKey="revenue" radius={[6, 6, 0, 0]}>
          {top10.map((_, index) => (
            <Cell key={`cell-${index}`} fill={COLORS[index % COLORS.length]} />
          ))}
        </Bar>
      </BarChart>
    </ResponsiveContainer>
  );
}
