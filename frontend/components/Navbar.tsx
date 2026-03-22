"use client";

import Link from "next/link";
import { usePathname } from "next/navigation";
import { ShoppingCart, BarChart3, Sparkles } from "lucide-react";
import { cn } from "@/lib/utils";

const links = [
  { href: "/", label: "Recommendations", icon: Sparkles },
  { href: "/analytics", label: "Analytics", icon: BarChart3 },
];

export function Navbar() {
  const pathname = usePathname();

  return (
    <nav className="border-b bg-white/80 dark:bg-gray-900/80 backdrop-blur-md sticky top-0 z-50 shadow-sm">
      <div className="max-w-7xl mx-auto px-4 sm:px-6 lg:px-8">
        <div className="flex items-center justify-between h-16">
          {/* Logo */}
          <div className="flex items-center gap-2.5">
            <div className="p-1.5 bg-brand-600 rounded-lg">
              <ShoppingCart className="h-5 w-5 text-white" />
            </div>
            <span className="font-bold text-lg tracking-tight gradient-text">
              RecoEngine
            </span>
          </div>

          {/* Nav links */}
          <div className="flex gap-1">
            {links.map((link) => (
              <Link
                key={link.href}
                href={link.href}
                className={cn(
                  "flex items-center gap-1.5 px-4 py-2 rounded-lg text-sm font-medium transition-all",
                  pathname === link.href
                    ? "bg-brand-600 text-white shadow-sm shadow-brand-200"
                    : "text-gray-600 hover:bg-gray-100 dark:text-gray-300 dark:hover:bg-gray-800"
                )}
              >
                <link.icon className="h-4 w-4" />
                {link.label}
              </Link>
            ))}
          </div>
        </div>
      </div>
    </nav>
  );
}
