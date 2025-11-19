import "./globals.css";
import type { ReactNode } from "react";
import { Plus_Jakarta_Sans } from "next/font/google";

const jakarta = Plus_Jakarta_Sans({
  subsets: ["latin"],
  display: "swap",
  variable: "--font-jakarta",
});

export const metadata = {
  title: "Finance Analytics",
  description: "Earnings analytics dashboard",
};

export default function RootLayout({ children }: { children: ReactNode }) {
  return (
    <html lang="en">
      <body className={`${jakarta.variable} bg-night-900 text-slate-200 min-h-screen`}>
        <div className="prime-bg" />
        <div className="relative z-10">
          {children}
        </div>
      </body>
    </html>
  );
}
