import type { Metadata } from "next";
import { Inter } from "next/font/google";
import "./globals.css";
import Sidebar from "@/components/Sidebar";
import TopNav from "@/components/TopNav";

const inter = Inter({ subsets: ["latin"] });

export const metadata: Metadata = {
  title: "VisionInspect AI",
  description: "Manufacturing Defect Detection & Quality Inspection System",
};

export default function RootLayout({
  children,
}: Readonly<{
  children: React.ReactNode;
}>) {
  return (
    <html lang="en">
      <body className={`${inter.className} bg-[#0f172a] text-slate-200 antialiased selection:bg-blue-500/30`}>
        <div className="flex min-h-screen">
          <Sidebar />
          <div className="flex-1 flex flex-col ml-0 md:ml-64 transition-all duration-300">
            <TopNav />
            <main className="flex-1 p-8 pt-28">
              {children}
            </main>
          </div>
        </div>
      </body>
    </html>
  );
}
