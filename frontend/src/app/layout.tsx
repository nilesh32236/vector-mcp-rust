import type { Metadata } from "next";
import "./globals.css";
import { ThemeProvider } from "next-themes";
import { ToasterClient } from "@/components/ToasterClient";

export const metadata: Metadata = {
  title: "VectorWiki - Local Code Intelligence",
  description: "Semantic search and documentation explorer for your codebase",
};

export default function RootLayout({
  children,
}: Readonly<{
  children: React.ReactNode;
}>) {
  return (
    <html lang="en" className="h-full" suppressHydrationWarning>
      <body className="min-h-full flex flex-col antialiased bg-background text-foreground">
        <ThemeProvider
          attribute="class"
          defaultTheme="dark"
          enableSystem
          disableTransitionOnChange={false}
        >
          {children}
          <ToasterClient />
        </ThemeProvider>
      </body>
    </html>
  );
}
