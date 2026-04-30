import type { NextConfig } from "next";

const nextConfig: NextConfig = {
  /* config options here */
  output: 'export',
  images: {
    unoptimized: true,
  },
  // Note: rewrites only work in development (next dev) or when using a custom server.
  // When running 'next build' with 'output: export', rewrites are ignored.
  // This is primarily for the 'npm run dev' experience.
  async rewrites() {
    return [
      {
        source: '/api/:path*',
        destination: 'http://localhost:47822/api/:path*',
      },
    ];
  },
};

export default nextConfig;
