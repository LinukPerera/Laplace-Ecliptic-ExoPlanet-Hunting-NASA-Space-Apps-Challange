import type { NextConfig } from "next";

const nextConfig: NextConfig = {
  /* config options here */
  async headers() {
    return [
      {
        source: '/api/:path*', // Apply to all API routes
        headers: [
          { key: 'Access-Control-Allow-Origin', value: '*' }, // Or specific origin
          { key: 'Access-Control-Allow-Methods', value: 'GET, POST, PUT, DELETE, OPTIONS' },
          { key: 'Access-Control-Allow-Headers', value: 'Content-Type, Authorization' },
          { key: 'Access-Control-Allow-Credentials', value: 'true' }, // If needed
        ],
      },
    ];
  },
};

export default nextConfig;
