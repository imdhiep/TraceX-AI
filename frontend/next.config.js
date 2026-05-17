/** @type {import('next').NextConfig} */
const rawApi = (process.env.NEXT_PUBLIC_API_BASE_URL || "").replace(/\/$/, "");

const nextConfig = {
  output: 'standalone',
  env: {
    NEXT_PUBLIC_API_BASE_URL: '/api-gw',
  },
  async rewrites() {
    if (!rawApi || rawApi.startsWith("/")) return [];
    const mediaUpstream = rawApi.replace(/\/api\/v\d+$/i, "");
    return [
      {
        source: "/api-gw/:path*",
        destination: `${rawApi}/:path*`,
      },
      {
        source: "/static/:path*",
        destination: `${mediaUpstream}/static/:path*`,
      },
    ];
  },
};
module.exports = nextConfig;
