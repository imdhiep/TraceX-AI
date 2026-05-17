/** @type {import('next').NextConfig} */
// process.env values are read at build time. The browser bundle gets the
// values from `env` below; the rewrites() block uses the raw build-time env.
const rawApi = (process.env.NEXT_PUBLIC_API_BASE_URL || "").replace(/\/$/, "");
const rawSearch = (process.env.NEXT_PUBLIC_SEARCH_API_BASE_URL || "").replace(/\/$/, "");
const searchConfigured = rawSearch && !rawSearch.startsWith("/");

const nextConfig = {
  output: 'standalone',
  env: {
    NEXT_PUBLIC_API_BASE_URL: '/api-gw',
    // Only route search through a dedicated gateway when the deploy env
    // actually configured one. Otherwise fall back to the metadata gateway
    // so existing deploys (e.g. Coolify without the new var) keep working.
    NEXT_PUBLIC_SEARCH_API_BASE_URL: searchConfigured ? '/search-gw' : '/api-gw',
  },
  async rewrites() {
    if (!rawApi || rawApi.startsWith("/")) return [];
    const mediaUpstream = rawApi.replace(/\/api\/v\d+$/i, "");
    const rewrites = [
      {
        source: "/api-gw/:path*",
        destination: `${rawApi}/:path*`,
      },
      {
        source: "/static/:path*",
        destination: `${mediaUpstream}/static/:path*`,
      },
    ];
    if (searchConfigured) {
      rewrites.push({
        source: "/search-gw/:path*",
        destination: `${rawSearch}/:path*`,
      });
    }
    return rewrites;
  },
};
module.exports = nextConfig;
