/** @type {import('next').NextConfig} */
const nextConfig = {
  output: 'standalone',
  // Override the browser bundle to always call the same-origin /api-gw prefix.
  // The real upstream URL is read from the OS env below (before this override applies).
  env: {
    NEXT_PUBLIC_API_BASE_URL: '/api-gw',
  },
  async rewrites() {
    // process.env here reads the real OS value from .env.local, not the /api-gw override above.
    const upstream = (process.env.NEXT_PUBLIC_API_BASE_URL || "").replace(/\/$/, "");
    if (!upstream || upstream.startsWith("/")) return [];
    const mediaUpstream = upstream.replace(/\/api\/v\d+$/i, "");
    return [
      {
        source: "/api-gw/:path*",
        destination: `${upstream}/:path*`,
      },
      {
        source: "/static/:path*",
        destination: `${mediaUpstream}/static/:path*`,
      },
    ];
  },
};
module.exports = nextConfig;
