import { defineConfig } from "vite";
import react from "@vitejs/plugin-react";
import mdx from "@mdx-js/rollup";

// For GitHub Pages project sites, assets must be served from /<repo>/.
// We make this dynamic so local dev still works:
export default defineConfig(() => ({
  plugins: [mdx(), react()],
  base: process.env.GHPAGES === "1" ? `/${process.env.REPO_NAME || ""}/` : "/",
}));
