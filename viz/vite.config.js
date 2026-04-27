import { defineConfig } from "vite";

export default defineConfig({
  base: "/viz/",
  server: {
    proxy: {
      "/api": "http://localhost:7860",
    },
  },
});
