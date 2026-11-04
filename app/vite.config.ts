import { defineConfig } from "vite";

// Relative base so the same build runs from GitHub Pages and offline from disk
// at the fair (plan Step 5.5).
export default defineConfig({ base: "./" });
