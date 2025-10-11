import { defineConfig } from 'vite';
import { resolve } from 'path';
import { execSync } from 'child_process';

function gitVersionPlugin() {
  let version = 'dev';
  let branch = 'unknown';
  let buildTime = new Date().toISOString();

  return {
    name: 'git-version',
    config() {
      try {
        version = execSync('git rev-parse --short HEAD', { encoding: 'utf-8' }).trim();
        branch = execSync('git rev-parse --abbrev-ref HEAD', { encoding: 'utf-8' }).trim();
      } catch (e) {
        console.warn('Git not available, using fallback version');
      }
      
      return {
        define: {
          __APP_VERSION__: JSON.stringify(version),
          __APP_BRANCH__: JSON.stringify(branch),
          __BUILD_TIME__: JSON.stringify(buildTime),
        },
      };
    },
  };
}

export default defineConfig({
  root: './',
  base: '/',
  
  build: {
    outDir: 'static/js/dist',
    emptyOutDir: true,
    sourcemap: true,
    manifest: true,
    
    rollupOptions: {
      input: resolve(__dirname, 'src/main.ts'),
      output: {
        entryFileNames: 'app.[hash].js',
        chunkFileNames: 'chunks/[name].[hash].js',
        assetFileNames: 'assets/[name].[hash][extname]',
      },
    },
    
    minify: 'esbuild',
    target: 'es2020',
  },
  
  plugins: [gitVersionPlugin()],
  
  server: {
    host: '0.0.0.0',
    port: 3000,
    https: {
      key: resolve(__dirname, '../../.secrets/localhost+2-key.pem'),
      cert: resolve(__dirname, '../../.secrets/localhost+2.pem'),
    },
    cors: true,
    strictPort: true,
    hmr: {
      protocol: 'wss',
      clientPort: 8443,
      path: '/@vite/hmr',
    },
  },
  
  resolve: {
    alias: {
      '@': resolve(__dirname, './src'),
    },
  },
});

