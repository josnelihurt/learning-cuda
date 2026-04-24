/**
 * Vite SPA (React only): build emits dist/index.html.
 * dist/.vite/manifest.json uses HTML path keys for Nginx static copies.
 */
import { defineConfig, loadEnv } from 'vite';
import type { Connect, Plugin } from 'vite';
import { resolve } from 'path';
import { execSync } from 'child_process';
import { createReadStream, existsSync, readFileSync } from 'node:fs';
import { extname, join } from 'node:path';
import react from '@vitejs/plugin-react';

function prettyFrontendRoutesPlugin(): Plugin {
  const rewrite: Connect.NextHandleFunction = (req, res, next) => {
    if (req.method !== 'GET' || !req.url) {
      next();
      return;
    }
    const q = req.url.indexOf('?');
    let pathname = q === -1 ? req.url : req.url.slice(0, q);
    const search = q === -1 ? '' : req.url.slice(q);
    if (pathname.length > 1 && pathname.endsWith('/')) {
      pathname = pathname.slice(0, -1);
    }
    // Root path serves React (index.html)
    if (pathname === '/' || pathname === '') {
      req.url = '/index.html' + search;
    }
    next();
  };

  return {
    name: 'pretty-frontend-routes',
    configureServer(server) {
      server.middlewares.use(rewrite);
    },
    configurePreviewServer(server) {
      server.middlewares.use(rewrite);
    },
  };
}

function gitVersionPlugin() {
  let version = 'dev';
  let versionNumber = 'dev';
  let branch = 'unknown';
  let buildTime = new Date().toISOString();

  return {
    name: 'git-version',
    config() {
      // Read VERSION file for version number (vite.config.ts is in src/front-end/)
      try {
        const versionPath = join(__dirname, 'VERSION');
        if (existsSync(versionPath)) {
          versionNumber = readFileSync(versionPath, 'utf-8').trim();
        }
      } catch {
        console.warn('VERSION file not found, using fallback version number');
      }

      // Check environment variables first (for Docker builds)
      if (process.env.VITE_COMMIT_HASH) {
        version = process.env.VITE_COMMIT_HASH;
      } else {
        try {
          version = execSync('git rev-parse --short HEAD', { encoding: 'utf-8' }).trim();
        } catch {
          console.warn('Git not available, using fallback version');
        }
      }

      if (process.env.VITE_BRANCH) {
        branch = process.env.VITE_BRANCH;
      } else {
        try {
          branch = execSync('git rev-parse --abbrev-ref HEAD', { encoding: 'utf-8' }).trim();
        } catch {
          console.warn('Git not available, using fallback branch');
        }
      }

      return {
        define: {
          __APP_VERSION__: JSON.stringify(version),
          __APP_VERSION_NUMBER__: JSON.stringify(versionNumber),
          __APP_BRANCH__: JSON.stringify(branch),
          __BUILD_TIME__: JSON.stringify(buildTime),
        },
      };
    },
  };
}

const DATA_MIME: Record<string, string> = {
  '.png': 'image/png',
  '.jpg': 'image/jpeg',
  '.jpeg': 'image/jpeg',
  '.mp4': 'video/mp4',
  '.webm': 'video/webm',
};

// Serves ./data/** at /data/** during Vite dev (nginx handles this in production).
function serveDataDirPlugin(): Plugin {
  return {
    name: 'serve-data-dir',
    configureServer(server) {
      server.middlewares.use('/data', (req, res, next) => {
        const filePath = join(__dirname, '../../data', req.url ?? '/');
        if (!existsSync(filePath)) { next(); return; }
        const ct = DATA_MIME[extname(filePath).toLowerCase()];
        if (ct) res.setHeader('Content-Type', ct);
        createReadStream(filePath).pipe(res);
      });
    },
  };
}

export default defineConfig(({ mode }) => {
  const env = loadEnv(mode, process.cwd(), '');
  const apiTarget = env.VITE_API_ORIGIN || 'https://localhost:8443';

  const backendHTTP = {
    target: apiTarget,
    changeOrigin: true,
    secure: false,
  };

  return {
    root: './',
    base: '/',

    publicDir: 'public',

    build: {
      outDir: 'dist',
      emptyOutDir: true,
      sourcemap: true,
      manifest: true,
      rollupOptions: {
        input: {
          main: resolve(__dirname, 'index.html'),
        },
      },
      minify: 'esbuild',
      target: 'es2020',
    },

    plugins: [gitVersionPlugin(), prettyFrontendRoutesPlugin(), serveDataDirPlugin(), react()],

    optimizeDeps: {
      include: [
        '@opentelemetry/api',
        '@opentelemetry/sdk-trace-web',
        '@opentelemetry/sdk-trace-base',
        '@opentelemetry/resources',
        '@opentelemetry/core',
        '@opentelemetry/exporter-trace-otlp-http',
      ],
    },

    server: {
      host: '0.0.0.0',
      port: 3000,
      https: {
        key: resolve(__dirname, '../../.secrets/localhost+2-key.pem'),
        cert: resolve(__dirname, '../../.secrets/localhost+2.pem'),
      },
      cors: true,
      strictPort: true,
      proxy: {
        '/api': backendHTTP,
        '/cuda_learning': backendHTTP,
        '/com.jrb': backendHTTP,
      },
    },

    preview: {
      host: '127.0.0.1',
      port: 4173,
      strictPort: true,
      https: false,
      proxy: {
        '/api': backendHTTP,
        '/cuda_learning': backendHTTP,
        '/com.jrb': backendHTTP,
      },
    },

    resolve: {
      alias: {
        '@': resolve(__dirname, './src'),
      },
    },
  };
});
