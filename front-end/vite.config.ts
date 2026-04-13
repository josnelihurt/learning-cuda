import { defineConfig, loadEnv } from 'vite';
import { resolve } from 'path';
import { execSync } from 'child_process';
import react from '@vitejs/plugin-react';

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
      } catch {
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

export default defineConfig(({ mode }) => {
  const env = loadEnv(mode, process.cwd(), '');
  const apiTarget = env.VITE_API_ORIGIN || 'https://localhost:8443';

  const backendHTTP = {
    target: apiTarget,
    changeOrigin: true,
    secure: false,
  };

  const backendWS = {
    ...backendHTTP,
    ws: true,
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
          react: resolve(__dirname, 'react.html'),
        },
      },
      minify: 'esbuild',
      target: 'es2020',
    },

    plugins: [gitVersionPlugin(), react()],

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
        key: resolve(__dirname, '../.secrets/localhost+2-key.pem'),
        cert: resolve(__dirname, '../.secrets/localhost+2.pem'),
      },
      cors: true,
      strictPort: true,
      proxy: {
        '/api': backendHTTP,
        '/flipt': backendHTTP,
        '/cuda_learning': backendHTTP,
        '/com.jrb': backendHTTP,
        '/ws': backendWS,
        '/grpc': {
          target: 'http://localhost:60061',
          changeOrigin: true,
          secure: false,
          rewrite: (path) => path.replace(/^\/grpc/, ''),
          configure: (proxy) => {
            proxy.on('proxyReq', (proxyReq, req) => {
              const contentType = req.headers['content-type'];
              if (contentType?.includes('application/grpc-web')) {
                proxyReq.setHeader('Content-Type', 'application/grpc');
                proxyReq.setHeader('grpc-encoding', 'identity');
                proxyReq.setHeader('grpc-accept-encoding', 'identity,deflate,gzip');
              }
            });
            proxy.on('proxyRes', (proxyRes, req) => {
              if (req.headers['accept']?.includes('application/grpc-web')) {
                proxyRes.headers['content-type'] = 'application/grpc-web+proto';
              }
            });
          },
        },
      },
    },

    resolve: {
      alias: {
        '@': resolve(__dirname, './src'),
      },
    },
  };
});
