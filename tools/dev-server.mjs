import http from "node:http";
import fs from "node:fs";
import path from "node:path";
import { fileURLToPath } from "node:url";

import { runBuild } from "./build-web.mjs";

const __filename = fileURLToPath(import.meta.url);
const __dirname = path.dirname(__filename);
const repoRoot = path.resolve(__dirname, "..");
const webRoot = path.join(repoRoot, "build-web-emscripten", "webapp");

function parseCliOptions(argv) {
  const options = {};
  for (let i = 0; i < argv.length; i += 1) {
    const arg = argv[i];
    if (arg === "--port" || arg === "-p") {
      options.port = argv[i + 1];
      i += 1;
      continue;
    }

    if (arg.startsWith("--port=")) {
      options.port = arg.slice("--port=".length);
      continue;
    }

    if (arg === "--host") {
      options.host = argv[i + 1];
      i += 1;
      continue;
    }

    if (arg.startsWith("--host=")) {
      options.host = arg.slice("--host=".length);
    }
  }
  return options;
}

const cliOptions = parseCliOptions(process.argv.slice(2));
const host = cliOptions.host || process.env.HOST || "127.0.0.1";
const port = Number.parseInt(cliOptions.port || process.env.PORT || "4174", 10);

if (!Number.isInteger(port) || port <= 0 || port > 65535) {
  console.error("Port must be an integer in the range 1-65535.");
  process.exit(1);
}

const mimeTypes = {
  ".css": "text/css; charset=utf-8",
  ".html": "text/html; charset=utf-8",
  ".ico": "image/x-icon",
  ".jpeg": "image/jpeg",
  ".jpg": "image/jpeg",
  ".js": "text/javascript; charset=utf-8",
  ".json": "application/json; charset=utf-8",
  ".map": "application/json; charset=utf-8",
  ".png": "image/png",
  ".svg": "image/svg+xml; charset=utf-8",
  ".txt": "text/plain; charset=utf-8",
  ".wasm": "application/wasm"
};

function sendError(res, statusCode, message) {
  res.writeHead(statusCode, { "Content-Type": "text/plain; charset=utf-8" });
  res.end(`${message}\n`);
}

function resolveRequestPath(urlPathname) {
  const decodedPath = decodeURIComponent(urlPathname);
  const candidate = decodedPath.endsWith("/")
    ? `${decodedPath}index.html`
    : decodedPath;
  const relativePath = candidate.replace(/^\/+/, "");
  const fullPath = path.normalize(path.join(webRoot, relativePath));
  if (!fullPath.startsWith(path.normalize(webRoot + path.sep)) && fullPath !== webRoot) {
    return null;
  }
  return fullPath;
}

async function main() {
  await runBuild();

  const server = http.createServer((req, res) => {
    try {
      const requestUrl = new URL(req.url || "/", `http://${host}:${port}`);
      const filePath = resolveRequestPath(requestUrl.pathname);
      if (!filePath) {
        sendError(res, 403, "Forbidden");
        return;
      }

      fs.stat(filePath, (statError, stats) => {
        if (statError || !stats.isFile()) {
          sendError(res, 404, "Not Found");
          return;
        }

        const ext = path.extname(filePath).toLowerCase();
        const contentType = mimeTypes[ext] || "application/octet-stream";
        res.writeHead(200, {
          "Content-Type": contentType,
          "Cache-Control": "no-cache"
        });

        const stream = fs.createReadStream(filePath);
        stream.on("error", () => sendError(res, 500, "Read Error"));
        stream.pipe(res);
      });
    } catch (error) {
      sendError(res, 500, error.message || "Server Error");
    }
  });

  server.on("error", (error) => {
    if (error && typeof error === "object" && "code" in error) {
      if (error.code === "EADDRINUSE") {
        console.error(`Port ${port} is already in use.`);
        process.exit(1);
      }

      if (error.code === "EPERM" || error.code === "EACCES") {
        console.error(`Could not bind http://${host}:${port}/. Check local sandbox or permissions.`);
        process.exit(1);
      }
    }

    console.error(error.message || String(error));
    process.exit(1);
  });

  server.listen(port, host, () => {
    console.log(`Serving ${webRoot}`);
    console.log(`Open http://${host}:${port}/index.html`);
  });

  const shutdown = () => {
    server.close(() => process.exit(0));
  };

  process.on("SIGINT", shutdown);
  process.on("SIGTERM", shutdown);
}

main().catch((error) => {
  console.error(error.message);
  process.exit(1);
});
