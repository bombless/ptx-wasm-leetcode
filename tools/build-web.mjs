import { spawn } from "node:child_process";
import path from "node:path";
import { fileURLToPath } from "node:url";

const __filename = fileURLToPath(import.meta.url);
const __dirname = path.dirname(__filename);
const repoRoot = path.resolve(__dirname, "..");

export function runBuild() {
  const command =
    "source /home/openclaw/.local/emsdk/emsdk_env.sh >/dev/null && " +
    "export EM_CACHE=/tmp/emscripten-cache && " +
    "export EMSDK_PYTHON=/home/openclaw/.local/bin/python3.11 && " +
    "cmake --build build-web-emscripten --target ptx_web -j\"$(nproc)\"";

  return new Promise((resolve, reject) => {
    const child = spawn("/usr/bin/zsh", ["-lc", command], {
      cwd: repoRoot,
      stdio: "inherit"
    });

    child.on("error", reject);
    child.on("exit", (code, signal) => {
      if (code === 0) {
        resolve();
        return;
      }

      if (signal) {
        reject(new Error(`build terminated by signal ${signal}`));
        return;
      }

      reject(new Error(`build failed with exit code ${code ?? "unknown"}`));
    });
  });
}

const invokedAsScript = process.argv[1] && path.resolve(process.argv[1]) === __filename;
if (invokedAsScript) {
  runBuild().catch((error) => {
    console.error(error.message);
    process.exit(1);
  });
}
