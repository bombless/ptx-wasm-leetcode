# GitHub Pages Deployment

This repository includes a GitHub Actions workflow at `.github/workflows/deploy-pages.yml`
that builds the wasm web app and publishes it to GitHub Pages.

## One-time setup

1. Push this repository to GitHub.
2. Open the repository on GitHub.
3. Go to `Settings` -> `Pages`.
4. Under `Build and deployment`, set `Source` to `GitHub Actions`.

## Publish flow

After Pages is configured, every push to the `master` branch will:

1. Install the latest Emscripten SDK.
2. Build the `ptx_web` target into `build-web/webapp`.
3. Upload the generated static site as a GitHub Pages artifact.
4. Deploy the artifact with the official Pages deploy action.

## Expected site URL

For the `CatAirlineClub/ptx-wasm-leetcode` repository, the Pages URL should be:

`https://catairlineclub.github.io/ptx-wasm-leetcode/`

GitHub may take a short time to publish the first deployment.
