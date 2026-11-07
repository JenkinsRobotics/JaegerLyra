# Jaeger Lyra project page

The site is the alpha overview and a saved source/reconstruction audio comparison. The live Python decoder runs locally; the hosted page does not claim to run it in the browser.

Canonical source: `docs/index.html` and `docs/assets/` on `master`. Publish those files at the root of the existing `gh-pages` branch. The configured site is https://jenkinsrobotics.github.io/JaegerLyra/ . Preserve this branch deployment configuration; do not enable a second deployment workflow for the same site.

The page reuses the Jaeger template's palette. Every implemented feature is distinguished from planned research. Source audio is synthetic Kokoro material already present in the bundled comparison, and the paired output is Lyra's saved reconstruction.
