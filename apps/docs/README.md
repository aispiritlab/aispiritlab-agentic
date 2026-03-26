# AI Spirit Workshops Docs

Workshop-first documentation site built with Fumadocs and Next.js static export.

## Local development

```bash
npm install
npm run dev
```

## Static build

```bash
npm run build
```

The exported site is written to `out/`.

## Base path

For local development the site runs at `/`.

For GitHub Project Pages, the build derives the base path from:

1. `DOCS_BASE_PATH` if set
2. `GITHUB_REPOSITORY` in CI

That keeps the same app usable both locally and under `/<repo>/` on Pages.
