const { chromium } = require("playwright");
const fs = require("fs");
const path = require("path");

const OUTPUT_DIR = "./collected-data-wikimedia";
const VIEWPORT = { width: 1280, height: 800 };
const MAX_PAGES = 300;

const urlQueue = [];

const TARGET_URLS = [
  "https://commons.wikimedia.org/wiki/Category:Scanned_books",
  "https://commons.wikimedia.org/wiki/Category:Historical_documents",
  "https://commons.wikimedia.org/wiki/Category:Newspapers"
];

TARGET_URLS.forEach(url => urlQueue.push(url));

const visitedUrlsPath = "./visited_urls_wikimedia.json";
const visitedUrlsData = fs.existsSync(visitedUrlsPath)
  ? JSON.parse(fs.readFileSync(visitedUrlsPath, "utf8"))
  : { visited: [] };

const visitedUrls = new Set(visitedUrlsData.visited);

let SAMPLE_ID = 0;
let VISITED_COUNT = 0;

function nextSampleId() {
  SAMPLE_ID += 1;
  return String(SAMPLE_ID).padStart(6, "0");
}

function ensureDir(dir) {
  if (!fs.existsSync(dir)) {
    fs.mkdirSync(dir, { recursive: true });
  }
}

function saveVisitedUrls() {
  fs.writeFileSync(
    visitedUrlsPath,
    JSON.stringify({ visited: [...visitedUrls] }, null, 2)
  );
}

function isValidCommonsLink(href) {
  if (!href) return false;
  if (!href.startsWith("/wiki/")) return false;
  return true;
}

async function run() {
  console.log("[INFO] Starting Wikimedia crawler");
  ensureDir(OUTPUT_DIR);

  const browser = await chromium.launch({ headless: true });
  const context = await browser.newContext({ viewport: VIEWPORT });
  const page = await context.newPage();

  while (urlQueue.length > 0 && VISITED_COUNT < MAX_PAGES) {
    const url = urlQueue.shift();

    if (visitedUrls.has(url)) continue;

    visitedUrls.add(url);
    saveVisitedUrls();

    console.log(`[INFO] Visiting (${VISITED_COUNT + 1}/${MAX_PAGES}): ${url}`);

    try {
      await page.goto(url, { waitUntil: "networkidle", timeout: 30000 });
    } catch {
      console.warn(`[WARN] Failed to load: ${url}`);
      continue;
    }

    // ---------- If this is a file page ----------
    const isFilePage = url.includes("/wiki/File:");

    if (isFilePage) {
      try {
        const image = await page.$(".fullImageLink img");
        if (!image) throw new Error("No image found");

        const imageUrl = await image.getAttribute("src");
        const description = await page.$eval(
          "#mw-content-text",
          el => el.innerText.slice(0, 500)
        ).catch(() => "");

        const id = nextSampleId();
        const sampleDir = path.join(OUTPUT_DIR, `sample_${id}`);
        ensureDir(sampleDir);

        const view = await page.request.get(imageUrl);
        fs.writeFileSync(
          path.join(sampleDir, "image.png"),
          await view.body()
        );

        fs.writeFileSync(
          path.join(sampleDir, "label.txt"),
          description || "",
          "utf-8"
        );

        fs.writeFileSync(
          path.join(sampleDir, "meta.json"),
          JSON.stringify({
            id,
            source: "wikimedia",
            url,
            image_url: imageUrl,
            text_length: description.length,
            timestamp: new Date().toISOString()
          }, null, 2)
        );

        console.log(`[INFO] Saved sample ${id}`);
      } catch (err) {
        console.warn(`[WARN] Failed to extract image from ${url}`);
      }
    }

    // ---------- Collect new links ----------
    const links = await page.$$eval("a[href]", as =>
      as.map(a => a.getAttribute("href"))
    );

    for (const href of links) {
      if (!isValidCommonsLink(href)) continue;
      const fullUrl = `https://commons.wikimedia.org${href}`;
      if (!visitedUrls.has(fullUrl)) {
        urlQueue.push(fullUrl);
      }
    }

    VISITED_COUNT += 1;
  }

  await browser.close();
  console.log("[INFO] Wikimedia crawling finished");
}

run().catch(err => {
  console.error(err);
  process.exit(1);
});
