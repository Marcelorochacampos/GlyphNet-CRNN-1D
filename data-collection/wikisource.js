const { chromium } = require("playwright");
const fs = require("fs");
const path = require("path");

const OUTPUT_DIR = "./collected-data-wikisource";
const VIEWPORT = { width: 1280, height: 800 };
const MAX_PAGES = 300;
const urlQueue = [];

const TARGET_URLS = [
  "https://pt.wikisource.org/wiki/Dom_Casmurro",
];

TARGET_URLS.forEach(url => urlQueue.push(url));

const VISITED_FILE = "./visited_urls_wikisource.json";

let visitedUrls = new Set();
if (fs.existsSync(VISITED_FILE)) {
  const data = JSON.parse(fs.readFileSync(VISITED_FILE, "utf-8"));
  visitedUrls = new Set(data.visited || []);
}

let SAMPLE_ID = 606;
let CURRENT_AMOUNT_OF_VISITED_URLS = 0;

function nextSampleId() {
  SAMPLE_ID += 1;
  return String(SAMPLE_ID).padStart(6, "0");
}

function ensureDir(dir) {
  if (!fs.existsSync(dir)) {
    fs.mkdirSync(dir, { recursive: true });
  }
}

function updateVisitedUrlsJson() {
  fs.writeFileSync(
    VISITED_FILE,
    JSON.stringify({ visited: [...visitedUrls] }, null, 2)
  );
}

function isValidWikisourceLink(href) {
  if (!href) return false;
  if (!href.startsWith("/wiki/")) return false;
  if (href.includes(":")) return false;
  return true;
}

function isTextClean(text) {
  if (text.length < 20 || text.length > 600) return false;
  if ((text.match(/[{}<>]/g) || []).length > 2) return false;
  return true;
}

async function run() {
  console.log("Starting Wikisource crawler...");
  ensureDir(OUTPUT_DIR);

  const browser = await chromium.launch({ headless: true });
  const context = await browser.newContext({ viewport: VIEWPORT });
  const page = await context.newPage();

  console.log(`Initial queue: ${urlQueue.length}`);
  console.log(`Already visited: ${visitedUrls.size}`);

  while (urlQueue.length > 0 && CURRENT_AMOUNT_OF_VISITED_URLS < MAX_PAGES) {
    const url = urlQueue.shift();

    if (visitedUrls.has(url)) {
      console.log(`Skipping visited: ${url}`);
      continue;
    }

    visitedUrls.add(url);
    updateVisitedUrlsJson();
    console.log(`Visiting (${visitedUrls.size}/${MAX_PAGES}): ${url}`);

    try {
      await page.goto(url, { waitUntil: "networkidle", timeout: 30000 });
    } catch {
      console.warn(`Failed to load: ${url}`);
      continue;
    }

    // Conteúdo principal do Wikisource
    const elements = await page.$$(
      "#mw-content-text p, #mw-content-text h2, #mw-content-text h3"
    );

    for (const el of elements) {
      const text = (await el.innerText()).trim();
      if (!isTextClean(text)) continue;

      const box = await el.boundingBox();
      if (!box || box.width < 150 || box.height < 40) continue;

      const id = nextSampleId();
      const sampleDir = path.join(OUTPUT_DIR, `sample_${id}`);
      ensureDir(sampleDir);

      await el.screenshot({
        path: path.join(sampleDir, "image.png")
      });

      fs.writeFileSync(
        path.join(sampleDir, "label.txt"),
        text,
        "utf-8"
      );

      fs.writeFileSync(
        path.join(sampleDir, "meta.json"),
        JSON.stringify({
          id,
          source: "wikisource",
          url,
          text_length: text.length,
          bounding_box: box,
          timestamp: new Date().toISOString()
        }, null, 2),
        "utf-8"
      );

      console.log(`Saved sample ${id}`);
    }

    // Descobrir novos links
    const links = await page.$$eval("a[href]", as =>
      as.map(a => a.getAttribute("href"))
    );

    for (const href of links) {
      if (!isValidWikisourceLink(href)) continue;
      const fullUrl = `https://pt.wikisource.org${href}`;
      if (!visitedUrls.has(fullUrl)) {
        urlQueue.push(fullUrl);
      }
    }

    CURRENT_AMOUNT_OF_VISITED_URLS += 1;
  }

  await browser.close();
  console.log("Wikisource crawling finished.");
}

run().catch(err => {
  console.error(err);
  process.exit(1);
});
