const { chromium } = require("playwright");
const fs = require("fs");
const path = require("path");

const OUTPUT_DIR = "./data";
const VIEWPORT = { width: 1280, height: 800 };
const TARGET_URLS = [
  "https://en.wikipedia.org/wiki/Optical_character_recognition"
];

let SAMPLE_ID = 0;

function nextSampleId() {
  SAMPLE_ID += 1;
  return String(SAMPLE_ID).padStart(6, "0");
}

function ensureDir(dir) {
  if (!fs.existsSync(dir)) {
    fs.mkdirSync(dir, { recursive: true });
  }
}

async function run() {
  ensureDir(OUTPUT_DIR);

  const browser = await chromium.launch({ headless: true });
  const context = await browser.newContext({ viewport: VIEWPORT });
  const page = await context.newPage();

  for (const url of TARGET_URLS) {
    console.log(`Visiting: ${url}`);
    await page.goto(url, { waitUntil: "networkidle" });

    const elements = await page.$$("p, h1, h2, h3, li");

    for (const el of elements) {
      const text = (await el.innerText()).trim();

      if (!text) continue;
      if (text.length < 10) continue;
      if (text.length > 500) continue;

      const box = await el.boundingBox();
      if (!box) continue;
      if (box.width < 100 || box.height < 30) continue;

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

      const meta = {
        id,
        url,
        tag: await el.evaluate(node => node.tagName),
        bounding_box: box,
        viewport: VIEWPORT,
        text_length: text.length,
        timestamp: new Date().toISOString()
      };

      fs.writeFileSync(
        path.join(sampleDir, "meta.json"),
        JSON.stringify(meta, null, 2),
        "utf-8"
      );

      console.log(`Saved sample ${id}`);
    }
  }

  await browser.close();
}

run().catch(err => {
  console.error(err);
  process.exit(1);
});
