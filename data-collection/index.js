const { chromium } = require("playwright");
const fs = require("fs");
const path = require("path");

const OUTPUT_DIR = "./data";
const VIEWPORT = { width: 1280, height: 800 };
const MAX_PAGES = 300; // limite de páginas visitadas
const visitedUrls = new Set();
const urlQueue = [];

const TARGET_URLS = [
  // "https://pt.wikipedia.org/wiki/Calígula",
  // "https://pt.wikipedia.org/wiki/Império_do_Japão",
  // "https://pt.wikipedia.org/wiki/Pré-história",
  // "https://pt.wikipedia.org/wiki/Guerras_Napoleônicas",
  // "https://pt.wikipedia.org/wiki/Moscovo",
  // "https://pt.wikipedia.org/wiki/Fortaleza_(arquitetura_militar)",
  // "https://pt.wikipedia.org/wiki/Língua_grega",
  // "https://pt.wikipedia.org/wiki/Antíoco_IV_Epifânio",
  // "https://pt.wikipedia.org/wiki/Grammy_Awards",
  // "https://pt.wikipedia.org/wiki/Monte_do_Templo",
  // "https://pt.wikipedia.org/wiki/Período_helenístico",
  // "https://pt.wikipedia.org/wiki/Idade_da_Pedra_na_Grécia",
  // "https://pt.wikipedia.org/wiki/Civilização_egeia",
  // "https://pt.wikipedia.org/wiki/Grécia_Antiga",
  // "https://pt.wikipedia.org/wiki/Idade_Média",
  // "https://pt.wikipedia.org/wiki/Idade_Moderna",
  // "https://pt.wikipedia.org/wiki/Idade_Contemporânea",
  // "https://pt.wikipedia.org/wiki/Violino",
  // "https://pt.wikipedia.org/wiki/Leonardo_da_Vinci",
  // "https://pt.wikipedia.org/wiki/Matemático",

  // "https://pt.wikipedia.org/wiki/História_do_Brasil",
  // "https://pt.wikipedia.org/wiki/República_Romana",
  // "https://pt.wikipedia.org/wiki/Império_Romano",
  // "https://pt.wikipedia.org/wiki/Revolução_Industrial",
  // "https://pt.wikipedia.org/wiki/Revolução_Francesa",
  // "https://pt.wikipedia.org/wiki/Iluminismo",
  // "https://pt.wikipedia.org/wiki/Renascimento",
  // "https://pt.wikipedia.org/wiki/Barroco",
  // "https://pt.wikipedia.org/wiki/Filosofia",
  // "https://pt.wikipedia.org/wiki/Filosofia_antiga",
  // "https://pt.wikipedia.org/wiki/Filosofia_moderna",
  // "https://pt.wikipedia.org/wiki/Filosofia_contemporânea",

  // "https://pt.wikipedia.org/wiki/Albert_Einstein",
  // "https://pt.wikipedia.org/wiki/Isaac_Newton",
  // "https://pt.wikipedia.org/wiki/Galileu_Galilei",
  // "https://pt.wikipedia.org/wiki/Nikola_Tesla",
  // "https://pt.wikipedia.org/wiki/Charles_Darwin",
  // "https://pt.wikipedia.org/wiki/Stephen_Hawking",

  // "https://pt.wikipedia.org/wiki/Teoria_da_relatividade",
  // "https://pt.wikipedia.org/wiki/Mecânica_quântica",
  // "https://pt.wikipedia.org/wiki/Física",
  // "https://pt.wikipedia.org/wiki/Química",
  // "https://pt.wikipedia.org/wiki/Biologia",
  // "https://pt.wikipedia.org/wiki/Genética",
  // "https://pt.wikipedia.org/wiki/Evolução",
  // "https://pt.wikipedia.org/wiki/Seleção_natural",

  // "https://pt.wikipedia.org/wiki/Língua_portuguesa",
  // "https://pt.wikipedia.org/wiki/Linguística",
  // "https://pt.wikipedia.org/wiki/Gramática",
  // "https://pt.wikipedia.org/wiki/Fonética",
  // "https://pt.wikipedia.org/wiki/Morfologia_(linguística)",
  // "https://pt.wikipedia.org/wiki/Sintaxe",

  // "https://pt.wikipedia.org/wiki/Brasil",
  // "https://pt.wikipedia.org/wiki/Portugal",
  // "https://pt.wikipedia.org/wiki/Espanha",
  // "https://pt.wikipedia.org/wiki/França",
  // "https://pt.wikipedia.org/wiki/Alemanha",
  // "https://pt.wikipedia.org/wiki/Itália",
  // "https://pt.wikipedia.org/wiki/Reino_Unido",
  // "https://pt.wikipedia.org/wiki/Estados_Unidos",
  // "https://pt.wikipedia.org/wiki/China",
  // "https://pt.wikipedia.org/wiki/Japão",

  // "https://pt.wikipedia.org/wiki/São_Paulo",
  // "https://pt.wikipedia.org/wiki/Rio_de_Janeiro",
  // "https://pt.wikipedia.org/wiki/Belo_Horizonte",
  // "https://pt.wikipedia.org/wiki/Salvador",
  // "https://pt.wikipedia.org/wiki/Porto_Alegre",

  // "https://pt.wikipedia.org/wiki/Música",
  // "https://pt.wikipedia.org/wiki/Música_clássica",
  // "https://pt.wikipedia.org/wiki/Rock",
  // "https://pt.wikipedia.org/wiki/Jazz",
  // "https://pt.wikipedia.org/wiki/Samba",

  // "https://pt.wikipedia.org/wiki/Arte",
  // "https://pt.wikipedia.org/wiki/Pintura",
  // "https://pt.wikipedia.org/wiki/Escultura",
  // "https://pt.wikipedia.org/wiki/Arquitetura",
  // "https://pt.wikipedia.org/wiki/Literatura",
  // "https://pt.wikipedia.org/wiki/Romance",
  // "https://pt.wikipedia.org/wiki/Poesia",

  // "https://pt.wikipedia.org/wiki/Computação",
  // "https://pt.wikipedia.org/wiki/Inteligência_artificial",
  // "https://pt.wikipedia.org/wiki/Aprendizado_de_máquina",
  // "https://pt.wikipedia.org/wiki/Redes_neurais_artificiais",
  // "https://pt.wikipedia.org/wiki/Algoritmo",
  // "https://pt.wikipedia.org/wiki/Estrutura_de_dados"

  "https://pt.wikipedia.org/wiki/História_da_ciência",
  "https://pt.wikipedia.org/wiki/História_da_filosofia",
  "https://pt.wikipedia.org/wiki/História_da_matemática",
  "https://pt.wikipedia.org/wiki/História_da_arte",
  "https://pt.wikipedia.org/wiki/História_da_literatura",

  "https://pt.wikipedia.org/wiki/Sociologia",
  "https://pt.wikipedia.org/wiki/Antropologia",
  "https://pt.wikipedia.org/wiki/Economia",
  "https://pt.wikipedia.org/wiki/Política",
  "https://pt.wikipedia.org/wiki/Democracia",

  "https://pt.wikipedia.org/wiki/Direito",
  "https://pt.wikipedia.org/wiki/Constituição",
  "https://pt.wikipedia.org/wiki/Cidadania",
  "https://pt.wikipedia.org/wiki/Estado",
  "https://pt.wikipedia.org/wiki/Nação",

  "https://pt.wikipedia.org/wiki/Astronomia",
  "https://pt.wikipedia.org/wiki/Cosmologia",
  "https://pt.wikipedia.org/wiki/Sistema_solar",
  "https://pt.wikipedia.org/wiki/Estrela",
  "https://pt.wikipedia.org/wiki/Galáxia",

  "https://pt.wikipedia.org/wiki/Geografia",
  "https://pt.wikipedia.org/wiki/Clima",
  "https://pt.wikipedia.org/wiki/Ecossistema",
  "https://pt.wikipedia.org/wiki/Biodiversidade",
  "https://pt.wikipedia.org/wiki/Meio_ambiente",

  "https://pt.wikipedia.org/wiki/Educação",
  "https://pt.wikipedia.org/wiki/Pedagogia",
  "https://pt.wikipedia.org/wiki/Ensino_superior",
  "https://pt.wikipedia.org/wiki/Ciência",
  "https://pt.wikipedia.org/wiki/Tecnologia",

  "https://pt.wikipedia.org/wiki/Engenharia",
  "https://pt.wikipedia.org/wiki/Engenharia_de_software",
  "https://pt.wikipedia.org/wiki/Programação",
  "https://pt.wikipedia.org/wiki/Linguagem_de_programação",
  "https://pt.wikipedia.org/wiki/Computador",

  "https://pt.wikipedia.org/wiki/Internet",
  "https://pt.wikipedia.org/wiki/World_Wide_Web",
  "https://pt.wikipedia.org/wiki/Segurança_da_informação",
  "https://pt.wikipedia.org/wiki/Criptografia",
  "https://pt.wikipedia.org/wiki/Banco_de_dados",

  "https://pt.wikipedia.org/wiki/Medicina",
  "https://pt.wikipedia.org/wiki/Saúde",
  "https://pt.wikipedia.org/wiki/Anatomia",
  "https://pt.wikipedia.org/wiki/Fisiologia",
  "https://pt.wikipedia.org/wiki/Doença",

  "https://pt.wikipedia.org/wiki/Psicologia",
  "https://pt.wikipedia.org/wiki/Neurociência",
  "https://pt.wikipedia.org/wiki/Cognição",
  "https://pt.wikipedia.org/wiki/Comportamento",
  "https://pt.wikipedia.org/wiki/Aprendizagem",

  "https://pt.wikipedia.org/wiki/História_militar",
  "https://pt.wikipedia.org/wiki/Guerra",
  "https://pt.wikipedia.org/wiki/Exército",
  "https://pt.wikipedia.org/wiki/Estratégia_militar",
  "https://pt.wikipedia.org/wiki/Tática",

  "https://pt.wikipedia.org/wiki/Religião",
  "https://pt.wikipedia.org/wiki/Filosofia_da_religião",
  "https://pt.wikipedia.org/wiki/Mitologia",
  "https://pt.wikipedia.org/wiki/Mitologia_grega",
  "https://pt.wikipedia.org/wiki/Mitologia_romana"
];

TARGET_URLS.forEach(url => urlQueue.push(url));
let SAMPLE_ID = 33086;

function nextSampleId() {
  SAMPLE_ID += 1;
  return String(SAMPLE_ID).padStart(6, "0");
}

function ensureDir(dir) {
  if (!fs.existsSync(dir)) {
    fs.mkdirSync(dir, { recursive: true });
  }
}

function isValidWikiLink(href) {
  if (!href) return false;
  if (!href.startsWith("/wiki/")) return false;
  if (href.includes(":")) return false;
  return true;
}

async function run() {
  ensureDir(OUTPUT_DIR);

  const browser = await chromium.launch({ headless: true });
  const context = await browser.newContext({ viewport: VIEWPORT });
  const page = await context.newPage();

  while (urlQueue.length > 0 && visitedUrls.size < MAX_PAGES) {
    const url = urlQueue.shift();
    if (visitedUrls.has(url)) continue;

    visitedUrls.add(url);
    console.log(`Visiting (${visitedUrls.size}/${MAX_PAGES}): ${url}`);

    try {
      await page.goto(url, { waitUntil: "networkidle", timeout: 30000 });
    } catch {
      console.warn(`Failed to load: ${url}`);
      continue;
    }

    const elements = await page.$$("p, h1, h2, h3, li");

    for (const el of elements) {
      const text = (await el.innerText()).trim();
      if (!text || text.length < 10 || text.length > 500) continue;

      const box = await el.boundingBox();
      if (!box || box.width < 100 || box.height < 30) continue;

      const id = nextSampleId();
      const sampleDir = path.join(OUTPUT_DIR, `sample_${id}`);
      ensureDir(sampleDir);

      await el.screenshot({ path: path.join(sampleDir, "image.png") });
      fs.writeFileSync(path.join(sampleDir, "label.txt"), text, "utf-8");

      fs.writeFileSync(
        path.join(sampleDir, "meta.json"),
        JSON.stringify({
          id,
          url,
          text_length: text.length,
          bounding_box: box,
          timestamp: new Date().toISOString()
        }, null, 2),
        "utf-8"
      );

      console.log(`Saved sample ${id}`);
    }

    const links = await page.$$eval("a[href]", as =>
      as.map(a => a.getAttribute("href"))
    );

    for (const href of links) {
      if (!isValidWikiLink(href)) continue;
      const fullUrl = `https://pt.wikipedia.org${href}`;
      if (!visitedUrls.has(fullUrl)) {
        urlQueue.push(fullUrl);
      }
    }
  }

  await browser.close();
  console.log("Crawling finalizado.");
}

run().catch(err => {
  console.error(err);
  process.exit(1);
});
