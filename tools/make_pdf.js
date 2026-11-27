// Print the single-page HTML build of the book to a PDF with headless Chromium.
//
//   jupyter-book build book --builder singlehtml
//   node tools/make_pdf.js            # writes book/_build/molecules-data-python.pdf
//
// Requires Node.js and the `playwright` package with Chromium installed
// (npm install playwright && npx playwright install chromium).
const path = require("path");
const { chromium } = require("playwright");

const root = path.resolve(__dirname, "..");
const src = "file://" + path.join(root, "book/_build/singlehtml/intro.html");
const out = path.join(root, "book/_build/molecules-data-python.pdf");

const PRINT_CSS = `
  .bd-header, .bd-sidebar-primary, .bd-sidebar-secondary, .header-article,
  .bd-footer, .bd-footer-article, .prev-next-area, .bd-search, .search-button,
  .theme-switch-button, .headerlink, .copybtn, .bd-header-announcement,
  .skip-link, .dropdown-buttons-trigger, #pst-back-to-top { display: none !important; }
  .bd-main, .bd-content, .bd-article-container, .bd-article { max-width: 100% !important; width: 100% !important; padding: 0 !important; margin: 0 !important; }
  .bd-page-width { max-width: 100% !important; }
  body { font-size: 10.5pt; }
  /* each chapter is a section inside the root toctree; start each on a new page */
  .toctree-wrapper > section { break-before: page; }
  #mdp-cover { break-after: page; height: 255mm; display: flex; align-items: center; justify-content: center; background: #14142b; }
  #mdp-cover img { max-height: 255mm !important; width: auto !important; }
  pre, .cell_output img, figure, table, .admonition { break-inside: avoid; }
  details { display: block; }
  details > summary { list-style: none; font-weight: 600; }
  img { max-width: 100% !important; height: auto !important; }
  a { color: inherit; text-decoration: none; }
`;

(async () => {
  const browser = await chromium.launch();
  const page = await browser.newPage();
  await page.goto(src, { waitUntil: "load", timeout: 300000 });
  await page.emulateMedia({ media: "print", colorScheme: "light" });
  await page.addStyleTag({ content: PRINT_CSS });
  // a full-page cover first
  await page.evaluate(() => {
    const d = document.createElement("div");
    d.id = "mdp-cover";
    d.innerHTML = '<img src="_static/cover.png" alt="cover">';
    const host = document.querySelector(".bd-article-container") || document.body;
    host.prepend(d);
  });
  // open every drop-down (exercise solutions) so they appear in print
  await page.evaluate(() => document.querySelectorAll("details").forEach(d => (d.open = true)));
  await page.waitForTimeout(1500);
  await page.pdf({
    path: out,
    format: "A4",
    printBackground: true,
    margin: { top: "18mm", bottom: "18mm", left: "16mm", right: "16mm" },
    displayHeaderFooter: true,
    headerTemplate: `<div style="font-size:7pt;width:100%;text-align:right;padding-right:16mm;color:#888">Molecules, Data and Python</div>`,
    footerTemplate: `<div style="font-size:7pt;width:100%;text-align:center;color:#888"><span class="pageNumber"></span></div>`,
  });
  await browser.close();
  console.log("wrote", path.relative(root, out));
})();
