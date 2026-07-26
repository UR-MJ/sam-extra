// Behavioural tests for the injected LoRA Manager "Manage" tab.
//
// The Python asset tests can only assert that the source still contains the
// fix. These load the real script into a jsdom document that mirrors Forge's
// extra-networks tab strip and then click things, which is what actually caught
// the "Manage pane never closes" bug: Gradio 4.40 keeps non-selected TabItems
// mounted and only writes an inline display, so the synthetic pane is ours to
// hide and nothing else will do it.
import { test } from "node:test";
import assert from "node:assert/strict";
import { readFileSync } from "node:fs";
import { fileURLToPath } from "node:url";
import path from "node:path";
import { JSDOM } from "jsdom";

const ROOT = path.resolve(path.dirname(fileURLToPath(import.meta.url)), "..", "..");
const SCRIPT = readFileSync(path.join(ROOT, "javascript", "lora_manager.js"), "utf8");

// A minimal stand-in for Forge's #txt2img_extra_tabs: a tab-nav of buttons and
// one .tabitem per tab, with Gradio's inline-display convention already applied.
const TABS = ["Generation", "Textual Inversion", "Checkpoints", "Lora"];

function strip(tab) {
  const panes = TABS.map(
    (name, i) =>
      `<div id="${tab}_${name.toLowerCase().replace(/ /g, "_")}" class="tabitem"` +
      ` style="display: ${i === 0 ? "block" : "none"};">${name} body</div>`
  ).join("");
  const buttons = TABS.map(
    (name, i) => `<button class="svelte-tab${i === 0 ? " selected" : ""}">${name}</button>`
  ).join("");
  return `<div id="${tab}_extra_tabs">
            <div class="tab-nav">${buttons}</div>
            ${panes}
          </div>`;
}

function buildDom() {
  // Both strips exist so injection completes for txt2img AND img2img; otherwise
  // the script keeps its bootstrap observer and 800 ms retry alive for 300 s and
  // the test process never exits.
  const dom = new JSDOM(
    `<!doctype html><html><body>${strip("txt2img")}${strip("img2img")}</body></html>`,
    { runScripts: "outside-only", pretendToBeVisual: true }
  );

  const { window } = dom;
  // Forge globals the script reaches for. gradioApp() is the document here;
  // onUiLoaded fires immediately so injection happens synchronously.
  window.gradioApp = () => window.document;
  window.onUiLoaded = (fn) => fn();
  window.eval(SCRIPT);
  return dom;
}

function state(dom) {
  const doc = dom.window.document;
  const container = doc.querySelector("#txt2img_extra_tabs");
  const nav = container.querySelector(":scope > div.tab-nav");
  return {
    container,
    nav,
    buttons: [...nav.querySelectorAll(":scope > button")],
    panes: [...container.querySelectorAll(":scope > .tabitem")],
    managePane: doc.querySelector("#txt2img_loramanager"),
    manageBtn: doc.querySelector("[data-sam3-lm-btn]"),
  };
}

// jsdom delivers MutationObserver callbacks as microtasks, same as a browser.
const settle = () => new Promise((resolve) => setTimeout(resolve, 0));

// The script defers start() by 500 ms behind onUiLoaded, so wait for the
// injection to actually land instead of guessing a delay.
async function ready(dom) {
  for (let i = 0; i < 60; i += 1) {
    if (dom.window.document.querySelector("#txt2img_loramanager")) return;
    await new Promise((resolve) => setTimeout(resolve, 50));
  }
  throw new Error("Manage tab was never injected");
}

test("injects a Manage tab into the existing strip", async () => {
  const dom = buildDom();
  await ready(dom);
  const s = state(dom);

  assert.ok(s.manageBtn, "Manage button was not injected");
  assert.ok(s.managePane, "Manage pane was not injected");
  assert.equal(s.manageBtn.textContent, "Manage");
  assert.equal(s.buttons.length, TABS.length + 1);
  assert.equal(s.panes.length, TABS.length + 1);
  // Injected hidden, and it must not steal the initial selection.
  assert.equal(s.managePane.style.display, "none");
  assert.equal(s.manageBtn.classList.contains("selected"), false);
  dom.window.close();
});

test("clicking Manage shows only the manager", async () => {
  const dom = buildDom();
  await ready(dom);
  let s = state(dom);

  s.manageBtn.click();
  await settle();
  s = state(dom);

  assert.equal(s.managePane.style.display, "block");
  assert.ok(s.manageBtn.classList.contains("selected"));
  for (const pane of s.panes) {
    if (pane === s.managePane) continue;
    assert.equal(pane.style.display, "none", `${pane.id} should be hidden`);
  }
  dom.window.close();
});

test("a native tab closes the manager on the first click", async () => {
  const dom = buildDom();
  await ready(dom);
  let s = state(dom);

  s.manageBtn.click();
  await settle();
  assert.equal(state(dom).managePane.style.display, "block");

  // Click "Checkpoints". Gradio would normally also react, but the point of the
  // fix is that we must not depend on it: here nothing else handles the click.
  s = state(dom);
  s.buttons[2].click();
  await settle();
  s = state(dom);

  assert.equal(s.managePane.style.display, "none", "manager stayed open");
  assert.equal(s.manageBtn.classList.contains("selected"), false);
  dom.window.close();
});

test("returning to the tab Gradio still thinks is selected also closes it", async () => {
  // The regression case. Gradio never saw our synthetic tab take over, so when
  // the user clicks the tab they came from its reactive block is not dirty and
  // it does nothing at all — only our handler can close the manager.
  const dom = buildDom();
  await ready(dom);
  let s = state(dom);
  const generation = s.panes[0];

  s.manageBtn.click();
  await settle();
  assert.equal(state(dom).managePane.style.display, "block");
  assert.equal(generation.style.display, "none");

  s = state(dom);
  s.buttons[0].click(); // "Generation" — the originally selected tab
  await settle();
  s = state(dom);

  assert.equal(s.managePane.style.display, "none", "manager stayed open");
  assert.equal(generation.style.display, "block", "previous tab was not restored");
  assert.ok(s.buttons[0].classList.contains("selected"));
  dom.window.close();
});

test("the observer closes the manager when Gradio marks a tab selected late", async () => {
  // Covers the ordering we cannot control: Gradio flushes its tab switch
  // asynchronously, so the selection can land after our click handler ran.
  const dom = buildDom();
  await ready(dom);
  let s = state(dom);

  s.manageBtn.click();
  await settle();
  assert.equal(state(dom).managePane.style.display, "block");

  // No click at all — just Gradio's own class update arriving afterwards.
  s = state(dom);
  s.buttons[3].classList.add("selected");
  await settle();

  assert.equal(
    state(dom).managePane.style.display,
    "none",
    "observer did not close the manager"
  );
  dom.window.close();
});

test("visibility survives a tab strip that is rebuilt after injection", async () => {
  // Listeners bound to nodes captured at injection time would be lost here.
  const dom = buildDom();
  await ready(dom);
  let s = state(dom);

  s.manageBtn.click();
  await settle();
  assert.equal(state(dom).managePane.style.display, "block");

  // Replace every native button with a fresh clone, as a re-render would.
  s = state(dom);
  for (const btn of s.buttons) {
    if (btn.hasAttribute("data-sam3-lm-btn")) continue;
    btn.replaceWith(btn.cloneNode(true));
  }
  await settle();

  s = state(dom);
  const rebuilt = s.buttons.find((b) => b.textContent === "Checkpoints");
  rebuilt.click();
  await settle();

  assert.equal(
    state(dom).managePane.style.display,
    "none",
    "delegation did not survive the rebuild"
  );
  dom.window.close();
});
