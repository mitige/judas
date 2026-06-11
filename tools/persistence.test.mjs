import assert from "node:assert/strict";
import fs from "node:fs";
import test from "node:test";

// app/ et viz/ embarquent chacun leur copie (apps Electron séparées) :
// la suite tourne sur LES DEUX et un test garantit qu'elles restent identiques.
const COPIES = {
  app: new URL("../app/src/persistence.mjs", import.meta.url),
  viz: new URL("../viz/src/persistence.mjs", import.meta.url),
};

test("app and viz persistence copies are byte-identical", () => {
  assert.equal(
    fs.readFileSync(COPIES.app, "utf8"),
    fs.readFileSync(COPIES.viz, "utf8"),
  );
});

function memoryStorage(seed = {}) {
  const data = new Map(Object.entries(seed));
  return {
    getItem: (key) => data.has(key) ? data.get(key) : null,
    setItem: (key, value) => data.set(key, String(value)),
  };
}

for (const [name, url] of Object.entries(COPIES)) {
  const { loadPersistedState, mergeDefaults, savePersistedState } =
    await import(url);

  test(`${name}: loadPersistedState merges saved values with new defaults`, () => {
    const storage = memoryStorage({
      "judas:test": JSON.stringify({ page: "live", nested: { x: 5 } }),
    });

    assert.deepEqual(
      loadPersistedState(storage, "judas:test", {
        page: "dashboard",
        connected: false,
        nested: { x: 0, y: 2 },
      }),
      { page: "live", connected: false, nested: { x: 5, y: 2 } },
    );
  });

  test(`${name}: loadPersistedState falls back to defaults on invalid JSON`, () => {
    const defaults = { page: "dashboard" };
    const storage = memoryStorage({ "judas:test": "not json" });

    assert.deepEqual(loadPersistedState(storage, "judas:test", defaults), defaults);
  });

  test(`${name}: savePersistedState writes JSON without throwing`, () => {
    const storage = memoryStorage();

    savePersistedState(storage, "judas:test", { cps: 12, model: "m.pts" });

    assert.deepEqual(
      loadPersistedState(storage, "judas:test", { cps: 0, model: "" }),
      { cps: 12, model: "m.pts" },
    );
  });

  test(`${name}: mergeDefaults preserves primitive stored values`, () => {
    assert.equal(mergeDefaults("dashboard", "training"), "training");
  });
}
