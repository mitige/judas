import { useCallback, useEffect, useState } from "react";

const isRecord = (value) =>
  value != null && typeof value === "object" && !Array.isArray(value);

export function mergeDefaults(defaults, stored) {
  if (!isRecord(defaults)) {
    return stored === undefined ? defaults : stored;
  }
  if (!isRecord(stored)) {
    return defaults;
  }
  const merged = { ...defaults };
  for (const [key, value] of Object.entries(stored)) {
    if (!(key in defaults)) continue;
    merged[key] = mergeDefaults(defaults[key], value);
  }
  return merged;
}

export function loadPersistedState(storage, key, defaults) {
  if (!storage) return defaults;
  try {
    const raw = storage.getItem(key);
    if (raw == null) return defaults;
    return mergeDefaults(defaults, JSON.parse(raw));
  } catch {
    return defaults;
  }
}

export function savePersistedState(storage, key, value) {
  if (!storage) return;
  try {
    storage.setItem(key, JSON.stringify(value));
  } catch {
    // Ignore storage quota / privacy mode failures; UI state still works in memory.
  }
}

function browserStorage() {
  return typeof window === "undefined" ? null : window.localStorage;
}

export function usePersistentState(key, defaults) {
  const [value, setValue] = useState(() =>
    loadPersistedState(browserStorage(), key, defaults));

  const setPersistentValue = useCallback((next) => {
    setValue((current) => {
      const resolved = typeof next === "function" ? next(current) : next;
      savePersistedState(browserStorage(), key, resolved);
      return resolved;
    });
  }, [key]);

  useEffect(() => {
    savePersistedState(browserStorage(), key, value);
  }, [key, value]);

  return [value, setPersistentValue];
}
