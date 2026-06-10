const { app, BrowserWindow, shell } = require("electron");
const path = require("path");

const DEV_URL = "http://localhost:5174";

function createWindow() {
  const win = new BrowserWindow({
    width: 1480,
    height: 900,
    minWidth: 1100,
    minHeight: 700,
    backgroundColor: "#05070d",
    autoHideMenuBar: true,
    title: "Judas — Arène",
    webPreferences: {
      preload: path.join(__dirname, "preload.cjs"),
      contextIsolation: true,
      nodeIntegration: false,
    },
  });

  win.webContents.setWindowOpenHandler(({ url }) => {
    shell.openExternal(url);
    return { action: "deny" };
  });

  if (process.env.NODE_ENV !== "production" && !app.isPackaged) {
    win.loadURL(DEV_URL);
  } else {
    win.loadFile(path.join(__dirname, "..", "dist", "index.html"));
  }
}

app.whenReady().then(() => {
  createWindow();
  app.on("activate", () => {
    if (BrowserWindow.getAllWindows().length === 0) createWindow();
  });
});

app.on("window-all-closed", () => {
  if (process.platform !== "darwin") app.quit();
});
