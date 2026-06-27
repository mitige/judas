package dev.judas.bridge;

import com.google.gson.JsonObject;
import com.google.gson.JsonParser;
import net.minecraft.client.Minecraft;
import net.minecraft.client.gui.inventory.GuiContainer;
import net.minecraftforge.client.event.RenderGameOverlayEvent;
import net.minecraftforge.common.MinecraftForge;
import net.minecraftforge.fml.client.registry.ClientRegistry;
import net.minecraftforge.fml.common.Mod;
import net.minecraftforge.fml.common.event.FMLInitializationEvent;
import net.minecraftforge.fml.common.eventhandler.SubscribeEvent;
import net.minecraftforge.fml.common.gameevent.TickEvent;
import net.minecraft.client.settings.KeyBinding;
import net.minecraft.entity.player.EntityPlayer;
import net.minecraft.inventory.Slot;
import net.minecraft.item.ItemStack;
import net.minecraft.scoreboard.Score;
import net.minecraft.scoreboard.ScoreObjective;
import net.minecraft.scoreboard.ScorePlayerTeam;
import net.minecraft.scoreboard.Scoreboard;
import org.lwjgl.input.Keyboard;

import java.io.BufferedReader;
import java.io.File;
import java.io.FileReader;
import java.io.FileWriter;
import java.io.IOException;
import java.io.PrintWriter;
import java.lang.reflect.Field;
import java.util.Collection;
import java.util.Locale;

/**
 * Judas Bridge — pont entre Minecraft 1.8.9 et le daemon Judas.
 *
 * À chaque tick client : envoie l'état (soi + cible) au daemon via WebSocket
 * localhost, applique la dernière action reçue. Latence visée < 2 ms.
 *
 * Touches : K = toggle bot, L = kill-switch, J = enregistrer une trace golden.
 */
@Mod(modid = "judasbridge", name = "Judas Bridge", version = "0.1.0",
     clientSideOnly = true)
public final class JudasMod {

    private static final String WS_HOST = "127.0.0.1";
    private static final int WS_PORT = 8765;
    private static final String WS_PATH = "/live";
    private static final int RECONNECT_TICKS = 40;   // 2 s

    private final StateCollector collector = new StateCollector();
    private final ActionApplier applier = new ActionApplier(collector);
    private final AutoGapple autoGapple = new AutoGapple();
    private final AutoJump autoJump = new AutoJump();
    private final TraceRecorder recorder = new TraceRecorder();
    private final PacketOrderProbe packetProbe = new PacketOrderProbe();
    private final PacketOrderGuard packetGuard = new PacketOrderGuard();
    private final WsClient ws = new WsClient(WS_HOST, WS_PORT, WS_PATH);
    private final JsonParser parser = new JsonParser();

    private KeyBinding keyToggle;
    private KeyBinding keyKill;
    private KeyBinding keyTrace;
    private KeyBinding keyNative;

    private boolean enabled = false;
    private int reconnectIn = 0;
    private File controlFile;
    private File statusLogFile;
    private File screenDumpFile;
    private long controlLastModified = 0L;
    private boolean requireBoxingMatch = true;
    private boolean manualOverride = false;
    private String pauseReason = "none";

    @Mod.EventHandler
    public void init(FMLInitializationEvent event) {
        keyToggle = new KeyBinding("Toggle bot", Keyboard.KEY_K, "Judas");
        keyKill = new KeyBinding("Kill-switch", Keyboard.KEY_L, "Judas");
        keyTrace = new KeyBinding("Enregistrer trace", Keyboard.KEY_J, "Judas");
        keyNative = new KeyBinding("Entree native (souris OS)", Keyboard.KEY_O, "Judas");
        ClientRegistry.registerKeyBinding(keyToggle);
        ClientRegistry.registerKeyBinding(keyKill);
        ClientRegistry.registerKeyBinding(keyTrace);
        ClientRegistry.registerKeyBinding(keyNative);
        MinecraftForge.EVENT_BUS.register(this);
    }

    @SubscribeEvent
    public void onClientTick(TickEvent.ClientTickEvent event) {
        Minecraft mc = Minecraft.getMinecraft();
        handleControlFile(mc);
        if (mc.thePlayer == null || mc.theWorld == null) {
            if (enabled) {
                if (manualOverride) pauseBot(mc, "no_world");
                else stopBot(mc);
            }
            return;
        }
        if (event.phase == TickEvent.Phase.START) {
            applier.clearKeystrokeMouseVisualsBeforeInput(mc);
        }

        // Phase.END : applique l'attaque native differee uniquement. En mode
        // DIRECT, l'attaque part dans ActionApplier.apply() pendant START, comme
        // un clic vanilla dans la phase input du tick.
        if (event.phase == TickEvent.Phase.END) {
            if (enabled && manualOverride && shouldPauseManual(mc)) {
                pauseBot(mc, manualPauseReason(mc));
            } else if (enabled && shouldAutoStop(mc)) {
                stopBot(mc);
            } else if (enabled && ws.isOpen() && !recorder.isRecording()) {
                if (autoGapple.isActive()) {
                    applier.cancelPendingAttack();
                } else {
                    applier.applyDeferredAttack(mc);
                    applier.applyKeystrokeMouseVisuals(mc);
                }
            }
            return;
        }

        if (enabled && !manualOverride && shouldAutoStop(mc)) {
            stopBot(mc);
            return;
        }
        packetProbe.beginClientTick(mc);
        packetGuard.beginClientTick(mc);
        handleKeys(mc);
        if (enabled && manualOverride && shouldPauseManual(mc)) {
            pauseBot(mc, manualPauseReason(mc));
            return;
        }
        if (enabled && !manualOverride && shouldAutoStop(mc)) {
            stopBot(mc);
            return;
        }
        if (enabled && manualOverride) {
            clearPause();
        }

        if (recorder.isRecording()) {
            recorder.onTick(mc);
            return;
        }
        if (!enabled) return;

        // connexion / reconnexion
        if (!ws.isOpen()) {
            applier.releaseAll(mc);
            if (reconnectIn-- <= 0) {
                reconnectIn = RECONNECT_TICKS;
                try {
                    ws.connect();
                } catch (IOException ignored) {
                }
            }
            return;
        }

        // 1. applique la dernière action reçue (réponse à l'état précédent)
        String msg = ws.pollLatest();
        JsonObject action = null;
        if (msg != null) {
            try {
                action = parser.parse(msg).getAsJsonObject();
                if ("action".equals(action.get("t").getAsString())) {
                    autoGapple.updateFromAction(action);
                    autoJump.updateFromAction(action);
                    collector.updateFriendsFromAction(action);
                    collector.updateKnockbackDumpFromAction(action);
                } else {
                    action = null;
                }
            } catch (RuntimeException ignored) {
                action = null;
            }
        }
        boolean eating = autoGapple.tickStart(mc, collector.getTarget());
        if (eating) {
            applier.cancelPendingAttack();
        } else {
            boolean hasModelAction = action != null;
            if (action == null) {
                action = autoGapple.fallbackRetreatAction(mc, collector.getTarget());
            }
            if (action != null) {
                boolean retreating = autoGapple.applyRetreatToAction(mc, collector.getTarget(), action);
                if (hasModelAction || retreating) {
                    autoJump.applyToAction(mc, action);
                    applier.apply(mc, action);
                }
            }
        }

        // 2. envoie l'état du tick courant
        JsonObject state = collector.collect(mc);
        if (state != null) {
            ws.sendText(state.toString());
        }
    }

    private void handleKeys(Minecraft mc) {
        while (keyToggle.isPressed()) {
            enabled = !enabled;
            if (!enabled) {
                manualOverride = false;
                pauseReason = "none";
                applier.releaseAll(mc);
                autoGapple.cancel(mc);
                autoJump.cancel();
                ws.close();
                logStatus("manual_toggle", "stopped");
            } else {
                manualOverride = true;
                requireBoxingMatch = false;
                pauseReason = "none";
                collector.resetCounters();
                reconnectIn = 0;
                logStatus("manual_toggle", "armed");
            }
        }
        while (keyKill.isPressed()) {
            stopBot(mc);
            if (recorder.isRecording()) recorder.stop(mc);
        }
        while (keyTrace.isPressed()) {
            if (recorder.isRecording()) recorder.stop(mc);
            else recorder.start(mc);
        }
        while (keyNative.isPressed()) {
            applier.setNative(!applier.nativeInput);
        }
    }

    private void handleControlFile(Minecraft mc) {
        ensureControlFiles(mc);
        if (controlFile == null || !controlFile.isFile()) return;
        long modified = controlFile.lastModified();
        if (modified == 0L || modified == controlLastModified) return;
        controlLastModified = modified;

        String command = readControlCommand();
        if (command.length() == 0) return;

        if ("arm_native".equals(command)) {
            applier.setNative(true);
            armBot(mc, command, true);
        } else if ("arm_native_live".equals(command)) {
            applier.setNative(true);
            armBot(mc, command, false);
        } else if ("arm".equals(command)) {
            armBot(mc, command, true);
        } else if ("arm_live".equals(command)) {
            armBot(mc, command, false);
        } else if ("native_on".equals(command)) {
            applier.setNative(true);
            logStatus(command, "ok");
        } else if ("native_off".equals(command)) {
            applier.setNative(false);
            logStatus(command, "ok");
        } else if ("disarm".equals(command) || "kill".equals(command)) {
            stopBot(mc);
            logStatus(command, "stopped");
        } else if ("status".equals(command)) {
            String reason = stopReason(mc, true);
            logStatus(command, "none".equals(reason) ? "ok" : "ok:" + reason);
        } else if ("status_live".equals(command)) {
            String reason = stopReason(mc, false);
            logStatus(command, "none".equals(reason) ? "ok" : "ok:" + reason);
        } else if ("dump_screen".equals(command)) {
            logStatus(command, dumpScreen(mc));
        } else {
            logStatus(command, "unknown");
        }
    }

    private void ensureControlFiles(Minecraft mc) {
        if (mc.mcDataDir == null) return;
        if (controlFile == null) {
            controlFile = new File(mc.mcDataDir, "judas-control.txt");
            if (controlFile.isFile()) {
                controlLastModified = controlFile.lastModified();
            }
        }
        if (statusLogFile == null) {
            statusLogFile = new File(mc.mcDataDir, "judas-bridge-status.log");
        }
        if (screenDumpFile == null) {
            screenDumpFile = new File(mc.mcDataDir, "judas-screen-dump.log");
        }
    }

    private String readControlCommand() {
        try (BufferedReader reader = new BufferedReader(new FileReader(controlFile))) {
            String line = reader.readLine();
            if (line == null) return "";
            return line.trim().toLowerCase(Locale.ROOT);
        } catch (IOException ignored) {
            return "";
        }
    }

    private void armBot(Minecraft mc, String command, boolean requireBoxing) {
        String reason = stopReason(mc, requireBoxing);
        if (!"none".equals(reason)) {
            enabled = false;
            applier.releaseAll(mc);
            autoJump.cancel();
            ws.close();
            logStatus(command, "blocked:" + reason);
            return;
        }
        manualOverride = false;
        pauseReason = "none";
        requireBoxingMatch = requireBoxing;
        enabled = true;
        collector.resetCounters();
        reconnectIn = 0;
        logStatus(command, "armed");
    }

    private boolean shouldAutoStop(Minecraft mc) {
        if (manualOverride) return false;
        return !"none".equals(stopReason(mc, requireBoxingMatch));
    }

    private void stopBot(Minecraft mc) {
        enabled = false;
        manualOverride = false;
        pauseReason = "none";
        applier.releaseAll(mc);
        autoGapple.cancel(mc);
        autoJump.cancel();
        ws.close();
        logStatus("auto_stop", stopReason(mc, requireBoxingMatch));
    }

    private boolean shouldPauseManual(Minecraft mc) {
        return !"none".equals(manualPauseReason(mc));
    }

    private String manualPauseReason(Minecraft mc) {
        if (mc.theWorld == null) return "no_world";
        if (mc.currentScreen != null) {
            return "screen:" + mc.currentScreen.getClass().getSimpleName();
        }
        if (!mc.inGameHasFocus) return "no_focus";
        return "none";
    }

    private void pauseBot(Minecraft mc, String reason) {
        applier.releaseAll(mc);
        autoGapple.cancel(mc);
        autoJump.cancel();
        ws.close();
        reconnectIn = 0;
        if (!reason.equals(pauseReason)) {
            pauseReason = reason;
            logStatus("auto_pause", reason);
        }
    }

    private void clearPause() {
        if (!"none".equals(pauseReason)) {
            pauseReason = "none";
            logStatus("auto_resume", "ok");
        }
    }

    private String stopReason(Minecraft mc, boolean requireBoxing) {
        if (mc.theWorld == null) return "no_world";
        if (mc.currentScreen != null) {
            return "screen:" + mc.currentScreen.getClass().getSimpleName();
        }
        if (!mc.inGameHasFocus) return "no_focus";
        if (requireBoxing) {
            if (!isLikelyBoxingMatch(mc)) return "not_boxing_match";
        } else if (!isLikelyLiveMatch(mc)) {
            return "not_live_match";
        }
        return "none";
    }

    private boolean isLikelyBoxingMatch(Minecraft mc) {
        String sidebar = sidebarText(mc);
        return sidebar.contains("hits")
                && sidebar.contains("you:")
                && sidebar.contains("them:")
                && sidebar.contains("1st to 100");
    }

    private boolean isLikelyLiveMatch(Minecraft mc) {
        if (!hasVisiblePlayerTarget(mc)) return false;
        String sidebar = sidebarText(mc);
        if (isLikelyBoxingMatch(mc)) return true;
        if (sidebar.contains("lobby") || sidebar.contains("hub")
                || sidebar.contains("queue") || sidebar.contains("click to play")) {
            return false;
        }
        return sidebar.contains("boxing")
                || sidebar.contains("hits")
                || sidebar.contains("duel")
                || sidebar.contains("opponent")
                || sidebar.contains("bed fight")
                || sidebar.contains("match")
                || sidebar.contains("round")
                || sidebar.contains("mode");
    }

    private boolean hasVisiblePlayerTarget(Minecraft mc) {
        if (mc.thePlayer == null || mc.theWorld == null) return false;
        double maxDistSq = 64.0 * 64.0;
        for (Object o : mc.theWorld.playerEntities) {
            EntityPlayer p = (EntityPlayer) o;
            if (p == mc.thePlayer || p.isInvisible() || p.isDead) continue;
            if (mc.thePlayer.getDistanceSqToEntity(p) <= maxDistSq) return true;
        }
        return false;
    }

    private String sidebarText(Minecraft mc) {
        if (mc.theWorld == null) return "";
        Scoreboard scoreboard = mc.theWorld.getScoreboard();
        if (scoreboard == null) return "";
        ScoreObjective objective = scoreboard.getObjectiveInDisplaySlot(1);
        if (objective == null) return "";

        StringBuilder text = new StringBuilder();
        text.append(cleanForLog(objective.getDisplayName()).toLowerCase(Locale.ROOT));
        Collection<Score> scores = scoreboard.getSortedScores(objective);
        for (Score score : scores) {
            String name = score.getPlayerName();
            if (name == null || name.startsWith("#")) continue;
            ScorePlayerTeam team = scoreboard.getPlayersTeam(name);
            String line = ScorePlayerTeam.formatPlayerName(team, name);
            text.append(' ').append(cleanForLog(line).toLowerCase(Locale.ROOT));
        }

        return text.toString();
    }

    private String dumpScreen(Minecraft mc) {
        if (screenDumpFile == null) return "dump:no_file";
        String screenName = mc.currentScreen == null
                ? "none" : mc.currentScreen.getClass().getSimpleName();
        int guiLeft = 0;
        int guiTop = 0;
        if (mc.currentScreen instanceof GuiContainer) {
            guiLeft = reflectInt(mc.currentScreen, "guiLeft");
            guiTop = reflectInt(mc.currentScreen, "guiTop");
        }
        int count = 0;
        try (PrintWriter writer = new PrintWriter(new FileWriter(screenDumpFile, false))) {
            writer.println("screen=" + screenName + " guiLeft=" + guiLeft + " guiTop=" + guiTop);
            if (mc.thePlayer != null && mc.thePlayer.openContainer != null) {
                for (Slot slot : mc.thePlayer.openContainer.inventorySlots) {
                    if (slot == null || !slot.getHasStack()) continue;
                    ItemStack stack = slot.getStack();
                    if (stack == null) continue;
                    int x = guiLeft + slot.xDisplayPosition + 8;
                    int y = guiTop + slot.yDisplayPosition + 8;
                    writer.println(String.format(Locale.ROOT,
                            "slot=%d x=%d y=%d item=%s name=%s",
                            slot.slotNumber, x, y,
                            stack.getItem().getRegistryName(),
                            cleanForLog(stack.getDisplayName())));
                    count++;
                }
            }
        } catch (IOException ex) {
            return "dump:error";
        }
        return "dump:" + screenName + ":" + count;
    }

    private static int reflectInt(Object target, String fieldName) {
        try {
            Field field = GuiContainer.class.getDeclaredField(fieldName);
            field.setAccessible(true);
            return field.getInt(target);
        } catch (ReflectiveOperationException ignored) {
            return 0;
        }
    }

    private static String cleanForLog(String text) {
        if (text == null) return "";
        return text.replaceAll("\u00A7.", "").replace('\n', ' ').replace('\r', ' ');
    }

    private void logStatus(String command, String result) {
        if (statusLogFile == null) return;
        try (PrintWriter writer = new PrintWriter(new FileWriter(statusLogFile, true))) {
            writer.println(String.format(Locale.ROOT,
                    "time=%d command=%s result=%s enabled=%s ws=%s input=%s",
                    System.currentTimeMillis(), command, result,
                    Boolean.toString(enabled), Boolean.toString(ws.isOpen()),
                    applier.inputModeLabel()));
        } catch (IOException ignored) {
        }
    }
    @SubscribeEvent
    public void onOverlay(RenderGameOverlayEvent.Text event) {
        Minecraft mc = Minecraft.getMinecraft();
        if (mc.thePlayer == null || mc.theWorld == null) {
            if (enabled) {
                if (manualOverride) pauseBot(mc, "no_world");
                else stopBot(mc);
            }
            return;
        }
        String status = enabled
                ? (ws.isOpen() ? "\u00A7bACTIF" : "\u00A7econnexion...")
                : "\u00A78inactif";
        if (recorder.isRecording()) status = "\u00A7dTRACE...";
        event.left.add("\u00A77[\u00A7bJudas\u00A77] " + status
                + " \u00A77| entree " + applier.inputModeLabel() + "\u00A78[O]");
        if (enabled) event.left.add(applier.aimStatusLine(mc));
        if (enabled) event.left.add(autoGapple.statusLine());
        if (enabled) event.left.add(autoJump.statusLine());
        if (enabled) event.left.add(collector.friendStatusLine());
        if (enabled) event.left.add(collector.knockbackDumpStatusLine());

        // Diagnostic combo/sprint : voir en direct si le mod\u00E8le encha\u00EEne
        // (d\u00E9ploiement fid\u00E8le) ou recule passivement (souci d'entra\u00EEnement).
        if (enabled && collector.getTarget() != null) {
            event.left.add(packetProbe.statusLine());
            event.left.add(packetGuard.statusLine());
            boolean spr = mc.thePlayer.isSprinting();
            event.left.add(String.format(
                    "\u00A77 d=\u00A7f%.2f \u00A77sprint=%s\u00A77 combo=\u00A7e%d",
                    collector.getLastDist(),
                    spr ? "\u00A7aon" : "\u00A78off",
                    collector.getMyCombo()));
            event.left.add(String.format(
                    "\u00A77 hits \u00A7a%d\u00A77/\u00A7c%d \u00A77(combo subi \u00A7c%d\u00A77)",
                    collector.getHitsDealt(), collector.getHitsTaken(),
                    collector.getOppCombo()));
        }
    }
}
