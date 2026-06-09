package dev.judas.bridge;

import com.google.gson.JsonObject;
import com.google.gson.JsonParser;
import net.minecraft.client.Minecraft;
import net.minecraftforge.client.event.RenderGameOverlayEvent;
import net.minecraftforge.common.MinecraftForge;
import net.minecraftforge.fml.client.registry.ClientRegistry;
import net.minecraftforge.fml.common.Mod;
import net.minecraftforge.fml.common.event.FMLInitializationEvent;
import net.minecraftforge.fml.common.eventhandler.SubscribeEvent;
import net.minecraftforge.fml.common.gameevent.TickEvent;
import net.minecraft.client.settings.KeyBinding;
import org.lwjgl.input.Keyboard;

import java.io.IOException;

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
    private final TraceRecorder recorder = new TraceRecorder();
    private final WsClient ws = new WsClient(WS_HOST, WS_PORT, WS_PATH);
    private final JsonParser parser = new JsonParser();

    private KeyBinding keyToggle;
    private KeyBinding keyKill;
    private KeyBinding keyTrace;

    private boolean enabled = false;
    private int reconnectIn = 0;

    @Mod.EventHandler
    public void init(FMLInitializationEvent event) {
        keyToggle = new KeyBinding("Toggle bot", Keyboard.KEY_K, "Judas");
        keyKill = new KeyBinding("Kill-switch", Keyboard.KEY_L, "Judas");
        keyTrace = new KeyBinding("Enregistrer trace", Keyboard.KEY_J, "Judas");
        ClientRegistry.registerKeyBinding(keyToggle);
        ClientRegistry.registerKeyBinding(keyKill);
        ClientRegistry.registerKeyBinding(keyTrace);
        MinecraftForge.EVENT_BUS.register(this);
    }

    @SubscribeEvent
    public void onClientTick(TickEvent.ClientTickEvent event) {
        if (event.phase != TickEvent.Phase.START) return;
        Minecraft mc = Minecraft.getMinecraft();
        if (mc.thePlayer == null) return;

        handleKeys(mc);

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
        if (msg != null) {
            try {
                JsonObject action = parser.parse(msg).getAsJsonObject();
                if ("action".equals(action.get("t").getAsString())) {
                    applier.apply(mc, action);
                }
            } catch (RuntimeException ignored) {
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
                applier.releaseAll(mc);
                ws.close();
            } else {
                collector.resetCounters();
                reconnectIn = 0;
            }
        }
        while (keyKill.isPressed()) {
            enabled = false;
            applier.releaseAll(mc);
            ws.close();
            if (recorder.isRecording()) recorder.stop(mc);
        }
        while (keyTrace.isPressed()) {
            if (recorder.isRecording()) recorder.stop(mc);
            else recorder.start(mc);
        }
    }

    @SubscribeEvent
    public void onOverlay(RenderGameOverlayEvent.Text event) {
        Minecraft mc = Minecraft.getMinecraft();
        if (mc.thePlayer == null) return;
        String status = enabled
                ? (ws.isOpen() ? "\u00A7bACTIF" : "\u00A7econnexion...")
                : "\u00A78inactif";
        if (recorder.isRecording()) status = "\u00A7dTRACE...";
        event.left.add("\u00A77[\u00A7bJudas\u00A77] " + status);
    }
}
