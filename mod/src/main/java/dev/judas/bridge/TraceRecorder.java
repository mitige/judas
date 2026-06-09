package dev.judas.bridge;

import com.google.gson.JsonObject;
import net.minecraft.client.Minecraft;
import net.minecraft.client.entity.EntityPlayerSP;
import net.minecraft.client.settings.KeyBinding;

import java.io.File;
import java.io.FileWriter;
import java.io.IOException;
import java.text.SimpleDateFormat;
import java.util.Date;

/**
 * Enregistreur de traces golden : exécute une séquence d'inputs scriptée et
 * écrit l'état exact du joueur à chaque tick (JSONL). Ces fichiers servent à
 * calibrer sim_ref contre le vrai jeu (tools/golden_compare.py).
 *
 * Séquence : idle 20t -> marche 60t -> sprint 60t -> sauts maintenus 60t ->
 * sprint-jump 60t -> strafe diagonale 60t -> idle 20t.
 */
public final class TraceRecorder {

    private FileWriter writer;
    private int tick = -1;
    private static final int TOTAL = 340;

    public boolean isRecording() {
        return tick >= 0;
    }

    public void start(Minecraft mc) {
        if (isRecording()) return;
        try {
            File dir = new File(mc.mcDataDir, "judas-traces");
            dir.mkdirs();
            String name = "trace-" + new SimpleDateFormat("yyyyMMdd-HHmmss").format(new Date()) + ".jsonl";
            writer = new FileWriter(new File(dir, name));
            tick = 0;
        } catch (IOException e) {
            writer = null;
            tick = -1;
        }
    }

    /** À appeler chaque ClientTickEvent.START quand l'enregistrement est actif. */
    public void onTick(Minecraft mc) {
        EntityPlayerSP p = mc.thePlayer;
        if (!isRecording() || p == null) return;

        boolean fwd = false, jump = false, sprint = false, left = false;
        if (tick < 20) {                       // idle
        } else if (tick < 80) {                // marche
            fwd = true;
        } else if (tick < 140) {               // sprint
            fwd = true; sprint = true;
        } else if (tick < 200) {               // sauts maintenus
            jump = true;
        } else if (tick < 260) {               // sprint-jump
            fwd = true; sprint = true; jump = true;
        } else if (tick < 320) {               // diagonale
            fwd = true; left = true;
        }
        key(mc.gameSettings.keyBindForward, fwd);
        key(mc.gameSettings.keyBindJump, jump);
        key(mc.gameSettings.keyBindSprint, sprint);
        key(mc.gameSettings.keyBindLeft, left);

        JsonObject o = new JsonObject();
        o.addProperty("tick", tick);
        o.addProperty("x", p.posX);
        o.addProperty("y", p.posY);
        o.addProperty("z", p.posZ);
        o.addProperty("vx", p.motionX);
        o.addProperty("vy", p.motionY);
        o.addProperty("vz", p.motionZ);
        o.addProperty("yaw", p.rotationYaw);
        o.addProperty("pitch", p.rotationPitch);
        o.addProperty("onGround", p.onGround);
        o.addProperty("sprinting", p.isSprinting());
        o.addProperty("in_fwd", fwd);
        o.addProperty("in_jump", jump);
        o.addProperty("in_sprint", sprint);
        o.addProperty("in_left", left);
        try {
            writer.write(o.toString() + "\n");
        } catch (IOException ignored) {
        }

        tick++;
        if (tick >= TOTAL) stop(mc);
    }

    public void stop(Minecraft mc) {
        if (!isRecording()) return;
        key(mc.gameSettings.keyBindForward, false);
        key(mc.gameSettings.keyBindJump, false);
        key(mc.gameSettings.keyBindSprint, false);
        key(mc.gameSettings.keyBindLeft, false);
        try {
            if (writer != null) writer.close();
        } catch (IOException ignored) {
        }
        writer = null;
        tick = -1;
    }

    private static void key(KeyBinding kb, boolean down) {
        KeyBinding.setKeyBindState(kb.getKeyCode(), down);
    }
}
