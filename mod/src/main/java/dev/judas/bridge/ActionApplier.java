package dev.judas.bridge;

import com.google.gson.JsonObject;
import net.minecraft.client.Minecraft;
import net.minecraft.client.entity.EntityPlayerSP;
import net.minecraft.client.settings.KeyBinding;
import net.minecraft.entity.player.EntityPlayer;
import net.minecraft.util.AxisAlignedBB;
import net.minecraft.util.MathHelper;
import net.minecraft.util.MovingObjectPosition;
import net.minecraft.util.Vec3;

/**
 * Applique l'action reçue de serve/ : rotations (déjà clampées par
 * l'humanisation côté serveur), touches via KeyBinding, clic d'attaque.
 *
 * L'attaque réplique le clic vanilla : raycast œil -> regard (portée 3.0)
 * contre l'AABB de la cible étendue de getCollisionBorderSize() ; si touché
 * -> playerController.attackEntity, sinon swing à vide.
 */
public final class ActionApplier {

    private final StateCollector collector;

    public ActionApplier(StateCollector collector) {
        this.collector = collector;
    }

    public void apply(Minecraft mc, JsonObject a) {
        EntityPlayerSP p = mc.thePlayer;
        if (p == null) return;

        // --- rotations ---
        p.rotationYaw += (float) a.get("dyaw").getAsDouble();
        p.rotationPitch = MathHelper.clamp_float(
                p.rotationPitch + (float) a.get("dpitch").getAsDouble(),
                -90.0F, 90.0F);

        // --- touches ---
        int fwd = a.get("forward").getAsInt();
        int strafe = a.get("strafe").getAsInt();
        key(mc.gameSettings.keyBindForward, fwd > 0);
        key(mc.gameSettings.keyBindBack, fwd < 0);
        key(mc.gameSettings.keyBindLeft, strafe > 0);    // strafe +1 = gauche (vanilla)
        key(mc.gameSettings.keyBindRight, strafe < 0);
        key(mc.gameSettings.keyBindJump, a.get("jump").getAsBoolean());
        key(mc.gameSettings.keyBindSprint, a.get("sprint").getAsBoolean());

        // --- clic d'attaque ---
        if (a.get("attack").getAsBoolean()) {
            doAttack(mc, p);
        }
    }

    private void doAttack(Minecraft mc, EntityPlayerSP p) {
        EntityPlayer target = collector.getTarget();
        collector.onOwnAttack();
        if (target != null && rayHitsTarget(p, target)) {
            mc.playerController.attackEntity(p, target);
        }
        p.swingItem();
    }

    /** Réplique EntityRenderer.getMouseOver (entités, reach 3.0). */
    private static boolean rayHitsTarget(EntityPlayerSP p, EntityPlayer target) {
        Vec3 eye = new Vec3(p.posX, p.posY + (double) p.getEyeHeight(), p.posZ);
        Vec3 look = p.getLook(1.0F);
        Vec3 end = eye.addVector(look.xCoord * 3.0, look.yCoord * 3.0, look.zCoord * 3.0);
        float border = target.getCollisionBorderSize();
        AxisAlignedBB box = target.getEntityBoundingBox().expand(border, border, border);
        if (box.isVecInside(eye)) return true;
        MovingObjectPosition hit = box.calculateIntercept(eye, end);
        return hit != null;
    }

    public void releaseAll(Minecraft mc) {
        key(mc.gameSettings.keyBindForward, false);
        key(mc.gameSettings.keyBindBack, false);
        key(mc.gameSettings.keyBindLeft, false);
        key(mc.gameSettings.keyBindRight, false);
        key(mc.gameSettings.keyBindJump, false);
        key(mc.gameSettings.keyBindSprint, false);
    }

    private static void key(KeyBinding kb, boolean down) {
        KeyBinding.setKeyBindState(kb.getKeyCode(), down);
    }
}
