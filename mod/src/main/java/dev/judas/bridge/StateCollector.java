package dev.judas.bridge;

import com.google.gson.JsonObject;
import net.minecraft.client.Minecraft;
import net.minecraft.client.entity.EntityPlayerSP;
import net.minecraft.entity.player.EntityPlayer;

/**
 * Collecte l'état du jeu à chaque tick : soi + cible, au format JSON attendu
 * par serve/ (voir serve/protocol.py). Les vitesses de la cible sont estimées
 * par delta de position (le client ne connaît pas le motion exact des autres
 * joueurs). Les compteurs de hits sont heuristiques : front montant de
 * hurtTime, attribué si on a swingé récemment.
 */
public final class StateCollector {

    private EntityPlayer target;
    private int hitsDealt = 0;
    private int hitsTaken = 0;
    private int prevTargetHurt = 0;
    private int prevSelfHurt = 0;
    private int ticksSinceOwnSwing = 100;
    private long tick = 0;

    public void onOwnAttack() {
        ticksSinceOwnSwing = 0;
    }

    public EntityPlayer getTarget() {
        return target;
    }

    public void resetCounters() {
        hitsDealt = 0;
        hitsTaken = 0;
        tick = 0;
    }

    /** Cible = joueur le plus proche (hors soi), max 64 blocs. */
    private EntityPlayer findTarget(Minecraft mc, EntityPlayerSP self) {
        EntityPlayer best = null;
        double bestDist = 64.0 * 64.0;
        for (Object o : mc.theWorld.playerEntities) {
            EntityPlayer p = (EntityPlayer) o;
            if (p == self || p.isInvisible() || p.isDead) continue;
            double d = self.getDistanceSqToEntity(p);
            if (d < bestDist) {
                bestDist = d;
                best = p;
            }
        }
        return best;
    }

    /** Construit le message d'état du tick courant, ou null si pas de cible. */
    public JsonObject collect(Minecraft mc) {
        EntityPlayerSP self = mc.thePlayer;
        if (self == null || mc.theWorld == null) return null;
        target = findTarget(mc, self);
        if (target == null) return null;

        ticksSinceOwnSwing++;
        tick++;

        // fronts montants de hurtTime -> compteurs de hits
        int tHurt = target.hurtTime;
        if (tHurt > prevTargetHurt && tHurt >= 9 && ticksSinceOwnSwing <= 3) {
            hitsDealt++;
        }
        prevTargetHurt = tHurt;
        int sHurt = self.hurtTime;
        if (sHurt > prevSelfHurt && sHurt >= 9) {
            hitsTaken++;
        }
        prevSelfHurt = sHurt;

        JsonObject msg = new JsonObject();
        msg.addProperty("t", "state");
        msg.addProperty("tick", tick);
        msg.add("self", playerJson(self,
                self.motionX, self.motionY, self.motionZ, hitsDealt));
        msg.add("target", playerJson(target,
                target.posX - target.lastTickPosX,
                target.posY - target.lastTickPosY,
                target.posZ - target.lastTickPosZ, hitsTaken));
        return msg;
    }

    private static JsonObject playerJson(EntityPlayer p, double vx, double vy,
                                         double vz, int hits) {
        JsonObject o = new JsonObject();
        o.addProperty("x", p.posX);
        o.addProperty("y", p.posY);
        o.addProperty("z", p.posZ);
        o.addProperty("vx", vx);
        o.addProperty("vy", vy);
        o.addProperty("vz", vz);
        o.addProperty("yaw", p.rotationYaw);
        o.addProperty("pitch", p.rotationPitch);
        o.addProperty("onGround", p.onGround);
        o.addProperty("sprinting", p.isSprinting());
        o.addProperty("hurtTime", p.hurtTime);
        o.addProperty("hits", hits);
        return o;
    }
}
