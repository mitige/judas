package dev.judas.bridge;

import com.google.gson.JsonArray;
import com.google.gson.JsonElement;
import com.google.gson.JsonObject;
import net.minecraft.block.state.IBlockState;
import net.minecraft.client.Minecraft;
import net.minecraft.client.entity.EntityPlayerSP;
import net.minecraft.entity.player.EntityPlayer;
import net.minecraft.util.BlockPos;
import net.minecraft.util.MathHelper;
import net.minecraft.world.World;

import java.io.File;
import java.io.FileWriter;
import java.io.IOException;
import java.io.PrintWriter;
import java.util.HashSet;
import java.util.Locale;
import java.util.Set;

/**
 * Collecte l'état du jeu à chaque tick : soi + cible, au format JSON attendu
 * par serve/ (voir serve/protocol.py). Les vitesses de la cible sont estimées
 * par delta de position (le client ne connaît pas le motion exact des autres
 * joueurs). Les compteurs de hits sont heuristiques : front montant de
 * hurtTime, attribué si on a swingé récemment.
 *
 * Fidélité (thread jeu) : hurtResistantTime (i-frames 0..20) est lu exactement
 * sur les deux joueurs — champ public vanilla — au lieu d'être approximé
 * hurtTime*2 ; jumpTicks de soi est suivi par un compteur miroir (le champ
 * vanilla est privé). Les chaînes de combo sont suivies pour le HUD diagnostic.
 */
public final class StateCollector {

    private EntityPlayer target;
    private int hitsDealt = 0;
    private int hitsTaken = 0;
    private int myCombo = 0;
    private int oppCombo = 0;
    private int prevTargetHurt = 0;
    private int prevSelfHurt = 0;
    private int ticksSinceOwnSwing = 100;
    private int selfJumpTicks = 0;       // miroir de EntityLivingBase.jumpTicks (privé)
    private double lastDist = 0.0;
    private long tick = 0;
    private boolean friendsEnabled = false;
    private final Set<String> friends = new HashSet<>();
    private boolean knockbackDumpEnabled = false;
    private File knockbackDumpFile;
    private int knockbackDumpSamples = 0;
    private String knockbackDumpStatus = "off";

    private static final int ARENA_SCAN_RADIUS = 96;
    private static final int ARENA_MIN_SIZE = 6;
    private static final int ARENA_MAX_SIZE = 128;
    private static final int[] ARENA_SIDE_SAMPLES = new int[] {-2, -1, 0, 1, 2};

    public void onOwnAttack() {
        ticksSinceOwnSwing = 0;
    }

    /** Appelé par ActionApplier quand un saut est commandé au sol. */
    public void onJump(boolean onGround) {
        if (onGround && selfJumpTicks == 0) selfJumpTicks = 10;
    }

    public EntityPlayer getTarget() {
        return target;
    }

    public void updateFriendsFromAction(JsonObject action) {
        if (action == null || !action.has("friends")) return;
        JsonElement raw = action.get("friends");
        if (raw.isJsonPrimitive()) {
            friendsEnabled = raw.getAsBoolean();
            return;
        }
        if (!raw.isJsonObject()) return;
        JsonObject cfg = raw.getAsJsonObject();
        if (cfg.has("enabled")) friendsEnabled = cfg.get("enabled").getAsBoolean();
        if (!cfg.has("names") || !cfg.get("names").isJsonArray()) return;
        friends.clear();
        JsonArray names = cfg.getAsJsonArray("names");
        for (JsonElement element : names) {
            String name = cleanName(element.getAsString());
            if (!name.isEmpty()) friends.add(name);
        }
    }

    public boolean isFriend(EntityPlayer p) {
        return friendsEnabled && p != null && friends.contains(cleanName(p.getName()));
    }

    public boolean isTargetFriend() {
        return isFriend(target);
    }

    public String friendStatusLine() {
        return "\u00A77 friends=" + (friendsEnabled ? "\u00A7aon" : "\u00A78off")
                + "\u00A77 count=\u00A7f" + friends.size();
    }

    public void updateKnockbackDumpFromAction(JsonObject action) {
        if (action == null || !action.has("knockback_dump")) return;
        JsonElement raw = action.get("knockback_dump");
        if (raw == null || raw.isJsonNull()) return;
        if (raw.isJsonPrimitive()) {
            knockbackDumpEnabled = raw.getAsBoolean();
        } else if (raw.isJsonObject()) {
            JsonObject cfg = raw.getAsJsonObject();
            if (cfg.has("enabled")) knockbackDumpEnabled = cfg.get("enabled").getAsBoolean();
        }
        if (!knockbackDumpEnabled) {
            knockbackDumpStatus = "off";
        } else if ("off".equals(knockbackDumpStatus)) {
            knockbackDumpStatus = knockbackDumpSamples > 0
                    ? "samples=" + knockbackDumpSamples
                    : "ready";
        }
    }

    public String knockbackDumpStatusLine() {
        return "\u00A77 kb-dump=" + (knockbackDumpEnabled ? "\u00A7aon" : "\u00A78off")
                + "\u00A77 " + knockbackDumpStatus;
    }

    public int getHitsDealt() {
        return hitsDealt;
    }

    public int getHitsTaken() {
        return hitsTaken;
    }

    public int getMyCombo() {
        return myCombo;
    }

    public int getOppCombo() {
        return oppCombo;
    }

    public double getLastDist() {
        return lastDist;
    }

    public void resetCounters() {
        hitsDealt = 0;
        hitsTaken = 0;
        myCombo = 0;
        oppCombo = 0;
        tick = 0;
    }

    /** Cible = joueur le plus proche (hors soi), max 64 blocs. */
    private EntityPlayer findTarget(Minecraft mc, EntityPlayerSP self) {
        EntityPlayer best = null;
        double bestDist = 64.0 * 64.0;
        for (Object o : mc.theWorld.playerEntities) {
            EntityPlayer p = (EntityPlayer) o;
            if (p == self || p.isInvisible() || p.isDead) continue;
            if (isFriend(p)) continue;
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
        if (selfJumpTicks > 0) selfJumpTicks--;
        tick++;
        lastDist = self.getDistanceToEntity(target);

        // fronts montants de hurtTime -> compteurs de hits + chaînes de combo
        int tHurt = target.hurtTime;
        if (tHurt > prevTargetHurt && tHurt >= 9 && ticksSinceOwnSwing <= 3) {
            hitsDealt++;
            myCombo++;
            oppCombo = 0;
            dumpKnockbackEvent(mc, "target", target, self);
        }
        prevTargetHurt = tHurt;
        int sHurt = self.hurtTime;
        if (sHurt > prevSelfHurt && sHurt >= 9) {
            hitsTaken++;
            oppCombo++;
            myCombo = 0;
            dumpKnockbackEvent(mc, "self", self, target);
        }
        prevSelfHurt = sHurt;

        JsonObject msg = new JsonObject();
        msg.addProperty("t", "state");
        msg.addProperty("tick", tick);
        msg.add("self", playerJson(self,
                self.motionX, self.motionY, self.motionZ, hitsDealt, selfJumpTicks));
        msg.add("target", playerJson(target,
                target.posX - target.lastTickPosX,
                target.posY - target.lastTickPosY,
                target.posZ - target.lastTickPosZ, hitsTaken, 0));
        JsonObject arena = detectArena(mc.theWorld, self, target);
        if (arena != null) msg.add("arena", arena);
        return msg;
    }

    private JsonObject detectArena(World world, EntityPlayerSP self, EntityPlayer target) {
        if (world == null || self == null || target == null) return null;
        int selfFloor = MathHelper.floor_double(self.getEntityBoundingBox().minY + 0.01);
        int targetFloor = MathHelper.floor_double(target.getEntityBoundingBox().minY + 0.01);
        int floorY = Math.min(selfFloor, targetFloor);
        double centerX = (self.posX + target.posX) * 0.5;
        double centerZ = (self.posZ + target.posZ) * 0.5;

        Double minX = findArenaBoundaryX(world, centerX, centerZ, floorY, -1);
        Double maxX = findArenaBoundaryX(world, centerX, centerZ, floorY, 1);
        Double minZ = findArenaBoundaryZ(world, centerX, centerZ, floorY, -1);
        Double maxZ = findArenaBoundaryZ(world, centerX, centerZ, floorY, 1);
        if (minX == null || maxX == null || minZ == null || maxZ == null) return null;
        double sizeX = maxX - minX;
        double sizeZ = maxZ - minZ;
        if (sizeX < ARENA_MIN_SIZE || sizeZ < ARENA_MIN_SIZE) return null;
        if (sizeX > ARENA_MAX_SIZE || sizeZ > ARENA_MAX_SIZE) return null;

        JsonObject arena = new JsonObject();
        arena.addProperty("origin_x", minX);
        arena.addProperty("origin_z", minZ);
        arena.addProperty("size_x", maxX - minX);
        arena.addProperty("size_z", maxZ - minZ);
        arena.addProperty("floor_y", floorY);
        return arena;
    }

    private Double findArenaBoundaryX(World world, double centerX, double centerZ,
                                      int floorY, int dir) {
        Double obstacle = findObstacleBoundaryX(world, centerX, centerZ, floorY, dir);
        return obstacle != null ? obstacle
                : findFloorBoundaryX(world, centerX, centerZ, floorY, dir);
    }

    private Double findArenaBoundaryZ(World world, double centerX, double centerZ,
                                      int floorY, int dir) {
        Double obstacle = findObstacleBoundaryZ(world, centerX, centerZ, floorY, dir);
        return obstacle != null ? obstacle
                : findFloorBoundaryZ(world, centerX, centerZ, floorY, dir);
    }

    private Double findObstacleBoundaryX(World world, double centerX, double centerZ,
                                         int floorY, int dir) {
        int startX = MathHelper.floor_double(centerX);
        int baseZ = MathHelper.floor_double(centerZ);
        for (int step = 1; step <= ARENA_SCAN_RADIUS; step++) {
            int x = startX + dir * step;
            int hits = 0;
            for (int dz : ARENA_SIDE_SAMPLES) {
                if (hasWallColumn(world, new BlockPos(x, floorY, baseZ + dz))) hits++;
            }
            if (hits >= 2) return dir < 0 ? x + 1.0 : (double) x;
        }
        return null;
    }

    private Double findObstacleBoundaryZ(World world, double centerX, double centerZ,
                                         int floorY, int dir) {
        int baseX = MathHelper.floor_double(centerX);
        int startZ = MathHelper.floor_double(centerZ);
        for (int step = 1; step <= ARENA_SCAN_RADIUS; step++) {
            int z = startZ + dir * step;
            int hits = 0;
            for (int dx : ARENA_SIDE_SAMPLES) {
                if (hasWallColumn(world, new BlockPos(baseX + dx, floorY, z))) hits++;
            }
            if (hits >= 2) return dir < 0 ? z + 1.0 : (double) z;
        }
        return null;
    }

    private Double findFloorBoundaryX(World world, double centerX, double centerZ,
                                      int floorY, int dir) {
        int startX = MathHelper.floor_double(centerX);
        int baseZ = MathHelper.floor_double(centerZ);
        boolean sawFloor = false;
        for (int step = 0; step <= ARENA_SCAN_RADIUS; step++) {
            int x = startX + dir * step;
            int floors = 0;
            for (int dz : ARENA_SIDE_SAMPLES) {
                if (hasFloor(world, new BlockPos(x, floorY, baseZ + dz))) floors++;
            }
            if (floors >= 2) {
                sawFloor = true;
            } else if (sawFloor && step > 0) {
                return dir < 0 ? x + 1.0 : (double) x;
            }
        }
        return null;
    }

    private Double findFloorBoundaryZ(World world, double centerX, double centerZ,
                                      int floorY, int dir) {
        int baseX = MathHelper.floor_double(centerX);
        int startZ = MathHelper.floor_double(centerZ);
        boolean sawFloor = false;
        for (int step = 0; step <= ARENA_SCAN_RADIUS; step++) {
            int z = startZ + dir * step;
            int floors = 0;
            for (int dx : ARENA_SIDE_SAMPLES) {
                if (hasFloor(world, new BlockPos(baseX + dx, floorY, z))) floors++;
            }
            if (floors >= 2) {
                sawFloor = true;
            } else if (sawFloor && step > 0) {
                return dir < 0 ? z + 1.0 : (double) z;
            }
        }
        return null;
    }

    private boolean hasWallColumn(World world, BlockPos pos) {
        return blocksMovement(world, pos)
                || blocksMovement(world, pos.up())
                || blocksMovement(world, pos.up().up());
    }

    private boolean hasFloor(World world, BlockPos pos) {
        return blocksMovement(world, pos.down())
                && !blocksMovement(world, pos)
                && !blocksMovement(world, pos.up());
    }

    private boolean blocksMovement(World world, BlockPos pos) {
        IBlockState state = world.getBlockState(pos);
        return state != null
                && state.getBlock() != null
                && state.getBlock().getMaterial().blocksMovement();
    }

    private void dumpKnockbackEvent(Minecraft mc, String role, EntityPlayer victim, EntityPlayer attacker) {
        if (!knockbackDumpEnabled || mc == null || victim == null) return;
        if (knockbackDumpFile == null) {
            knockbackDumpFile = new File(mc.mcDataDir, "judas-kb-dump.jsonl");
        }
        double vx = victim.motionX;
        double vy = victim.motionY;
        double vz = victim.motionZ;
        double dxTick = victim.posX - victim.lastTickPosX;
        double dyTick = victim.posY - victim.lastTickPosY;
        double dzTick = victim.posZ - victim.lastTickPosZ;
        if (Math.abs(vx) + Math.abs(vz) < 0.000001
                && Math.abs(dxTick) + Math.abs(dzTick) > 0.000001) {
            vx = dxTick;
            vy = dyTick;
            vz = dzTick;
        }
        double awayX = 0.0;
        double awayZ = 0.0;
        double dist = 0.0;
        if (attacker != null) {
            double ax = victim.posX - attacker.posX;
            double az = victim.posZ - attacker.posZ;
            dist = Math.sqrt(ax * ax + az * az);
            if (dist > 0.000001) {
                awayX = ax / dist;
                awayZ = az / dist;
            }
        }
        double horiz = Math.sqrt(vx * vx + vz * vz);
        double along = dist > 0.000001 ? vx * awayX + vz * awayZ : horiz;
        double kbHEst = along / 0.4;
        double kbVEst = vy / 0.4;
        String server = mc.getCurrentServerData() != null
                ? mc.getCurrentServerData().serverIP
                : "singleplayer";
        String victimName = victim.getName();
        String attackerName = attacker != null ? attacker.getName() : "";
        String line = String.format(Locale.ROOT,
                "{\"event\":\"knockback\",\"tick\":%d,\"server\":\"%s\","
                        + "\"role\":\"%s\",\"victim\":\"%s\",\"attacker\":\"%s\","
                        + "\"dist\":%.6f,\"motion_x\":%.6f,\"motion_y\":%.6f,"
                        + "\"motion_z\":%.6f,\"delta_x\":%.6f,\"delta_y\":%.6f,"
                        + "\"delta_z\":%.6f,\"horiz\":%.6f,\"along\":%.6f,"
                        + "\"kb_h_est\":%.6f,\"kb_v_est\":%.6f,"
                        + "\"hurt_time\":%d,\"hurt_resistant_time\":%d,"
                        + "\"on_ground\":%s,\"sprinting\":%s,"
                        + "\"my_combo\":%d,\"opp_combo\":%d}",
                tick,
                jsonString(server),
                jsonString(role),
                jsonString(victimName),
                jsonString(attackerName),
                dist,
                vx,
                vy,
                vz,
                dxTick,
                dyTick,
                dzTick,
                horiz,
                along,
                kbHEst,
                kbVEst,
                victim.hurtTime,
                victim.hurtResistantTime,
                String.valueOf(victim.onGround),
                String.valueOf(victim.isSprinting()),
                myCombo,
                oppCombo);
        try (PrintWriter out = new PrintWriter(new FileWriter(knockbackDumpFile, true))) {
            out.println(line);
            knockbackDumpSamples++;
            knockbackDumpStatus = "samples=" + knockbackDumpSamples;
        } catch (IOException ignored) {
            knockbackDumpStatus = "write_error";
        }
    }

    private static JsonObject playerJson(EntityPlayer p, double vx, double vy,
                                         double vz, int hits, int jumpTicks) {
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
        o.addProperty("hurtResistantTime", p.hurtResistantTime);  // i-frames exacts (0..20)
        o.addProperty("jumpTicks", jumpTicks);                    // soi: miroir ; cible: 0
        o.addProperty("clickCooldown", estimatedClickCooldown(p)); // cible: signal swing recent
        o.addProperty("hits", hits);
        return o;
    }

    private static int estimatedClickCooldown(EntityPlayer p) {
        if (!p.isSwingInProgress) return 0;
        if (p.swingProgressInt <= 1) return 2;
        if (p.swingProgressInt <= 2) return 1;
        return 0;
    }

    private static String cleanName(String text) {
        if (text == null) return "";
        return text.replaceAll("\u00A7.", "").trim().toLowerCase(Locale.ROOT);
    }

    private static String jsonString(String text) {
        if (text == null) return "";
        return text.replace("\\", "\\\\")
                .replace("\"", "\\\"")
                .replace("\r", " ")
                .replace("\n", " ");
    }
}
