package dev.judas.bridge;

import com.google.gson.JsonElement;
import com.google.gson.JsonObject;
import net.minecraft.block.material.Material;
import net.minecraft.block.state.IBlockState;
import net.minecraft.client.Minecraft;
import net.minecraft.client.entity.EntityPlayerSP;
import net.minecraft.client.settings.KeyBinding;
import net.minecraft.entity.player.EntityPlayer;
import net.minecraft.inventory.Slot;
import net.minecraft.item.ItemStack;
import net.minecraft.util.BlockPos;
import net.minecraft.util.MathHelper;
import net.minecraft.world.World;

import java.util.Locale;

final class AutoGapple {
    private boolean enabled = true;
    private float healthThreshold = 14.0F;
    private float criticalHealthThreshold = 8.0F;
    private float safeDistance = 11.50F;
    private boolean retreatEnabled = true;
    private float retreatDistance = 18.0F;
    private float absorptionThreshold = -1.0F;
    private boolean fastRetreat = true;
    private boolean retreatHops = true;
    private boolean sprintHopHold = true;
    private boolean avoidObstacles = true;
    private boolean retreatStrafe = true;
    private boolean wallSlide = true;
    private boolean retreatSpeedLock = true;
    private boolean retreatVelocityAssist = true;
    private boolean retreatSpeedFirst = true;
    private boolean retreatFullSpeed = true;
    private float retreatSpeedFloor = 4.50F;
    private float retreatMaxSpeed = 4.80F;
    private float retreatAccel = 5.50F;
    private boolean retreatSprintRetap = true;
    private int retreatSprintRetapMaxTicks = 2;
    private boolean retreatAirControl = true;
    private boolean retreatStepAssist = true;
    private float retreatStepHeight = 1.20F;
    private boolean fallbackRetreat = true;
    private boolean retreatInputLock = true;
    private boolean forceSprintRetreat = true;
    private boolean releaseRetreatOnHit = true;
    private boolean criticalRearmOnly = true;
    private boolean criticalTrappedEat = true;
    private float retreatTurnLimitDeg = 360.0F;
    private float eatingRetreatTurnLimitDeg = 360.0F;
    private int retreatPathHoldMaxTicks = 2;
    private int retreatStuckAbortTicks = 4;
    private int minRetreatTicks = 0;
    private int maxRetreatTicks = 64;
    private int criticalRetreatMaxTicks = 6;
    private int criticalEatCommitTicks = 12;
    private int combatRecoveryTicks = 6;
    private int retreatStrafeHoldMaxTicks = 5;
    private int retreatObstacleJumpHoldMaxTicks = 60;
    private int retreatObstacleEscapeTicks = 120;
    private boolean retreatPanicSpeed = true;
    private float retreatObstacleLookahead = 24.00F;
    private int criticalTrappedStuckTicks = 2;
    private boolean retreatInterrupted = false;
    private int retreatCombatTicks = 0;
    private int retreatPathHoldTicks = 0;
    private int retreatStrafeHoldTicks = 0;
    private int retreatObstacleJumpTicks = 0;
    private int retreatTicks = 0;
    private int retreatStuckTicks = 0;
    private int retreatSprintRetapTicks = 0;
    private int lastRetreatStrafe = 0;
    private double lastRetreatX = 0.0;
    private double lastRetreatZ = 0.0;
    private double previousRetreatPosX = 0.0;
    private double previousRetreatPosZ = 0.0;
    private float previousStepHeight = -1.0F;
    private int eatTicks = 0;
    private int previousHotbar = -1;
    private int restoreContainerSlot = -1;
    private int restoreHotbar = -1;
    private String status = "off";

    private static final int EAT_TICKS = 36;
    private static final float ACTIVE_ABSORPTION = 1.0F;
    private static final double DEFAULT_RETREAT_TURN_LIMIT_DEG = 360.0;
    private static final double DEFAULT_RETREAT_EAT_TURN_LIMIT_DEG = 360.0;
    private static final int RETREAT_PATH_HOLD_TICKS = 2;
    private static final int RETREAT_STRAFE_HOLD_TICKS = 5;
    private static final int RETREAT_OBSTACLE_JUMP_HOLD_TICKS = 48;
    private static final int RETREAT_OBSTACLE_ESCAPE_TICKS = 120;
    private static final double RETREAT_REPLAN_MARGIN = 0.00;
    private static final int DEFAULT_RETREAT_STUCK_ABORT_TICKS = 8;
    private static final int RETREAT_START_BURST_TICKS = 8;
    private static final int RETREAT_STUCK_JUMP_TICKS = 1;
    private static final double RETREAT_STUCK_EPS_SQ = 0.04 * 0.04;
    private static final double RETREAT_MIN_PROGRESS = 0.080;
    private static final int RETREAT_NEAR_BLOCKED_ESCAPE = 1;
    private static final int RETREAT_NEAR_UNSAFE_ESCAPE = 1;
    private static final double RETREAT_PANIC_MIN_AWAY_DOT = 0.00;
    private static final double RETREAT_PANIC_MIN_SCORE = -75.0;
    private static final double RETREAT_PANIC_SIDE_ESCAPE = 0.02;
    private static final double RETREAT_NEAR_BLOCK_AHEAD = 4.60;
    private static final double RETREAT_HARD_BLOCK_AHEAD = 18.00;
    private static final double RETREAT_SIDE_WIDTH = 0.48;
    private static final double RETREAT_WIDE_SIDE_WIDTH = 1.08;
    private static final double RETREAT_EXTRA_WIDE_SIDE_WIDTH = 1.32;
    private static final double RETREAT_MIN_AWAY_DOT = 0.04;
    private static final double RETREAT_SPEED_MIN_AWAY_DOT = 0.24;
    private static final double RETREAT_SPEED_FIRST_MIN_AWAY_DOT = 0.76;
    private static final double RETREAT_STRAFE_SPEED_BLEND = 0.55;
    private static final double RETREAT_ESCAPE_SPEED_BLEND = 1.25;
    private static final double RETREAT_FAST_STRAFE_SPEED_BLEND = 0.08;
    private static final double RETREAT_STRAFE_TURN_DEG = 6.0;
    private static final double RETREAT_STRAFE_SIDE_ESCAPE = 0.06;
    private static final double RETREAT_SPEED_STRAFE_SUPPRESS_SIDE = 0.44;
    private static final double RETREAT_STRAIGHT_OVERRIDE_CENTER_CLEAR = 0.90;
    private static final double RETREAT_STRAIGHT_OVERRIDE_SCORE_GAP = 110.0;
    private static final double RETREAT_CLEAN_STRAFE_SUPPRESS_SIDE = 0.88;
    private static final double RETREAT_SIDE_MOTION_DAMP = 0.92;
    private static final double RETREAT_SAFER_PATH_SCORE_MARGIN = 18.0;
    private static final double RETREAT_PANIC_SPEED_MULTIPLIER = 1.18;
    private static final double RETREAT_PANIC_ACCEL_MULTIPLIER = 1.15;
    private static final double RETREAT_START_BURST_SPEED_MULTIPLIER = 1.08;
    private static final double RETREAT_START_BURST_ACCEL_MULTIPLIER = 1.20;
    private static final double RETREAT_FULL_SPEED_CLEAN_SIDE_LIMIT = 0.90;
    private static final double[] RETREAT_ANGLE_OFFSETS = new double[] {
            0.0, 2.5, -2.5, 5.0, -5.0, 7.5, -7.5, 10.0, -10.0,
            11.25, -11.25, 12.5, -12.5, 15.0, -15.0,
            18.75, -18.75, 22.5, -22.5, 25.0, -25.0, 30.0, -30.0,
            32.5, -32.5, 37.5, -37.5, 42.5, -42.5, 45.0, -45.0, 52.5, -52.5,
            60.0, -60.0, 67.5, -67.5, 75.0, -75.0,
            82.5, -82.5, 90.0, -90.0,
            105.0, -105.0, 120.0, -120.0, 135.0, -135.0,
            150.0, -150.0, 165.0, -165.0, 180.0
    };
    private static final double[] RETREAT_AHEAD_SAMPLES = new double[] {
            0.08, 0.12, 0.20, 0.35, 0.50, 0.70, 0.90, 1.15, 1.45, 1.80, 2.20, 2.70,
            3.35, 4.15, 5.20, 6.40, 7.80, 9.40, 11.20, 12.80, 14.40, 16.00,
            18.00, 20.00
    };
    private static final double[] RETREAT_SIDE_SAMPLES = new double[] {
            -4.00,
            -3.20,
            -2.60,
            -2.10,
            -1.68,
            -RETREAT_EXTRA_WIDE_SIDE_WIDTH,
            -RETREAT_WIDE_SIDE_WIDTH, -0.72, -0.60, -RETREAT_SIDE_WIDTH, -0.36, -0.24,
            0.0,
            0.24, 0.36, RETREAT_SIDE_WIDTH, 0.60, 0.72, RETREAT_WIDE_SIDE_WIDTH,
            RETREAT_EXTRA_WIDE_SIDE_WIDTH,
            1.68,
            2.10,
            2.60,
            3.20,
            4.00
    };

    void updateFromAction(JsonObject action) {
        if (action == null || !action.has("auto_gapple")) return;
        JsonElement raw = action.get("auto_gapple");
        if (raw.isJsonPrimitive()) {
            enabled = raw.getAsBoolean();
            return;
        }
        if (!raw.isJsonObject()) return;
        JsonObject cfg = raw.getAsJsonObject();
        if (cfg.has("enabled")) enabled = cfg.get("enabled").getAsBoolean();
        if (cfg.has("health_threshold")) {
            healthThreshold = clampFloat(cfg.get("health_threshold").getAsFloat(), 1.0F, 20.0F);
        }
        if (cfg.has("critical_health_threshold")) {
            criticalHealthThreshold = clampFloat(
                    cfg.get("critical_health_threshold").getAsFloat(), 1.0F, healthThreshold);
        }
        if (cfg.has("safe_distance")) {
            safeDistance = clampFloat(cfg.get("safe_distance").getAsFloat(), 2.0F, 12.0F);
        }
        if (cfg.has("retreat_enabled")) {
            retreatEnabled = cfg.get("retreat_enabled").getAsBoolean();
        }
        if (cfg.has("retreat_distance")) {
            retreatDistance = clampFloat(cfg.get("retreat_distance").getAsFloat(), 2.0F, 40.0F);
        }
        if (cfg.has("absorption_threshold")) {
            absorptionThreshold = cfg.get("absorption_threshold").getAsFloat();
        }
        if (cfg.has("fast_retreat")) {
            fastRetreat = cfg.get("fast_retreat").getAsBoolean();
        }
        if (cfg.has("retreat_hops")) {
            retreatHops = cfg.get("retreat_hops").getAsBoolean();
        }
        if (cfg.has("sprint_hop_hold")) {
            sprintHopHold = cfg.get("sprint_hop_hold").getAsBoolean();
        }
        if (cfg.has("avoid_obstacles")) {
            avoidObstacles = cfg.get("avoid_obstacles").getAsBoolean();
        }
        if (cfg.has("retreat_strafe")) {
            retreatStrafe = cfg.get("retreat_strafe").getAsBoolean();
        }
        if (cfg.has("wall_slide")) {
            wallSlide = cfg.get("wall_slide").getAsBoolean();
        }
        if (cfg.has("retreat_speed_lock")) {
            retreatSpeedLock = cfg.get("retreat_speed_lock").getAsBoolean();
        }
        if (cfg.has("retreat_velocity_assist")) {
            retreatVelocityAssist = cfg.get("retreat_velocity_assist").getAsBoolean();
        }
        if (cfg.has("retreat_speed_first")) {
            retreatSpeedFirst = cfg.get("retreat_speed_first").getAsBoolean();
        }
        if (cfg.has("retreat_full_speed")) {
            retreatFullSpeed = cfg.get("retreat_full_speed").getAsBoolean();
        }
        if (cfg.has("retreat_speed_floor")) {
            retreatSpeedFloor = clampFloat(cfg.get("retreat_speed_floor").getAsFloat(), 0.05F, 6.00F);
        }
        if (cfg.has("retreat_max_speed")) {
            retreatMaxSpeed = clampFloat(cfg.get("retreat_max_speed").getAsFloat(), 0.10F, 6.00F);
        }
        if (cfg.has("retreat_accel")) {
            retreatAccel = clampFloat(cfg.get("retreat_accel").getAsFloat(), 0.01F, 8.00F);
        }
        if (cfg.has("retreat_sprint_retap")) {
            retreatSprintRetap = cfg.get("retreat_sprint_retap").getAsBoolean();
        }
        if (cfg.has("retreat_sprint_retap_ticks")) {
            retreatSprintRetapMaxTicks = clampInt(
                    cfg.get("retreat_sprint_retap_ticks").getAsInt(), 0, 8);
        }
        if (cfg.has("retreat_air_control")) {
            retreatAirControl = cfg.get("retreat_air_control").getAsBoolean();
        }
        if (cfg.has("retreat_step_assist")) {
            retreatStepAssist = cfg.get("retreat_step_assist").getAsBoolean();
        }
        if (cfg.has("retreat_step_height")) {
            retreatStepHeight = clampFloat(cfg.get("retreat_step_height").getAsFloat(), 0.60F, 1.25F);
        }
        if (cfg.has("fallback_retreat")) {
            fallbackRetreat = cfg.get("fallback_retreat").getAsBoolean();
        }
        if (cfg.has("retreat_input_lock")) {
            retreatInputLock = cfg.get("retreat_input_lock").getAsBoolean();
        }
        if (cfg.has("force_sprint_retreat")) {
            forceSprintRetreat = cfg.get("force_sprint_retreat").getAsBoolean();
        }
        if (cfg.has("release_retreat_on_hit")) {
            releaseRetreatOnHit = cfg.get("release_retreat_on_hit").getAsBoolean();
        }
        if (cfg.has("critical_rearm_only")) {
            criticalRearmOnly = cfg.get("critical_rearm_only").getAsBoolean();
        }
        if (cfg.has("critical_trapped_eat")) {
            criticalTrappedEat = cfg.get("critical_trapped_eat").getAsBoolean();
        }
        if (cfg.has("retreat_turn_limit_deg")) {
            retreatTurnLimitDeg = clampFloat(
                    cfg.get("retreat_turn_limit_deg").getAsFloat(), 45.0F, 360.0F);
        }
        if (cfg.has("eating_retreat_turn_limit_deg")) {
            eatingRetreatTurnLimitDeg = clampFloat(
                    cfg.get("eating_retreat_turn_limit_deg").getAsFloat(), 45.0F, 360.0F);
        }
        if (cfg.has("retreat_path_hold_ticks")) {
            retreatPathHoldMaxTicks = clampInt(
                    cfg.get("retreat_path_hold_ticks").getAsInt(), 0, 12);
        }
        if (cfg.has("retreat_stuck_abort_ticks")) {
            retreatStuckAbortTicks = clampInt(
                    cfg.get("retreat_stuck_abort_ticks").getAsInt(), 1, 20);
        }
        if (cfg.has("retreat_min_ticks")) {
            minRetreatTicks = clampInt(cfg.get("retreat_min_ticks").getAsInt(), 0, 40);
        }
        if (cfg.has("retreat_max_ticks")) {
            maxRetreatTicks = clampInt(cfg.get("retreat_max_ticks").getAsInt(), 1, 120);
        }
        if (cfg.has("critical_retreat_max_ticks")) {
            criticalRetreatMaxTicks = clampInt(
                    cfg.get("critical_retreat_max_ticks").getAsInt(), 1, 60);
        }
        if (cfg.has("critical_eat_commit_ticks")) {
            criticalEatCommitTicks = clampInt(
                    cfg.get("critical_eat_commit_ticks").getAsInt(), 0, EAT_TICKS);
        }
        if (cfg.has("combat_recovery_ticks")) {
            combatRecoveryTicks = clampInt(cfg.get("combat_recovery_ticks").getAsInt(), 0, 80);
        }
        if (cfg.has("retreat_strafe_hold_ticks")) {
            retreatStrafeHoldMaxTicks = clampInt(
                    cfg.get("retreat_strafe_hold_ticks").getAsInt(), 0, 12);
        }
        if (cfg.has("retreat_obstacle_jump_hold_ticks")) {
            retreatObstacleJumpHoldMaxTicks = clampInt(
                    cfg.get("retreat_obstacle_jump_hold_ticks").getAsInt(), 0, 60);
        }
        if (cfg.has("retreat_obstacle_escape_ticks")) {
            retreatObstacleEscapeTicks = clampInt(
                    cfg.get("retreat_obstacle_escape_ticks").getAsInt(), 0, 120);
        }
        if (cfg.has("retreat_panic_speed")) {
            retreatPanicSpeed = cfg.get("retreat_panic_speed").getAsBoolean();
        }
        if (cfg.has("retreat_obstacle_lookahead")) {
            retreatObstacleLookahead = clampFloat(
                    cfg.get("retreat_obstacle_lookahead").getAsFloat(), 1.20F, 24.00F);
        }
        if (cfg.has("critical_trapped_stuck_ticks")) {
            criticalTrappedStuckTicks = clampInt(
                    cfg.get("critical_trapped_stuck_ticks").getAsInt(), 1, retreatStuckAbortTicks);
        }
        if (criticalHealthThreshold > healthThreshold) {
            criticalHealthThreshold = healthThreshold;
        }
        if (retreatDistance < safeDistance) {
            retreatDistance = safeDistance;
        }
        if (maxRetreatTicks < minRetreatTicks) {
            maxRetreatTicks = minRetreatTicks;
        }
        if (criticalRetreatMaxTicks < minRetreatTicks) {
            criticalRetreatMaxTicks = minRetreatTicks;
        }
        if (retreatMaxSpeed < retreatSpeedFloor) {
            retreatMaxSpeed = retreatSpeedFloor;
        }
        if (criticalTrappedStuckTicks > retreatStuckAbortTicks) {
            criticalTrappedStuckTicks = retreatStuckAbortTicks;
        }
    }

    JsonObject fallbackRetreatAction(Minecraft mc, EntityPlayer target) {
        if (!enabled || !retreatEnabled || !fallbackRetreat) return null;
        EntityPlayerSP p = mc == null ? null : mc.thePlayer;
        if (p == null || target == null || target.isDead || !shouldEat(p)) return null;
        boolean critical = p.getHealth() <= criticalHealthThreshold;
        if (!critical && (retreatCombatTicks > 0 || p.hurtTime > 0
                || (criticalRearmOnly && retreatInterrupted))) {
            return null;
        }
        JsonObject action = new JsonObject();
        action.addProperty("dyaw", 0.0);
        action.addProperty("dpitch", 0.0);
        action.addProperty("retreat_turn", true);
        action.addProperty("forward", 1);
        action.addProperty("strafe", 0);
        action.addProperty("jump", retreatObstacleJumpTicks > 0 || p.isCollidedHorizontally);
        action.addProperty("sprint", true);
        action.addProperty("force_sprint", true);
        action.addProperty("attack", false);
        action.addProperty("use", false);
        action.addProperty("rightClick", false);
        return action;
    }

    boolean tickStart(Minecraft mc, EntityPlayer target) {
        EntityPlayerSP p = mc.thePlayer;
        if (!enabled || p == null || mc.theWorld == null || mc.playerController == null) {
            releaseUse(mc);
            releaseRetreatMovement(mc);
            resetUseState();
            resetRetreatState();
            status = enabled ? "no_player" : "off";
            return false;
        }
        if (eatTicks > 0) {
            if (shouldAbortEating(p, target)) {
                boolean critical = p.getHealth() <= criticalHealthThreshold;
                boolean commitCriticalEat = shouldCommitCriticalEating(p);
                boolean combatRecovery = false;
                if (commitCriticalEat) {
                    status = "critical eat commit:" + eatTicks;
                } else if (!critical && p.hurtTime > 0) {
                    startCombatRecovery(mc, "eat hit -> normal");
                    combatRecovery = true;
                }
                if (!commitCriticalEat) {
                    restorePreviousSlot(mc, p);
                    releaseRetreatMovement(mc);
                    resetUseState();
                    if (critical) {
                        resetRetreatState();
                        status = unsafeStatus(p, target);
                    } else if (!combatRecovery) {
                        status = unsafeStatus(p, target);
                    }
                    return false;
                }
            }
            key(mc.gameSettings.keyBindUseItem, true);
            boolean retreatingEat = maintainEatingRetreat(mc, p, target, eatTicks - 1);
            eatTicks--;
            if (!retreatingEat) {
                releaseRetreatMovement(mc);
                status = "eating:" + eatTicks;
            }
            if (eatTicks <= 0) finish(mc, p);
            return true;
        }
        if (!shouldEat(p)) {
            releaseRetreatMovement(mc);
            resetRetreatState();
            status = String.format(Locale.ROOT, "ready hp=%.1f abs=%.1f",
                    p.getHealth(), p.getAbsorptionAmount());
            return false;
        }
        boolean critical = p.getHealth() <= criticalHealthThreshold;
        if (critical) {
            retreatInterrupted = false;
            retreatCombatTicks = 0;
        } else if (retreatCombatTicks > 0) {
            retreatCombatTicks--;
            if (releaseRetreatOnHit) releaseRetreatMovement(mc);
            status = "combat recovery:" + retreatCombatTicks;
            return false;
        }
        if (retreatInterrupted && !critical && criticalRearmOnly) {
            if (releaseRetreatOnHit) releaseRetreatMovement(mc);
            status = "retreat hit lock -> normal";
            return false;
        }
        if (isRetreatCombatHit(p)) {
            startCombatRecovery(mc, "hit -> normal");
            return false;
        }
        if (!isSafeToStart(p, target)) {
            status = unsafeStatus(p, target);
            return false;
        }

        int hotbar = findGappleHotbar(p);
        if (hotbar < 0) {
            hotbar = moveGappleToHotbar(mc, p);
        }
        if (hotbar < 0) {
            status = "missing";
            return false;
        }

        previousHotbar = p.inventory.currentItem;
        selectHotbar(mc, p, hotbar);
        key(mc.gameSettings.keyBindUseItem, true);
        try {
            mc.playerController.sendUseItem(p, mc.theWorld, p.inventory.getCurrentItem());
        } catch (Throwable ignored) {
        }
        eatTicks = EAT_TICKS;
        boolean retreatingEat = maintainEatingRetreat(mc, p, target, eatTicks);
        status = retreatingEat ? "start+retreat:" + hotbar : "start:" + hotbar;
        return true;
    }

    boolean applyRetreatToAction(Minecraft mc, EntityPlayer target, JsonObject action) {
        if (action == null || !enabled || !retreatEnabled) return false;
        EntityPlayerSP p = mc.thePlayer;
        if (p == null || target == null || target.isDead) {
            resetRetreatState();
            return false;
        }
        if (!shouldEat(p)) {
            resetRetreatState();
            return false;
        }
        boolean critical = p.getHealth() <= criticalHealthThreshold;
        if (critical) {
            retreatInterrupted = false;
            retreatCombatTicks = 0;
        } else if (retreatCombatTicks > 0) {
            retreatCombatTicks--;
            if (releaseRetreatOnHit) releaseRetreatMovement(mc);
            resetRetreatMotionState();
            status = "combat recovery:" + retreatCombatTicks;
            return false;
        }
        if (retreatInterrupted && !critical && criticalRearmOnly) {
            if (releaseRetreatOnHit) releaseRetreatMovement(mc);
            resetRetreatMotionState();
            status = "retreat stopped -> normal";
            return false;
        }
        if (isRetreatCombatHit(p)) {
            startCombatRecovery(mc, "retreat hit -> normal");
            return false;
        }

        float need = retreatGoalDistance();
        float safe = safeEatDistance();
        float dist = p.getDistanceToEntity(target);
        if (dist >= need && retreatTicks >= minRetreatTicks) {
            releaseRetreatMovement(mc);
            retreatPathHoldTicks = 0;
            return false;
        }
        int tickLimit = retreatTickLimit(critical);
        if (retreatTicks >= tickLimit) {
            retreatPathHoldTicks = 0;
            if (critical || dist >= safe) {
                status = String.format(Locale.ROOT,
                        "retreat max -> eat d=%.1f/%.1f", dist, safe);
                return false;
            }
            retreatInterrupted = true;
            releaseRetreatMovement(mc);
            status = "retreat max -> normal";
            return false;
        }
        updateRetreatProgress(p);

        double awayX = p.posX - target.posX;
        double awayZ = p.posZ - target.posZ;
        double len = Math.sqrt(awayX * awayX + awayZ * awayZ);
        if (len < 1.0e-6) {
            releaseRetreatMovement(mc);
            return false;
        }
        awayX /= len;
        awayZ /= len;

        RetreatPath path = avoidObstacles
                ? chooseRetreatPath(mc.theWorld, p, awayX, awayZ)
                : new RetreatPath(awayX, awayZ, false, true, true, 0.0, 0, false);
        boolean blockedButUsable = canUseBlockedRetreatPath(path, awayX, awayZ);
        boolean obstacleEscape = shouldForceObstacleEscape(p, path, blockedButUsable);
        if (!path.viable && !blockedButUsable && !critical && !obstacleEscape) {
            if (dist >= safe) {
                releaseRetreatMovement(mc);
                status = String.format(Locale.ROOT,
                        "retreat blocked -> eat d=%.1f/%.1f", dist, safe);
                return false;
            }
            retreatInterrupted = true;
            retreatPathHoldTicks = 0;
            releaseRetreatMovement(mc);
            status = "retreat blocked -> normal";
            return false;
        }
        if (!path.viable && !blockedButUsable && !obstacleEscape
                && retreatStuckTicks >= retreatStuckAbortTicks) {
            status = "retreat blocked -> eat";
            return false;
        }
        double desiredYaw = Math.toDegrees(Math.atan2(-path.x, path.z));
        double dyaw = clampMag(wrapDeg(desiredYaw - p.rotationYaw), retreatTurnLimitDeg);
        double sideEscape = path.x * awayZ - path.z * awayX;
        int strafe = stabilizeRetreatStrafe(
                p, path, retreatStrafeForPath(p, path, desiredYaw, sideEscape), sideEscape);
        strafe = forceObstacleStrafe(strafe, p, path, sideEscape, obstacleEscape);
        boolean unstuckHop = (p.isCollidedHorizontally
                || retreatStuckTicks >= RETREAT_STUCK_JUMP_TICKS)
                && canJumpNow(p);
        boolean obstacleHop = shouldObstacleHop(mc.theWorld, p, path);
        boolean sprintHop = shouldSprintHop(mc.theWorld, p, path);
        boolean holdHop = shouldHoldSprintHop(p, path);
        boolean hop = updateRetreatHop(p, path,
                path.jump || obstacleHop || unstuckHop || obstacleEscape,
                sprintHop || holdHop);

        retreatTicks++;
        action.addProperty("dyaw", dyaw);
        action.addProperty("dpitch", 0.0);
        action.addProperty("retreat_turn", true);
        action.addProperty("forward", 1);
        if (strafe != 0) action.addProperty("strafe", strafe);
        else action.addProperty("strafe", 0);
        action.addProperty("sprint", true);
        action.addProperty("force_sprint", true);
        forceRetreatStepAssist(p, path);
        forceRetreatSprint(mc, p);
        forceRetreatInput(mc, p, strafe, hop || p.isCollidedHorizontally);
        forceRetreatVelocity(p, path, strafe);
        if (hop) action.addProperty("jump", true);
        else action.addProperty("jump", p.isCollidedHorizontally);
        action.addProperty("attack", false);
        action.addProperty("use", false);
        action.addProperty("rightClick", false);
        status = String.format(Locale.ROOT,
                "retreat%s%s%s%s d=%.1f/%.1f yaw=%.0f t=%d/%d stuck=%d str=%d spd=%.1f",
                fastRetreat ? "+fast" : "",
                retreatFullSpeed ? "+full" : "",
                hop || p.isCollidedHorizontally ? "+jump" : "",
                path.viable ? "" : (obstacleEscape ? "+escape" : "+blocked"),
                dist, need, dyaw, retreatTicks, tickLimit, retreatStuckTicks, strafe,
                horizontalSpeedPerSecond(p));
        return true;
    }

    boolean isActive() {
        return eatTicks > 0;
    }

    void cancel(Minecraft mc) {
        releaseUse(mc);
        releaseRetreatMovement(mc);
        resetUseState();
        resetRetreatState();
        status = enabled ? "cancelled" : "off";
    }

    String statusLine() {
        return "\u00A77 gapple=" + (enabled ? "\u00A7aon" : "\u00A78off")
                + "\u00A77 " + status;
    }

    private boolean shouldEat(EntityPlayerSP p) {
        if (p.getHealth() <= criticalHealthThreshold) return true;
        if (p.getAbsorptionAmount() >= ACTIVE_ABSORPTION) return false;
        if (p.getHealth() <= healthThreshold) return true;
        return absorptionThreshold >= 0.0F && p.getAbsorptionAmount() <= absorptionThreshold;
    }

    private boolean isSafeToStart(EntityPlayerSP p, EntityPlayer target) {
        if (target == null || target.isDead) return true;
        boolean critical = p.getHealth() <= criticalHealthThreshold;
        if (critical && retreatEnabled && targetNeedsRetreat(p, target)) {
            if (criticalTrappedEat
                    && (p.isCollidedHorizontally
                            || retreatStuckTicks >= criticalTrappedStuckTicks)) {
                return true;
            }
            if (retreatTicks < minRetreatTicks) return false;
            return retreatTicks >= retreatTickLimit(critical);
        }
        if (!targetTooCloseToEat(p, target)) return true;
        if (retreatEnabled && retreatTicks < minRetreatTicks) return false;
        if (p.getHealth() > criticalHealthThreshold) return false;
        if (!retreatEnabled) return true;
        if (criticalTrappedEat
                && critical
                && (p.isCollidedHorizontally
                        || retreatStuckTicks >= criticalTrappedStuckTicks)) {
            return true;
        }
        return retreatTicks >= retreatTickLimit(critical)
                || retreatStuckTicks >= retreatStuckAbortTicks;
    }

    private boolean shouldAbortEating(EntityPlayerSP p, EntityPlayer target) {
        if (p.hurtTime > 0 && p.getHealth() > criticalHealthThreshold) return true;
        if (target == null || target.isDead) return false;
        return targetTooCloseToEat(p, target)
                && p.getHealth() > criticalHealthThreshold;
    }

    private boolean shouldCommitCriticalEating(EntityPlayerSP p) {
        return p.getHealth() <= criticalHealthThreshold
                && criticalEatCommitTicks > 0
                && eatTicks <= criticalEatCommitTicks;
    }

    private boolean targetNeedsRetreat(EntityPlayerSP p, EntityPlayer target) {
        return target != null
                && !target.isDead
                && p.getDistanceToEntity(target) < retreatGoalDistance();
    }

    private boolean targetTooCloseToEat(EntityPlayerSP p, EntityPlayer target) {
        return target != null
                && !target.isDead
                && p.getDistanceToEntity(target) < safeEatDistance();
    }

    private float retreatGoalDistance() {
        return retreatEnabled ? Math.max(safeDistance, retreatDistance) : safeDistance;
    }

    private float safeEatDistance() {
        return safeDistance;
    }

    private int retreatTickLimit(boolean critical) {
        if (!critical) return maxRetreatTicks;
        return Math.min(maxRetreatTicks, Math.max(minRetreatTicks, criticalRetreatMaxTicks));
    }

    private RetreatPath chooseRetreatPath(World world, EntityPlayerSP p, double awayX, double awayZ) {
        RetreatPath best = null;
        boolean bestHeld = false;
        RetreatPath straight = scorePath(world, p, awayX, awayZ, awayX, awayZ);
        if (retreatPathHoldTicks > 0 && retreatStuckTicks <= 0 && !p.isCollidedHorizontally) {
            RetreatPath held = scorePath(world, p, lastRetreatX, lastRetreatZ, awayX, awayZ);
            if (held.viable && !held.blockedAhead) {
                best = held;
                bestHeld = true;
            }
        }
        for (double offset : RETREAT_ANGLE_OFFSETS) {
            double[] dir = rotate(awayX, awayZ, offset);
            RetreatPath candidate = scorePath(world, p, dir[0], dir[1], awayX, awayZ);
            double margin = bestHeld ? RETREAT_REPLAN_MARGIN : 0.0;
            if (best == null
                    || (candidate.viable && !best.viable)
                    || (candidate.viable == best.viable
                            && !candidate.blockedAhead
                            && best.blockedAhead)
                    || isSaferRetreatCandidate(candidate, best, margin)
                    || (candidate.viable == best.viable
                            && candidate.score > best.score + margin)) {
                best = candidate;
                bestHeld = false;
            }
        }
        if (shouldPreferStraightRetreat(straight, best, p)) {
            best = straight;
            bestHeld = false;
        }
        if (best == null) {
            best = new RetreatPath(awayX, awayZ, false, true, true, 0.0, 0, false);
            bestHeld = false;
        }
        lastRetreatX = best.x;
        lastRetreatZ = best.z;
        retreatPathHoldTicks = bestHeld
                ? Math.max(0, retreatPathHoldTicks - 1)
                : (best.blockedAhead || p.isCollidedHorizontally || retreatStuckTicks > 0 || !best.viable
                        ? 0
                        : retreatPathHoldMaxTicks);
        return best;
    }

    private int retreatStrafeForPath(EntityPlayerSP p, RetreatPath path,
                                     double desiredYaw, double sideEscape) {
        if (!retreatStrafe) return 0;
        boolean blockedEscape = !path.viable && (path.blockedAhead
                || p.isCollidedHorizontally
                || retreatStuckTicks > 0);
        if (!path.viable && !blockedEscape) return 0;
        double dyaw = wrapDeg(desiredYaw - p.rotationYaw);
        boolean sidePath = Math.abs(sideEscape) >= RETREAT_STRAFE_SIDE_ESCAPE;
        boolean slideOut = wallSlide && (p.isCollidedHorizontally
                || retreatStuckTicks > 0
                || path.blockedAhead
                || path.jump);
        if (retreatSpeedFirst
                && isCleanSpeedRetreatPath(path)
                && !slideOut
                && Math.abs(sideEscape) <= RETREAT_CLEAN_STRAFE_SUPPRESS_SIDE) {
            return 0;
        }
        if (retreatFullSpeed
                && isCleanSpeedRetreatPath(path)
                && !slideOut
                && Math.abs(sideEscape) <= RETREAT_FULL_SPEED_CLEAN_SIDE_LIMIT) {
            return 0;
        }
        if (retreatSpeedFirst
                && path.viable
                && !path.blockedAhead
                && !slideOut
                && Math.abs(sideEscape) <= RETREAT_SPEED_STRAFE_SUPPRESS_SIDE) {
            return 0;
        }
        boolean turnAssist = Math.abs(dyaw) >= RETREAT_STRAFE_TURN_DEG
                && (!retreatSpeedFirst
                        || slideOut
                        || !path.viable
                        || path.blockedAhead
                        || Math.abs(sideEscape) >= RETREAT_STRAFE_SIDE_ESCAPE);
        if (!turnAssist && !sidePath && !slideOut) return 0;
        if (slideOut && path.strafeAssist != 0) return path.strafeAssist;
        if (turnAssist) return dyaw > 0.0 ? 1 : -1;
        if (sidePath) return sideEscape < 0.0 ? 1 : -1;
        return sideEscape <= 0.0 ? 1 : -1;
    }

    private boolean shouldPreferStraightRetreat(RetreatPath straight,
                                                RetreatPath best,
                                                EntityPlayerSP p) {
        if (!(fastRetreat || forceSprintRetreat) || !retreatSpeedFirst) return false;
        if (straight == null || !straight.viable || straight.blockedAhead) return false;
        if (p.isCollidedHorizontally || retreatStuckTicks > 0) return false;
        if (!isCleanSpeedRetreatPath(straight)) return false;
        if (straight.centerClearFrac < RETREAT_STRAIGHT_OVERRIDE_CENTER_CLEAR) return false;
        if (straight.nearBlocked > 0 || straight.nearUnsafe > 0) return false;
        if (best == null || !best.viable || best.blockedAhead) return true;
        if (straight.awayDot > 0.98 && Math.abs(best.sideEscape) <= 0.55) return true;
        return best.score - straight.score <= RETREAT_STRAIGHT_OVERRIDE_SCORE_GAP;
    }

    private boolean isSaferRetreatCandidate(RetreatPath candidate,
                                            RetreatPath best,
                                            double margin) {
        if (candidate == null || best == null || candidate.viable != best.viable) return false;
        int candidateHazard = retreatHazardScore(candidate);
        int bestHazard = retreatHazardScore(best);
        return candidateHazard + 2 <= bestHazard
                && candidate.score + Math.max(margin, RETREAT_SAFER_PATH_SCORE_MARGIN) >= best.score;
    }

    private int retreatHazardScore(RetreatPath path) {
        int score = path.nearBlocked * 3 + path.nearUnsafe * 5;
        if (path.blockedAhead) score += 12;
        if (!path.viable) score += 16;
        return score;
    }

    private boolean isCleanSpeedRetreatPath(RetreatPath path) {
        return path != null
                && path.viable
                && !path.blockedAhead
                && path.centerClearFrac >= RETREAT_STRAIGHT_OVERRIDE_CENTER_CLEAR
                && path.nearBlocked == 0
                && path.nearUnsafe == 0;
    }

    private int stabilizeRetreatStrafe(EntityPlayerSP p, RetreatPath path,
                                       int strafe, double sideEscape) {
        if (!retreatStrafe) {
            retreatStrafeHoldTicks = 0;
            lastRetreatStrafe = 0;
            return 0;
        }
        boolean needsEscape = p.isCollidedHorizontally
                || retreatStuckTicks > 0
                || path.blockedAhead
                || !path.viable;
        if (strafe == 0 && needsEscape) {
            strafe = path.strafeAssist != 0
                    ? path.strafeAssist
                    : (sideEscape <= 0.0 ? 1 : -1);
        }
        if (strafe == 0) {
            if (retreatStrafeHoldTicks > 0
                    && lastRetreatStrafe != 0
                    && needsEscape) {
                retreatStrafeHoldTicks--;
                return lastRetreatStrafe;
            }
            retreatStrafeHoldTicks = 0;
            lastRetreatStrafe = 0;
            return 0;
        }
        if (lastRetreatStrafe != 0
                && strafe != lastRetreatStrafe
                && retreatStrafeHoldTicks > 0) {
            retreatStrafeHoldTicks--;
            return lastRetreatStrafe;
        }
        lastRetreatStrafe = strafe;
        retreatStrafeHoldTicks = retreatStrafeHoldMaxTicks;
        return strafe;
    }

    private int forceObstacleStrafe(int strafe, EntityPlayerSP p, RetreatPath path,
                                    double sideEscape, boolean obstacleEscape) {
        if (!retreatStrafe || strafe != 0 || path == null) return strafe;
        if (!obstacleEscape && !needsObstacleEscape(p, path)) return strafe;
        return path.strafeAssist != 0
                ? path.strafeAssist
                : (sideEscape <= 0.0 ? 1 : -1);
    }

    private boolean maintainEatingRetreat(Minecraft mc, EntityPlayerSP p,
                                          EntityPlayer target, int remainingTicks) {
        if (!retreatEnabled || target == null || target.isDead || !targetNeedsRetreat(p, target)) {
            releaseRetreatMovement(mc);
            return false;
        }
        updateRetreatProgress(p);
        double awayX = p.posX - target.posX;
        double awayZ = p.posZ - target.posZ;
        double len = Math.sqrt(awayX * awayX + awayZ * awayZ);
        if (len < 1.0e-6) return false;
        awayX /= len;
        awayZ /= len;

        RetreatPath path = avoidObstacles
                ? chooseRetreatPath(mc.theWorld, p, awayX, awayZ)
                : new RetreatPath(awayX, awayZ, false, true, true, 0.0, 0, false);
        boolean blockedButUsable = canUseBlockedRetreatPath(path, awayX, awayZ);
        boolean obstacleEscape = shouldForceObstacleEscape(p, path, blockedButUsable);
        if (!path.viable && !blockedButUsable && !obstacleEscape
                && retreatStuckTicks >= retreatStuckAbortTicks) {
            releaseRetreatMovement(mc);
            return false;
        }

        double desiredYaw = Math.toDegrees(Math.atan2(-path.x, path.z));
        double dyaw = clampMag(wrapDeg(desiredYaw - p.rotationYaw), eatingRetreatTurnLimitDeg);
        p.rotationYaw += (float) dyaw;

        double sideEscape = path.x * awayZ - path.z * awayX;
        int strafe = stabilizeRetreatStrafe(
                p, path, retreatStrafeForPath(p, path, desiredYaw, sideEscape), sideEscape);
        strafe = forceObstacleStrafe(strafe, p, path, sideEscape, obstacleEscape);
        boolean hop = path.jump
                || shouldObstacleHop(mc.theWorld, p, path);
        boolean speedHop = shouldSprintHop(mc.theWorld, p, path)
                || shouldHoldSprintHop(p, path);
        boolean unstuckHop = (p.isCollidedHorizontally
                || retreatStuckTicks >= RETREAT_STUCK_JUMP_TICKS)
                && canJumpNow(p);
        hop = updateRetreatHop(p, path, hop || unstuckHop || obstacleEscape, speedHop);

        key(mc.gameSettings.keyBindBack, false);
        key(mc.gameSettings.keyBindForward, true);
        key(mc.gameSettings.keyBindSprint, true);
        key(mc.gameSettings.keyBindLeft, strafe > 0);
        key(mc.gameSettings.keyBindRight, strafe < 0);
        key(mc.gameSettings.keyBindJump, hop || p.isCollidedHorizontally);
        forceRetreatStepAssist(p, path);
        forceRetreatSprint(mc, p);
        forceRetreatInput(mc, p, strafe, hop || p.isCollidedHorizontally);
        forceRetreatVelocity(p, path, strafe);
        retreatTicks++;
        status = String.format(Locale.ROOT,
                "eating+retreat%s%s%s:%d yaw=%.0f stuck=%d str=%d spd=%.1f",
                fastRetreat ? "+fast" : "",
                retreatFullSpeed ? "+full" : "",
                hop || p.isCollidedHorizontally ? "+jump" : "",
                remainingTicks, dyaw, retreatStuckTicks, strafe, horizontalSpeedPerSecond(p));
        return true;
    }

    private void updateRetreatProgress(EntityPlayerSP p) {
        if (retreatTicks <= 0) {
            previousRetreatPosX = p.posX;
            previousRetreatPosZ = p.posZ;
            retreatStuckTicks = 0;
            return;
        }
        double dx = p.posX - previousRetreatPosX;
        double dz = p.posZ - previousRetreatPosZ;
        double progress = dx * lastRetreatX + dz * lastRetreatZ;
        if (dx * dx + dz * dz < RETREAT_STUCK_EPS_SQ
                || (retreatTicks > 1 && progress < RETREAT_MIN_PROGRESS)) {
            retreatStuckTicks++;
            retreatPathHoldTicks = 0;
        } else {
            retreatStuckTicks = 0;
        }
        previousRetreatPosX = p.posX;
        previousRetreatPosZ = p.posZ;
    }

    private RetreatPath scorePath(World world, EntityPlayerSP p,
                                  double dirX, double dirZ,
                                  double awayX, double awayZ) {
        boolean hardBlocked = false;
        boolean blockedAhead = false;
        int centerHardBlocked = 0;
        int farCenterBlocked = 0;
        boolean jump = false;
        int centerChecked = 0;
        int centerClear = 0;
        int leftChecked = 0;
        int leftClear = 0;
        int rightChecked = 0;
        int rightClear = 0;
        int clear = 0;
        int blocked = 0;
        int unsafe = 0;
        int checked = 0;
        int centerUnsafe = 0;
        int nearBlocked = 0;
        int nearUnsafe = 0;
        int footY = MathHelper.floor_double(p.getEntityBoundingBox().minY + 0.01);
        double sideX = dirZ;
        double sideZ = -dirX;
        for (double ahead : RETREAT_AHEAD_SAMPLES) {
            for (double side : RETREAT_SIDE_SAMPLES) {
                checked++;
                double x = p.posX + dirX * ahead + sideX * side;
                double z = p.posZ + dirZ * ahead + sideZ * side;
                BlockPos pos = new BlockPos(
                        MathHelper.floor_double(x),
                        footY,
                        MathHelper.floor_double(z));
                StepProbe probe = probeStep(world, pos);
                boolean centerLane = Math.abs(side) < 1.0e-6;
                if (probe.hardBlocked) {
                    blocked++;
                    if (centerLane) {
                        centerHardBlocked++;
                        if (ahead <= RETREAT_HARD_BLOCK_AHEAD) {
                            hardBlocked = true;
                        } else {
                            farCenterBlocked++;
                        }
                        if (ahead <= retreatObstacleLookahead) {
                            blockedAhead = true;
                        }
                    } else if (ahead <= 0.95) {
                        blockedAhead = true;
                    }
                } else {
                    clear++;
                }
                if (ahead <= 1.20 && probe.hardBlocked) {
                    nearBlocked++;
                }
                if (ahead <= 1.20 && probe.unsafeGround) {
                    nearUnsafe++;
                }
                if (side > 1.0e-6) {
                    leftChecked++;
                    if (!probe.hardBlocked && !probe.unsafeGround) leftClear++;
                } else if (side < -1.0e-6) {
                    rightChecked++;
                    if (!probe.hardBlocked && !probe.unsafeGround) rightClear++;
                }
                if (centerLane) {
                    centerChecked++;
                    if (!probe.hardBlocked && !probe.unsafeGround) centerClear++;
                    if (probe.unsafeGround) centerUnsafe++;
                    if (probe.jump) {
                        jump = true;
                        if (ahead <= retreatObstacleLookahead) blockedAhead = true;
                    }
                }
                if (probe.unsafeGround) unsafe++;
            }
        }
        double awayDot = dirX * awayX + dirZ * awayZ;
        double clearFrac = checked == 0 ? 0.0 : (double) clear / (double) checked;
        double centerClearFrac = centerChecked == 0 ? 0.0 : (double) centerClear / (double) centerChecked;
        boolean cleanCenter = centerChecked > 0
                && centerClear == centerChecked
                && centerUnsafe == 0
                && !hardBlocked;
        boolean hopClear = cleanCenter
                && nearUnsafe == 0
                && nearBlocked <= Math.max(1, checked / 12);
        boolean escapeOnly = blockedAhead || p.isCollidedHorizontally || retreatStuckTicks > 0;
        boolean speedPriority = fastRetreat || forceSprintRetreat;
        double minAwayDot = speedPriority && !escapeOnly
                ? (retreatSpeedFirst
                        ? RETREAT_SPEED_FIRST_MIN_AWAY_DOT
                        : RETREAT_SPEED_MIN_AWAY_DOT)
                : RETREAT_MIN_AWAY_DOT;
        boolean movingTowardTarget = awayDot < (escapeOnly ? -0.04 : minAwayDot);
        boolean safeCenter = centerUnsafe <= Math.max(1, centerChecked / 5);
        boolean viable = !hardBlocked
                && centerClear > 0
                && safeCenter
                && nearUnsafe <= Math.max(1, checked / 8)
                && unsafe <= Math.max(1, checked / 4)
                && !movingTowardTarget;
        double sideEscape = Math.abs(dirX * awayZ - dirZ * awayX);
        double score = awayDot * 44.0 + clearFrac * 9.0 + centerClearFrac * 9.75;
        if (retreatStuckTicks > 0 || blockedAhead || p.isCollidedHorizontally) {
            score += sideEscape * 13.50;
        } else {
            score -= sideEscape * 2.10;
        }
        score -= blocked * 3.35;
        score -= unsafe * 4.50;
        score -= centerUnsafe * 7.00;
        score -= nearBlocked * 3.00;
        score -= nearUnsafe * 6.25;
        score -= farCenterBlocked * 1.10;
        score -= centerHardBlocked * 2.40;
        if (movingTowardTarget) score -= (-awayDot) * 20.0;
        if (p.isCollidedHorizontally && jump) score += 3.20;
        if (blockedAhead) score += sideEscape * 8.00;
        if (retreatStuckTicks >= RETREAT_STUCK_JUMP_TICKS) score += sideEscape * 9.50;
        if (speedPriority && cleanCenter && awayDot > 0.70 && !blockedAhead) score += 5.75;
        if (speedPriority && hopClear) score += 4.75;
        if (speedPriority && awayDot > 0.92 && centerClearFrac >= 0.80 && unsafe == 0) score += 7.10;
        if (speedPriority && !blockedAhead && awayDot > 0.80) score += 3.40;
        if (speedPriority && centerClearFrac >= 0.85 && unsafe == 0 && !blockedAhead) score += 3.80;
        if (speedPriority && !blockedAhead && nearBlocked == 0 && nearUnsafe == 0) score += 5.25;
        if (speedPriority && blockedAhead) score -= 3.00;
        if (speedPriority && blockedAhead && sideEscape >= 0.12) score += sideEscape * 10.00;
        if (speedPriority && blockedAhead && sideEscape < 0.18) score -= 22.00;
        if (speedPriority && (blockedAhead || nearBlocked > 0) && sideEscape >= 0.18) {
            score += sideEscape * 22.00;
        }
        if (speedPriority && nearBlocked == 0 && nearUnsafe == 0
                && centerClearFrac >= 0.95 && !blockedAhead) {
            score += 10.00;
        }
        if (speedPriority && p.isCollidedHorizontally && sideEscape >= 0.18 && !hardBlocked) {
            score += sideEscape * 9.25;
        }
        if (speedPriority && blocked == 0 && unsafe == 0 && awayDot > 0.80) score += 6.00;
        if (speedPriority && sideEscape <= 0.22 && awayDot > 0.94
                && centerClearFrac >= 0.80 && nearUnsafe == 0 && !blockedAhead) {
            score += 6.20;
        }
        if (retreatFullSpeed && speedPriority && cleanCenter && !blockedAhead
                && awayDot > 0.92 && sideEscape <= RETREAT_FULL_SPEED_CLEAN_SIDE_LIMIT) {
            score += awayDot * 22.0;
            score -= sideEscape * 18.0;
        }
        if (speedPriority && sideEscape <= 0.12 && awayDot > 0.96
                && centerClearFrac >= 0.90 && unsafe == 0 && !blockedAhead) {
            score += 9.40;
        }
        if (speedPriority && retreatSpeedFirst) {
            score += awayDot * 55.0;
            if (cleanCenter && awayDot > 0.90 && blocked == 0 && unsafe == 0 && !blockedAhead) {
                score += 24.00;
            }
            if (awayDot > 0.96 && centerClearFrac >= 0.95 && nearBlocked == 0 && nearUnsafe == 0) {
                score += 18.50;
            }
            if (!blockedAhead && retreatStuckTicks <= 0 && !p.isCollidedHorizontally) {
                score -= sideEscape * 26.00;
            }
            if (!escapeOnly && awayDot < RETREAT_SPEED_FIRST_MIN_AWAY_DOT) {
                score -= (RETREAT_SPEED_FIRST_MIN_AWAY_DOT - awayDot) * 120.0;
            }
            if (awayDot < 0.55) score -= (0.55 - awayDot) * 60.0;
            if ((blockedAhead || p.isCollidedHorizontally || retreatStuckTicks > 0)
                    && sideEscape >= 0.16) {
                score += sideEscape * 16.00;
            }
        }
        if (jump) score -= 0.08;
        if (hardBlocked) score -= 100.0;
        int strafeAssist = 0;
        if (leftChecked > 0 && rightChecked > 0) {
            double leftFrac = (double) leftClear / (double) leftChecked;
            double rightFrac = (double) rightClear / (double) rightChecked;
            if (Math.abs(leftFrac - rightFrac) >= 0.12) {
                strafeAssist = leftFrac > rightFrac ? 1 : -1;
            }
        }
        return new RetreatPath(dirX, dirZ, jump, hopClear, viable, score,
                strafeAssist, blockedAhead, awayDot, sideEscape, centerClearFrac,
                nearBlocked, nearUnsafe);
    }

    private boolean canUseBlockedRetreatPath(RetreatPath path, double awayX, double awayZ) {
        if (path == null || path.score < RETREAT_PANIC_MIN_SCORE) return false;
        double awayDot = path.x * awayX + path.z * awayZ;
        if (awayDot < RETREAT_PANIC_MIN_AWAY_DOT) return false;
        double sideEscape = Math.abs(path.x * awayZ - path.z * awayX);
        return path.blockedAhead
                || retreatStuckTicks > 0
                || sideEscape >= RETREAT_PANIC_SIDE_ESCAPE;
    }

    private boolean shouldForceObstacleEscape(EntityPlayerSP p, RetreatPath path, boolean blockedButUsable) {
        return avoidObstacles
                && retreatObstacleEscapeTicks > 0
                && path != null
                && ((!path.viable && !blockedButUsable)
                        || needsObstacleEscape(p, path))
                && retreatTicks < retreatObstacleEscapeTicks;
    }

    private boolean shouldSprintHop(World world, EntityPlayerSP p, RetreatPath path) {
        if (!(fastRetreat || forceSprintRetreat) || !retreatHops || !canFastHopPath(path)) return false;
        if (!canJumpNow(p)) return false;
        int footY = MathHelper.floor_double(p.getEntityBoundingBox().minY + 0.01);
        BlockPos feet = new BlockPos(
                MathHelper.floor_double(p.posX + path.x * 0.40),
                footY,
                MathHelper.floor_double(p.posZ + path.z * 0.40));
        return !blocksMovement(world, feet.up())
                && !blocksMovement(world, feet.up().up())
                && !isLiquid(world, feet)
                && !isLiquid(world, feet.down());
    }

    private boolean shouldObstacleHop(World world, EntityPlayerSP p, RetreatPath path) {
        if (!avoidObstacles || !retreatHops || !needsObstacleEscape(p, path)) {
            return false;
        }
        if (!canJumpNow(p)) return false;
        int footY = MathHelper.floor_double(p.getEntityBoundingBox().minY + 0.01);
        BlockPos feet = new BlockPos(
                MathHelper.floor_double(p.posX + path.x * 0.55),
                footY,
                MathHelper.floor_double(p.posZ + path.z * 0.55));
        return !blocksMovement(world, feet.up())
                && !blocksMovement(world, feet.up().up())
                && !isLiquid(world, feet)
                && !isLiquid(world, feet.down());
    }

    private boolean shouldHoldSprintHop(EntityPlayerSP p, RetreatPath path) {
        return (fastRetreat || forceSprintRetreat)
                && retreatHops
                && sprintHopHold
                && canFastHopPath(path)
                && !p.isInWater()
                && !p.isInLava()
                && !p.isOnLadder();
    }

    private boolean canFastHopPath(RetreatPath path) {
        return path != null
                && (path.hopClear
                        || (path.viable
                                && !path.blockedAhead
                                && path.nearUnsafe < RETREAT_NEAR_UNSAFE_ESCAPE
                                && retreatStuckTicks <= 0));
    }

    private boolean updateRetreatHop(EntityPlayerSP p, RetreatPath path,
                                     boolean obstacleHop, boolean speedHop) {
        if (obstacleHop || needsObstacleEscape(p, path)) {
            retreatObstacleJumpTicks = Math.max(
                    retreatObstacleJumpTicks, retreatObstacleJumpHoldMaxTicks);
        }
        boolean heldObstacleHop = retreatObstacleJumpTicks > 0;
        if (retreatObstacleJumpTicks > 0) retreatObstacleJumpTicks--;
        return obstacleHop || speedHop || heldObstacleHop;
    }

    private boolean canJumpNow(EntityPlayerSP p) {
        return p.onGround && !p.isInWater() && !p.isInLava() && !p.isOnLadder();
    }

    private void forceRetreatStepAssist(EntityPlayerSP p, RetreatPath path) {
        if (!retreatStepAssist || p == null || p.isInWater() || p.isInLava() || p.isOnLadder()) {
            restoreRetreatStepHeight(p);
            return;
        }
        if (previousStepHeight < 0.0F) {
            previousStepHeight = p.stepHeight;
        }
        float wanted = Math.max(retreatStepHeight, previousStepHeight);
        if (path != null && needsObstacleEscape(p, path)) {
            wanted = Math.max(wanted, 1.00F);
        }
        p.stepHeight = Math.max(p.stepHeight, wanted);
    }

    private void restoreRetreatStepHeight(EntityPlayerSP p) {
        if (previousStepHeight >= 0.0F) {
            if (p != null) p.stepHeight = previousStepHeight;
            previousStepHeight = -1.0F;
        }
    }

    private boolean isRetreatCombatHit(EntityPlayerSP p) {
        return p.hurtTime > 0 && p.getHealth() > criticalHealthThreshold;
    }

    private boolean hasNearRetreatObstacle(RetreatPath path) {
        return path != null
                && (path.nearBlocked >= RETREAT_NEAR_BLOCKED_ESCAPE
                        || path.nearUnsafe >= RETREAT_NEAR_UNSAFE_ESCAPE);
    }

    private boolean needsObstacleEscape(EntityPlayerSP p, RetreatPath path) {
        return path != null
                && (path.blockedAhead
                        || hasNearRetreatObstacle(path)
                        || p.isCollidedHorizontally
                        || retreatStuckTicks > 0);
    }

    private boolean panicRetreatActive(EntityPlayerSP p, RetreatPath path) {
        return retreatPanicSpeed
                && (p.getHealth() <= criticalHealthThreshold
                        || needsObstacleEscape(p, path)
                        || path == null
                        || !path.viable);
    }

    private void forceRetreatSprint(Minecraft mc, EntityPlayerSP p) {
        if (!(fastRetreat || forceSprintRetreat)
                || p.isInWater() || p.isInLava() || p.isOnLadder()) return;
        if (retreatSpeedLock) {
            p.setSneaking(false);
            if (p.movementInput != null) {
                p.movementInput.sneak = false;
                p.movementInput.moveForward = 1.0F;
            }
            p.moveForward = 1.0F;
        }
        if (retreatSprintRetap
                && retreatSpeedLock
                && retreatSprintRetapMaxTicks > 0
                && mc != null
                && mc.gameSettings != null) {
            if (!p.isSprinting() || p.isCollidedHorizontally || p.hurtTime > 0) {
                retreatSprintRetapTicks = Math.max(
                        retreatSprintRetapTicks, retreatSprintRetapMaxTicks);
            }
            if (retreatSprintRetapTicks > 0) {
                key(mc.gameSettings.keyBindSprint, false);
                key(mc.gameSettings.keyBindForward, false);
                retreatSprintRetapTicks--;
            }
            key(mc.gameSettings.keyBindForward, true);
            key(mc.gameSettings.keyBindSprint, true);
        }
        p.setSprinting(true);
    }

    private void forceRetreatInput(Minecraft mc, EntityPlayerSP p, int strafe, boolean jump) {
        if (!retreatInputLock || mc.gameSettings == null) return;
        key(mc.gameSettings.keyBindBack, false);
        key(mc.gameSettings.keyBindForward, true);
        key(mc.gameSettings.keyBindSprint, true);
        if (retreatSpeedLock) key(mc.gameSettings.keyBindSneak, false);
        key(mc.gameSettings.keyBindLeft, strafe > 0);
        key(mc.gameSettings.keyBindRight, strafe < 0);
        key(mc.gameSettings.keyBindJump, jump);
        if (p.movementInput != null) {
            p.movementInput.moveForward = 1.0F;
            p.movementInput.moveStrafe = strafe > 0 ? 1.0F : (strafe < 0 ? -1.0F : 0.0F);
            p.movementInput.jump = jump;
            if (retreatSpeedLock) p.movementInput.sneak = false;
        }
        p.moveForward = 1.0F;
        p.moveStrafing = strafe > 0 ? 1.0F : (strafe < 0 ? -1.0F : 0.0F);
        forceRetreatSprint(mc, p);
    }

    private void forceRetreatVelocity(EntityPlayerSP p, RetreatPath path, int strafe) {
        if (!retreatVelocityAssist
                || !(fastRetreat || forceSprintRetreat)
                || !retreatSpeedLock
                || path == null
                || p.isInWater()
                || p.isInLava()
                || p.isOnLadder()
                || p.isSneaking()) {
            return;
        }
        double moveX = path.x;
        double moveZ = path.z;
        boolean escape = needsObstacleEscape(p, path) || !path.viable;
        if (strafe != 0) {
            double sideBlend = escape ? RETREAT_ESCAPE_SPEED_BLEND
                    : (retreatSpeedFirst
                            ? RETREAT_FAST_STRAFE_SPEED_BLEND
                            : RETREAT_STRAFE_SPEED_BLEND);
            double sideX = path.z;
            double sideZ = -path.x;
            double sign = strafe > 0 ? 1.0 : -1.0;
            moveX += sideX * sign * sideBlend;
            moveZ += sideZ * sign * sideBlend;
        }
        double len = Math.sqrt(moveX * moveX + moveZ * moveZ);
        if (len < 1.0e-6) return;
        moveX /= len;
        moveZ /= len;

        double floor = retreatSpeedFloor;
        double accel = retreatAccel;
        boolean panicSpeed = panicRetreatActive(p, path);
        if (forceSprintRetreat) {
            floor = Math.max(floor, retreatMaxSpeed * (retreatSpeedFirst ? 1.00 : 0.92));
            accel = Math.max(accel, retreatSpeedFirst ? 1.00 : 0.62);
        }
        if (retreatTicks <= RETREAT_START_BURST_TICKS) {
            floor = Math.max(floor, retreatMaxSpeed * RETREAT_START_BURST_SPEED_MULTIPLIER);
            accel = Math.max(accel, retreatAccel * RETREAT_START_BURST_ACCEL_MULTIPLIER);
        }
        if (retreatFullSpeed && !escape && isCleanSpeedRetreatPath(path)) {
            floor = Math.max(floor, retreatMaxSpeed);
            accel = Math.max(accel, retreatAccel);
        }
        if (needsObstacleEscape(p, path) || !path.viable) {
            floor = Math.max(floor, retreatMaxSpeed);
            accel = Math.max(accel, retreatSpeedFirst ? 1.00 : 0.74);
        }
        if (panicSpeed) {
            floor = Math.max(floor, retreatMaxSpeed * RETREAT_PANIC_SPEED_MULTIPLIER);
            accel = Math.max(accel, retreatAccel * RETREAT_PANIC_ACCEL_MULTIPLIER);
        }
        if (!p.onGround) {
            floor *= retreatFullSpeed ? 1.00 : (forceSprintRetreat ? (retreatSpeedFirst ? 0.96 : 0.90) : 0.82);
            if (retreatAirControl) {
                floor = Math.max(floor, retreatMaxSpeed * (retreatFullSpeed ? 1.00 : 0.94));
                accel = Math.max(accel, retreatAccel * 0.55);
            }
        }
        double along = p.motionX * moveX + p.motionZ * moveZ;
        if (along < floor) {
            double boost = Math.min(accel, floor - along);
            p.motionX += moveX * boost;
            p.motionZ += moveZ * boost;
        }
        if (retreatSpeedFirst && !escape && isCleanSpeedRetreatPath(path)) {
            double sideX = moveZ;
            double sideZ = -moveX;
            double sideMotion = p.motionX * sideX + p.motionZ * sideZ;
            double damp = strafe == 0 ? RETREAT_SIDE_MOTION_DAMP : RETREAT_SIDE_MOTION_DAMP * 0.35;
            p.motionX -= sideX * sideMotion * damp;
            p.motionZ -= sideZ * sideMotion * damp;
        }
        double speed = Math.sqrt(p.motionX * p.motionX + p.motionZ * p.motionZ);
        double max = retreatMaxSpeed;
        if (panicSpeed) max = Math.max(max, retreatMaxSpeed * RETREAT_PANIC_SPEED_MULTIPLIER);
        if (!p.onGround) {
            max *= retreatFullSpeed ? 1.00 : (forceSprintRetreat ? (retreatSpeedFirst ? 1.00 : 0.98) : 0.96);
            if (retreatAirControl) {
                max = Math.max(max, retreatMaxSpeed);
            }
        }
        double alongAfter = p.motionX * moveX + p.motionZ * moveZ;
        boolean helpfulFastRetreat = alongAfter >= max * 0.98;
        if (!helpfulFastRetreat && speed > max && speed > 1.0e-6) {
            double scale = max / speed;
            p.motionX *= scale;
            p.motionZ *= scale;
        }
    }

    private StepProbe probeStep(World world, BlockPos pos) {
        boolean foot = blocksMovement(world, pos);
        boolean body = blocksMovement(world, pos.up());
        boolean head = blocksMovement(world, pos.up().up());
        boolean liquid = isLiquid(world, pos) || isLiquid(world, pos.down());
        boolean ground = blocksMovement(world, pos.down()) || foot;
        boolean unsafeGround = liquid || !ground;
        if (!foot && !body && !head) return new StepProbe(false, false, unsafeGround);
        if (isStepObstacle(world, pos)) return new StepProbe(false, true, liquid);
        return new StepProbe(true, false, true);
    }

    private boolean isStepObstacle(World world, BlockPos pos) {
        return blocksMovement(world, pos)
                && !blocksMovement(world, pos.up())
                && !blocksMovement(world, pos.up().up());
    }

    private boolean blocksMovement(World world, BlockPos pos) {
        IBlockState state = world.getBlockState(pos);
        return state != null
                && state.getBlock() != null
                && state.getBlock().getMaterial().blocksMovement();
    }

    private boolean isLiquid(World world, BlockPos pos) {
        IBlockState state = world.getBlockState(pos);
        if (state == null || state.getBlock() == null) return false;
        Material material = state.getBlock().getMaterial();
        return material != null && material.isLiquid();
    }

    private static double[] rotate(double x, double z, double deg) {
        double rad = Math.toRadians(deg);
        double c = Math.cos(rad);
        double s = Math.sin(rad);
        return new double[] { x * c - z * s, x * s + z * c };
    }

    private String unsafeStatus(EntityPlayerSP p, EntityPlayer target) {
        if (target == null || target.isDead) return "unsafe";
        return String.format(Locale.ROOT, "unsafe d=%.1f safe=%.1f goal=%.1f hp=%.1f",
                p.getDistanceToEntity(target), safeEatDistance(),
                retreatGoalDistance(), p.getHealth());
    }

    private static double horizontalSpeedPerSecond(EntityPlayerSP p) {
        return Math.sqrt(p.motionX * p.motionX + p.motionZ * p.motionZ) * 20.0;
    }

    private int findGappleHotbar(EntityPlayerSP p) {
        for (int i = 0; i < 9; i++) {
            if (isGapple(p.inventory.getStackInSlot(i))) return i;
        }
        return -1;
    }

    private int moveGappleToHotbar(Minecraft mc, EntityPlayerSP p) {
        Slot source = findGappleInventorySlot(p);
        if (source == null) return -1;
        int targetHotbar = findEmptyHotbar(p);
        boolean restore = false;
        if (targetHotbar < 0) {
            targetHotbar = p.inventory.currentItem;
            restore = true;
        }
        mc.playerController.windowClick(
                p.inventoryContainer.windowId,
                source.slotNumber,
                targetHotbar,
                2,
                p);
        if (restore) {
            restoreContainerSlot = source.slotNumber;
            restoreHotbar = targetHotbar;
        } else {
            restoreContainerSlot = -1;
            restoreHotbar = -1;
        }
        return targetHotbar;
    }

    private Slot findGappleInventorySlot(EntityPlayerSP p) {
        for (Object raw : p.inventoryContainer.inventorySlots) {
            Slot slot = (Slot) raw;
            if (slot == null || slot.inventory != p.inventory) continue;
            int idx = slot.getSlotIndex();
            if (idx < 9 || idx >= 36 || !slot.getHasStack()) continue;
            if (isGapple(slot.getStack())) return slot;
        }
        return null;
    }

    private int findEmptyHotbar(EntityPlayerSP p) {
        for (int i = 0; i < 9; i++) {
            ItemStack stack = p.inventory.getStackInSlot(i);
            if (stack == null) return i;
        }
        return -1;
    }

    private void finish(Minecraft mc, EntityPlayerSP p) {
        releaseUse(mc);
        releaseRetreatMovement(mc);
        restorePreviousSlot(mc, p);
        resetUseState();
        resetRetreatState();
        status = "done";
    }

    private void restorePreviousSlot(Minecraft mc, EntityPlayerSP p) {
        releaseUse(mc);
        if (restoreContainerSlot >= 0 && restoreHotbar >= 0) {
            mc.playerController.windowClick(
                    p.inventoryContainer.windowId,
                    restoreContainerSlot,
                    restoreHotbar,
                    2,
                    p);
        }
        if (previousHotbar >= 0 && previousHotbar < 9) {
            selectHotbar(mc, p, previousHotbar);
        }
    }

    private void resetUseState() {
        eatTicks = 0;
        previousHotbar = -1;
        restoreContainerSlot = -1;
        restoreHotbar = -1;
    }

    private void resetRetreatState() {
        retreatInterrupted = false;
        retreatCombatTicks = 0;
        resetRetreatMotionState();
    }

    private void resetRetreatMotionState() {
        retreatPathHoldTicks = 0;
        retreatStrafeHoldTicks = 0;
        retreatObstacleJumpTicks = 0;
        retreatTicks = 0;
        retreatStuckTicks = 0;
        retreatSprintRetapTicks = 0;
        lastRetreatStrafe = 0;
        lastRetreatX = 0.0;
        lastRetreatZ = 0.0;
        previousRetreatPosX = 0.0;
        previousRetreatPosZ = 0.0;
    }

    private void startCombatRecovery(Minecraft mc, String newStatus) {
        retreatInterrupted = true;
        retreatCombatTicks = Math.max(retreatCombatTicks, combatRecoveryTicks);
        if (releaseRetreatOnHit) releaseRetreatMovement(mc);
        resetRetreatMotionState();
        status = newStatus;
    }

    private static boolean isGapple(ItemStack stack) {
        if (stack == null || stack.getItem() == null) return false;
        String registry = String.valueOf(stack.getItem().getRegistryName()).toLowerCase(Locale.ROOT);
        if (registry.contains("golden_apple")) return true;
        String name = clean(stack.getDisplayName()).toLowerCase(Locale.ROOT);
        return name.contains("golden head")
                || name.contains("gold head")
                || (name.contains("gold") && name.contains("head"));
    }

    private static void selectHotbar(Minecraft mc, EntityPlayerSP p, int hotbar) {
        p.inventory.currentItem = hotbar;
        mc.playerController.updateController();
    }

    private static void releaseUse(Minecraft mc) {
        if (mc.gameSettings != null) {
            key(mc.gameSettings.keyBindUseItem, false);
        }
    }

    private void releaseRetreatMovement(Minecraft mc) {
        releaseRetreatMovement(mc, mc == null ? null : mc.thePlayer);
    }

    private void releaseRetreatMovement(Minecraft mc, EntityPlayerSP p) {
        restoreRetreatStepHeight(p);
        if (mc == null || mc.gameSettings == null) return;
        key(mc.gameSettings.keyBindForward, false);
        key(mc.gameSettings.keyBindBack, false);
        key(mc.gameSettings.keyBindLeft, false);
        key(mc.gameSettings.keyBindRight, false);
        key(mc.gameSettings.keyBindJump, false);
        key(mc.gameSettings.keyBindSprint, false);
        if (p == null) return;
        if (p.movementInput != null) {
            p.movementInput.moveForward = 0.0F;
            p.movementInput.moveStrafe = 0.0F;
            p.movementInput.jump = false;
        }
        p.moveForward = 0.0F;
        p.moveStrafing = 0.0F;
        retreatSprintRetapTicks = 0;
        p.setSprinting(false);
    }

    private static void key(KeyBinding kb, boolean down) {
        KeyBinding.setKeyBindState(kb.getKeyCode(), down);
    }

    private static float clampFloat(float v, float lo, float hi) {
        return Math.max(lo, Math.min(hi, v));
    }

    private static int clampInt(int v, int lo, int hi) {
        return Math.max(lo, Math.min(hi, v));
    }

    private static double clampMag(double v, double m) {
        if (v > m) return m;
        if (v < -m) return -m;
        return v;
    }

    private static double wrapDeg(double v) {
        v %= 360.0;
        if (v >= 180.0) v -= 360.0;
        if (v < -180.0) v += 360.0;
        return v;
    }

    private static String clean(String text) {
        if (text == null) return "";
        return text.replaceAll("\u00A7.", "").replace('\n', ' ').replace('\r', ' ');
    }

    private static final class RetreatPath {
        final double x;
        final double z;
        final boolean jump;
        final boolean hopClear;
        final boolean viable;
        final double score;
        final int strafeAssist;
        final boolean blockedAhead;
        final double awayDot;
        final double sideEscape;
        final double centerClearFrac;
        final int nearBlocked;
        final int nearUnsafe;

        RetreatPath(double x, double z, boolean jump, boolean hopClear,
                    boolean viable, double score, int strafeAssist,
                    boolean blockedAhead) {
            this(x, z, jump, hopClear, viable, score, strafeAssist,
                    blockedAhead, 1.0, 0.0, 1.0, 0, 0);
        }

        RetreatPath(double x, double z, boolean jump, boolean hopClear,
                    boolean viable, double score, int strafeAssist,
                    boolean blockedAhead, double awayDot, double sideEscape,
                    double centerClearFrac, int nearBlocked, int nearUnsafe) {
            this.x = x;
            this.z = z;
            this.jump = jump;
            this.hopClear = hopClear;
            this.viable = viable;
            this.score = score;
            this.strafeAssist = strafeAssist;
            this.blockedAhead = blockedAhead;
            this.awayDot = awayDot;
            this.sideEscape = sideEscape;
            this.centerClearFrac = centerClearFrac;
            this.nearBlocked = nearBlocked;
            this.nearUnsafe = nearUnsafe;
        }
    }

    private static final class StepProbe {
        final boolean hardBlocked;
        final boolean jump;
        final boolean unsafeGround;

        StepProbe(boolean hardBlocked, boolean jump, boolean unsafeGround) {
            this.hardBlocked = hardBlocked;
            this.jump = jump;
            this.unsafeGround = unsafeGround;
        }
    }
}
