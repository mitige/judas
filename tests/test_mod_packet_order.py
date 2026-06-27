import re
from random import Random
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]


def _method_body(source: str, signature: str) -> str:
    start = source.index(signature)
    brace = source.index("{", start)
    depth = 0
    for i in range(brace, len(source)):
        if source[i] == "{":
            depth += 1
        elif source[i] == "}":
            depth -= 1
            if depth == 0:
                return source[brace + 1:i]
    raise AssertionError(f"method body not closed: {signature}")


def test_direct_input_attack_is_applied_during_client_tick_start():
    source = (ROOT / "mod/src/main/java/dev/judas/bridge/ActionApplier.java").read_text()
    apply_body = _method_body(source, "public void apply(Minecraft mc, JsonObject a)")

    assert "boolean landed = doAttackDirect(mc, p);" in apply_body
    assert "pendingAttack = useNative && attack;" in apply_body
    assert apply_body.index("boolean landed = doAttackDirect(mc, p);") < apply_body.index(
        "pendingAttack = useNative && attack;"
    )


def test_legacy_direct_attack_sends_animation_before_interact():
    source = (ROOT / "mod/src/main/java/dev/judas/bridge/ActionApplier.java").read_text()
    attack_body = _method_body(source, "private boolean doAttackDirect(Minecraft mc, EntityPlayerSP p)")

    assert "p.swingItem();" in attack_body
    assert "mc.playerController.attackEntity(p, target);" in attack_body
    assert "return true;" in attack_body
    assert "return false;" in attack_body
    assert attack_body.index("p.swingItem();") < attack_body.index(
        "mc.playerController.attackEntity(p, target);"
    )


def test_direct_sprint_hit_releases_forward_before_movement():
    source = (ROOT / "mod/src/main/java/dev/judas/bridge/ActionApplier.java").read_text()
    apply_body = _method_body(source, "public void apply(Minecraft mc, JsonObject a)")

    assert "if (landed && sprint && fwd > 0 && !forceSprint)" in apply_body
    assert "key(mc.gameSettings.keyBindForward, false);" in apply_body
    assert "key(mc.gameSettings.keyBindSprint, false);" in apply_body
    assert "p.setSprinting(false);" in apply_body


def test_keystrokes_overlay_mirrors_bot_mouse_buttons():
    applier = (ROOT / "mod/src/main/java/dev/judas/bridge/ActionApplier.java").read_text()
    mod = (ROOT / "mod/src/main/java/dev/judas/bridge/JudasMod.java").read_text()
    apply_body = _method_body(applier, "public void apply(Minecraft mc, JsonObject a)")
    visual_body = _method_body(applier, "public void applyKeystrokeMouseVisuals(Minecraft mc)")
    release_body = _method_body(applier, "public void releaseAll(Minecraft mc)")
    tick_body = _method_body(mod, "public void onClientTick(TickEvent.ClientTickEvent event)")

    assert "KEYSTROKE_MOUSE_HOLD_TICKS = 2" in applier
    assert "boolean useItem = jsonBool(a, \"use\") || jsonBool(a, \"rightClick\");" in apply_body
    assert "queueKeystrokeMouseVisual(attack, useItem);" in apply_body
    assert apply_body.index("queueKeystrokeMouseVisual") < apply_body.index("if (attack && !useNative)")
    assert "key(mc.gameSettings.keyBindAttack, showAttack);" in visual_body
    assert "key(mc.gameSettings.keyBindUseItem, showUse);" in visual_body
    assert "key(mc.gameSettings.keyBindAttack, false);" in release_body
    assert "key(mc.gameSettings.keyBindUseItem, false);" in release_body
    assert "applier.clearKeystrokeMouseVisualsBeforeInput(mc);" in tick_body
    assert "applier.applyKeystrokeMouseVisuals(mc);" in tick_body
    assert tick_body.index("applier.clearKeystrokeMouseVisualsBeforeInput(mc);") < tick_body.index(
        "if (event.phase == TickEvent.Phase.END)"
    )


def test_state_collector_sends_detected_arena_calibration():
    source = (ROOT / "mod/src/main/java/dev/judas/bridge/StateCollector.java").read_text()
    collect_body = _method_body(source, "public JsonObject collect(Minecraft mc)")
    detect_body = _method_body(source, "private JsonObject detectArena(World world, EntityPlayerSP self, EntityPlayer target)")

    assert "JsonObject arena = detectArena(mc.theWorld, self, target);" in collect_body
    assert 'msg.add("arena", arena);' in collect_body
    assert 'arena.addProperty("origin_x", minX);' in detect_body
    assert 'arena.addProperty("origin_z", minZ);' in detect_body
    assert 'arena.addProperty("size_x", maxX - minX);' in detect_body
    assert 'arena.addProperty("size_z", maxZ - minZ);' in detect_body
    assert 'arena.addProperty("floor_y", floorY);' in detect_body
    assert "findArenaBoundaryX(world, centerX, centerZ, floorY, -1)" in detect_body
    assert "findArenaBoundaryZ(world, centerX, centerZ, floorY, 1)" in detect_body
    assert "private Double findObstacleBoundaryX" in source
    assert "private Double findFloorBoundaryX" in source
    assert "private boolean blocksMovement(World world, BlockPos pos)" in source


def test_live_knockback_dump_can_log_server_kb_samples():
    source = (ROOT / "mod/src/main/java/dev/judas/bridge/StateCollector.java").read_text()
    mod = (ROOT / "mod/src/main/java/dev/judas/bridge/JudasMod.java").read_text()
    collect_body = _method_body(source, "public JsonObject collect(Minecraft mc)")
    tick_body = _method_body(mod, "public void onClientTick(TickEvent.ClientTickEvent event)")
    dump_body = _method_body(source, "private void dumpKnockbackEvent(Minecraft mc, String role, EntityPlayer victim, EntityPlayer attacker)")

    assert "void updateKnockbackDumpFromAction(JsonObject action)" in source
    assert '"knockback_dump"' in source
    assert "collector.updateKnockbackDumpFromAction(action);" in tick_body
    assert "collector.knockbackDumpStatusLine()" in mod
    assert 'new File(mc.mcDataDir, "judas-kb-dump.jsonl")' in source
    assert "dumpKnockbackEvent(mc, \"target\", target, self);" in collect_body
    assert "dumpKnockbackEvent(mc, \"self\", self, target);" in collect_body
    assert "victim.motionX" in dump_body
    assert "victim.motionY" in dump_body
    assert "victim.motionZ" in dump_body
    assert "kb_h_est" in dump_body
    assert "kb_v_est" in dump_body
    assert "server" in dump_body


def test_live_auto_gapple_scans_inventory_and_holds_use_item():
    auto = (ROOT / "mod/src/main/java/dev/judas/bridge/AutoGapple.java").read_text()
    mod = (ROOT / "mod/src/main/java/dev/judas/bridge/JudasMod.java").read_text()
    tick_body = _method_body(mod, "public void onClientTick(TickEvent.ClientTickEvent event)")

    assert "class AutoGapple" in auto
    assert "void updateFromAction(JsonObject action)" in auto
    assert '"auto_gapple"' in auto
    assert "health_threshold" in auto
    assert "findGappleHotbar" in auto
    assert "findGappleInventorySlot" in auto
    assert "windowClick(" in auto
    assert "sendUseItem" in auto
    assert "key(mc.gameSettings.keyBindUseItem, true);" in auto
    assert "golden_apple" in auto
    assert "golden head" in auto
    assert "private final AutoGapple autoGapple = new AutoGapple();" in mod
    assert "autoGapple.updateFromAction(action);" in tick_body
    assert "boolean eating = autoGapple.tickStart(mc, collector.getTarget());" in tick_body
    assert "if (eating)" in tick_body
    assert "applier.apply(mc, action);" in tick_body
    assert tick_body.index("boolean eating = autoGapple.tickStart(mc, collector.getTarget());") < tick_body.index(
        "applier.apply(mc, action);"
    )
    assert "autoGapple.statusLine()" in mod


def test_live_auto_gapple_triggers_immediately_below_health_threshold():
    auto = (ROOT / "mod/src/main/java/dev/judas/bridge/AutoGapple.java").read_text()
    tick_body = _method_body(auto, "boolean tickStart(Minecraft mc, EntityPlayer target)")
    should_eat_body = _method_body(auto, "private boolean shouldEat(EntityPlayerSP p)")

    assert "COOLDOWN_TICKS" not in auto
    assert "cooldownTicks" not in auto
    assert '"cooldown:' not in auto
    assert "if (!shouldEat(p))" in tick_body
    assert "if (!isSafeToStart(p, target))" in tick_body
    assert "int hotbar = findGappleHotbar(p);" in tick_body
    assert tick_body.index("if (!shouldEat(p))") < tick_body.index("int hotbar = findGappleHotbar(p);")
    assert tick_body.index("if (!isSafeToStart(p, target))") < tick_body.index("int hotbar = findGappleHotbar(p);")
    assert "p.getHealth() <= criticalHealthThreshold" in should_eat_body
    assert "p.getAbsorptionAmount() >= ACTIVE_ABSORPTION" in should_eat_body
    assert "p.getHealth() <= healthThreshold" in should_eat_body


def test_live_auto_gapple_avoids_chain_eating_under_pressure():
    auto = (ROOT / "mod/src/main/java/dev/judas/bridge/AutoGapple.java").read_text()
    tick_body = _method_body(auto, "boolean tickStart(Minecraft mc, EntityPlayer target)")
    should_eat_body = _method_body(auto, "private boolean shouldEat(EntityPlayerSP p)")
    safe_body = _method_body(auto, "private boolean isSafeToStart(EntityPlayerSP p, EntityPlayer target)")
    abort_body = _method_body(auto, "private boolean shouldAbortEating(EntityPlayerSP p, EntityPlayer target)")

    assert "critical_health_threshold" in auto
    assert "critical_eat_commit_ticks" in auto
    assert "safe_distance" in auto
    assert "ACTIVE_ABSORPTION = 1.0F" in auto
    assert "if (p.getAbsorptionAmount() >= ACTIVE_ABSORPTION) return false;" in should_eat_body
    assert "private boolean shouldCommitCriticalEating(EntityPlayerSP p)" in auto
    assert "boolean commitCriticalEat = shouldCommitCriticalEating(p);" in auto
    assert 'status = "critical eat commit:" + eatTicks;' in auto
    assert "if (!targetTooCloseToEat(p, target)) return true;" in safe_body
    assert "if (retreatEnabled && retreatTicks < minRetreatTicks) return false;" in safe_body
    assert "return retreatTicks >= retreatTickLimit(critical)" in safe_body
    assert "if (isRetreatCombatHit(p))" in tick_body
    assert tick_body.index("if (isRetreatCombatHit(p))") < tick_body.index(
        "if (!isSafeToStart(p, target))"
    )
    assert "if (p.hurtTime > 0 && p.getHealth() > criticalHealthThreshold) return true;" in abort_body
    assert "targetTooCloseToEat(p, target)" in abort_body
    assert abort_body.index("if (p.hurtTime > 0 && p.getHealth() > criticalHealthThreshold) return true;") < abort_body.index(
        "if (target == null || target.isDead) return false;"
    )
    assert "targetTooCloseToEat(p, target)" in abort_body
    assert "&& p.getHealth() > criticalHealthThreshold" in abort_body
    assert "restorePreviousSlot(mc, p);" in auto
    assert "unsafeStatus(p, target)" in auto


def test_live_auto_gapple_critical_hp_retreats_before_starting_gapple():
    auto = (ROOT / "mod/src/main/java/dev/judas/bridge/AutoGapple.java").read_text()
    safe_body = _method_body(auto, "private boolean isSafeToStart(EntityPlayerSP p, EntityPlayer target)")

    assert "boolean critical = p.getHealth() <= criticalHealthThreshold;" in safe_body
    assert "if (critical && retreatEnabled && targetNeedsRetreat(p, target))" in safe_body
    assert safe_body.index(
        "if (critical && retreatEnabled && targetNeedsRetreat(p, target))"
    ) < safe_body.index(
        "if (!targetTooCloseToEat(p, target)) return true;"
    )
    assert "return retreatTicks >= retreatTickLimit(critical);" in safe_body


def test_live_auto_gapple_can_retreat_away_before_eating():
    auto = (ROOT / "mod/src/main/java/dev/judas/bridge/AutoGapple.java").read_text()
    mod = (ROOT / "mod/src/main/java/dev/judas/bridge/JudasMod.java").read_text()
    tick_body = _method_body(mod, "public void onClientTick(TickEvent.ClientTickEvent event)")
    retreat_body = _method_body(auto, "boolean applyRetreatToAction(Minecraft mc, EntityPlayer target, JsonObject action)")
    eating_retreat_body = _method_body(auto, "private boolean maintainEatingRetreat(Minecraft mc, EntityPlayerSP p,")
    choose_body = _method_body(auto, "private RetreatPath chooseRetreatPath(World world, EntityPlayerSP p, double awayX, double awayZ)")
    score_body = _method_body(auto, "private RetreatPath scorePath(World world, EntityPlayerSP p,")
    probe_body = _method_body(auto, "private StepProbe probeStep(World world, BlockPos pos)")
    retreat_goal_body = _method_body(auto, "private float retreatGoalDistance()")
    safe_eat_body = _method_body(auto, "private float safeEatDistance()")

    assert "retreat_enabled" in auto
    assert "retreat_distance" in auto
    assert "fast_retreat" in auto
    assert "retreat_hops" in auto
    assert "sprint_hop_hold" in auto
    assert "avoid_obstacles" in auto
    assert "retreat_strafe" in auto
    assert "wall_slide" in auto
    assert "retreat_speed_lock" in auto
    assert "retreat_velocity_assist" in auto
    assert "retreat_speed_first" in auto
    assert "retreat_speed_floor" in auto
    assert "retreat_max_speed" in auto
    assert "retreat_accel" in auto
    assert "retreat_sprint_retap" in auto
    assert "retreat_sprint_retap_ticks" in auto
    assert "retreat_air_control" in auto
    assert "retreat_step_assist" in auto
    assert "retreat_step_height" in auto
    assert "fallback_retreat" in auto
    assert "retreat_input_lock" in auto
    assert "force_sprint_retreat" in auto
    assert "release_retreat_on_hit" in auto
    assert "critical_rearm_only" in auto
    assert "critical_trapped_eat" in auto
    assert "retreat_turn_limit_deg" in auto
    assert "eating_retreat_turn_limit_deg" in auto
    assert "retreat_path_hold_ticks" in auto
    assert "retreat_stuck_abort_ticks" in auto
    assert "retreat_min_ticks" in auto
    assert "retreat_max_ticks" in auto
    assert "critical_retreat_max_ticks" in auto
    assert "critical_eat_commit_ticks" in auto
    assert "combat_recovery_ticks" in auto
    assert "retreat_strafe_hold_ticks" in auto
    assert "retreat_obstacle_jump_hold_ticks" in auto
    assert "retreat_obstacle_escape_ticks" in auto
    assert "retreat_panic_speed" in auto
    assert "retreat_obstacle_lookahead" in auto
    assert "critical_trapped_stuck_ticks" in auto
    assert 'clampFloat(cfg.get("retreat_distance").getAsFloat(), 2.0F, 40.0F)' in auto
    assert 'cfg.get("retreat_turn_limit_deg").getAsFloat(), 45.0F, 360.0F' in auto
    assert 'cfg.get("eating_retreat_turn_limit_deg").getAsFloat(), 45.0F, 360.0F' in auto
    assert 'cfg.get("retreat_path_hold_ticks").getAsInt(), 0, 12' in auto
    assert 'cfg.get("retreat_stuck_abort_ticks").getAsInt(), 1, 20' in auto
    assert 'clampInt(cfg.get("retreat_min_ticks").getAsInt(), 0, 40)' in auto
    assert 'clampInt(cfg.get("retreat_max_ticks").getAsInt(), 1, 120)' in auto
    assert 'cfg.get("retreat_strafe_hold_ticks").getAsInt(), 0, 12' in auto
    assert 'cfg.get("retreat_obstacle_jump_hold_ticks").getAsInt(), 0, 60' in auto
    assert 'cfg.get("retreat_obstacle_escape_ticks").getAsInt(), 0, 120' in auto
    assert 'cfg.get("retreat_speed_floor").getAsFloat(), 0.05F, 6.00F' in auto
    assert 'cfg.get("retreat_max_speed").getAsFloat(), 0.10F, 6.00F' in auto
    assert 'cfg.get("retreat_accel").getAsFloat(), 0.01F, 8.00F' in auto
    assert 'cfg.get("retreat_sprint_retap_ticks").getAsInt(), 0, 8' in auto
    assert 'cfg.get("retreat_step_height").getAsFloat(), 0.60F, 1.25F' in auto
    assert 'cfg.get("retreat_obstacle_lookahead").getAsFloat(), 1.20F, 24.00F' in auto
    assert "private float safeDistance = 11.50F;" in auto
    assert "private boolean retreatEnabled = true;" in auto
    assert "private float retreatDistance = 18.0F;" in auto
    assert "private boolean fastRetreat = true;" in auto
    assert "private boolean retreatHops = true;" in auto
    assert "private boolean sprintHopHold = true;" in auto
    assert "private boolean avoidObstacles = true;" in auto
    assert "private boolean retreatStrafe = true;" in auto
    assert "private boolean wallSlide = true;" in auto
    assert "private boolean retreatSpeedLock = true;" in auto
    assert "private boolean retreatVelocityAssist = true;" in auto
    assert "private boolean retreatSpeedFirst = true;" in auto
    assert "private boolean retreatFullSpeed = true;" in auto
    assert "private float retreatSpeedFloor = 4.50F;" in auto
    assert "private float retreatMaxSpeed = 4.80F;" in auto
    assert "private float retreatAccel = 5.50F;" in auto
    assert "private boolean retreatSprintRetap = true;" in auto
    assert "private int retreatSprintRetapMaxTicks = 2;" in auto
    assert "private boolean retreatAirControl = true;" in auto
    assert "private boolean retreatStepAssist = true;" in auto
    assert "private float retreatStepHeight = 1.20F;" in auto
    assert "private boolean fallbackRetreat = true;" in auto
    assert "private boolean retreatInputLock = true;" in auto
    assert "private boolean forceSprintRetreat = true;" in auto
    assert "private boolean releaseRetreatOnHit = true;" in auto
    assert "private boolean criticalRearmOnly = true;" in auto
    assert "private boolean criticalTrappedEat = true;" in auto
    assert "private float retreatTurnLimitDeg = 360.0F;" in auto
    assert "private float eatingRetreatTurnLimitDeg = 360.0F;" in auto
    assert "private int retreatPathHoldMaxTicks = 2;" in auto
    assert "private boolean enabled = true;" in auto
    assert "private int retreatStuckAbortTicks = 4;" in auto
    assert "private int minRetreatTicks = 0;" in auto
    assert "private int maxRetreatTicks = 64;" in auto
    assert "private int criticalRetreatMaxTicks = 6;" in auto
    assert "private int criticalEatCommitTicks = 12;" in auto
    assert "private int combatRecoveryTicks = 6;" in auto
    assert "private int retreatStrafeHoldMaxTicks = 5;" in auto
    assert "private int retreatObstacleJumpHoldMaxTicks = 60;" in auto
    assert "private int retreatObstacleEscapeTicks = 120;" in auto
    assert "private boolean retreatPanicSpeed = true;" in auto
    assert "private float retreatObstacleLookahead = 24.00F;" in auto
    assert "private int criticalTrappedStuckTicks = 2;" in auto
    assert "private boolean retreatInterrupted = false;" in auto
    assert "private int retreatCombatTicks = 0;" in auto
    assert "private int retreatPathHoldTicks = 0;" in auto
    assert "private int retreatStrafeHoldTicks = 0;" in auto
    assert "private int retreatObstacleJumpTicks = 0;" in auto
    assert "private int retreatTicks = 0;" in auto
    assert "private int retreatStuckTicks = 0;" in auto
    assert "private int retreatSprintRetapTicks = 0;" in auto
    assert "private float previousStepHeight = -1.0F;" in auto
    assert "private int lastRetreatStrafe = 0;" in auto
    assert "JsonObject fallbackRetreatAction(Minecraft mc, EntityPlayer target)" in auto
    assert "if (!enabled || !retreatEnabled || !fallbackRetreat) return null;" in auto
    assert "retreatCombatTicks > 0 || p.hurtTime > 0" in auto
    assert "criticalRearmOnly && retreatInterrupted" in auto
    assert "criticalTrappedEat" in auto
    assert "retreatStuckTicks >= criticalTrappedStuckTicks" in auto
    assert 'action.addProperty("retreat_turn", true);' in auto
    assert 'action.addProperty("forward", 1);' in auto
    assert 'action.addProperty("sprint", true);' in auto
    assert 'action.addProperty("force_sprint", true);' in auto
    assert 'action.addProperty("attack", false);' in auto
    assert "awayX = p.posX - target.posX" in retreat_body
    assert "awayZ = p.posZ - target.posZ" in retreat_body
    assert "? chooseRetreatPath(mc.theWorld, p, awayX, awayZ)" in retreat_body
    assert ': new RetreatPath(awayX, awayZ, false, true, true, 0.0, 0, false);' in retreat_body
    assert "double desiredYaw = Math.toDegrees(Math.atan2(-path.x, path.z));" in retreat_body
    assert "wrapDeg(desiredYaw - p.rotationYaw)" in retreat_body
    assert "boolean unstuckHop = (p.isCollidedHorizontally" in retreat_body
    assert "boolean sprintHop = shouldSprintHop(mc.theWorld, p, path);" in retreat_body
    assert "boolean obstacleHop = shouldObstacleHop(mc.theWorld, p, path);" in retreat_body
    assert "boolean holdHop = shouldHoldSprintHop(p, path);" in retreat_body
    assert "boolean hop = updateRetreatHop(p, path," in retreat_body
    assert "sprintHop || holdHop" in retreat_body
    assert 'action.addProperty("dyaw", dyaw);' in retreat_body
    assert 'action.addProperty("dpitch", 0.0);' in retreat_body
    assert 'action.addProperty("retreat_turn", true);' in retreat_body
    assert 'action.addProperty("forward", 1);' in retreat_body
    assert "retreatStrafeForPath(p, path, desiredYaw, sideEscape)" in retreat_body
    assert 'action.addProperty("strafe", strafe);' in retreat_body
    assert 'else action.addProperty("strafe", 0);' in retreat_body
    assert 'action.addProperty("sprint", true);' in retreat_body
    assert 'action.addProperty("force_sprint", true);' in retreat_body
    assert "forceRetreatStepAssist(p, path);" in retreat_body
    assert "forceRetreatSprint(mc, p);" in retreat_body
    assert "forceRetreatInput(mc, p, strafe, hop || p.isCollidedHorizontally);" in retreat_body
    assert "forceRetreatVelocity(p, path, strafe);" in retreat_body
    assert 'if (hop) action.addProperty("jump", true);' in retreat_body
    assert 'else action.addProperty("jump", p.isCollidedHorizontally);' in retreat_body
    assert 'action.addProperty("attack", false);' in retreat_body
    assert 'action.addProperty("use", false);' in retreat_body
    assert 'action.addProperty("rightClick", false);' in retreat_body
    assert "if (!shouldEat(p))" in retreat_body
    assert "resetRetreatState();" in retreat_body
    assert "boolean critical = p.getHealth() <= criticalHealthThreshold;" in retreat_body
    assert "if (critical) {" in retreat_body
    assert "retreatInterrupted = false;" in retreat_body
    assert "retreatCombatTicks = 0;" in retreat_body
    assert "else if (retreatCombatTicks > 0)" in retreat_body
    assert "if (retreatInterrupted && !critical && criticalRearmOnly)" in retreat_body
    assert 'status = "retreat stopped -> normal";' in retreat_body
    assert "private boolean isRetreatCombatHit(EntityPlayerSP p)" in auto
    assert "p.hurtTime > 0 && p.getHealth() > criticalHealthThreshold" in auto
    assert "if (isRetreatCombatHit(p))" in retreat_body
    assert "startCombatRecovery(mc, \"retreat hit -> normal\");" in retreat_body
    assert '"retreat hit -> normal"' in retreat_body
    assert retreat_body.index("if (isRetreatCombatHit(p))") < retreat_body.index("float need = retreatGoalDistance();")
    assert "int tickLimit = retreatTickLimit(critical);" in retreat_body
    assert "if (retreatTicks >= tickLimit)" in retreat_body
    assert '"retreat max -> eat d=%.1f/%.1f"' in retreat_body
    assert 'status = "retreat max -> normal";' in retreat_body
    assert "updateRetreatProgress(p);" in retreat_body
    assert "boolean blockedButUsable = canUseBlockedRetreatPath(path, awayX, awayZ);" in retreat_body
    assert "boolean obstacleEscape = shouldForceObstacleEscape(p, path, blockedButUsable);" in retreat_body
    assert "if (!path.viable && !blockedButUsable && !critical && !obstacleEscape)" in retreat_body
    assert 'status = "retreat blocked -> normal";' in retreat_body
    assert '"retreat blocked -> eat d=%.1f/%.1f"' in retreat_body
    assert "p.getHealth() <= criticalHealthThreshold" in retreat_body
    assert "retreat%s%s%s%s d=%.1f/%.1f yaw=%.0f t=%d/%d stuck=%d str=%d spd=%.1f" in retreat_body
    assert "RETREAT_COUNTER_COOLDOWN_TICKS" not in auto
    assert "RETREAT_COUNTER_REACH" not in auto
    assert 'action.addProperty("attack", true);' not in auto
    assert "RETREAT_PATH_HOLD_TICKS = 2" in auto
    assert "RETREAT_STRAFE_HOLD_TICKS = 5" in auto
    assert "RETREAT_OBSTACLE_JUMP_HOLD_TICKS = 48" in auto
    assert "RETREAT_OBSTACLE_ESCAPE_TICKS = 120" in auto
    assert "RETREAT_REPLAN_MARGIN = 0.00" in auto
    assert "RETREAT_ANGLE_OFFSETS" in auto
    assert "RETREAT_AHEAD_SAMPLES" in auto
    assert "RETREAT_SIDE_SAMPLES" in auto
    assert "105.0, -105.0" in auto
    assert "5.0, -5.0" in auto
    assert "7.5, -7.5" in auto
    assert "18.75, -18.75" in auto
    assert "82.5, -82.5" in auto
    assert "150.0, -150.0, 165.0, -165.0, 180.0" in auto
    assert "0.20" in auto
    assert "0.50" in auto
    assert "5.20" in auto
    assert "6.40" in auto
    assert "7.80" in auto
    assert "9.40" in auto
    assert "11.20" in auto
    assert "14.40" in auto
    assert "16.00" in auto
    assert "1.68" in auto
    assert "2.10" in auto
    assert "2.60" in auto
    assert "RETREAT_PANIC_MIN_AWAY_DOT" in auto
    assert "RETREAT_PANIC_MIN_SCORE" in auto
    assert "RETREAT_PANIC_SIDE_ESCAPE" in auto
    assert "RETREAT_NEAR_BLOCKED_ESCAPE = 1" in auto
    assert "RETREAT_NEAR_UNSAFE_ESCAPE = 1" in auto
    assert "RETREAT_STUCK_JUMP_TICKS = 1" in auto
    assert "RETREAT_MIN_PROGRESS" in auto
    assert "RETREAT_SPEED_MIN_AWAY_DOT" in auto
    assert "RETREAT_WIDE_SIDE_WIDTH" in auto
    assert "RETREAT_EXTRA_WIDE_SIDE_WIDTH" in auto
    assert "RETREAT_EAT_TURN_LIMIT_DEG" in auto
    assert "if (retreatPathHoldTicks > 0 && retreatStuckTicks <= 0 && !p.isCollidedHorizontally)" in choose_body
    assert "scorePath(world, p, lastRetreatX, lastRetreatZ, awayX, awayZ)" in choose_body
    assert "held.viable && !held.blockedAhead" in choose_body
    assert "for (double offset : RETREAT_ANGLE_OFFSETS)" in choose_body
    assert "scorePath(world, p, dir[0], dir[1], awayX, awayZ)" in choose_body
    assert "RetreatPath straight = scorePath(world, p, awayX, awayZ, awayX, awayZ);" in choose_body
    assert "shouldPreferStraightRetreat(straight, best, p)" in choose_body
    assert "candidate.score > best.score + margin" in choose_body
    assert "isSaferRetreatCandidate(candidate, best, margin)" in choose_body
    assert "private int retreatHazardScore(RetreatPath path)" in auto
    assert "!candidate.blockedAhead" in choose_body
    assert "best.blockedAhead" in choose_body
    assert "candidate.viable && !best.viable" in choose_body
    assert "lastRetreatX = best.x;" in choose_body
    assert "retreatPathHoldTicks = bestHeld" in choose_body
    assert "best.blockedAhead || p.isCollidedHorizontally || retreatStuckTicks > 0 || !best.viable" in choose_body
    assert "private void updateRetreatProgress(EntityPlayerSP p)" in auto
    assert "retreatStuckTicks++;" in auto
    assert "retreatPathHoldTicks = 0;" in auto
    assert "centerUnsafe" in score_body
    assert "nearBlocked" in score_body
    assert "nearUnsafe" in score_body
    assert "safeCenter" in score_body
    assert "centerChecked" in score_body
    assert "centerClear" in score_body
    assert "hopClear" in score_body
    assert "viable" in score_body
    assert "probeStep(world, pos)" in score_body
    assert "score -= 100.0" in score_body
    assert "centerClearFrac" in score_body
    assert "cleanCenter" in score_body
    assert "score = awayDot * 44.0" in score_body
    assert "score += sideEscape * 13.50" in score_body
    assert "score -= blocked * 3.35" in score_body
    assert "score -= unsafe * 4.50" in score_body
    assert "score -= centerUnsafe * 7.00" in score_body
    assert "score -= nearBlocked * 3.00" in score_body
    assert "score -= nearUnsafe * 6.25" in score_body
    assert "p.isCollidedHorizontally && jump" in score_body
    assert "RETREAT_NEAR_BLOCK_AHEAD" in auto
    assert "RETREAT_HARD_BLOCK_AHEAD" in auto
    assert "farCenterBlocked" in score_body
    assert "centerHardBlocked" in score_body
    assert "sideEscape" in score_body
    assert "retreatStuckTicks >= RETREAT_STUCK_JUMP_TICKS" in score_body
    assert "blockedAhead" in score_body
    assert "strafeAssist" in score_body
    assert "boolean speedPriority = fastRetreat || forceSprintRetreat;" in score_body
    assert "RETREAT_SPEED_MIN_AWAY_DOT" in score_body
    assert "RETREAT_SPEED_FIRST_MIN_AWAY_DOT" in score_body
    assert "speedPriority && hopClear" in score_body
    assert "speedPriority && cleanCenter" in score_body
    assert "speedPriority && awayDot > 0.92" in score_body
    assert "speedPriority && !blockedAhead && nearBlocked == 0 && nearUnsafe == 0" in score_body
    assert "speedPriority && blockedAhead" in score_body
    assert "speedPriority && p.isCollidedHorizontally && sideEscape >= 0.18" in score_body
    assert "speedPriority && blocked == 0 && unsafe == 0 && awayDot > 0.80" in score_body
    assert "score += 9.40" in score_body
    assert "speedPriority && retreatSpeedFirst" in score_body
    assert "score += awayDot * 55.0" in score_body
    assert "score += 24.00" in score_body
    assert "score -= sideEscape * 26.00" in score_body
    assert "score -= (RETREAT_SPEED_FIRST_MIN_AWAY_DOT - awayDot) * 120.0" in score_body
    assert "score += sideEscape * 16.00" in score_body
    assert "score += sideEscape * 8.00" in score_body
    assert "score += sideEscape * 9.50" in score_body
    assert "score += sideEscape * 10.00" in score_body
    assert "private boolean canUseBlockedRetreatPath(RetreatPath path, double awayX, double awayZ)" in auto
    assert "private boolean shouldForceObstacleEscape(EntityPlayerSP p, RetreatPath path, boolean blockedButUsable)" in auto
    assert "retreatTicks < retreatObstacleEscapeTicks" in auto
    assert "path.blockedAhead" in auto
    assert "hasNearRetreatObstacle(path)" in auto
    assert "blockedButUsable" in auto
    assert "p.isCollidedHorizontally" in auto
    assert "private int retreatStrafeForPath(EntityPlayerSP p, RetreatPath path," in auto
    assert "private int stabilizeRetreatStrafe(EntityPlayerSP p, RetreatPath path," in auto
    assert "RETREAT_STRAFE_TURN_DEG" in auto
    assert "RETREAT_STRAFE_SIDE_ESCAPE" in auto
    assert "RETREAT_FAST_STRAFE_SPEED_BLEND" in auto
    assert "RETREAT_SPEED_STRAFE_SUPPRESS_SIDE" in auto
    assert "RETREAT_STRAIGHT_OVERRIDE_CENTER_CLEAR" in auto
    assert "RETREAT_STRAIGHT_OVERRIDE_SCORE_GAP" in auto
    assert "RETREAT_CLEAN_STRAFE_SUPPRESS_SIDE" in auto
    assert "RETREAT_SIDE_MOTION_DAMP" in auto
    assert "boolean turnAssist = Math.abs(dyaw) >= RETREAT_STRAFE_TURN_DEG" in auto
    assert "private boolean shouldPreferStraightRetreat(RetreatPath straight," in auto
    assert "private boolean isCleanSpeedRetreatPath(RetreatPath path)" in auto
    assert "straight.centerClearFrac < RETREAT_STRAIGHT_OVERRIDE_CENTER_CLEAR" in auto
    assert "isCleanSpeedRetreatPath(straight)" in auto
    assert "Math.abs(sideEscape) <= RETREAT_SPEED_STRAFE_SUPPRESS_SIDE" in auto
    assert "Math.abs(sideEscape) <= RETREAT_CLEAN_STRAFE_SUPPRESS_SIDE" in auto
    assert "|| !path.viable" in auto
    assert "retreatStrafeHoldTicks--;" in auto
    assert "lastRetreatStrafe = strafe;" in auto
    assert "private boolean updateRetreatHop(EntityPlayerSP p, RetreatPath path," in auto
    assert "private boolean canFastHopPath(RetreatPath path)" in auto
    assert "&& !path.blockedAhead" in auto
    assert "path.nearUnsafe < RETREAT_NEAR_UNSAFE_ESCAPE" in auto
    assert "retreatObstacleJumpTicks = Math.max(" in auto
    assert "return obstacleHop || speedHop || heldObstacleHop;" in auto
    assert "wallSlide && (p.isCollidedHorizontally" in auto
    assert "path.blockedAhead" in auto
    assert "path.strafeAssist" in auto
    assert "private boolean maintainEatingRetreat(Minecraft mc, EntityPlayerSP p," in auto
    assert 'key(mc.gameSettings.keyBindForward, true);' in eating_retreat_body
    assert 'key(mc.gameSettings.keyBindSprint, true);' in eating_retreat_body
    assert "key(mc.gameSettings.keyBindSneak, false);" in auto
    assert "forceRetreatSprint(mc, p);" in eating_retreat_body
    assert "forceRetreatStepAssist(p, path);" in eating_retreat_body
    assert "forceRetreatInput(mc, p, strafe, hop || p.isCollidedHorizontally);" in eating_retreat_body
    assert "forceRetreatVelocity(p, path, strafe);" in eating_retreat_body
    assert "shouldObstacleHop(mc.theWorld, p, path)" in eating_retreat_body
    assert "updateRetreatHop(p, path, hop || unstuckHop || obstacleEscape, speedHop)" in eating_retreat_body
    assert "p.setSneaking(false);" in auto
    assert "p.movementInput.sneak = false;" in auto
    assert "p.setSprinting(true);" in auto
    assert "p.stepHeight = Math.max(p.stepHeight, wanted);" in auto
    assert "p.stepHeight = previousStepHeight;" in auto
    assert "releaseRetreatMovement(mc);" in auto
    assert "private void forceRetreatSprint(Minecraft mc, EntityPlayerSP p)" in auto
    assert "private void forceRetreatStepAssist(EntityPlayerSP p, RetreatPath path)" in auto
    assert "private void restoreRetreatStepHeight(EntityPlayerSP p)" in auto
    assert "private void forceRetreatInput(Minecraft mc, EntityPlayerSP p, int strafe, boolean jump)" in auto
    assert "private void forceRetreatVelocity(EntityPlayerSP p, RetreatPath path, int strafe)" in auto
    assert "retreatVelocityAssist" in auto
    assert "double accel = retreatAccel;" in auto
    assert "retreatMaxSpeed * (retreatSpeedFirst ? 1.00 : 0.92)" in auto
    assert "floor = Math.max(floor, retreatMaxSpeed);" in auto
    assert "accel = Math.max(accel, retreatSpeedFirst ? 1.00 : 0.74);" in auto
    assert "p.motionX += moveX * boost;" in auto
    assert "p.motionZ += moveZ * boost;" in auto
    assert "double sideMotion = p.motionX * sideX + p.motionZ * sideZ;" in auto
    assert "p.motionX -= sideX * sideMotion * damp;" in auto
    assert "p.motionZ -= sideZ * sideMotion * damp;" in auto
    assert "double scale = max / speed;" in auto
    assert "boolean helpfulFastRetreat = alongAfter >= max * 0.98;" in auto
    assert "p.moveForward = 1.0F;" in auto
    assert "p.moveStrafing = strafe > 0 ? 1.0F : (strafe < 0 ? -1.0F : 0.0F);" in auto
    assert "p.movementInput.moveForward = 1.0F;" in auto
    assert "p.movementInput.moveStrafe = strafe > 0 ? 1.0F : (strafe < 0 ? -1.0F : 0.0F);" in auto
    assert "p.movementInput.jump = jump;" in auto
    assert "p.movementInput.moveForward = 0.0F;" in auto
    assert "p.movementInput.moveStrafe = 0.0F;" in auto
    assert "p.movementInput.sneak = false;" in auto
    assert "chooseRetreatPath(mc.theWorld, p, awayX, awayZ)" in eating_retreat_body
    assert "p.rotationYaw += (float) dyaw;" in eating_retreat_body
    assert "private int retreatTickLimit(boolean critical)" in auto
    assert "Math.min(maxRetreatTicks, Math.max(minRetreatTicks, criticalRetreatMaxTicks))" in auto
    assert "private boolean shouldSprintHop(World world, EntityPlayerSP p, RetreatPath path)" in auto
    assert "private boolean shouldObstacleHop(World world, EntityPlayerSP p, RetreatPath path)" in auto
    assert "private boolean shouldHoldSprintHop(EntityPlayerSP p, RetreatPath path)" in auto
    assert "sprintHopHold" in auto
    assert "!(fastRetreat || forceSprintRetreat) || !retreatHops || !canFastHopPath(path)" in auto
    assert "private boolean canJumpNow(EntityPlayerSP p)" in auto
    assert "p.onGround && !p.isInWater() && !p.isInLava() && !p.isOnLadder()" in auto
    assert "isLiquid(world, pos)" in probe_body
    assert "blocksMovement(world, pos.down()) || foot" in probe_body
    assert "if (isStepObstacle(world, pos)) return new StepProbe(false, true, liquid);" in probe_body
    assert "return new StepProbe(true, false, true);" in probe_body
    assert "retreatEnabled ? Math.max(safeDistance, retreatDistance) : safeDistance" in retreat_goal_body
    assert "return safeDistance;" in safe_eat_body
    assert "private boolean targetNeedsRetreat(EntityPlayerSP p, EntityPlayer target)" in auto
    assert "private boolean targetTooCloseToEat(EntityPlayerSP p, EntityPlayer target)" in auto
    assert "autoGapple.applyRetreatToAction(mc, collector.getTarget(), action);" in tick_body
    assert "boolean hasModelAction = action != null;" in tick_body
    assert "action = autoGapple.fallbackRetreatAction(mc, collector.getTarget());" in tick_body
    assert "if (hasModelAction || retreating)" in tick_body
    assert tick_body.index("autoGapple.applyRetreatToAction(mc, collector.getTarget(), action);") < tick_body.index(
        "autoJump.applyToAction(mc, action);"
    )
    assert tick_body.index("autoGapple.applyRetreatToAction(mc, collector.getTarget(), action);") < tick_body.index(
        "applier.apply(mc, action);"
    )


def test_live_auto_gapple_fast_retreat_defaults_are_exposed_to_live_app():
    live_py = (ROOT / "serve/live.py").read_text()
    live_js = (ROOT / "app/src/pages/Live.jsx").read_text()

    assert "retreat_speed_first: bool = True" in live_py
    assert "safe_distance: float = 11.50" in live_py
    assert "retreat_distance: float = 18.0" in live_py
    assert "retreat_full_speed: bool = True" in live_py
    assert "retreat_speed_floor: float = 4.50" in live_py
    assert "retreat_max_speed: float = 4.80" in live_py
    assert "retreat_accel: float = 5.50" in live_py
    assert "retreat_sprint_retap: bool = True" in live_py
    assert "retreat_sprint_retap_ticks: int = 2" in live_py
    assert "retreat_air_control: bool = True" in live_py
    assert "retreat_step_assist: bool = True" in live_py
    assert "retreat_step_height: float = 1.20" in live_py
    assert "critical_rearm_only: bool = True" in live_py
    assert "critical_trapped_eat: bool = True" in live_py
    assert "retreat_path_hold_ticks: int = 2" in live_py
    assert "retreat_stuck_abort_ticks: int = 4" in live_py
    assert "retreat_max_ticks: int = 64" in live_py
    assert "critical_retreat_max_ticks: int = 6" in live_py
    assert "retreat_strafe_hold_ticks: int = 5" in live_py
    assert "retreat_obstacle_jump_hold_ticks: int = 60" in live_py
    assert "retreat_obstacle_escape_ticks: int = 120" in live_py
    assert "retreat_panic_speed: bool = True" in live_py
    assert "retreat_obstacle_lookahead: float = 24.00" in live_py
    assert "critical_trapped_stuck_ticks: int = 2" in live_py

    assert "autoGappleSpeedFirst: true" in live_js
    assert "autoGappleSafeDistance: 11.50" in live_js
    assert "autoGappleRetreatDistance: 18" in live_js
    assert "autoGappleFullSpeed: true" in live_js
    assert "autoGappleSpeedFloor: 4.50" in live_js
    assert "autoGappleMaxSpeed: 4.80" in live_js
    assert "autoGappleAccel: 5.50" in live_js
    assert "autoGappleSprintRetap: true" in live_js
    assert "autoGappleSprintRetapTicks: 2" in live_js
    assert "autoGappleAirControl: true" in live_js
    assert "autoGappleStepAssist: true" in live_js
    assert "autoGappleStepHeight: 1.20" in live_js
    assert "autoGappleCriticalRearmOnly: true" in live_js
    assert "autoGappleCriticalTrappedEat: true" in live_js
    assert "autoGappleRetreatPathHoldTicks: 2" in live_js
    assert "autoGappleRetreatStuckAbortTicks: 4" in live_js
    assert "autoGappleRetreatMaxTicks: 64" in live_js
    assert "autoGappleCriticalRetreatMaxTicks: 6" in live_js
    assert "autoGappleRetreatStrafeHoldTicks: 5" in live_js
    assert "autoGappleRetreatObstacleJumpHoldTicks: 60" in live_js
    assert "autoGappleRetreatObstacleEscapeTicks: 120" in live_js
    assert "autoGappleRetreatPanicSpeed: true" in live_js
    assert "autoGappleRetreatObstacleLookahead: 24.00" in live_js
    assert "autoGappleCriticalTrappedStuckTicks: 2" in live_js
    assert 'usePersistentState("judas:app:live:v47", LIVE_DEFAULTS)' in live_js
    assert "knockbackDump: false" in live_js
    assert "knockback_dump: { enabled: !!knockbackDump }" in live_js
    assert "kb dump" in live_js
    assert "retreat_speed_first: !!autoGappleSpeedFirst" in live_js
    assert "retreat_full_speed: !!autoGappleFullSpeed" in live_js
    assert "retreat_speed_floor: +autoGappleSpeedFloor" in live_js
    assert "retreat_max_speed: +autoGappleMaxSpeed" in live_js
    assert "retreat_sprint_retap: !!autoGappleSprintRetap" in live_js
    assert "retreat_sprint_retap_ticks: +autoGappleSprintRetapTicks" in live_js
    assert "retreat_air_control: !!autoGappleAirControl" in live_js
    assert "retreat_step_assist: !!autoGappleStepAssist" in live_js
    assert "retreat_step_height: +autoGappleStepHeight" in live_js
    assert "retreat_panic_speed: !!autoGappleRetreatPanicSpeed" in live_js
    assert "retreat_obstacle_lookahead: +autoGappleRetreatObstacleLookahead" in live_js
    assert "critical_rearm_only: !!autoGappleCriticalRearmOnly" in live_js
    assert "critical_trapped_eat: !!autoGappleCriticalTrappedEat" in live_js
    assert "critical_trapped_stuck_ticks: +autoGappleCriticalTrappedStuckTicks" in live_js
    assert "full-speed retreat" in live_js
    assert "sprint retap" in live_js
    assert "air control" in live_js
    assert "step assist" in live_js
    assert "step height" in live_js


def test_live_auto_gapple_speed_first_retreat_keeps_obstacle_escape_scoped():
    auto = (ROOT / "mod/src/main/java/dev/judas/bridge/AutoGapple.java").read_text()
    score_body = _method_body(auto, "private RetreatPath scorePath(World world, EntityPlayerSP p,")
    obstacle_body = _method_body(auto, "private boolean shouldObstacleHop(World world, EntityPlayerSP p, RetreatPath path)")
    velocity_body = _method_body(auto, "private void forceRetreatVelocity(EntityPlayerSP p, RetreatPath path, int strafe)")
    helper_body = _method_body(auto, "private boolean needsObstacleEscape(EntityPlayerSP p, RetreatPath path)")

    assert "RETREAT_SPEED_FIRST_MIN_AWAY_DOT = 0.76" in auto
    assert "RETREAT_SPEED_STRAFE_SUPPRESS_SIDE = 0.44" in auto
    assert "RETREAT_STRAIGHT_OVERRIDE_SCORE_GAP = 110.0" in auto
    assert "RETREAT_CLEAN_STRAFE_SUPPRESS_SIDE = 0.88" in auto
    assert "score += awayDot * 55.0;" in score_body
    assert "score -= sideEscape * 26.00;" in score_body
    assert "(RETREAT_SPEED_FIRST_MIN_AWAY_DOT - awayDot) * 120.0" in score_body
    assert "private boolean needsObstacleEscape(EntityPlayerSP p, RetreatPath path)" in auto
    assert "path.blockedAhead" in helper_body
    assert "hasNearRetreatObstacle(path)" in helper_body
    assert "p.isCollidedHorizontally" in helper_body
    assert "retreatStuckTicks > 0" in helper_body
    assert "|| !needsObstacleEscape(p, path)" in obstacle_body
    assert "boolean escape = needsObstacleEscape(p, path) || !path.viable;" in velocity_body
    assert "boolean panicSpeed = panicRetreatActive(p, path);" in velocity_body
    assert "RETREAT_PANIC_SPEED_MULTIPLIER" in auto
    assert "RETREAT_START_BURST_TICKS = 8" in auto
    assert "RETREAT_START_BURST_SPEED_MULTIPLIER" in auto
    assert "RETREAT_START_BURST_ACCEL_MULTIPLIER" in auto
    assert "RETREAT_FULL_SPEED_CLEAN_SIDE_LIMIT" in auto
    assert "retreatFullSpeed" in score_body
    assert "private boolean panicRetreatActive(EntityPlayerSP p, RetreatPath path)" in auto


def test_live_auto_jump_toggles_and_sets_jump_before_action_apply():
    auto = (ROOT / "mod/src/main/java/dev/judas/bridge/AutoJump.java").read_text()
    mod = (ROOT / "mod/src/main/java/dev/judas/bridge/JudasMod.java").read_text()
    tick_body = _method_body(mod, "public void onClientTick(TickEvent.ClientTickEvent event)")
    apply_body = _method_body(auto, "boolean applyToAction(Minecraft mc, JsonObject action)")
    obstacle_body = _method_body(auto, "private boolean isStepObstacle(World world, BlockPos pos)")

    assert "class AutoJump" in auto
    assert "void updateFromAction(JsonObject action)" in auto
    assert '"auto_jump"' in auto
    assert "action.addProperty(\"jump\", true);" in apply_body
    assert "if (shouldJump(mc.theWorld, p, action))" in apply_body
    assert "HOLD_TICKS = 4" in auto
    assert "blocksMovement(world, pos)" in obstacle_body
    assert "!blocksMovement(world, pos.up())" in obstacle_body
    assert "!blocksMovement(world, pos.up().up())" in obstacle_body
    assert "private final AutoJump autoJump = new AutoJump();" in mod
    assert "autoJump.updateFromAction(action);" in tick_body
    assert "autoJump.applyToAction(mc, action);" in tick_body
    assert "applier.apply(mc, action);" in tick_body
    assert tick_body.index("autoJump.applyToAction(mc, action);") < tick_body.index(
        "applier.apply(mc, action);"
    )
    assert "autoJump.statusLine()" in mod
    assert "autoJump.cancel();" in mod


def test_live_friend_mode_filters_targets_and_blocks_attacks():
    collector = (ROOT / "mod/src/main/java/dev/judas/bridge/StateCollector.java").read_text()
    applier = (ROOT / "mod/src/main/java/dev/judas/bridge/ActionApplier.java").read_text()
    mod = (ROOT / "mod/src/main/java/dev/judas/bridge/JudasMod.java").read_text()
    apply_body = _method_body(applier, "public void apply(Minecraft mc, JsonObject a)")
    native_body = _method_body(applier, "public void applyDeferredAttack(Minecraft mc)")

    assert "void updateFriendsFromAction(JsonObject action)" in collector
    assert '"friends"' in collector
    assert "friendsEnabled" in collector
    assert "friends.contains(cleanName(p.getName()))" in collector
    assert "if (isFriend(p)) continue;" in collector
    assert "collector.updateFriendsFromAction(action);" in mod
    assert "collector.friendStatusLine()" in mod
    assert "if (attack && collector.isTargetFriend()) attack = false;" in apply_body
    assert "if (collector.isTargetFriend()) return;" in native_body


def test_packet_order_probe_observes_without_mutating_outbound_packets():
    source = (ROOT / "mod/src/main/java/dev/judas/bridge/PacketOrderProbe.java").read_text()

    assert "extends ChannelDuplexHandler" in source
    assert "msg instanceof C0APacketAnimation" in source
    assert "C02PacketUseEntity.Action.ATTACK" in source
    assert "ctx.write(msg, promise);" in source


def test_packet_order_probe_matches_grim_pre_attack_state_machine():
    source = (ROOT / "mod/src/main/java/dev/judas/bridge/PacketOrderProbe.java").read_text()

    assert "private boolean sentAnimation = false;" in source
    assert "private boolean sentSlotSwitch = false;" in source
    assert "msg instanceof C09PacketHeldItemChange" in source
    assert "resetOrderState();" in source
    assert "if (!isAsync(msg)) resetOrderState();" in source
    assert "msg instanceof C00PacketKeepAlive" in source
    assert '"BAD pre-attack"' in source
    assert '"OK A->I"' in source
    assert "sawFlyingThisTick" not in source
    assert '"BAD post-combat"' not in source


def test_packet_order_probe_logs_attack_order_to_minecraft_dir():
    source = (ROOT / "mod/src/main/java/dev/judas/bridge/PacketOrderProbe.java").read_text()

    assert 'new File(mc.mcDataDir, "judas-packet-order.log")' in source
    assert "private void logAttack()" in source
    assert 'writer.println("tick="' in source
    assert "seq=\" + lastSequence" in source
    assert "order=\" + lastOrder" in source
    assert "ok=\" + lastOk" in source


def test_packet_order_probe_logs_when_installed():
    source = (ROOT / "mod/src/main/java/dev/judas/bridge/PacketOrderProbe.java").read_text()

    assert "logProbeInstalled();" in source
    assert '" probe=installed"' in source
    assert source.index('new File(mc.mcDataDir, "judas-packet-order.log")') < source.index(
        "install(mc);"
    )


def test_mod_installs_packet_order_probe_on_client_tick():
    source = (ROOT / "mod/src/main/java/dev/judas/bridge/JudasMod.java").read_text()

    assert "private final PacketOrderProbe packetProbe = new PacketOrderProbe();" in source
    assert "packetProbe.beginClientTick(mc);" in source
    assert "packetProbe.statusLine()" in source


def test_packet_order_guard_injects_missing_animation_before_attack():
    source = (ROOT / "mod/src/main/java/dev/judas/bridge/PacketOrderGuard.java").read_text()

    assert "extends ChannelDuplexHandler" in source
    assert "msg instanceof C02PacketUseEntity" in source
    assert "C02PacketUseEntity.Action.ATTACK" in source
    assert "if (isAttack(msg) && !sentAnimation)" in source
    assert "ctx.write(new C0APacketAnimation(), ctx.newPromise());" in source
    assert "ctx.write(msg, promise);" in source
    assert "if (!isAsync(msg)) resetOrderState();" in source


def test_mod_installs_guard_after_probe_so_probe_sees_corrected_order():
    source = (ROOT / "mod/src/main/java/dev/judas/bridge/JudasMod.java").read_text()
    tick_body = _method_body(source, "public void onClientTick(TickEvent.ClientTickEvent event)")

    assert "private final PacketOrderGuard packetGuard = new PacketOrderGuard();" in source
    assert "packetGuard.beginClientTick(mc);" in tick_body
    assert tick_body.index("packetProbe.beginClientTick(mc);") < tick_body.index(
        "packetGuard.beginClientTick(mc);"
    )


def test_packet_order_guard_reports_when_it_had_to_inject_animation():
    guard_source = (ROOT / "mod/src/main/java/dev/judas/bridge/PacketOrderGuard.java").read_text()
    mod_source = (ROOT / "mod/src/main/java/dev/judas/bridge/JudasMod.java").read_text()

    assert "private volatile int injectedAnimations = 0;" in guard_source
    assert "injectedAnimations++;" in guard_source
    assert "public String statusLine()" in guard_source
    assert 'guard="' in guard_source
    assert 'injected=' in guard_source
    assert "packetGuard.statusLine()" in mod_source


def test_packet_order_guard_logs_injections_to_minecraft_dir():
    source = (ROOT / "mod/src/main/java/dev/judas/bridge/PacketOrderGuard.java").read_text()

    assert 'new File(mc.mcDataDir, "judas-packet-order.log")' in source
    assert "private void logInjection()" in source
    assert "logInjection();" in source
    assert 'writer.println("tick="' in source
    assert '" guard=injected"' in source
    assert "total=\" + injectedAnimations" in source


def test_packet_order_probe_marks_other_non_async_resets_in_sequence():
    source = (ROOT / "mod/src/main/java/dev/judas/bridge/PacketOrderProbe.java").read_text()

    assert 'append("R");' in source
    assert "msg instanceof C03PacketPlayer" in source
    assert "if (!isAsync(msg))" in source


def test_mod_manual_toggle_pauses_gui_focus_but_does_not_disarm_for_match_gate():
    source = (ROOT / "mod/src/main/java/dev/judas/bridge/JudasMod.java").read_text()
    tick_body = _method_body(source, "public void onClientTick(TickEvent.ClientTickEvent event)")
    keys_body = _method_body(source, "private void handleKeys(Minecraft mc)")
    auto_stop_body = _method_body(source, "private boolean shouldAutoStop(Minecraft mc)")
    manual_pause_body = _method_body(source, "private String manualPauseReason(Minecraft mc)")

    assert "private boolean manualOverride = false;" in source
    assert "private String pauseReason = \"none\";" in source
    assert "private boolean shouldAutoStop(Minecraft mc)" in source
    assert "if (manualOverride) return false;" in auto_stop_body
    assert "manualOverride = true;" in keys_body
    assert "requireBoxingMatch = false;" in keys_body
    assert 'logStatus("manual_toggle", "armed");' in keys_body
    assert "mc.theWorld == null" in source
    assert "mc.currentScreen != null" in manual_pause_body
    assert "!mc.inGameHasFocus" in manual_pause_body
    assert "pauseBot(mc, \"no_world\")" in tick_body
    assert "if (enabled && manualOverride && shouldPauseManual(mc))" in tick_body
    assert "pauseBot(mc, manualPauseReason(mc));" in tick_body
    assert "if (enabled && !manualOverride && shouldAutoStop(mc))" in tick_body
    assert tick_body.index("if (enabled && !manualOverride && shouldAutoStop(mc))") < tick_body.index(
        "packetProbe.beginClientTick(mc);"
    )
    handle_idx = tick_body.index("handleKeys(mc);")
    assert handle_idx < tick_body.index(
        "if (enabled && manualOverride && shouldPauseManual(mc))",
        handle_idx,
    )



def test_native_input_exposes_unavailable_reason_for_overlay():
    native = (ROOT / "mod/src/main/java/dev/judas/bridge/NativeInput.java").read_text()
    applier = (ROOT / "mod/src/main/java/dev/judas/bridge/ActionApplier.java").read_text()
    mod = (ROOT / "mod/src/main/java/dev/judas/bridge/JudasMod.java").read_text()

    assert "private String status = \"ok\";" in native
    assert "String status()" in native
    assert "fail(\"pointer=null\")" in native
    assert "fail(t.getClass().getSimpleName())" in native
    assert "public String inputModeLabel()" in applier
    assert "souris OS indispo(" in applier
    assert "ni.status()" in applier
    assert "applier.inputModeLabel()" in mod
    assert "applier.isNative() ?" not in mod

def test_mod_overlay_exposes_native_aim_diagnostics():
    applier = (ROOT / "mod/src/main/java/dev/judas/bridge/ActionApplier.java").read_text()
    mod = (ROOT / "mod/src/main/java/dev/judas/bridge/JudasMod.java").read_text()
    body = _method_body(applier, "public String aimStatusLine(Minecraft mc)")

    assert "targetYawErrorDeg(p, target)" in body
    assert "targetPitchErrorDeg(p, target)" in body
    assert "String.format(Locale.ROOT" in body
    assert "aim OS err=" in body
    assert "cmd=%.1f/%.1f" in body
    assert "os=%d/%d" in body
    assert "sent=%.1f/%.1f" in body
    assert "pend=%.1f/%.1f" in body
    assert "step=%.3f/%.3f" in body
    assert "stall=%d" in body
    assert "lastNativeAppliedYaw = appliedYaw;" in applier
    assert "lastNativeIssuedYaw = sentYaw;" in applier
    assert "event.left.add(applier.aimStatusLine(mc));" in mod

def test_native_aim_diagnostics_are_logged_to_minecraft_dir():
    source = (ROOT / "mod/src/main/java/dev/judas/bridge/ActionApplier.java").read_text()
    apply_body = _method_body(source, "public void apply(Minecraft mc, JsonObject a)")
    log_body = _method_body(source, "private void logNativeAim(Minecraft mc,")

    assert "import java.io.File;" in source
    assert "import java.io.FileWriter;" in source
    assert "import java.io.PrintWriter;" in source
    assert 'new File(mc.mcDataDir, "judas-aim-os.log")' in source
    assert 'event=start player=' in source
    assert "event=no_target" in source
    for field in (
        "yawErr=%.3f", "pitchErr=%.3f", "cmdYaw=%.3f",
        "cmdPitch=%.3f", "dx=%d", "dy=%d", "sentYaw=%.3f",
        "sentPitch=%.3f", "appliedYaw=%.3f", "appliedPitch=%.3f",
        "pendingYaw=%.3f", "pendingPitch=%.3f", "stepYaw=%.5f",
        "stepPitch=%.5f", "stall=%d", "yawSign=%d", "pitchSign=%d",
    ):
        assert field in log_body
    assert "logNativeAimNoTarget(mc, p, aim[0], aim[1]);" in apply_body
    assert "logNativeAim(mc, p, target, aim[0], aim[1]);" in apply_body
    assert apply_body.index("applyLookNative") < apply_body.index("logNativeAim")
    assert "pitchSign, yawReversalSettleTicks, pitchReversalSettleTicks" in log_body
    assert "aimLogTicks = 0;" in source

def test_stabilized_pitch_targets_body_band_instead_of_space_above_player():
    source = (ROOT / "mod/src/main/java/dev/judas/bridge/ActionApplier.java").read_text()
    body = _method_body(source, "private static double targetAimY(EntityPlayer target, double eyeY)")

    assert "MathHelper.clamp_double(eyeY," in body
    assert "target.posY + AIM_MIN_Y" in body
    assert "target.posY + AIM_MAX_Y" in body
    aim_min = _java_double_const(source, "AIM_MIN_Y")
    aim_max = _java_double_const(source, "AIM_MAX_Y")
    assert 0.15 <= aim_min <= 0.40
    assert 1.20 <= aim_max <= 1.50
    assert aim_max < 1.62
def test_input_modes_stabilize_aim_before_applying_rotation():
    source = (ROOT / "mod/src/main/java/dev/judas/bridge/ActionApplier.java").read_text()
    apply_body = _method_body(source, "public void apply(Minecraft mc, JsonObject a)")
    direct_body = _method_body(source, "private void applyLookDirect(EntityPlayerSP p, double dyaw, double dpitch, double step)")
    native_body = _method_body(source, "private void applyLookNative(EntityPlayerSP p, double dyaw, double dpitch,")

    assert "EntityPlayer target = collector.getTarget();" in apply_body
    assert 'boolean freeLook = jsonBool(a, "retreat_turn");' in apply_body
    assert 'boolean forceSprint = jsonBool(a, "force_sprint") || freeLook;' in apply_body
    assert "double[] aim = freeLook" in apply_body
    assert "? new double[] { dyaw, dpitch }" in apply_body
    assert ": stabilizeDirectAim(p, target, dyaw, dpitch);" in apply_body
    assert "recordAimDebug(aim[0], aim[1]);" in apply_body
    assert "applyLookNative(p, aim[0], aim[1], step" in apply_body
    assert "applyLookDirect(p, aim[0], aim[1], step);" in apply_body
    assert apply_body.index("double[] aim = freeLook") < apply_body.index("recordAimDebug") < apply_body.index("if (useNative)")
    assert "targetPitchErrorDeg" in source
    assert "targetYawErrorDeg" in source
    assert "targetAimY" in source
    assert "AIM_MAX_Y" in source
    assert "private static final double AIM_LOCK_BLEND = 1.0;" in source
    assert "private static final double AIM_FINE_LOCK_BLEND = 1.0;" in source
    assert "double limit = 180.0;" in source
    assert "stabilizeAxis(dyaw, yawErr, limit, 0.05)" in source
    assert "stabilizeAxis(dpitch, pitchErr, limit, 0.05)" in source
    assert "p.setSprinting(sprint && fwd > 0 && (forceSprint || !p.isCollidedHorizontally));" in source
    assert "return clampMag(err, limit);" in source
    assert "if (Math.abs(dpitch) < 1.0e-6) resPitch = 0.0;" in direct_body
    assert "appliedPitch - pitchStep" in direct_body
    assert "resPitch = Math.abs" in direct_body
    assert "if (Math.abs(dyaw) < 1.0e-6) resYaw = 0.0;" in native_body
    assert "if (resYaw * dyaw < 0.0) resYaw = 0.0;" in native_body
    assert "wantYaw = nativeYawDemand(resYaw, dyaw, appliedYaw, step);" in native_body
    assert "double sendYaw = holdYaw ? 0.0 : nativeIssue(wantYaw, nativePendingYaw, dyaw, step);" in native_body
    assert "sendYaw / step" in native_body
    assert "nativeYawStep = step;" in native_body
    assert "nativePendingYaw = updateNativePending(nativePendingYaw, appliedYaw, step);" in native_body
    assert "pendingDir(nativePendingYaw, step)" in native_body
    assert "NATIVE_MAX_DEG_PER_TICK" in source
    assert "NATIVE_MAX_COUNTS_PER_TICK" in source
    assert "NATIVE_SIGN_FLIP_TICKS" in source
    assert "NATIVE_CMD_FLIP_GUARD_TICKS" in source
    assert "NATIVE_PENDING_STALE_TICKS" in source
    assert "NATIVE_REVERSAL_SETTLE_TICKS" in source
    assert "NATIVE_STEP_EMA" not in source
    assert "updatePendingStaleTicks" in native_body
    assert "yawPendingStaleTicks = 0;" in native_body
    assert "pitchPendingStaleTicks = 0;" in native_body
    assert "yawCmdGuardTicks <= 0 && diverged(intendedYawDir, dyaw, appliedYaw, step)" in native_body
    assert "yawReversalSettleTicks = NATIVE_REVERSAL_SETTLE_TICKS;" in native_body
    assert "intendedYawDir = 0;" in native_body
    assert "pitchReversalSettleTicks = NATIVE_REVERSAL_SETTLE_TICKS;" in native_body
    assert "boolean holdYaw = yawReversalSettleTicks > 0;" in native_body
    assert "double sendYaw = holdYaw ? 0.0 : nativeIssue" in native_body
    assert "clampCounts((int) Math.round" in native_body
    assert "clampMag(resYaw + dyaw - appliedYaw, 180.0)" not in native_body

def _java_double_const(source: str, name: str) -> float:
    m = re.search(rf"private static final double {name} = ([0-9.]+);", source)
    assert m, f"missing Java double constant {name}"
    return float(m.group(1))


def _java_int_const(source: str, name: str) -> int:
    m = re.search(rf"private static final int {name} = ([0-9]+);", source)
    assert m, f"missing Java int constant {name}"
    return int(m.group(1))


def test_native_mouse_os_command_stays_bounded_under_os_latency_contract():
    source = (ROOT / "mod/src/main/java/dev/judas/bridge/ActionApplier.java").read_text()
    native_max = _java_double_const(source, "NATIVE_MAX_DEG_PER_TICK")
    max_counts = _java_int_const(source, "NATIVE_MAX_COUNTS_PER_TICK")
    demand_gain = _java_double_const(source, "NATIVE_DEMAND_GAIN")
    yaw_demand_gain = _java_double_const(source, "NATIVE_YAW_DEMAND_GAIN")
    yaw_fine_one_to_one = _java_double_const(source, "NATIVE_YAW_FINE_ONE_TO_ONE_DEG")
    reversal_settle_ticks = _java_int_const(source, "NATIVE_REVERSAL_SETTLE_TICKS")
    fine_one_to_one = _java_double_const(source, "NATIVE_FINE_ONE_TO_ONE_DEG")

    assert native_max >= 260.0
    assert max_counts >= 30000
    assert 0.55 <= demand_gain <= 0.65
    assert 25.0 <= fine_one_to_one <= 35.0
    assert 0.80 <= yaw_demand_gain <= 0.95
    assert yaw_fine_one_to_one >= 75.0
    assert reversal_settle_ticks == 0
    assert "double demand = residual + cmd;" in source
    assert "nativeYawDemand(resYaw, dyaw, appliedYaw, step)" in source
    assert "NATIVE_YAW_FINE_ONE_TO_ONE_DEG" in source
    assert "double fineCap = Math.min(Math.abs(cmd), fineOneToOneDeg);" in source
    assert "Math.max(fineCap, dampedCap)" in source
    assert "double demand = residual + cmd - applied;" not in source
    assert "double issue = demand - pending;" not in source
    assert "resYaw = holdYaw ? 0.0 : wantYaw - sentYaw;" in source
    assert "resPitch = holdPitch ? 0.0 : wantPitch - sentPitch;" in source
    assert "double sentYaw = yawSign * dx * step;" in source
    assert "updateNativeStep(nativeYawStep" not in source

    from tools.native_aim_sim import load_constants, simulate_suite, verdict_text

    report = simulate_suite(load_constants(ROOT / "mod/src/main/java/dev/judas/bridge/ActionApplier.java"))
    assert report.field_stable is True
    assert report.failures <= 2100
    assert report.worst_growth_over_limit <= 100.0
    assert report.worst_final_error <= 110.0
    assert report.nonzero_reversal_sends == 0
    assert "worst_cmd_err=" in verdict_text(report)
