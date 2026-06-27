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

import java.io.File;
import java.io.FileWriter;
import java.io.PrintWriter;
import java.util.Locale;

/**
 * Applique l'action reçue de serve/ : rotations + touches + clic d'attaque.
 *
 * Deux voies pour souris/clic (bascule 'O' en jeu, nativeInput) :
 *  - NATIVE (optionnel) : vrais events OS via Robot (souris relative en boucle
 *    fermée + vrai clic gauche) -> un anticheat qui hook l'input CLIENT voit de
 *    vraies entrées, pas une écriture de rotationYaw. Repli auto sur DIRECT si
 *    Robot est indisponible ou reste sans effet (souris non captée).
 *  - DIRECT : écriture de rotationYaw quantifiée en pas ENTIERS de souris (cube
 *    de sensibilité 1.8.9) -> équivalent vraie-souris pour un anticheat côté
 *    serveur (mêmes paquets, signature GCD identique à une vraie souris).
 *
 * dyaw/dpitch arrivent déjà humanisés (réaction + lissage + tremblement) du
 * daemon. Le mouvement reste piloté par KeyBinding (le serveur voit le
 * déplacement, pas les events clavier ; mouvement physiquement fidèle).
 */
public final class ActionApplier {

    private final StateCollector collector;
    // résidu de rotation (pas entiers de souris) : report pour ne pas dériver.
    private double resYaw = 0.0;
    private double resPitch = 0.0;

    // Entrée OS réelle (souris/clic) : OFF par défaut -> mode DIRECT = visée du
    // modèle appliquée INSTANTANÉMENT (byte-perfect, quantifiée sous le degré),
    // la plus fidèle au modèle. 'O' en jeu active le natif (paquets vanilla pour
    // l'anticheat, au prix d'un retard sub-tick).
    public boolean nativeInput = false;
    private final NativeInput ni = new NativeInput();
    private double lastYaw = 0.0;
    private double lastPitch = 0.0;
    private boolean primed = false;
    private int nativeStallTicks = 0;   // détecte une souris OS sans effet
    private int yawSign = 1;
    private int pitchSign = 1;          // base = invertMouse, auto-corrigée si divergence
    private int yawDiverge = 0;
    private int pitchDiverge = 0;
    private int intendedYawDir = 0;
    private int intendedPitchDir = 0;
    private int lastYawCmdDir = 0;
    private int lastPitchCmdDir = 0;
    private int yawCmdGuardTicks = 0;
    private int pitchCmdGuardTicks = 0;
    private int yawPendingStaleTicks = 0;
    private int pitchPendingStaleTicks = 0;
    private int yawReversalSettleTicks = 0;
    private int pitchReversalSettleTicks = 0;
    private int lastNativeDx = 0;
    private int lastNativeDy = 0;
    private double nativeYawStep = 0.0;
    private double nativePitchStep = 0.0;
    private double nativePendingYaw = 0.0;
    private double nativePendingPitch = 0.0;
    private double lastAimDyaw = 0.0;
    private double lastAimDpitch = 0.0;
    private double lastNativeAppliedYaw = 0.0;
    private double lastNativeAppliedPitch = 0.0;
    private double lastNativeIssuedYaw = 0.0;
    private double lastNativeIssuedPitch = 0.0;
    private File aimLogFile;
    private int aimLogTicks = 0;
    private boolean pendingAttack = false;   // attaque native differee -> Phase.END
    private boolean visualAttackQueued = false;
    private boolean visualUseQueued = false;
    private int visualAttackTicks = 0;
    private int visualUseTicks = 0;

    private static final double AIM_MIN_Y = 0.25;
    private static final double AIM_MAX_Y = 1.45;
    private static final double AIM_LOCK_BLEND = 1.0;
    private static final double AIM_FINE_LOCK_DEG = 8.0;
    private static final double AIM_FINE_LOCK_BLEND = 1.0;
    private static final double NATIVE_MAX_DEG_PER_TICK = 260.0;
    private static final int NATIVE_MAX_COUNTS_PER_TICK = 30000;
    private static final int NATIVE_SIGN_FLIP_TICKS = 1;
    private static final int NATIVE_CMD_FLIP_GUARD_TICKS = 3;
    private static final int NATIVE_PENDING_STALE_TICKS = 4;
    private static final int NATIVE_REVERSAL_SETTLE_TICKS = 0;
    private static final double NATIVE_DEMAND_GAIN = 0.60;
    private static final double NATIVE_FINE_ONE_TO_ONE_DEG = 30.0;
    private static final double NATIVE_YAW_DEMAND_GAIN = 0.90;
    private static final double NATIVE_YAW_FINE_ONE_TO_ONE_DEG = 180.0;
    private static final int KEYSTROKE_MOUSE_HOLD_TICKS = 2;

    public ActionApplier(StateCollector collector) {
        this.collector = collector;
    }

    public void setNative(boolean n) {
        nativeInput = n;
        primed = false;
        resYaw = resPitch = 0.0;
        nativeStallTicks = 0;
        lastNativeDx = lastNativeDy = 0;
        lastYawCmdDir = lastPitchCmdDir = 0;
        yawCmdGuardTicks = pitchCmdGuardTicks = 0;
        yawPendingStaleTicks = pitchPendingStaleTicks = 0;
        yawReversalSettleTicks = pitchReversalSettleTicks = 0;
        nativeYawStep = nativePitchStep = 0.0;
        nativePendingYaw = nativePendingPitch = 0.0;
        lastNativeAppliedYaw = lastNativeAppliedPitch = 0.0;
        lastNativeIssuedYaw = lastNativeIssuedPitch = 0.0;
        aimLogTicks = 0;
        pendingAttack = false;
    }

    public boolean isNative() {
        return nativeInput && ni.available();
    }

    public String inputModeLabel() {
        if (nativeInput && ni.available()) {
            return "\u00A7asouris OS";
        }
        if (nativeInput) {
            return "\u00A7csouris OS indispo(" + ni.status() + ")";
        }
        return "\u00A7edirecte";
    }

    public String aimStatusLine(Minecraft mc) {
        EntityPlayerSP p = mc.thePlayer;
        EntityPlayer target = collector.getTarget();
        if (p == null || target == null) {
            return "\u00A77 aim: \u00A78no target";
        }
        double yawErr = targetYawErrorDeg(p, target);
        double pitchErr = targetPitchErrorDeg(p, target);
        if (!isNative()) {
            return String.format(Locale.ROOT,
                    "\u00A77 aim direct err=\u00A7f%.1f/%.1f\u00A77 cmd=%.1f/%.1f",
                    yawErr, pitchErr, lastAimDyaw, lastAimDpitch);
        }
        return String.format(Locale.ROOT,
                "\u00A77 aim OS err=\u00A7f%.1f/%.1f\u00A77 cmd=%.1f/%.1f os=%d/%d sent=%.1f/%.1f app=%.1f/%.1f pend=%.1f/%.1f step=%.3f/%.3f stall=%d settle=%d/%d",
                yawErr, pitchErr, lastAimDyaw, lastAimDpitch,
                lastNativeDx, lastNativeDy, lastNativeIssuedYaw,
                lastNativeIssuedPitch, lastNativeAppliedYaw,
                lastNativeAppliedPitch, nativePendingYaw, nativePendingPitch,
                nativeYawStep, nativePitchStep, nativeStallTicks,
                yawReversalSettleTicks, pitchReversalSettleTicks);
    }

    private void recordAimDebug(double dyaw, double dpitch) {
        lastAimDyaw = dyaw;
        lastAimDpitch = dpitch;
    }
    public void apply(Minecraft mc, JsonObject a) {
        EntityPlayerSP p = mc.thePlayer;
        if (p == null) return;

        final double dyaw = a.get("dyaw").getAsDouble();
        final double dpitch = a.get("dpitch").getAsDouble();
        final double step = mouseStep(mc);
        final boolean useNative = nativeInput && ni.available();

        // --- rotations ---
        EntityPlayer target = collector.getTarget();
        boolean freeLook = jsonBool(a, "retreat_turn");
        double[] aim = freeLook
                ? new double[] { dyaw, dpitch }
                : stabilizeDirectAim(p, target, dyaw, dpitch);
        recordAimDebug(aim[0], aim[1]);
        if (useNative) {
            applyLookNative(p, aim[0], aim[1], step, mc.gameSettings.invertMouse);
            if (target == null) {
                logNativeAimNoTarget(mc, p, aim[0], aim[1]);
            } else {
                logNativeAim(mc, p, target, aim[0], aim[1]);
            }
        } else {
            applyLookDirect(p, aim[0], aim[1], step);
        }

        // --- touches (mouvement) ---
        int fwd = a.get("forward").getAsInt();
        int strafe = a.get("strafe").getAsInt();
        boolean jump = a.get("jump").getAsBoolean();
        boolean sprint = a.get("sprint").getAsBoolean();
        boolean forceSprint = jsonBool(a, "force_sprint") || freeLook;
        key(mc.gameSettings.keyBindForward, fwd > 0);
        key(mc.gameSettings.keyBindBack, fwd < 0);
        key(mc.gameSettings.keyBindLeft, strafe > 0);    // strafe +1 = gauche (vanilla)
        key(mc.gameSettings.keyBindRight, strafe < 0);
        key(mc.gameSettings.keyBindJump, jump);
        key(mc.gameSettings.keyBindSprint, sprint);

        // --- sprint : toggle-sprint identique au sim (sim_ref/match.py:109-115) ---
        p.setSprinting(sprint && fwd > 0 && (forceSprint || !p.isCollidedHorizontally));
        if (jump) collector.onJump(p.onGround);

        // --- attaque ---
        // DIRECT doit rester dans la phase input du tick, comme un clic vanilla.
        // Seul NATIVE est differe: Robot injecte un vrai clic OS que Minecraft
        // traitera au tick suivant via son propre pipeline d'input.
        boolean attack = a.get("attack").getAsBoolean();
        if (attack && collector.isTargetFriend()) attack = false;
        boolean useItem = jsonBool(a, "use") || jsonBool(a, "rightClick");
        queueKeystrokeMouseVisual(attack, useItem);
        if (attack && !useNative) {
            collector.onOwnAttack();
            boolean landed = doAttackDirect(mc, p);
            if (landed && sprint && fwd > 0 && !forceSprint) {
                key(mc.gameSettings.keyBindForward, false);
                key(mc.gameSettings.keyBindSprint, false);
                p.setSprinting(false);
            }
        }
        pendingAttack = useNative && attack;
    }

    public void clearKeystrokeMouseVisualsBeforeInput(Minecraft mc) {
        key(mc.gameSettings.keyBindAttack, false);
        key(mc.gameSettings.keyBindUseItem, false);
    }

    public void applyKeystrokeMouseVisuals(Minecraft mc) {
        if (visualAttackQueued) visualAttackTicks = KEYSTROKE_MOUSE_HOLD_TICKS;
        if (visualUseQueued) visualUseTicks = KEYSTROKE_MOUSE_HOLD_TICKS;
        visualAttackQueued = false;
        visualUseQueued = false;

        boolean showAttack = visualAttackTicks > 0;
        boolean showUse = visualUseTicks > 0;
        key(mc.gameSettings.keyBindAttack, showAttack);
        key(mc.gameSettings.keyBindUseItem, showUse);
        if (visualAttackTicks > 0) visualAttackTicks--;
        if (visualUseTicks > 0) visualUseTicks--;
    }

    private void queueKeystrokeMouseVisual(boolean attack, boolean useItem) {
        visualAttackQueued = visualAttackQueued || attack;
        visualUseQueued = visualUseQueued || useItem;
    }

    private static boolean jsonBool(JsonObject obj, String key) {
        return obj.has(key) && obj.get(key).getAsBoolean();
    }

    public void cancelPendingAttack() {
        pendingAttack = false;
        visualAttackQueued = false;
        visualAttackTicks = 0;
    }

    /** Clic OS memorise, a appeler en Phase.END uniquement pour la voie NATIVE. */
    public void applyDeferredAttack(Minecraft mc) {
        if (!pendingAttack) return;
        pendingAttack = false;
        EntityPlayerSP p = mc.thePlayer;
        if (p == null) return;
        if (collector.isTargetFriend()) return;
        if (nativeInput && ni.available()) {
            collector.onOwnAttack();
            ni.leftClick();          // vrai clic OS -> MC attaque la cible au viseur
        }
    }

    /** DIRECT : ecriture de rotation quantifiee en pas souris (residu reporte). */
    private void logNativeAim(Minecraft mc, EntityPlayerSP p, EntityPlayer target,
                              double cmdYaw, double cmdPitch) {
        if (target == null || !ensureAimLogFile(mc)) return;
        double yawErr = targetYawErrorDeg(p, target);
        double pitchErr = targetPitchErrorDeg(p, target);
        try (PrintWriter writer = new PrintWriter(new FileWriter(aimLogFile, true))) {
            logNativeAimStart(writer, mc, p);
            writer.println(String.format(Locale.ROOT,
                    "tick=%d player=%s yawErr=%.3f pitchErr=%.3f cmdYaw=%.3f cmdPitch=%.3f dx=%d dy=%d sentYaw=%.3f sentPitch=%.3f appliedYaw=%.3f appliedPitch=%.3f pendingYaw=%.3f pendingPitch=%.3f stepYaw=%.5f stepPitch=%.5f stall=%d yawSign=%d pitchSign=%d settleYaw=%d settlePitch=%d",
                    aimLogTicks++, p.getName(), yawErr, pitchErr, cmdYaw, cmdPitch,
                    lastNativeDx, lastNativeDy, lastNativeIssuedYaw,
                    lastNativeIssuedPitch, lastNativeAppliedYaw,
                    lastNativeAppliedPitch, nativePendingYaw, nativePendingPitch,
                    nativeYawStep, nativePitchStep, nativeStallTicks, yawSign,
                    pitchSign, yawReversalSettleTicks, pitchReversalSettleTicks));
        } catch (Throwable ignored) {
        }
    }

    private void logNativeAimNoTarget(Minecraft mc, EntityPlayerSP p,
                                      double cmdYaw, double cmdPitch) {
        if (!ensureAimLogFile(mc)) return;
        try (PrintWriter writer = new PrintWriter(new FileWriter(aimLogFile, true))) {
            logNativeAimStart(writer, mc, p);
            writer.println(String.format(Locale.ROOT,
                    "event=no_target tick=%d player=%s cmdYaw=%.3f cmdPitch=%.3f dx=%d dy=%d sentYaw=%.3f sentPitch=%.3f appliedYaw=%.3f appliedPitch=%.3f stall=%d",
                    aimLogTicks++, p.getName(), cmdYaw, cmdPitch, lastNativeDx,
                    lastNativeDy, lastNativeIssuedYaw, lastNativeIssuedPitch,
                    lastNativeAppliedYaw, lastNativeAppliedPitch, nativeStallTicks));
        } catch (Throwable ignored) {
        }
    }

    private boolean ensureAimLogFile(Minecraft mc) {
        if (mc.mcDataDir == null) return false;
        if (aimLogFile == null) {
            aimLogFile = new File(mc.mcDataDir, "judas-aim-os.log");
        }
        return true;
    }

    private void logNativeAimStart(PrintWriter writer, Minecraft mc, EntityPlayerSP p) {
        if (aimLogTicks == 0) {
            writer.println("event=start player=" + p.getName()
                    + " sensitivity=" + mc.gameSettings.mouseSensitivity
                    + " invert=" + mc.gameSettings.invertMouse);
        }
    }
    private void applyLookDirect(EntityPlayerSP p, double dyaw, double dpitch, double step) {
        if (Math.abs(dyaw) < 1.0e-6) resYaw = 0.0;
        if (Math.abs(dpitch) < 1.0e-6) resPitch = 0.0;
        double wantYaw = resYaw + dyaw;
        long dx = Math.round(wantYaw / step);
        double yawStep = dx * step;
        resYaw = wantYaw - yawStep;
        p.rotationYaw += (float) yawStep;

        double wantPitch = resPitch + dpitch;
        long dy = Math.round(wantPitch / step);
        double pitchStep = dy * step;
        float oldPitch = p.rotationPitch;
        float newPitch = MathHelper.clamp_float(
                oldPitch + (float) pitchStep, -90.0F, 90.0F);
        p.rotationPitch = newPitch;
        double appliedPitch = newPitch - oldPitch;
        // Si le clamp vertical bloque une partie du mouvement, ne garde pas un
        // résidu impossible qui pousserait encore vers le ciel/sol au tick suivant.
        resPitch = Math.abs(appliedPitch - pitchStep) > 1.0e-6
                ? 0.0 : wantPitch - pitchStep;
    }

    private static double[] stabilizeDirectAim(EntityPlayerSP p, EntityPlayer target,
                                               double dyaw, double dpitch) {
        if (target == null) return new double[] { dyaw, dpitch };
        double yawErr = targetYawErrorDeg(p, target);
        double pitchErr = targetPitchErrorDeg(p, target);
        double limit = 180.0;
        return new double[] {
                stabilizeAxis(dyaw, yawErr, limit, 0.05),
                stabilizeAxis(dpitch, pitchErr, limit, 0.05)
        };
    }

    private static double stabilizeAxis(double cmd, double err, double limit, double deadband) {
        double absErr = Math.abs(err);
        if (absErr <= deadband) return 0.0;
        return clampMag(err, limit);
    }

    private static double targetYawErrorDeg(EntityPlayerSP p, EntityPlayer target) {
        double dx = target.posX - p.posX;
        double dz = target.posZ - p.posZ;
        double yawTo = Math.toDegrees(Math.atan2(-dx, dz));
        return wrapDeg(yawTo - p.rotationYaw);
    }

    private static double targetPitchErrorDeg(EntityPlayerSP p, EntityPlayer target) {
        double dx = target.posX - p.posX;
        double dz = target.posZ - p.posZ;
        double distH = Math.sqrt(dx * dx + dz * dz);
        if (distH <= 1.0e-9) return 0.0;
        double eyeY = p.posY + p.getEyeHeight();
        double aimY = targetAimY(target, eyeY);
        double pitchTo = -Math.toDegrees(Math.atan2(aimY - eyeY, distH));
        return pitchTo - p.rotationPitch;
    }

    private static double targetAimY(EntityPlayer target, double eyeY) {
        return MathHelper.clamp_double(eyeY,
                target.posY + AIM_MIN_Y, target.posY + AIM_MAX_Y);
    }

    /** NATIVE : vrais moves souris OS en boucle fermee. On demande la meme
     *  rotation quantifiee que DIRECT/le visualiseur ce tick, puis on mesure au
     *  tick suivant ce que Minecraft a applique. Bascule en direct si les moves
     *  OS restent sans effet (souris non captee par le jeu). */
    private void applyLookNative(EntityPlayerSP p, double dyaw, double dpitch,
                                 double step, boolean invertMouse) {
        if (!primed) {
            lastYaw = p.rotationYaw;
            lastPitch = p.rotationPitch;
            yawSign = 1;
            pitchSign = invertMouse ? -1 : 1;   // base dérivée ; auto-corrigée plus bas
            yawDiverge = pitchDiverge = 0;
            intendedYawDir = intendedPitchDir = 0;
            lastYawCmdDir = lastPitchCmdDir = 0;
            yawCmdGuardTicks = pitchCmdGuardTicks = 0;
            yawPendingStaleTicks = pitchPendingStaleTicks = 0;
            yawReversalSettleTicks = pitchReversalSettleTicks = 0;
            lastNativeDx = lastNativeDy = 0;
            nativeYawStep = step;
            nativePitchStep = step;
            nativePendingYaw = nativePendingPitch = 0.0;
            primed = true;
        }
        double appliedYaw = wrapDeg(p.rotationYaw - lastYaw);
        double appliedPitch = p.rotationPitch - lastPitch;
        lastNativeAppliedYaw = appliedYaw;
        lastNativeAppliedPitch = appliedPitch;
        // Command-fidelity mode: the OS path must ask for the same quantized
        // rotation as DIRECT/the visualizer. Actual OS gain is observed below
        // through appliedYaw/appliedPitch, but it must not rescale this tick's
        // command or the bot starts drifting away from the visualizer.
        nativeYawStep = step;
        nativePitchStep = step;
        nativePendingYaw = updateNativePending(nativePendingYaw, appliedYaw, step);
        nativePendingPitch = updateNativePending(nativePendingPitch, appliedPitch, step);
        yawPendingStaleTicks = updatePendingStaleTicks(
                yawPendingStaleTicks, nativePendingYaw, appliedYaw, dyaw, step);
        pitchPendingStaleTicks = updatePendingStaleTicks(
                pitchPendingStaleTicks, nativePendingPitch, appliedPitch, dpitch, step);
        if (yawPendingStaleTicks >= NATIVE_PENDING_STALE_TICKS) {
            nativePendingYaw = 0.0;
            yawPendingStaleTicks = 0;
        }
        if (pitchPendingStaleTicks >= NATIVE_PENDING_STALE_TICKS) {
            nativePendingPitch = 0.0;
            pitchPendingStaleTicks = 0;
        }

        if (Math.abs(dyaw) < 1.0e-6) resYaw = 0.0;
        if (Math.abs(dpitch) < 1.0e-6) resPitch = 0.0;
        if (resYaw * dyaw < 0.0) resYaw = 0.0;
        if (resPitch * dpitch < 0.0) resPitch = 0.0;

        int yawCmdDir = commandDir(dyaw);
        if (yawCmdDir != 0 && lastYawCmdDir != 0 && yawCmdDir != lastYawCmdDir) {
            yawCmdGuardTicks = NATIVE_CMD_FLIP_GUARD_TICKS;
            yawDiverge = 0;
            nativePendingYaw = 0.0;
            yawPendingStaleTicks = 0;
            yawReversalSettleTicks = NATIVE_REVERSAL_SETTLE_TICKS;
        } else if (yawCmdGuardTicks > 0) {
            yawCmdGuardTicks--;
        }
        if (yawCmdDir != 0) lastYawCmdDir = yawCmdDir;

        int pitchCmdDir = commandDir(dpitch);
        if (pitchCmdDir != 0 && lastPitchCmdDir != 0 && pitchCmdDir != lastPitchCmdDir) {
            pitchCmdGuardTicks = NATIVE_CMD_FLIP_GUARD_TICKS;
            pitchDiverge = 0;
            nativePendingPitch = 0.0;
            pitchPendingStaleTicks = 0;
            pitchReversalSettleTicks = NATIVE_REVERSAL_SETTLE_TICKS;
        } else if (pitchCmdGuardTicks > 0) {
            pitchCmdGuardTicks--;
        }
        if (pitchCmdDir != 0) lastPitchCmdDir = pitchCmdDir;

        // stall : on commande mais rien ne bouge -> Robot inopérant, repli direct
        boolean commanded = Math.abs(dyaw) > step || Math.abs(dpitch) > step;
        if (commanded && Math.abs(appliedYaw) < 1.0e-4 && Math.abs(appliedPitch) < 1.0e-4) {
            if (++nativeStallTicks > 20) { setNative(false); return; }
        } else {
            nativeStallTicks = 0;
        }

        // auto-correction de signe : si la rotation est partie à l'OPPOSÉ de
        // l'intention du tick précédent (invertMouse/quirk OS), on bascule le
        // signe d'injection (converge en quelques ticks, ne peut pas osciller).
        if (yawCmdGuardTicks <= 0 && diverged(intendedYawDir, dyaw, appliedYaw, step)) {
            if (++yawDiverge >= NATIVE_SIGN_FLIP_TICKS) {
                yawSign = -yawSign;
                yawDiverge = 0;
                nativePendingYaw = 0.0;
                yawPendingStaleTicks = 0;
                yawCmdGuardTicks = NATIVE_CMD_FLIP_GUARD_TICKS;
                yawReversalSettleTicks = NATIVE_REVERSAL_SETTLE_TICKS;
                intendedYawDir = 0;
            }
        } else if (Math.abs(appliedYaw) > 1.0e-6) {
            yawDiverge = 0;
        }
        if (pitchCmdGuardTicks <= 0 && diverged(intendedPitchDir, dpitch, appliedPitch, step)) {
            if (++pitchDiverge >= NATIVE_SIGN_FLIP_TICKS) {
                pitchSign = -pitchSign;
                pitchDiverge = 0;
                nativePendingPitch = 0.0;
                pitchPendingStaleTicks = 0;
                pitchCmdGuardTicks = NATIVE_CMD_FLIP_GUARD_TICKS;
                pitchReversalSettleTicks = NATIVE_REVERSAL_SETTLE_TICKS;
                intendedPitchDir = 0;
            }
        } else if (Math.abs(appliedPitch) > 1.0e-6) {
            pitchDiverge = 0;
        }

        boolean holdYaw = yawReversalSettleTicks > 0;
        boolean holdPitch = pitchReversalSettleTicks > 0;
        double wantYaw;
        double wantPitch;
        if (holdYaw) {
            yawReversalSettleTicks--;
            wantYaw = 0.0;
            resYaw = 0.0;
        } else {
            wantYaw = nativeYawDemand(resYaw, dyaw, appliedYaw, step);
        }
        if (holdPitch) {
            pitchReversalSettleTicks--;
            wantPitch = 0.0;
            resPitch = 0.0;
        } else {
            wantPitch = nativeDemand(resPitch, dpitch, appliedPitch, step);
        }
        double sendYaw = holdYaw ? 0.0 : nativeIssue(wantYaw, nativePendingYaw, dyaw, step);
        double sendPitch = holdPitch ? 0.0 : nativeIssue(wantPitch, nativePendingPitch, dpitch, step);
        int dx = clampCounts((int) Math.round(yawSign * sendYaw / step));
        int dy = clampCounts((int) Math.round(pitchSign * sendPitch / step));
        ni.mouseMoveRel(dx, dy);
        lastNativeDx = dx;
        lastNativeDy = dy;
        double sentYaw = yawSign * dx * step;
        double sentPitch = pitchSign * dy * step;
        lastNativeIssuedYaw = sentYaw;
        lastNativeIssuedPitch = sentPitch;
        resYaw = holdYaw ? 0.0 : wantYaw - sentYaw;
        resPitch = holdPitch ? 0.0 : wantPitch - sentPitch;
        nativePendingYaw = nativePendingYaw + sentYaw;
        nativePendingPitch = nativePendingPitch + sentPitch;
        // mémorise le sens visé, même si on attend encore un ancien move OS.
        intendedYawDir = dx != 0 ? (sentYaw > 0 ? 1 : -1)
                : pendingDir(nativePendingYaw, step);
        intendedPitchDir = dy != 0 ? (sentPitch > 0 ? 1 : -1)
                : pendingDir(nativePendingPitch, step);
        lastYaw = p.rotationYaw;
        lastPitch = p.rotationPitch;
    }

    private static double nativeDemand(double residual, double cmd,
                                       double applied, double step) {
        return nativeDemandCapped(residual, cmd, applied, step,
                NATIVE_DEMAND_GAIN, NATIVE_FINE_ONE_TO_ONE_DEG);
    }

    private static double nativeYawDemand(double residual, double cmd,
                                          double applied, double step) {
        return nativeDemandCapped(residual, cmd, applied, step,
                NATIVE_YAW_DEMAND_GAIN, NATIVE_YAW_FINE_ONE_TO_ONE_DEG);
    }

    private static double nativeDemandCapped(double residual, double cmd,
                                             double applied, double step,
                                             double demandGain,
                                             double fineOneToOneDeg) {
        if (Math.abs(cmd) < 1.0e-6) return 0.0;
        double demand = residual + cmd;
        if (demand * cmd < 0.0) return 0.0;
        double fineCap = Math.min(Math.abs(cmd), fineOneToOneDeg);
        double dampedCap = Math.abs(cmd) * demandGain + step;
        double cap = Math.max(step, Math.min(NATIVE_MAX_DEG_PER_TICK,
                Math.max(fineCap, dampedCap)));
        return clampMag(demand, cap);
    }

    private static double nativeIssue(double demand, double pending,
                                      double cmd, double step) {
        if (Math.abs(cmd) < 1.0e-6) return 0.0;
        if (demand * cmd < 0.0) return 0.0;
        return clampMag(demand, Math.max(step, NATIVE_MAX_DEG_PER_TICK));
    }

    private static double updateNativePending(double pending, double applied,
                                              double step) {
        if (Math.abs(pending) <= step * 0.5) return 0.0;
        if (pending * applied > 0.0 && Math.abs(applied) > step * 0.5) {
            return 0.0;
        }
        return clampMag(pending, NATIVE_MAX_DEG_PER_TICK);
    }

    private static int updatePendingStaleTicks(int staleTicks, double pending,
                                               double applied, double cmd,
                                               double step) {
        if (Math.abs(pending) <= step * 0.5 || Math.abs(cmd) <= step
                || Math.abs(applied) > step * 0.5) {
            return 0;
        }
        return staleTicks + 1;
    }

    private static int pendingDir(double pending, double step) {
        return Math.abs(pending) > step * 0.5 ? (pending > 0.0 ? 1 : -1) : 0;
    }

    private static int clampCounts(int counts) {
        if (counts > NATIVE_MAX_COUNTS_PER_TICK) return NATIVE_MAX_COUNTS_PER_TICK;
        if (counts < -NATIVE_MAX_COUNTS_PER_TICK) return -NATIVE_MAX_COUNTS_PER_TICK;
        return counts;
    }
    private static int commandDir(double cmd) {
        return Math.abs(cmd) > 1.0e-6 ? (cmd > 0.0 ? 1 : -1) : 0;
    }

    private static boolean diverged(int intendedDir, double cmd,
                                    double applied, double step) {
        return intendedDir != 0
                && cmd * intendedDir > 0.0
                && Math.abs(applied) > Math.max(1.0e-6, step * 0.5)
                && applied * intendedDir < 0.0;
    }


    private boolean doAttackDirect(Minecraft mc, EntityPlayerSP p) {
        EntityPlayer target = collector.getTarget();
        p.swingItem();
        if (target != null && rayHitsTarget(p, target)) {
            mc.playerController.attackEntity(p, target);
            return true;
        }
        return false;
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
        primed = false;
        resYaw = resPitch = 0.0;
        nativeStallTicks = 0;
        lastNativeDx = lastNativeDy = 0;
        lastYawCmdDir = lastPitchCmdDir = 0;
        yawCmdGuardTicks = pitchCmdGuardTicks = 0;
        yawPendingStaleTicks = pitchPendingStaleTicks = 0;
        yawReversalSettleTicks = pitchReversalSettleTicks = 0;
        nativeYawStep = nativePitchStep = 0.0;
        nativePendingYaw = nativePendingPitch = 0.0;
        lastNativeAppliedYaw = lastNativeAppliedPitch = 0.0;
        lastNativeIssuedYaw = lastNativeIssuedPitch = 0.0;
        aimLogTicks = 0;
        pendingAttack = false;
        visualAttackQueued = false;
        visualUseQueued = false;
        visualAttackTicks = 0;
        visualUseTicks = 0;
        key(mc.gameSettings.keyBindAttack, false);
        key(mc.gameSettings.keyBindUseItem, false);
    }

    private static void key(KeyBinding kb, boolean down) {
        KeyBinding.setKeyBindState(kb.getKeyCode(), down);
    }

    /** Degrés de rotation par count de souris (formule 1.8.9 :
     *  f = sensibilité*0.6 + 0.2 ; pas = f^3 * 8 * 0.15). Toujours > 0. */
    private static double mouseStep(Minecraft mc) {
        double f = mc.gameSettings.mouseSensitivity * 0.6 + 0.2;
        return f * f * f * 8.0 * 0.15;
    }

    private static double wrapDeg(double v) {
        v %= 360.0;
        if (v >= 180.0) v -= 360.0;
        if (v < -180.0) v += 360.0;
        return v;
    }

    private static double clampMag(double v, double m) {
        return v < -m ? -m : (v > m ? m : v);
    }
}
