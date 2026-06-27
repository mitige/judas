package dev.judas.bridge;

import com.google.gson.JsonElement;
import com.google.gson.JsonObject;
import net.minecraft.block.state.IBlockState;
import net.minecraft.client.Minecraft;
import net.minecraft.client.entity.EntityPlayerSP;
import net.minecraft.util.BlockPos;
import net.minecraft.util.MathHelper;
import net.minecraft.world.World;

final class AutoJump {
    private boolean enabled = false;
    private int holdTicks = 0;
    private String status = "off";

    private static final int HOLD_TICKS = 4;
    private static final double SIDE_SAMPLE = 0.28;
    private static final double[] FORWARD_SAMPLES = new double[] {0.55, 0.85};
    private static final double[] SIDE_SAMPLES = new double[] {-SIDE_SAMPLE, 0.0, SIDE_SAMPLE};

    void updateFromAction(JsonObject action) {
        if (action == null || !action.has("auto_jump")) return;
        JsonElement raw = action.get("auto_jump");
        if (raw.isJsonPrimitive()) {
            enabled = raw.getAsBoolean();
            return;
        }
        if (!raw.isJsonObject()) return;
        JsonObject cfg = raw.getAsJsonObject();
        if (cfg.has("enabled")) enabled = cfg.get("enabled").getAsBoolean();
    }

    boolean applyToAction(Minecraft mc, JsonObject action) {
        if (action == null) return false;
        if (!enabled) {
            holdTicks = 0;
            status = "off";
            return false;
        }
        EntityPlayerSP p = mc.thePlayer;
        if (p == null || mc.theWorld == null) {
            holdTicks = 0;
            status = "no_player";
            return false;
        }
        if (action.has("jump") && action.get("jump").getAsBoolean()) {
            holdTicks = 0;
            status = "model";
            return false;
        }
        if (shouldJump(mc.theWorld, p, action)) {
            holdTicks = HOLD_TICKS;
        }
        if (holdTicks <= 0) {
            status = "ready";
            return false;
        }
        action.addProperty("jump", true);
        status = "jump:" + holdTicks;
        holdTicks--;
        return true;
    }

    String statusLine() {
        return "\u00A77 autojump=" + (enabled ? "\u00A7aon" : "\u00A78off")
                + "\u00A77 " + status;
    }

    void cancel() {
        holdTicks = 0;
        status = enabled ? "cancelled" : "off";
    }

    private boolean shouldJump(World world, EntityPlayerSP p, JsonObject action) {
        if (!p.onGround || p.isInWater() || p.isInLava() || p.isOnLadder()) return false;
        int fwd = jsonInt(action, "forward");
        int strafe = jsonInt(action, "strafe");
        if (fwd == 0 && strafe == 0 && !p.isCollidedHorizontally) return false;

        double yaw = Math.toRadians(p.rotationYaw);
        double forwardX = -Math.sin(yaw);
        double forwardZ = Math.cos(yaw);
        double leftX = Math.cos(yaw);
        double leftZ = Math.sin(yaw);
        double moveX = fwd * forwardX + strafe * leftX * 0.75;
        double moveZ = fwd * forwardZ + strafe * leftZ * 0.75;
        double len = Math.sqrt(moveX * moveX + moveZ * moveZ);
        if (len < 1.0e-6) {
            moveX = forwardX;
            moveZ = forwardZ;
        } else {
            moveX /= len;
            moveZ /= len;
        }
        return hasStepObstacle(world, p, moveX, moveZ);
    }

    private boolean hasStepObstacle(World world, EntityPlayerSP p, double moveX, double moveZ) {
        double sideX = moveZ;
        double sideZ = -moveX;
        int footY = MathHelper.floor_double(p.getEntityBoundingBox().minY + 0.01);
        for (double ahead : FORWARD_SAMPLES) {
            for (double side : SIDE_SAMPLES) {
                double x = p.posX + moveX * ahead + sideX * side;
                double z = p.posZ + moveZ * ahead + sideZ * side;
                BlockPos pos = new BlockPos(
                        MathHelper.floor_double(x),
                        footY,
                        MathHelper.floor_double(z));
                if (isStepObstacle(world, pos)) return true;
            }
        }
        return false;
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

    private static int jsonInt(JsonObject action, String key) {
        return action.has(key) ? action.get(key).getAsInt() : 0;
    }
}
