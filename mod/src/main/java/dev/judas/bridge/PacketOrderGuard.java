package dev.judas.bridge;

import io.netty.channel.Channel;
import io.netty.channel.ChannelDuplexHandler;
import io.netty.channel.ChannelHandlerContext;
import io.netty.channel.ChannelPipeline;
import io.netty.channel.ChannelPromise;
import net.minecraft.client.Minecraft;
import net.minecraft.network.play.client.C00PacketKeepAlive;
import net.minecraft.network.play.client.C02PacketUseEntity;
import net.minecraft.network.play.client.C09PacketHeldItemChange;
import net.minecraft.network.play.client.C0APacketAnimation;
import net.minecraft.network.play.client.C19PacketResourcePackStatus;

import java.io.File;
import java.io.FileWriter;
import java.io.PrintWriter;

/**
 * Outbound guard for the Grim 1.8 PacketOrderB invariant:
 * ANIMATION must immediately arm ATTACK, with one slot switch tolerated.
 */
public final class PacketOrderGuard {
    private static final String HANDLER_NAME = "judas_packet_order_guard";

    private Channel channel;
    private volatile int injectedAnimations = 0;
    private volatile int ticksSinceInjection = 999;
    private volatile long clientTick = 0;
    private volatile String playerName = "?";
    private File logFile;

    public void beginClientTick(Minecraft mc) {
        install(mc);
        clientTick++;
        if (mc.thePlayer != null) playerName = mc.thePlayer.getName();
        if (logFile == null && mc.mcDataDir != null) {
            logFile = new File(mc.mcDataDir, "judas-packet-order.log");
        }
        if (ticksSinceInjection < 999) ticksSinceInjection++;
    }

    public String statusLine() {
        String color = injectedAnimations == 0 ? "\u00A7a" : "\u00A7e";
        return "\u00A77 guard=" + color + (injectedAnimations == 0 ? "natural" : "injected")
                + "\u00A77 injected=\u00A7f" + injectedAnimations
                + "\u00A78[" + ticksSinceInjection + "t]";
    }

    private void install(Minecraft mc) {
        try {
            if (mc.thePlayer == null || mc.thePlayer.sendQueue == null) return;
            Channel ch = mc.thePlayer.sendQueue.getNetworkManager().channel();
            if (ch == null || !ch.isOpen()) return;
            if (ch == channel && ch.pipeline().get(HANDLER_NAME) != null) return;
            ChannelPipeline pipeline = ch.pipeline();
            if (pipeline.get(HANDLER_NAME) == null) {
                pipeline.addLast(HANDLER_NAME, new GuardHandler());
            }
            channel = ch;
        } catch (Throwable ignored) {
            channel = null;
        }
    }

    private final class GuardHandler extends ChannelDuplexHandler {
        private boolean sentAnimation = false;
        private boolean sentSlotSwitch = false;

        @Override
        public void write(ChannelHandlerContext ctx, Object msg,
                          ChannelPromise promise) throws Exception {
            if (msg instanceof C0APacketAnimation) {
                sentAnimation = true;
                sentSlotSwitch = false;
                ctx.write(msg, promise);
                return;
            }
            if (msg instanceof C09PacketHeldItemChange && !sentSlotSwitch) {
                sentSlotSwitch = true;
                ctx.write(msg, promise);
                return;
            }
            if (isAttack(msg) && !sentAnimation) {
                ctx.write(new C0APacketAnimation(), ctx.newPromise());
                sentAnimation = true;
                injectedAnimations++;
                ticksSinceInjection = 0;
                logInjection();
            }

            ctx.write(msg, promise);
            if (!isAsync(msg)) resetOrderState();
        }

        private boolean isAttack(Object msg) {
            return msg instanceof C02PacketUseEntity
                    && ((C02PacketUseEntity) msg).getAction() == C02PacketUseEntity.Action.ATTACK;
        }

        private void resetOrderState() {
            sentAnimation = false;
            sentSlotSwitch = false;
        }

        private boolean isAsync(Object msg) {
            return msg instanceof C00PacketKeepAlive
                    || msg instanceof C19PacketResourcePackStatus;
        }

        private void logInjection() {
            if (logFile == null) return;
            try (PrintWriter writer = new PrintWriter(new FileWriter(logFile, true))) {
                writer.println("tick=" + clientTick
                        + " player=" + playerName
                        + " guard=injected"
                        + " total=" + injectedAnimations);
            } catch (Throwable ignored) {
            }
        }
    }
}
