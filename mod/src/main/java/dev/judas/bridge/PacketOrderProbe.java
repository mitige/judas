package dev.judas.bridge;

import io.netty.channel.Channel;
import io.netty.channel.ChannelDuplexHandler;
import io.netty.channel.ChannelHandlerContext;
import io.netty.channel.ChannelPipeline;
import io.netty.channel.ChannelPromise;
import net.minecraft.client.Minecraft;
import net.minecraft.network.play.client.C00PacketKeepAlive;
import net.minecraft.network.play.client.C02PacketUseEntity;
import net.minecraft.network.play.client.C03PacketPlayer;
import net.minecraft.network.play.client.C09PacketHeldItemChange;
import net.minecraft.network.play.client.C0APacketAnimation;
import net.minecraft.network.play.client.C19PacketResourcePackStatus;

import java.io.File;
import java.io.FileWriter;
import java.io.PrintWriter;

/**
 * Passive outbound packet observer for the packet-order HUD diagnostic.
 * It never mutates or cancels packets.
 */
public final class PacketOrderProbe {
    private static final String HANDLER_NAME = "judas_packet_order_probe";

    private Channel channel;
    private volatile String lastOrder = "n/a";
    private volatile String lastSequence = "";
    private volatile boolean lastOk = true;
    private volatile int lastAgeTicks = 999;
    private volatile long clientTick = 0;
    private volatile String playerName = "?";
    private File logFile;

    public void beginClientTick(Minecraft mc) {
        clientTick++;
        if (mc.thePlayer != null) playerName = mc.thePlayer.getName();
        if (logFile == null && mc.mcDataDir != null) {
            logFile = new File(mc.mcDataDir, "judas-packet-order.log");
        }
        install(mc);
        if (lastAgeTicks < 999) lastAgeTicks++;
    }

    public String statusLine() {
        String color = lastOk ? "\u00A7a" : "\u00A7c";
        String seq = lastSequence.length() == 0 ? "n/a" : lastSequence;
        return "\u00A77 pkt=\u00A7f" + seq + " " + color + lastOrder
                + "\u00A78[" + lastAgeTicks + "t]";
    }

    private void install(Minecraft mc) {
        try {
            if (mc.thePlayer == null || mc.thePlayer.sendQueue == null) return;
            Channel ch = mc.thePlayer.sendQueue.getNetworkManager().channel();
            if (ch == null || !ch.isOpen()) return;
            if (ch == channel && ch.pipeline().get(HANDLER_NAME) != null) return;
            ChannelPipeline pipeline = ch.pipeline();
            if (pipeline.get(HANDLER_NAME) == null) {
                pipeline.addLast(HANDLER_NAME, new ProbeHandler());
                logProbeInstalled();
            }
            channel = ch;
        } catch (Throwable ignored) {
            channel = null;
        }
    }

    private void logProbeInstalled() {
        if (logFile == null) return;
        try (PrintWriter writer = new PrintWriter(new FileWriter(logFile, true))) {
            writer.println("tick=" + clientTick
                    + " player=" + playerName
                    + " probe=installed");
        } catch (Throwable ignored) {
        }
    }

    private final class ProbeHandler extends ChannelDuplexHandler {
        private boolean sentAnimation = false;
        private boolean sentSlotSwitch = false;
        private final StringBuilder recent = new StringBuilder();

        @Override
        public void write(ChannelHandlerContext ctx, Object msg,
                          ChannelPromise promise) throws Exception {
            try {
                observe(msg);
            } catch (Throwable ignored) {
            }
            ctx.write(msg, promise);
        }

        private void observe(Object msg) {
            if (msg instanceof C0APacketAnimation) {
                append("A");
                sentAnimation = true;
                sentSlotSwitch = false;
                return;
            }
            if (msg instanceof C02PacketUseEntity) {
                C02PacketUseEntity packet = (C02PacketUseEntity) msg;
                if (packet.getAction() == C02PacketUseEntity.Action.ATTACK) {
                    append("I");
                    lastOk = sentAnimation;
                    lastOrder = sentAnimation ? "OK A->I" : "BAD pre-attack";
                    lastSequence = recent.toString();
                    lastAgeTicks = 0;
                    logAttack();
                    resetOrderState();
                    return;
                }
            }
            if (msg instanceof C09PacketHeldItemChange && !sentSlotSwitch) {
                append("S");
                sentSlotSwitch = true;
                return;
            }
            if (msg instanceof C03PacketPlayer) {
                append("F");
            } else if (!isAsync(msg)) {
                append("R");
            }
            if (!isAsync(msg)) resetOrderState();
        }

        private void resetOrderState() {
            sentAnimation = false;
            sentSlotSwitch = false;
        }

        private boolean isAsync(Object msg) {
            return msg instanceof C00PacketKeepAlive
                    || msg instanceof C19PacketResourcePackStatus;
        }

        private void logAttack() {
            if (logFile == null) return;
            try (PrintWriter writer = new PrintWriter(new FileWriter(logFile, true))) {
                writer.println("tick=" + clientTick
                        + " player=" + playerName
                        + " seq=" + lastSequence
                        + " order=" + lastOrder
                        + " ok=" + lastOk);
            } catch (Throwable ignored) {
            }
        }

        private void append(String token) {
            if (recent.length() > 0) recent.append('>');
            recent.append(token);
            while (recent.length() > 32) {
                int cut = recent.indexOf(">");
                if (cut < 0) {
                    recent.setLength(0);
                    return;
                }
                recent.delete(0, cut + 1);
            }
        }
    }
}
