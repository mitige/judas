package dev.judas.bridge;

import java.io.DataInputStream;
import java.io.IOException;
import java.io.InputStream;
import java.io.OutputStream;
import java.net.Socket;
import java.nio.charset.StandardCharsets;
import java.security.SecureRandom;
import java.util.Base64;
import java.util.concurrent.atomic.AtomicBoolean;
import java.util.concurrent.atomic.AtomicReference;

/**
 * Client WebSocket minimal (RFC 6455, frames texte) pour localhost.
 * Zéro dépendance externe — utilisable tel quel dans un mod 1.8.9.
 *
 * Usage : connect() ; sendText(json) chaque tick ; pollLatest() pour la
 * dernière action reçue (non bloquant). Reconnexion gérée par l'appelant.
 */
public final class WsClient {

    private final String host;
    private final int port;
    private final String path;

    private Socket socket;
    private OutputStream out;
    private Thread readerThread;
    private final AtomicBoolean open = new AtomicBoolean(false);
    private final AtomicReference<String> latest = new AtomicReference<String>(null);
    private final SecureRandom rng = new SecureRandom();

    public WsClient(String host, int port, String path) {
        this.host = host;
        this.port = port;
        this.path = path;
    }

    public boolean isOpen() {
        return open.get();
    }

    /** Dernier message texte reçu (et l'efface), ou null. */
    public String pollLatest() {
        return latest.getAndSet(null);
    }

    public synchronized void connect() throws IOException {
        close();
        socket = new Socket(host, port);
        socket.setTcpNoDelay(true);
        out = socket.getOutputStream();
        InputStream in = socket.getInputStream();

        byte[] keyBytes = new byte[16];
        rng.nextBytes(keyBytes);
        String key = Base64.getEncoder().encodeToString(keyBytes);
        String req = "GET " + path + " HTTP/1.1\r\n"
                + "Host: " + host + ":" + port + "\r\n"
                + "Upgrade: websocket\r\n"
                + "Connection: Upgrade\r\n"
                + "Sec-WebSocket-Key: " + key + "\r\n"
                + "Sec-WebSocket-Version: 13\r\n\r\n";
        out.write(req.getBytes(StandardCharsets.US_ASCII));
        out.flush();

        // lit la réponse HTTP jusqu'à la ligne vide
        StringBuilder sb = new StringBuilder();
        int c, state = 0;
        while ((c = in.read()) != -1) {
            sb.append((char) c);
            state = (c == '\r' && (state == 0 || state == 2)) ? state + 1
                  : (c == '\n' && (state == 1 || state == 3)) ? state + 1 : 0;
            if (state == 4) break;
            if (sb.length() > 8192) throw new IOException("handshake trop long");
        }
        if (!sb.toString().contains("101")) {
            throw new IOException("handshake WebSocket refuse: " + sb.toString().split("\r\n")[0]);
        }

        open.set(true);
        final DataInputStream din = new DataInputStream(in);
        readerThread = new Thread(new Runnable() {
            @Override
            public void run() {
                readLoop(din);
            }
        }, "judas-ws-reader");
        readerThread.setDaemon(true);
        readerThread.start();
    }

    private void readLoop(DataInputStream in) {
        try {
            while (open.get()) {
                int b0 = in.readUnsignedByte();
                int opcode = b0 & 0x0F;
                int b1 = in.readUnsignedByte();
                boolean masked = (b1 & 0x80) != 0;
                long len = b1 & 0x7F;
                if (len == 126) len = in.readUnsignedShort();
                else if (len == 127) len = in.readLong();
                byte[] mask = new byte[4];
                if (masked) in.readFully(mask);
                byte[] payload = new byte[(int) len];
                in.readFully(payload);
                if (masked) {
                    for (int i = 0; i < payload.length; i++) payload[i] ^= mask[i & 3];
                }
                if (opcode == 0x1) {                       // texte
                    latest.set(new String(payload, StandardCharsets.UTF_8));
                } else if (opcode == 0x9) {                // ping -> pong
                    sendFrame(0xA, payload);
                } else if (opcode == 0x8) {                // close
                    break;
                }
            }
        } catch (IOException ignored) {
            // socket fermée / serveur arrêté
        } finally {
            open.set(false);
        }
    }

    public void sendText(String text) {
        if (!open.get()) return;
        try {
            sendFrame(0x1, text.getBytes(StandardCharsets.UTF_8));
        } catch (IOException e) {
            open.set(false);
        }
    }

    private synchronized void sendFrame(int opcode, byte[] payload) throws IOException {
        if (out == null) return;
        int len = payload.length;
        byte[] header;
        if (len < 126) {
            header = new byte[]{(byte) (0x80 | opcode), (byte) (0x80 | len)};
        } else if (len < 65536) {
            header = new byte[]{(byte) (0x80 | opcode), (byte) (0x80 | 126),
                    (byte) (len >> 8), (byte) len};
        } else {
            throw new IOException("frame trop grande");
        }
        byte[] mask = new byte[4];
        rng.nextBytes(mask);
        byte[] body = new byte[len];
        for (int i = 0; i < len; i++) body[i] = (byte) (payload[i] ^ mask[i & 3]);
        out.write(header);
        out.write(mask);
        out.write(body);
        out.flush();
    }

    public synchronized void close() {
        open.set(false);
        try {
            if (socket != null) socket.close();
        } catch (IOException ignored) {
        }
        socket = null;
        out = null;
    }
}
