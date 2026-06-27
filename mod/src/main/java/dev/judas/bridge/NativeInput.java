package dev.judas.bridge;

import java.awt.MouseInfo;
import java.awt.Point;
import java.awt.PointerInfo;
import java.awt.Robot;
import java.awt.event.InputEvent;

/**
 * Entree OS reelle via java.awt.Robot (aucune dependance native a shader) :
 * vrais deplacements de souris relatifs + vrais clics gauche, pour qu'un
 * anticheat qui hook l'input CLIENT (events souris) voie de vraies entrees et
 * pas une ecriture directe de rotationYaw.
 *
 * Souris relative : Robot ne fait que de l'absolu, donc on lit la position
 * courante (curseur verrouille au centre quand MC capture la souris) et on
 * deplace de (dx,dy) -> le delta brut est lu par LWJGL comme un vrai mouvement.
 * La boucle fermee cote ActionApplier corrige l'acceleration/precision Windows.
 *
 * Tout est garde par try/catch : a la moindre erreur (headless, securite,
 * pointeur nul) on bascule available()=false et l'appelant reprend la voie
 * directe (ecriture quantifiee), sans casser le bot.
 */
final class NativeInput {

    private Robot robot;
    private boolean ok;
    private String status = "ok";

    NativeInput() {
        try {
            robot = new Robot();
            robot.setAutoDelay(0);
            robot.setAutoWaitForIdle(false);
            ok = true;
            status = "ok";
        } catch (Throwable t) {
            fail(t);
        }
    }

    boolean available() {
        return ok;
    }

    String status() {
        return status;
    }

    /** Deplacement souris RELATIF (counts entiers) depuis la position courante. */
    void mouseMoveRel(int dx, int dy) {
        if (!ok || (dx == 0 && dy == 0)) return;
        try {
            PointerInfo pi = MouseInfo.getPointerInfo();
            if (pi == null) { fail("pointer=null"); return; }
            Point p = pi.getLocation();
            robot.mouseMove(p.x + dx, p.y + dy);
        } catch (Throwable t) {
            fail(t);
        }
    }

    /** Vrai clic gauche (press+release) - MC le traite comme une attaque au viseur. */
    void leftClick() {
        if (!ok) return;
        try {
            robot.mousePress(InputEvent.BUTTON1_DOWN_MASK);
            robot.mouseRelease(InputEvent.BUTTON1_DOWN_MASK);
        } catch (Throwable t) {
            fail(t);
        }
    }

    private void fail(Throwable t) {
        fail(t.getClass().getSimpleName());
    }

    private void fail(String reason) {
        ok = false;
        status = reason;
    }
}