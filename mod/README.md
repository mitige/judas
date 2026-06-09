# judas-bridge — mod Forge 1.8.9

Pont état/actions entre le client Minecraft 1.8.9 et le daemon Judas
(`python -m serve.daemon`, WebSocket `ws://127.0.0.1:8765/live`).

## Build (PC Windows, JDK 8+ installé)

```bat
:: une seule fois : installer Gradle 8 (ex: scoop install gradle / choco install gradle)
:: puis générer le wrapper :
cd mod
gradle wrapper --gradle-version 8.5

:: builds suivants :
gradlew build
```

Le jar est produit dans `build/libs/judas-bridge-0.1.0.jar`.
Copier dans `.minecraft/mods` (profil Forge 1.8.9 — 11.15.1.2318).

Toolchain : Gradle 8 + essential architectury-loom (support moderne du 1.8.9),
mappings MCP `stable_22`. Premier build : ~5 min (décompilation du jeu).

Pour tester en développement : `gradlew runClient`.

## In-game

| Touche | Effet |
|---|---|
| `K` | Toggle bot ON/OFF (se connecte au daemon) |
| `L` | Kill-switch : tout couper immédiatement |
| `J` | Enregistrer une trace golden (340 ticks d'inputs scriptés) |

Les traces golden sont écrites dans `.minecraft/judas-traces/*.jsonl` et se
comparent à la physique de référence avec :

```bat
python tools/golden_compare.py .minecraft/judas-traces/trace-XXXX.jsonl
```

## Protocole (JSON texte, 1 message/tick)

mod -> daemon :
```json
{"t":"state","tick":123,
 "self":{"x":..,"y":..,"z":..,"vx":..,"vy":..,"vz":..,"yaw":..,"pitch":..,
          "onGround":true,"sprinting":true,"hurtTime":0,"hits":12},
 "target":{...}}
```

daemon -> mod :
```json
{"t":"action","dyaw":2.4,"dpitch":-0.7,"forward":1,"strafe":0,
 "jump":false,"sprint":true,"attack":true}
```

Notes :
- les rotations reçues sont déjà clampées par l'humanisation côté daemon ;
- le CPS est limité côté daemon, le mod exécute ;
- la vitesse de la cible est estimée par delta de position (le client ne
  connaît pas le motion des autres joueurs) ;
- les compteurs de hits sont heuristiques (front montant de `hurtTime`).
