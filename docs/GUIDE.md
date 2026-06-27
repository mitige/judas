# Entraîner un dieu du PvP — le guide

Tout ce qui suit a été payé en milliards de steps. Lisez-le avant de lancer
un long run.

## La philosophie

**Le jeu est le reward.** Hit +1, hurt −0.85, win ±10 (zéro-somme). Tout le
reste est de l'échafaudage qui doit s'effacer ou rester marginal :

- `reward_dist` (rapprochement) : annealé automatiquement par la rampe
  adaptative jusqu'à un plancher (`shaping_floor_frac`, 25 % par défaut) —
  une pression de fermeture permanente, pas un objectif.
- `reward_combo` (0.25/maillon, fenêtre 25 ticks, cap 5) : **zéro-somme** —
  le bonus du frappeur est retiré à la victime, qui apprend donc à s'échapper.
  Le premier hit d'une chaîne ne paie rien : le poke isolé n'est pas
  subventionné.
- `reward_sprint_hit` (0.35) : bonus **zéro-somme** sur les hits portés en
  sprint, pour rendre le sprint-reset/Z-tap plus rentable que le trade frontal.
- `reward_trade_penalty` (0.4) : malus appliqué aux deux agents quand ils
  touchent le même tick. Il est volontairement non zéro-somme pour casser
  l'équilibre dégénéré « 1 coup / 1 coup ».
- Le **timeout est une égalité** : mener au score puis fuir ne gagne jamais.
  La passivité ne peut pas être optimale, structurellement.

Si une métrique de shaping monte alors que `eval vs bot` stagne, c'est du
reward hacking : **réduisez le shaping, n'ajoutez pas de contre-shaping.**

## Les métriques (Dashboard)

Les seuils de ce tableau sont **encodés dans le Dashboard** : courbes et
valeurs restent calmes quand c'est sain, passent **ambre** (à surveiller) ou
**braise** (malsain) sinon (`app/src/health.js`). Un dashboard tout calme =
laissez tourner.

| Métrique | C'est quoi | Sain |
|---|---|---|
| **safe state** | verdict du garde anti-regression. `safe` = checkpoint validé ; `stop` = rollback ; `await_fresh_eval` = pas encore assez de preuve fresh. | `safe`, avec `safe ckpt=safe_latest.pt` ou source validée |
| **eval vs bot** | winrate vs le chase-bot à visée parfaite — **le juge absolu**. L'ELO self-play peut stagner pendant que le niveau réel monte : c'est cette courbe qui tranche. | 0 longtemps, puis décollage net (heures) |
| elo / population | ELO du meilleur membre ; « population » montre les 4 membres (★ = leader) | montée, écarts qui se font et se défont |
| eval vs past | winrate vs le dernier snapshot — un quasi-miroir | ~0.4–0.6 (parité bruitée) |
| league WR | vs les adversaires de league du rollout. Ce n'est pas le juge de niveau et ne doit pas remplacer `eval vs bot` ou `safe state`. | 0.6–0.9 ; chute = la league a rattrapé |
| hit rate /min | hits du learner par minute | croît puis se stabilise (45–60) |
| fresh combo12 / fresh sky | éval reset séparée du rollout courant, utilisée pour accepter/refuser `safe_latest.pt`. | combo12 > 0, sky proche de 0 |
| sprint hits | % de hits portés en sprint — le « Z-tap mesurable » | doit MONTER (0.3 → 0.7+) ; plat ≈ 0.1 = signal d'alarme architecture/env |
| combo hits | % de hits en chaîne (hors trades) | 0.3–0.5 en régime ; dépend aussi de la qualité de l'adversaire |
| engage | % de ticks à portée (<3.5 blocs) — le thermomètre d'agressivité | 0.2–0.5 |
| entropy | exploration restante (différentielle : peut être négative sans drame) | décroissance douce ; chute brutale = collapse |
| shaping / spawn gap | état de la rampe (curriculum) | spawn 2→6, shaping → plancher |

## Les pièges (vécus, chers)

1. **L'architecture n'est pas un détail de vitesse.** Le trunk MLP fait
   ×25 en débit mais **ne sait pas apprendre le combo** : 6,8 Mds de steps,
   33 évals consécutives à 0.00 contre le bot, sprint_hits mort. Le combo est
   une compétence *temporelle* (phase du knockback adverse, expiration des
   i-frames, rythme sprint-release) — il faut l'attention sur l'historique
   (transformer d96, le profil par défaut). Utilisez le MLP
   (`attention: false`) uniquement pour prototyper des rewards à 1M+ sps.
   **Règle : ~50 itérations sans aucun signe de vie de sprint_hits ni
   d'eval_bot ⇒ suspectez l'architecture ou l'environnement, pas la patience.**
2. **La taille d'arène est un paramètre de gameplay, pas de décor.** 45×45 =
   spawn éloignés, densité de combat effondrée, kiting infini, eval_bot gelé
   à 0. Le boxing se joue sur ~18×18. (Le spawn standard est borné à 8 blocs
   par garde-fou, mais l'arène géante reste un désert.)
3. **Un miroir entre agents égaux ne montre jamais de gros combos** — le
   défenseur de même force s'échappe ; c'est le zéro-somme qui fonctionne.
   Pour VOIR le niveau : latest vs un vieux snapshot, latest vs le bot, ou
   in-game contre des humains.
4. **Les lignées se protègent.** `best.pt` ne peut pas être écrasé par un
   eval_bot inférieur (protection codée), mais le pruning des checkpoints ne
   sépare pas les lignées d'un même nom de run : **changez de nom de run
   quand vous changez d'architecture.** Un seed entre architectures
   incompatibles est signalé bruyamment dans le log.
5. **Vérifiez le formulaire avant d'accuser la policy.** Spawn gap à 15 ?
   → champ arena. Combo absent d'un run lancé via l'app ? → les champs combo
   existent depuis la v2 du formulaire. Le payload de l'app EST la config.

## Le PBT en deux phrases

4 policies co-entraînées sur des tranches d'envs : elles s'affrontent
(cross-play), partagent la league et le bot, et toutes les 25 itérations le
bas du classement ELO copie le haut avec hyperparamètres perturbés (lr,
entropie, clip — bornés). L'annealing classique est désactivé : c'est la
sélection qui pilote. La tête de checkpoint reste le meilleur membre —
l'export et le déploiement ignorent le PBT.

## Tuner l'agressivité

| Levier | Effet | Défaut |
|---|---|---|
| `reward_hurt` (champ « hurt (−) ») | > −1.0 rend le trade EV+ → initiative | −0.85 |
| `reward_sprint_hit` | paie les hits portés en sprint | 0.35 |
| `reward_trade_penalty` | punit les trades hit+hit du même tick | 0.4 |
| `shaping_floor_frac` (« dist floor ») | plancher du shaping de rapprochement | 0.25 |
| `reward_combo` / window / cap | paie les chaînes (zéro-somme) | 0.25 / 25 / 5 |
| timeout = égalité | la fuite ne gagne pas | câblé |
| `kb_idle_mult` | <1 = profil CustomKB : un adversaire immobile prend moins de KB | 0.6 |

Sur-agression (feed mutuel, win rate qui s'effondre) → remonter `hurt (−)`
vers 0.95. Passivité → vérifier l'arène AVANT de toucher aux rewards.

## Reprendre, changer, mesurer

- **Resume** : cocher « resume latest » — poids, optimiseur, league, RNG,
  rampe et population reviennent. Les itérations non sauvegardées (< 25)
  sont perdues et les courbes sont automatiquement tronquées des doublons.
- **Changer la physique** : sim_ref d'abord (vérité), puis
  `sim/csrc/boxing_core.h`, puis `scripts\verify.bat` — l'équivalence à 1e-6
  est l'invariant du projet. Les deux implémentations doivent diverger de
  zéro, pas « presque ».
- **Mesurer un changement** : 12 itérations suffisent pour voir hit_rate,
  engage et combo bouger. Comparez à config égale par ailleurs, et jugez
  les runs longs sur eval_bot par heure murale — jamais sur les sps.

## Vérifier le PacketOrder serveur

Le check serveur `PacketOrderB pre-attack` attend l'ordre vanilla 1.8 :
`C0A Animation` puis `C02 UseEntity ATTACK` (`C09 HeldItemChange` est toléré
entre les deux). La voie **directe** du mod envoie donc `swingItem()` avant
`attackEntity()`, et une garde Netty injecte une animation si une autre voie
produit une attaque nue.

Fermez Minecraft avant un test sur serveur : Forge garde les jars chargés tant
que le client tourne. Reconstruisez ensuite le jar, copiez-le dans `mods/`,
désactivez les anciens `judas-bridge-*.jar`, puis remettez le log local à zéro :

```bat
scripts\prepare_packet_order_test.bat
```

Si le vieux jar Judas est verrouillé et que vous voulez fermer le client
automatiquement :

```bat
scripts\prepare_packet_order_test.bat -StopMinecraft
```

Le premier argument peut cibler un dossier `mods` temporaire :

```bat
scripts\prepare_packet_order_test.bat C:\chemin\vers\.minecraft\mods
```

Ensuite : lancez Minecraft, chargez le modèle dans la page **Live**, gardez
l'entrée en **directe** (`O` ne doit pas afficher souris OS), activez avec `K`,
puis faites quelques attaques sur le serveur qui vérifie l'ordre des paquets.
Pour suivre le verdict pendant que Minecraft tourne :

```bat
scripts\watch_packet_order.bat
```

Après le test :

```bat
scripts\check_packet_order.bat -Strict
```

Verdicts :

- `CLEAN` : attaques vues, aucun `BAD pre-attack`, aucune injection de garde.
- `GUARDED` : ordre final envoyé au serveur corrigé par la garde ; le serveur
  ne devrait pas voir `PacketOrderB`, mais il faut regarder le log/HUD pour
  savoir quelle voie a produit l'attaque sans swing naturel. En `-Strict`, ce
  verdict échoue : l'objectif du test serveur est `CLEAN`.
- `BAD` : au moins une attaque est partie sans animation avant elle. Ce cas
  doit reproduire le message serveur `PacketOrderB pre-attack`.
- `NO_ATTACKS` : le test n'a pas généré d'attaque réseau exploitable ; refaire
  avec une cible réellement dans le raycast.
## Verifier l'aim souris OS

La voie **souris OS** depend du focus Windows, de la sensibilite Minecraft et
de la capture souris par le client. Le mod ecrit donc un log local a chaque run
natif : `.minecraft\judas-aim-os.log`. Ce log mesure l'erreur de visee reelle
sur la cible, les counts souris envoyes, le mouvement applique et les stalls.

Avant un essai terrain, preparez le mod et repartez d'un log court. Ce script lance aussi `scripts\check_native_aim_sim.bat`; si cette simulation echoue, la jar n'est pas deployee :

```bat
scripts\prepare_aim_os_test.bat
```

Le build Gradle du mod est borne a 300 secondes par defaut. En cas de timeout
ou d'echec, les sorties sont dans `runs\build_mod\gradle_*.out.log` et
`runs\build_mod\gradle_*.err.log`. Pour un premier build lent, lancez
`scripts\build_mod.bat -BuildTimeoutSeconds 600`.

Si le vieux jar Judas est verrouille et que vous voulez fermer le client
automatiquement :

```bat
scripts\prepare_aim_os_test.bat -StopMinecraft
```

Le premier argument peut cibler un dossier `mods` temporaire :

```bat
scripts\prepare_aim_os_test.bat C:\chemin\vers\.minecraft\mods
```

Pour le flux terrain complet en une commande :

```bat
scripts\field_test_aim_os.bat
```

Pour une machine fragile, vous pouvez borner plus court la compilation du mod :

```bat
scripts\field_test_aim_os.bat -BuildTimeoutSeconds 180
```

Si `scripts\check_mod_deploy.bat` affiche deja `OK_DEPLOY` et que vous voulez
produire les logs terrain sans relancer Gradle, utilisez le jar Judas deja
deploye :

```bat
scripts\field_test_aim_os.bat -UseDeployedMod
```

Raccourci equivalent :

```bat
scripts\field_test_aim_os_quick.bat
```

Ce raccourci exige que Minecraft soit deja lance : s'il ne detecte pas le
client, il s'arrete avant de demarrer le daemon live et affiche
`minecraft_not_running`. Cette verification arrive avant la fermeture des
anciennes instances Judas app/viz/training, donc un quick proof lance trop tot
ne ferme pas le visualiseur pour rien.

Ce mode verifie quand meme que la jar active dans `mods` est identique au build
local et writable avant de reset les logs et de demarrer le live.

Avant de lancer Minecraft, le preflight passif verifie export safe, jar active,
processus Judas/port live, statut Minecraft, puis resume les logs terrain sans
rien demarrer :

```bat
scripts\check_field_preflight.bat
```

Le script attend si la jar est verrouillee, deploie quand Minecraft est ferme,
ferme d'abord les anciennes instances Judas app/viz/training/combo, verifie
`OK_DEPLOY`, remet a zero `.minecraft\judas-aim-os.log`, lance `judas_live`
avec le modele combo safe, verifie le WebSocket live, remet a zero le log des
actions live apres ce precheck, puis surveille le log aim jusqu'a `PRECISE` ou
echec clair. Passez `-KeepOtherJudasProcesses` uniquement si vous voulez garder
ces processus ouverts pendant la preuve, ou `-KeepAimLog` pour conserver
l'ancien log aim. La sortie complete est aussi capturee dans
`runs\field_proof\field_*.log` sauf avec `-NoProofLog`; utilisez `-ProofLog`
pour choisir un chemin precis. Les chemins `-PacketLog`, `-MinecraftLog` et
`-PacketSession` sont relayes au reset packet-order, au check strict et au
status final. Quand l'aim est precise, il verifie aussi :

- `runs\judas-live-actions.log` avec `scripts\check_live_actions.bat -Strict`
  pour refuser back/tap-back/jump, exiger un minimum de strafe boxing, et
  refuser la visee ciel sur les actions Minecraft reelles. Ce garde exige par
  defaut le modele `combo_god_leaderboard10_combo12-safe_latest` : un ancien
  countertap/directpad ne peut plus faire passer la preuve terrain de cette IA.
  Il calcule aussi `attack_cps` depuis les ticks `attack=true` et echoue au
  dessus de `max_attack_cps=10.0`, y compris pendant un counter-hit urgent.
  Il exige aussi `min_strafe_frac=0.50`, donc une IA qui fonce droit et ne
  strafe que ponctuellement echoue. Les 20 premiers samples live ont leur
  propre garde `min_opener_strafe_frac=0.75`, pour refuser un depart tout droit
  suivi d'un strafe tardif.
  Le meme garde compte aussi `strafe_flips` et `strafe_hold_avg` pour refuser
  le spam gauche/droite tremblant, y compris les bursts d'un tick separes par
  du neutre.
- `.minecraft\judas-packet-order.log` avec `scripts\check_packet_order.bat -Strict`
  pour refuser les attaques qui ne peuvent pas register a cause de l'ordre des
  paquets.

Pour relire la preuve terrain sans lancer de daemon, sans reset les logs et sans
toucher a Minecraft :

```bat
scripts\check_field_status.bat -Strict
```

Il resume en une sortie `AIM_OS`, `LIVE_ACTIONS`, `PACKET_ORDER`, puis
`SUMMARY PASS` seulement si les trois preuves reelles sont presentes. Chaque
ligne affiche aussi le chemin du log lu (`path=... exists=... size=... mtime=...`) ; la
ligne `LIVE_ACTIONS` affiche `daemon=127.0.0.1:8765:up/down`. Si
`LIVE_ACTIONS` indique `samples=0` avec `exists=false`, aucun live terrain n'a
encore ecrit `runs\judas-live-actions.log` ; lancez `scripts\field_test_aim_os.bat`
pour demarrer le live avec le bon `JUDAS_LIVE_ACTION_LOG`.
Quand `field_test_aim_os.bat` affiche ce statut en fin de run, il passe aussi
un seuil `FreshAfter` : les logs presents avant le demarrage du test sont
marques `STALE` et ne peuvent pas valider la preuve terrain. Le daemon live
lance pour la preuve est stoppe avant ce statut final, donc `daemon=...:down`
est attendu si le script nettoie correctement.

Note : `judas_live` verifie maintenant que
`models\combo_god_leaderboard10_combo12-safe_latest.pts` est plus recent que
`runs\combo_god_leaderboard10_combo12\safe_latest.pt` quand `-NoExport` est
utilise. Si un nouveau checkpoint safe apparait, relancez sans `-NoExport`
pour regenerer l'export. Les anciens exports `countertap96` et `directpad`
restent des fallbacks si le nouveau safe n'existe pas, notamment
`models\combo_god_countertap96_combo12-safe_latest.pts`.
Pour verifier ce point sans lancer le daemon :

```bat
scripts\check_safe_export.bat
```

Le daemon est aussi protege contre les doubles lancements. `scripts\daemon.bat`
demarre au plus un daemon, retourne `ALREADY_RUNNING`/`ALREADY_LISTENING` si un
daemon existe deja, et accepte `-Force` pour fermer un ancien daemon Judas avant
de redemarrer. Les preuves `field_test_aim_os.bat` et `prove_combo_god.bat`
utilisent ce mode force pour attacher un log d'actions propre au bon daemon.

Pour verifier sans ecrire que la jar active dans `mods` est bien celle qui vient
d'etre construite :

```bat
scripts\check_mod_deploy.bat
scripts\check_mod_deploy.bat -RequireWritable
```

Verdicts utiles :

- `OK_DEPLOY` : la jar active dans `mods` est identique au build local.
- `STALE_DEPLOY` : Minecraft charge encore une ancienne jar ; relancer
  `scripts\prepare_aim_os_test.bat -StopMinecraft` avant de juger l'aim.
- `LOCKED_DEPLOY` : Minecraft tient la jar ouverte ; fermez le client ou utilisez
  `-StopMinecraft`, puis relancez la preparation. Si le sandbox ne peut pas
  fermer le client, lancez `scripts\watch_deploy_aim_os.bat`, puis fermez
  Minecraft : le script deployera la jar des que le verrou disparait.

Pour remettre seulement le log a zero :

```bat
scripts\check_aim_os.bat -Reset
```

Lancez Minecraft, chargez le modele dans **Live**, appuyez sur `O` jusqu'a ce
que l'overlay affiche `souris OS`, activez avec `K`, puis laissez le bot viser
une cible pendant au moins une ou deux secondes. L'overlay doit afficher une
ligne `aim OS err=...` ; si elle reste absente, le mode natif n'est pas actif ou
le bot n'a pas de cible.

Apres le test :

```bat
scripts\check_aim_os.bat -Strict
```

Pendant que Minecraft tourne, vous pouvez surveiller en continu jusqu au verdict :

```bat
scripts\watch_aim_os.bat
```

Par defaut, l'analyse ne juge que la derniere session `event=start` du log
append-only et refuse un log plus vieux que la jar Judas active dans `mods/` avec
`STALE_LOG`. `scripts\field_test_aim_os.bat` passe aussi `-FreshAfter` au
watcher : un log ecrit avant le demarrage du test ne peut pas valider la preuve.
Utilisez `-All` uniquement pour inspecter tout l'historique, et `-AllowStale`
uniquement pour relire volontairement un ancien log.

Verdicts :

- `WARMUP` : moins de 20 samples, laissez le bot viser encore.
- `PRECISE` : au moins 20 samples, p95 yaw/pitch <= 5 degres, max <= 15 degres,
  aucun stall long. C'est le minimum attendu avant de juger le niveau in-game.
- `NOT_1TO1` : la souris OS n'envoie pas la meme rotation cumulee que `cmdYaw` /
  `cmdPitch` au pas souris pres ; verifier `cmd_drift_p95` et `cmd_drift_max`.
- `LOOSE` : la souris OS bouge en 1:1, mais l'erreur reste trop large ; regardez
  l'overlay `step`, `pend`, `app` et `cmd` pour savoir si Minecraft applique bien.
- `DIVERGE` : les rotations appliquees partent souvent dans le sens oppose de
  l'erreur precedente ; suspecter inversion souris/quirk OS avant de juger la
  policy.
- `STALL` : des moves OS sont commandes mais Minecraft ne les consomme pas ;
  verifier focus/capture souris, ou repasser en mode direct.
- `NO_SAMPLES` : aucun run souris OS exploitable n'a ete ecrit.
- `NO_TARGET` : le mode souris OS tourne et loggue, mais Judas ne detecte pas
  de cible ; rapprochez une cible ou verifiez la detection avant de juger la visee.
- `STALE_LOG` : le log a ete ecrit avant la jar active ou avant le seuil
  `-FreshAfter` demande ; relancez Minecraft avec la jar courante, ou passez
  `-AllowStale` seulement pour relire un ancien run.

## Verifier le dieu du combo en arene

Le visualiseur doit prouver le style demande avant un long run Minecraft : pas
de miroir sterile, pas de draws dans les preuves role A/B, pas de
back/jump escape, pas de visee ciel, et au moins une chaine nette de 12+ hits
avec le profil combo safe par defaut.

```bat
scripts\check_arena_combo.bat -Events
```

Le check charge par defaut
`models/combo_god_leaderboard10_combo12-safe_latest.pts` contre `__combo_spar__`,
puis relance `__combo_spar__` contre ce meme modele en role B, puis ajoute
`---MIRROR PROOF---` safe-vs-safe pour refuser le vieux cas ou deux IAs
tournaient sans se toucher. Le miroir entre deux agents identiques peut draw,
mais il doit rester actif, toucher, viser le corps et ne pas back/jump. Il utilise
`cps=10`, `rot=190`, arene 40, spawn gap 8, actions samplees, KB custom et
target 50. En strict implicite, il echoue si les matchs ne terminent pas, si un
draw apparait, si le modele teste reste sous 12 combo dans un des deux roles,
ou si A/B utilisent du back/jump escape, manquent de strafe boxing, ou
regardent le ciel.

Verdicts :

- `PASS` : la regression observee initialement est absente sur 8 matchs.
- `FAIL` : draws, combo trop court ou matchs non termines ; verifier les modeles
  charges et les parametres arene/live avant d'accuser l'entrainement.
- `MISSING` : un checkpoint attendu n'existe pas dans `runs/`.

Pour lancer seulement un petit cycle training combo, utilisez le wrapper borne :

```bat
scripts\start_combo_god.bat -Force -Iters 8 -TimeoutMinutes 20
scripts\status_combo_god.bat
scripts\stop_combo_god.bat
```

`-Force` ferme le run possede par `train.pid` et nettoie aussi les vieux jobs
combo Judas detectes par ligne de commande (`train.run`, `train_combo_god.bat`,
config `combo_god_leaderboard10_combo12`). `stop_combo_god.bat` fait le meme nettoyage
d'orphelins, meme si le PID file a disparu. `-TimeoutMinutes 20` tue l'arbre
Python et sort code `124` si CUDA/JIT/training reste bloque ; mettez
`-TimeoutMinutes 0` seulement pour des runs longs surveilles.

Les reprises combo preferent maintenant `safe_latest.pt` avant `latest.pt`.
Le daemon fait aussi cette substitution si l'app demande `latest.pt` et qu'un
`safe_latest.pt` existe, pour eviter de repartir depuis un checkpoint ecrit
avant la validation anti-ciel/anti-escape.
Judas Arene applique la meme regle : pour `combo_god_leaderboard10_combo12`
et les fallbacks combo,
si `safe_latest.pt` existe, le menu modele propose ce safe au lieu du
`latest.pt` ou des checkpoints bruts recents du meme run.
Le daemon applique aussi la redirection aux endpoints `/arena/load` et
`/live/load`, donc un ancien choix persiste ou un appel REST manuel vers
`runs\combo_god_leaderboard10_combo12\latest.pt` repart sur le safe valide
au lieu de charger un checkpoint brut.
La page Models cache aussi les checkpoints bruts de ce run quand le safe existe,
et `/models/export` redirige un export manuel de `latest.pt`/`ckpt_*.pt` combo
vers `safe_latest.pt`.
Enfin, `/live/load` refuse l'export
`models\combo_god_leaderboard10_combo12-safe_latest.pts` si son `.json` ne
pointe pas vers `safe_latest.pt` avec la bonne taille et le bon SHA256. Un
export stale doit etre regenere avant le test terrain.
Le endpoint `/models` annote aussi cet export avec `export_status=fresh/stale` :
Live et Judas Arene ignorent les exports stale, tandis que la page Models les
affiche avec le statut pour diagnostiquer pourquoi un load est refuse.

Pour lancer la preuve courte complete sans entrainement long :

```bat
scripts\prove_combo_god.bat
```

Cette commande enchaine l'arene deux roles, un daemon live temporaire avec log
synthetique separe, le check WebSocket live, puis affiche le statut terrain. Elle
stoppe le daemon a la fin. Elle ne remplace pas `field_test_aim_os.bat` :
`SUMMARY PASS` terrain exige encore Minecraft, OS mouse, actions live reelles et
packet-order propre. Avec `-RequireField`, le statut terrain utilise aussi
`FreshAfter` : seuls les logs ecrits apres le demarrage de la preuve peuvent
valider le run.
## Verifier la boucle aim souris OS hors Minecraft

Avant un test terrain, le controleur natif peut etre stresse hors Minecraft sur
les cas qui provoquent les oscillations : gain Windows variable, signe inverse,
latence, jitter, et retournement brutal de commande.

```bat
scripts\check_native_aim_sim.bat
```

Verdicts :

- `PASS` : toutes les trajectoires simulees convergent, et aucun ordre oppose
  n'est envoye au tick exact ou la commande se retourne.
- `FAIL` : une constante ou une modification de la boucle native peut recreer
  les zigzags ; verifier `NATIVE_REVERSAL_SETTLE_TICKS`, pending et signe.

Ce check ne remplace pas `scripts\check_aim_os.bat -Strict` : il prouve la
boucle de controle, pas la capture souris reelle par Minecraft.
