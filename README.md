# Analyse Numérique — Cascade Trophique

Projet MAM3 (Polytech Nice Sophia / Université Côte d'Azur) portant sur la résolution numérique d'un système d'équations différentielles modélisant une cascade trophique inspirée des écosystèmes de Yellowstone, Isle Royale et Banff.

**Auteurs du projet original :** Thomas Begotti et 3 autres étudiants

## Contexte

Une cascade trophique décrit comment l'ajout d'un prédateur au sommet de la chaîne alimentaire modifie la structure de tout un écosystème (herbivores, végétation). Ce projet modélise une cascade à 3 niveaux :

1. **Végétation** (V)
2. **Proies herbivores** : wapiti (N) et cerf (D)
3. **Prédateurs** : loup (W) et ours (B)

Le système est décrit par 5 équations différentielles ordinaires (EDO) non-linéaires et fortement couplées, intégrant croissance logistique, pâturage de type Holling, prédation multi-proies et variations saisonnières (mortalité des loups modulée par un terme en sin²).

## Objectifs

- Résoudre le système de 5 EDO avec plusieurs méthodes numériques et justifier leur choix.
- Représenter graphiquement l'évolution des cinq populations.
- Scénariser des phénomènes plausibles (réintroduction de prédateurs, sécheresse, chasse intensive, extinction) et en analyser les effets.

## Méthodes numériques implémentées

- **Euler explicite** : simple mais instable au-delà d'un pas de temps critique (système raide, rapport spectral ≈ 39.5).
- **Euler implicite (Newton-Raphson)** : stable inconditionnellement, mais coûteux (inversion de la jacobienne 5×5 à chaque itération).
- **Runge-Kutta d'ordre 4 (RK4)** : meilleur compromis précision/temps, méthode retenue pour les scénarios finaux car elle redémarre proprement à chaque pas (essentiel lors d'événements discontinus comme une réintroduction).
- **Adams-Bashforth 4 (AB4)** : méthode multipas la plus rapide, mais dépendante des 4 états précédents — problématique en cas de discontinuité.

Toutes les méthodes sont validées par comparaison à la référence **BDF** (`scipy.integrate.solve_ivp`).

## Scénarios étudiés

1. **Réintroduction des loups (Yellowstone)** — cascade top-down : hausse de la végétation après la baisse des ongulés.
2. **Sécheresse** — contrôle bottom-up : réduction de la croissance de la végétation impactant toute la chaîne.
3. **Chasse intensive** — pression sur les loups menant à leur quasi-extinction.
4. **Extinction des ours** — hausse des ongulés et de la végétation impactée en conséquence.

## Conclusion

RK4 a été retenu comme méthode principale pour les simulations : il offre une précision d'ordre 4 à un coût modéré et, contrairement à AB4, ne dépend pas de l'historique des pas précédents — ce qui garantit une transition numérique stable lors d'événements discontinus (ex. réintroduction des loups à t = 20 ans).

## Contenu du dépôt

- `code` — implémentation Python des schémas numériques (Euler explicite/implicite, RK4, AB4) et des scénarios de simulation.

## Contribution personnelle

Mes contributions sur ce projet sont :

- Implémentation de la méthode de Runge-Kutta d'ordre 4 (RK4)
- Implémentation de la méthode multipas d'Adams-Bashforth 4 (AB4)
- Réalisation des graphiques
