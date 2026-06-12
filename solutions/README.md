# Solutions longues des exercices

Ce répertoire contient les corrections longues associées aux exercices du Tome I.

Le livre garde volontairement des exercices lisibles et des indications de solution courtes. Le dépôt compagnon sert à aller plus loin : montrer le raisonnement complet, détailler les calculs, expliciter les choix d'architecture et, quand c'est utile, relier l'exercice au code exécutable.

## Principe

Chaque exercice possède un identifiant stable :

```text
EX-CCC-NNN
```

Avec :

- `CCC` : numéro du chapitre sur trois chiffres ;
- `NNN` : numéro de l'exercice dans le chapitre.

La solution longue correspondante est placée dans :

```text
solutions/chCCC/ex-CCC-NNN.md
```

Exemple :

```text
solutions/ch016/ex-016-003.md
```

## Rôle du fichier d'index

Le fichier `solutions/index-solutions.json` est le contrat de génération des corrections.

Il contient, pour chaque exercice :

- l'identifiant ;
- le chapitre ;
- le titre ;
- l'énoncé ;
- le niveau ;
- le type d'exercice ;
- la solution attendue courte déjà présente dans le livre ;
- le chemin de la solution longue à produire ;
- le batch de génération associé.

Ce fichier permet de vérifier que le livre, l'annexe M et le dépôt compagnon restent synchronisés.

Deux fichiers humains complètent cet index :

- `solutions/catalogue-solutions.md` : catalogue navigable des 187 corrections ;
- `solutions/qa-report.md` : rapport de vérification structurelle du dossier `solutions/`.
- `solutions/validate-solutions.py` : validateur local reproductible du dossier `solutions/`.
- `solutions/chCCC/README.md` : index local des corrections de chaque chapitre.

## Format d'une solution longue

Les fichiers de solution doivent suivre une structure stable :

```md
# EX-CCC-NNN - Titre de l'exercice

## Énoncé

Rappel court de l'exercice.

## Ce que l'exercice vérifie

Explication de l'objectif réel : notion mathématique, réflexe d'architecture, lecture de forme tensorielle, diagnostic, protocole, etc.

## Solution détaillée

Réponse développée étape par étape.

## Points importants

- Ce qu'il faut retenir.
- Les pièges fréquents.
- Les limites de la réponse.

## Prolongement

Optionnel : variante, test complémentaire, lien avec un exemple de code ou un chapitre suivant.
```

## Exercices de code

Certains exercices annoncent un fichier `.cs` dans l'annexe M. Dans ce cas, la correction longue reste tout de même un fichier Markdown.

Le fichier Markdown doit expliquer :

- le problème à résoudre ;
- le raisonnement ;
- le lien vers le fichier C# associé, s'il existe ;
- la commande de validation ;
- le résultat attendu ;
- les limites du prototype.

Le code ne doit pas remplacer l'explication. Dans un livre technique, le lecteur doit comprendre pourquoi le code est écrit ainsi, pas seulement voir un fichier qui compile.

## Batches de génération

Les solutions sont découpées ainsi :

| Batch | Chapitres | Contenu |
|---|---:|---|
| `SOL-02` | 1 à 4 | Fondements historiques, algèbre, optimisation, systèmes dynamiques |
| `SOL-03` | 5 à 10 | Perceptron, MLP, backpropagation, RNN, LSTM, GRU |
| `SOL-04` | 11, 12, 20, 29 | Transformers, SSM modernes, comparatif global, Transformer² |
| `SOL-05` | 13 à 19 | Temps continu, Neural ODE, CT-RNN, LTC, CfC, LFM, théorie LNN |
| `SOL-06` | 21 à 28 | Mixture-of-Experts, routage, GShard, Switch, Mixtral, DeepSeekMoE |
| `SOL-07` | 30 à 35 | Architecture modulaire, mémoire, concepts, agentique, gouvernance, Qaya |
| `SOL-08` | 36 à 38 | TorchSharp, entraînement distribué, déploiement |
| `SOL-09` | fichiers `.cs` | Fichiers de code de soutien annoncés par l'annexe M et l'index |
| `SOL-10` | catalogue + QA | Catalogue navigable et rapport de vérification structurelle |
| `SOL-11` | validation | Script local `validate-solutions.py` |
| `SOL-12` | index par chapitre | README local dans chaque dossier `solutions/chCCC` |

## Validation locale

Depuis la racine du dépôt :

```powershell
python solutions/validate-solutions.py
```

Sortie attendue :

```text
solutions validation: ok
```

Pour une sortie exploitable par script :

```powershell
python solutions/validate-solutions.py --json
```

## Règles d'écriture

Les corrections doivent respecter la voix DomiGeek :

- expliquer le problème avant la formule ;
- rester technique, direct et pédagogique ;
- éviter le ton marketing ;
- éviter l'anthropomorphisme naïf ;
- distinguer résultat établi, observation expérimentale, hypothèse et choix d'architecture ;
- ne pas transformer Qaya en argument d'autorité.

Pour les exercices mathématiques, chaque symbole important doit être introduit avant d'être utilisé.

Pour les exercices d'architecture, la réponse doit préciser au moins :

- les responsabilités ;
- les entrées et sorties ;
- les limites ;
- les métriques ou traces à observer.

Pour les exercices de réflexion, la réponse doit montrer le raisonnement, pas seulement donner une conclusion.

## Validation

Avant publication, les contrôles suivants doivent passer :

- tous les exercices de l'annexe M ont un fichier de solution longue ;
- tous les liens `solutions/chCCC/ex-CCC-NNN.md` existent ;
- les exercices de code pointent vers un exemple ou une commande vérifiable quand c'est pertinent ;
- aucun fichier interne de génération du manuscrit n'est publié par erreur dans le dépôt compagnon public.
