# Rapport QA des solutions

Date : 2026-06-11

## Portée

Ce rapport couvre uniquement le dossier `solutions/`.
Aucun export PDF, EPUB ou site web n’est régénéré par ce lot.

## Résultats

| Vérification | Résultat |
|---|---:|
| Exercices attendus | `187` |
| Solutions Markdown longues présentes | `187` |
| Fichiers C# de soutien présents | `16` |
| Liens du catalogue cassés | `0` |
| Index README par chapitre | `38` |
| Liens des README de chapitre cassés | `0` |
| Fichiers scannés pour l'encodage | `246` |
| Duplicats dans l’annexe | `0` |
| Duplicats dans les chapitres | `0` |
| Exercices absents des chapitres | `0` |
| Exercices absents de l’annexe | `0` |
| Validateur local | `ok` |
| Statut global | `ok` |

## Notes

- Les fichiers Markdown sont les corrections longues principales.
- Les fichiers `.cs` sont des supports de lecture et de validation locale, alignés sur les exercices qui annoncent du code.
- La validation de ce lot est structurelle : présence des fichiers, cohérence de l’index, titres, liens et encodage UTF-8.
- La compilation complète des snippets `.cs` dépend de leur intégration éventuelle dans un projet de tests ou un projet console dédié.

## Commandes de contrôle utilisées

```powershell
python solutions/validate-solutions.py
python solutions/validate-solutions.py --json
```
