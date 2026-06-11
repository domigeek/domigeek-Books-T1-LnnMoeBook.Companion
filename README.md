# Les Dimensions de l'Intelligence Artificielle - Tome I - Vers les Réseaux Neuronaux Liquides

Dépôt compagnon du Tome I : _Les Dimensions de l'Intelligence Artificielle - Tome I - Vers les Réseaux Neuronaux Liquides_.

URL du dépôt compagnon :

```text
https://github.com/domigeek/domigeek-Books-T1-LnnMoeBook.Companion
```

## Contenu publié

- `code/csharp/` : code compagnon C# / TorchSharp.
- `figures/export/` : images exportées utilisées par le livre.
- `exports/pdf/tome-i.pdf` : version PDF du Tome I.
- `exports/epub/tome-i.epub` : version EPUB du Tome I.

Le dépôt contient uniquement les artefacts listés ci-dessus.

## Commandes utiles

```powershell
dotnet build code/csharp/LnnMoeBook.sln
dotnet test code/csharp/LnnMoeBook.sln -m:1
dotnet run --project code/csharp/LnnMoeBook.Examples
```

## Note

Qaya est cité dans le livre uniquement comme étude de cas architecturale. Ce dépôt ne contient pas le projet Qaya complet.
