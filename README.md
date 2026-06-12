# Les Dimensions de l'Intelligence Artificielle - Tome I - Vers les Réseaux Neuronaux Liquides

Dépôt compagnon du Tome I : _Les Dimensions de l'Intelligence Artificielle - Tome I - Vers les Réseaux Neuronaux Liquides_.

URL du dépôt compagnon :

```text
https://github.com/domigeek/domigeek-Books-T1-LnnMoeBook.Companion
```

## Contenu publié

- `code/csharp/` : code compagnon C# / TorchSharp.
- `code/python/` : compléments Python / PyTorch ciblés pour comparer quelques exemples clés.
- `figures/export/` : images exportées utilisées par le livre.
- `solutions/` : corrections longues des exercices du Tome I.
- `exports/pdf/tome-i.pdf` : version PDF du Tome I.
- `exports/pdf/tome-i-justifie.pdf` : version PDF justifiée du Tome I.
- `exports/epub/tome-i.epub` : version EPUB du Tome I.

Le dépôt contient les artefacts publiés du livre, le code compagnon et les corrections longues.

Le dépôt compagnon est appelé à évoluer dans le temps. De nouveaux exemples, variantes, tests et mesures pourront être ajoutés pour nourrir la communauté technique, tout en conservant un objectif pédagogique : rendre les mécanismes lisibles, exécutables et vérifiables sans prétendre fournir un framework de production.

## Commandes utiles

```powershell
dotnet build code/csharp/LnnMoeBook.sln
dotnet test code/csharp/LnnMoeBook.sln -m:1
dotnet run --project code/csharp/LnnMoeBook.Examples
python -m pip install -r code/python/requirements.txt
python code/python/examples/tensor_creation.py
python code/python/examples/simple_rnn_forecast.py
python code/python/examples/simple_ltc_cell.py
python solutions/validate-solutions.py
```

## Note

Qaya est cité dans le livre uniquement comme étude de cas architecturale. Ce dépôt ne contient pas le projet Qaya complet.
