using System.Globalization;
using TorchSharp;

namespace LnnMoeBook.Examples.Memory;

public sealed record MiniRagDocument(
    string Id,
    string Title,
    string Content,
    float[] Embedding,
    IReadOnlyList<string> Tags,
    int Timestamp,
    float Confidence,
    IReadOnlyDictionary<string, string> Metadata);

public sealed record MiniRagRequest(
    string Question,
    int TopK,
    string? RequiredTag = null,
    float MinimumSimilarity = -1.0f);

public sealed record MiniRagSource(
    int Rank,
    string Id,
    string Title,
    string Excerpt,
    float Score,
    IReadOnlyList<string> Tags,
    IReadOnlyDictionary<string, string> Metadata);

public sealed record MiniRagResponse(
    MiniRagRequest Request,
    string Answer,
    string Context,
    IReadOnlyList<MiniRagSource> Sources,
    bool UsedFallback)
{
    public torch.Tensor ToSourceScoreTensor()
    {
        return torch.tensor(Sources.Select(source => source.Score).ToArray(), dtype: torch.float32);
    }
}

public sealed record MiniRagReport(
    int DocumentCount,
    int EmbeddingWidth,
    MiniRagResponse RagResponse,
    MiniRagResponse EmptyResponse);

public static class MiniRagPipeline
{
    public const int EmbeddingWidth = 4;

    public static MiniRagReport RunDefault()
    {
        var documents = GenerateToyDocuments();
        var store = BuildMemoryStore(documents);
        var ragResponse = Answer(
            store,
            new MiniRagRequest(
                Question: "Comment un pipeline RAG utilise des sources de memoire et un contexte ?",
                TopK: 3,
                RequiredTag: "memory",
                MinimumSimilarity: 0.10f));
        var emptyResponse = Answer(
            store,
            new MiniRagRequest(
                Question: "Question hors corpus sur un sujet absent.",
                TopK: 2,
                RequiredTag: "absent",
                MinimumSimilarity: 0.10f));

        return new MiniRagReport(
            documents.Count,
            EmbeddingWidth,
            ragResponse,
            emptyResponse);
    }

    public static IReadOnlyList<MiniRagDocument> GenerateToyDocuments()
    {
        var specs = new[]
        {
            NewDocument(
                "doc-moe-routing",
                "Routage MoE",
                "Un routeur top-k selectionne quelques experts MoE pour chaque token et laisse les autres inactifs.",
                ["moe", "routing"],
                timestamp: 10,
                confidence: 0.94f,
                chapter: "19"),
            NewDocument(
                "doc-moe-balance",
                "Equilibrage MoE",
                "Une penalite de load balancing aide a diagnostiquer un effondrement du routage vers peu d'experts.",
                ["moe", "monitoring"],
                timestamp: 12,
                confidence: 0.91f,
                chapter: "19"),
            NewDocument(
                "doc-ltc-state",
                "Etat liquide",
                "Une cellule LTC maintient un etat continu dont la constante de temps effective varie avec le gate.",
                ["lnn", "ltc"],
                timestamp: 20,
                confidence: 0.93f,
                chapter: "14"),
            NewDocument(
                "doc-liquid-orchestrator",
                "Orchestrateur liquide",
                "Un orchestrateur liquide peut utiliser un etat temporel pour modifier les scores de routage MoE.",
                ["lnn", "moe", "orchestration"],
                timestamp: 25,
                confidence: 0.86f,
                chapter: "25"),
            NewDocument(
                "doc-rag-sources",
                "Sources RAG",
                "Un pipeline RAG recupere des sources, assemble un contexte et produit une reponse qui cite ces sources.",
                ["memory", "rag"],
                timestamp: 60,
                confidence: 0.95f,
                chapter: "26"),
            NewDocument(
                "doc-memory-forgetting",
                "Oubli controle",
                "Une politique de retention peut marquer des episodes obsoletes pour limiter la contamination memoire.",
                ["memory", "safety"],
                timestamp: 35,
                confidence: 0.88f,
                chapter: "26"),
            NewDocument(
                "doc-vector-store",
                "Vector store local",
                "Un vector store compare une requete et des documents par similarite cosinus sur des embeddings.",
                ["memory", "vector"],
                timestamp: 38,
                confidence: 0.90f,
                chapter: "26"),
            NewDocument(
                "doc-deploy-api",
                "API inference",
                "Un service d'inference expose une reponse JSON versionnee et observable via une API.",
                ["deployment", "api"],
                timestamp: 45,
                confidence: 0.87f,
                chapter: "31"),
            NewDocument(
                "doc-observability",
                "Observabilite",
                "Le monitoring mesure la latence, les erreurs et les versions de modele en deploiement.",
                ["deployment", "monitoring"],
                timestamp: 48,
                confidence: 0.84f,
                chapter: "31"),
            NewDocument(
                "doc-rag-limits",
                "Limites RAG",
                "La similarite vectorielle ne garantit pas la verite; les sources recuperees doivent rester auditables.",
                ["memory", "rag", "safety"],
                timestamp: 50,
                confidence: 0.89f,
                chapter: "26")
        };

        return specs
            .Select(spec => new MiniRagDocument(
                spec.Id,
                spec.Title,
                spec.Content,
                EmbedText(spec.Title + " " + spec.Content),
                spec.Tags,
                spec.Timestamp,
                spec.Confidence,
                new Dictionary<string, string>
                {
                    ["chapter"] = spec.Chapter,
                    ["title"] = spec.Title
                }))
            .ToArray();
    }

    public static EpisodicMemoryStore BuildMemoryStore(IReadOnlyList<MiniRagDocument> documents)
    {
        ValidateDocuments(documents);

        var store = new EpisodicMemoryStore(EmbeddingWidth);
        foreach (var document in documents)
        {
            store.Upsert(new EpisodicMemoryEntry(
                document.Id,
                document.Content,
                document.Embedding.ToArray(),
                document.Tags.ToArray(),
                document.Timestamp,
                document.Confidence,
                document.Metadata.ToDictionary(pair => pair.Key, pair => pair.Value, StringComparer.Ordinal)));
        }

        return store;
    }

    public static MiniRagResponse Answer(
        EpisodicMemoryStore store,
        MiniRagRequest request)
    {
        ValidateRequest(request);

        var query = new EpisodicMemoryQuery(
            Embedding: EmbedText(request.Question),
            TopK: request.TopK,
            RequiredTag: request.RequiredTag,
            MinimumSimilarity: request.MinimumSimilarity);
        var retrieved = store.Recall(query);
        var sources = retrieved.Select(ToSource).ToArray();
        var context = BuildContext(sources);
        var answer = BuildAnswer(request, sources);

        return new MiniRagResponse(
            request,
            answer,
            context,
            sources,
            UsedFallback: sources.Length == 0);
    }

    public static float[] EmbedText(string text)
    {
        if (string.IsNullOrWhiteSpace(text))
        {
            throw new ArgumentException("Text must not be empty.", nameof(text));
        }

        var lower = text.ToLowerInvariant();
        var vector = new float[EmbeddingWidth];

        vector[0] += CountKeywords(lower, "moe", "expert", "experts", "routing", "routeur", "top-k", "switch", "mixtral");
        vector[1] += CountKeywords(lower, "lnn", "ltc", "liquid", "liquide", "etat", "state", "temps", "time", "gate");
        vector[2] += CountKeywords(lower, "memory", "memoire", "rag", "source", "sources", "retrieval", "recupere", "contexte", "context", "vector", "cosinus");
        vector[3] += CountKeywords(lower, "api", "deployment", "deploiement", "service", "json", "monitoring", "observable", "latence", "version");

        if (vector.All(value => value == 0.0f))
        {
            vector[2] = 0.25f;
        }

        return vector;
    }

    public static string BuildContext(IReadOnlyList<MiniRagSource> sources)
    {
        if (sources.Count == 0)
        {
            return "";
        }

        return string.Join(
            Environment.NewLine,
            sources.Select(source => string.Create(
                CultureInfo.InvariantCulture,
                $"[S{source.Rank}] {source.Id}: {source.Excerpt}")));
    }

    public static string ToSourcesCsv(MiniRagResponse response)
    {
        var lines = new List<string>
        {
            "rank,id,score,content,tags,metadata"
        };

        foreach (var source in response.Sources)
        {
            lines.Add(string.Create(
                CultureInfo.InvariantCulture,
                $"{source.Rank},{Csv(source.Id)},{source.Score:0.######},{Csv(source.Excerpt)},{Csv(string.Join("|", source.Tags))},{Csv(FormatMetadata(source.Metadata))}"));
        }

        return string.Join(Environment.NewLine, lines) + Environment.NewLine;
    }

    public static string ToAnswerCsv(MiniRagResponse response)
    {
        var sourceIds = string.Join("|", response.Sources.Select(source => source.Id));
        var context = response.Context.Replace(Environment.NewLine, "\\n", StringComparison.Ordinal);
        var lines = new[]
        {
            "query,answer,source_count,sources,context",
            $"{Csv(response.Request.Question)},{Csv(response.Answer)},{response.Sources.Count},{Csv(sourceIds)},{Csv(context)}"
        };

        return string.Join(Environment.NewLine, lines) + Environment.NewLine;
    }

    public static string ToContextCsv(MiniRagResponse response)
    {
        var lines = new List<string>
        {
            "rank,context_line"
        };

        foreach (var line in response.Context.Split(Environment.NewLine, StringSplitOptions.RemoveEmptyEntries))
        {
            var separator = line.IndexOf(']', StringComparison.Ordinal);
            var rank = separator > 2 && line.StartsWith("[S", StringComparison.Ordinal)
                ? line[2..separator]
                : "";
            lines.Add($"{rank},{Csv(line)}");
        }

        return string.Join(Environment.NewLine, lines) + Environment.NewLine;
    }

    public static torch.Tensor ToDocumentEmbeddingTensor(IReadOnlyList<MiniRagDocument> documents)
    {
        ValidateDocuments(documents);

        return torch.tensor(documents.SelectMany(document => document.Embedding).ToArray(), dtype: torch.float32)
            .reshape(documents.Count, EmbeddingWidth);
    }

    public static string FormatReport(MiniRagReport report)
    {
        var sourceIds = report.RagResponse.Sources.Count == 0
            ? "none"
            : string.Join(",", report.RagResponse.Sources.Select(source => source.Id));
        var best = report.RagResponse.Sources.Count == 0
            ? "none"
            : report.RagResponse.Sources[0].Id;
        var tag = string.IsNullOrWhiteSpace(report.RagResponse.Request.RequiredTag)
            ? "none"
            : report.RagResponse.Request.RequiredTag;

        return string.Create(
            CultureInfo.InvariantCulture,
            $"mini rag: documents={report.DocumentCount}, width={report.EmbeddingWidth}, top_k={report.RagResponse.Request.TopK}, retrieved={report.RagResponse.Sources.Count}, tag={tag}, best={best}, has_sources={!report.RagResponse.UsedFallback}, source_ids=[{sourceIds}], fallback={report.EmptyResponse.UsedFallback}");
    }

    private static MiniRagSource ToSource(EpisodicMemorySearchResult result)
    {
        var metadata = result.Entry.Metadata?.ToDictionary(pair => pair.Key, pair => pair.Value, StringComparer.Ordinal)
            ?? new Dictionary<string, string>();
        var title = metadata.TryGetValue("title", out var value)
            ? value
            : result.Entry.Id;

        return new MiniRagSource(
            result.Rank,
            result.Entry.Id,
            title,
            FirstSentence(result.Entry.Text),
            result.Similarity,
            result.Entry.Tags.ToArray(),
            metadata);
    }

    private static string BuildAnswer(
        MiniRagRequest request,
        IReadOnlyList<MiniRagSource> sources)
    {
        if (sources.Count == 0)
        {
            return "Aucune source locale suffisante n'a ete trouvee pour cette question.";
        }

        var sourceIds = string.Join(", ", sources.Select(source => source.Id));
        return string.Create(
            CultureInfo.InvariantCulture,
            $"Reponse locale: {sources[0].Excerpt} Cette reponse utilise {sources.Count} source(s): {sourceIds}. Question: {request.Question}");
    }

    private static int CountKeywords(
        string text,
        params string[] keywords)
    {
        var count = 0;
        foreach (var keyword in keywords)
        {
            if (text.Contains(keyword, StringComparison.Ordinal))
            {
                count++;
            }
        }

        return count;
    }

    private static string FirstSentence(string text)
    {
        var index = text.IndexOf('.', StringComparison.Ordinal);
        return index < 0
            ? text.Trim()
            : text[..(index + 1)].Trim();
    }

    private static void ValidateDocuments(IReadOnlyList<MiniRagDocument> documents)
    {
        if (documents.Count == 0)
        {
            throw new ArgumentException("At least one document is required.", nameof(documents));
        }

        var ids = new HashSet<string>(StringComparer.Ordinal);
        foreach (var document in documents)
        {
            if (string.IsNullOrWhiteSpace(document.Id))
            {
                throw new ArgumentException("Document id must not be empty.", nameof(documents));
            }

            if (!ids.Add(document.Id))
            {
                throw new ArgumentException("Document ids must be unique.", nameof(documents));
            }

            if (string.IsNullOrWhiteSpace(document.Title) || string.IsNullOrWhiteSpace(document.Content))
            {
                throw new ArgumentException("Document title and content must not be empty.", nameof(documents));
            }

            if (document.Embedding.Length != EmbeddingWidth)
            {
                throw new ArgumentException("Document embedding width is inconsistent.", nameof(documents));
            }

            if (document.Embedding.Any(value => float.IsNaN(value) || float.IsInfinity(value))
                || document.Embedding.All(value => value == 0.0f))
            {
                throw new ArgumentException("Document embeddings must be finite and non-zero.", nameof(documents));
            }

            if (document.Tags.Any(string.IsNullOrWhiteSpace))
            {
                throw new ArgumentException("Document tags must not be empty.", nameof(documents));
            }
        }
    }

    private static void ValidateRequest(MiniRagRequest request)
    {
        if (string.IsNullOrWhiteSpace(request.Question))
        {
            throw new ArgumentException("Question must not be empty.", nameof(request));
        }

        if (request.TopK <= 0)
        {
            throw new ArgumentOutOfRangeException(nameof(request), "TopK must be positive.");
        }

        if (float.IsNaN(request.MinimumSimilarity) || float.IsInfinity(request.MinimumSimilarity))
        {
            throw new ArgumentOutOfRangeException(nameof(request), "Minimum similarity must be finite.");
        }
    }

    private static MiniRagDocumentSpec NewDocument(
        string id,
        string title,
        string content,
        string[] tags,
        int timestamp,
        float confidence,
        string chapter)
    {
        return new MiniRagDocumentSpec(id, title, content, tags, timestamp, confidence, chapter);
    }

    private static string Csv(string value)
    {
        if (!value.Contains(',') && !value.Contains('"') && !value.Contains('\n') && !value.Contains('\r'))
        {
            return value;
        }

        return "\"" + value.Replace("\"", "\"\"", StringComparison.Ordinal) + "\"";
    }

    private static string FormatMetadata(IReadOnlyDictionary<string, string> metadata)
    {
        return string.Join(
            "|",
            metadata
                .OrderBy(pair => pair.Key, StringComparer.Ordinal)
                .Select(pair => $"{pair.Key}={pair.Value}"));
    }

    private sealed record MiniRagDocumentSpec(
        string Id,
        string Title,
        string Content,
        string[] Tags,
        int Timestamp,
        float Confidence,
        string Chapter);
}
