using System.Globalization;
using TorchSharp;

namespace LnnMoeBook.Examples.Memory;

public sealed record EpisodicMemoryEntry(
    string Id,
    string Text,
    float[] Embedding,
    IReadOnlyList<string> Tags,
    int Timestamp,
    float Confidence,
    IReadOnlyDictionary<string, string>? Metadata = null,
    int InsertedIndex = -1,
    bool IsDeleted = false);

public sealed record EpisodicMemoryQuery(
    float[] Embedding,
    int TopK,
    string? RequiredTag = null,
    float MinimumSimilarity = -1.0f);

public sealed record EpisodicMemorySearchResult(
    EpisodicMemoryEntry Entry,
    float Similarity,
    int Rank);

public sealed record EpisodicMemoryReport(
    int Entries,
    int Active,
    int Deleted,
    int EmbeddingWidth,
    int TopK,
    int PrunedCount,
    IReadOnlyList<EpisodicMemorySearchResult> MoeResults,
    IReadOnlyList<EpisodicMemorySearchResult> MemoryResults);

public sealed class EpisodicMemoryStore
{
    private readonly List<EpisodicMemoryEntry> _entries = [];
    private int _nextInsertedIndex;

    public EpisodicMemoryStore(int embeddingWidth)
    {
        if (embeddingWidth <= 0)
        {
            throw new ArgumentOutOfRangeException(nameof(embeddingWidth), "Embedding width must be positive.");
        }

        EmbeddingWidth = embeddingWidth;
    }

    public int EmbeddingWidth { get; }

    public int Count => _entries.Count(entry => !entry.IsDeleted);

    public int TotalCount => _entries.Count;

    public int DeletedCount => _entries.Count(entry => entry.IsDeleted);

    public IReadOnlyList<EpisodicMemoryEntry> Entries => _entries
        .Where(entry => !entry.IsDeleted)
        .Select(CloneEntry)
        .ToArray();

    public IReadOnlyList<EpisodicMemoryEntry> AuditEntries => _entries
        .Select(CloneEntry)
        .ToArray();

    public static EpisodicMemoryReport RunDefault()
    {
        var store = FromEntries(GenerateSyntheticEpisodes());
        var topK = 3;
        var moeResults = store.Recall(new EpisodicMemoryQuery(
            Embedding: new[] { 1.0f, 0.12f, 0.02f, 0.0f },
            TopK: topK,
            RequiredTag: "moe"));
        var memoryResults = store.Recall(new EpisodicMemoryQuery(
            Embedding: new[] { 0.0f, 0.08f, 0.95f, 0.22f },
            TopK: 2,
            RequiredTag: "memory"));
        var pruned = store.PruneOlderThan(minTimestamp: 15);

        return new EpisodicMemoryReport(
            store.TotalCount,
            store.Count,
            store.DeletedCount,
            store.EmbeddingWidth,
            topK,
            pruned,
            moeResults,
            memoryResults);
    }

    public static EpisodicMemoryStore FromEntries(IEnumerable<EpisodicMemoryEntry> entries)
    {
        var materialized = entries.ToArray();
        if (materialized.Length == 0)
        {
            throw new ArgumentException("At least one memory entry is required.", nameof(entries));
        }

        var store = new EpisodicMemoryStore(materialized[0].Embedding.Length);
        foreach (var entry in materialized)
        {
            store.Upsert(entry);
        }

        return store;
    }

    public static IReadOnlyList<EpisodicMemoryEntry> GenerateSyntheticEpisodes()
    {
        return new[]
        {
            NewEntry(
                "evt-moe-routing",
                "Top-k router selected two parametric experts for a token batch.",
                [0.96f, 0.12f, 0.03f, 0.00f],
                ["moe", "routing"],
                timestamp: 10,
                confidence: 0.95f,
                metadata: new Dictionary<string, string>
                {
                    ["chapter"] = "19",
                    ["kind"] = "routing"
                }),
            NewEntry(
                "evt-moe-balance",
                "Load balancing penalty increased after routed expert collapse.",
                [0.88f, 0.26f, 0.10f, 0.02f],
                ["moe", "monitoring"],
                timestamp: 12,
                confidence: 0.90f,
                metadata: new Dictionary<string, string>
                {
                    ["chapter"] = "19",
                    ["kind"] = "diagnostic"
                }),
            NewEntry(
                "evt-ltc-tau",
                "LTC effective time constants changed with the input gate.",
                [0.06f, 0.96f, 0.08f, 0.02f],
                ["lnn", "ltc"],
                timestamp: 20,
                confidence: 0.93f,
                metadata: new Dictionary<string, string>
                {
                    ["chapter"] = "14",
                    ["kind"] = "state"
                }),
            NewEntry(
                "evt-rag-source",
                "Retriever returned a source document before generation.",
                [0.02f, 0.08f, 0.96f, 0.21f],
                ["memory", "rag"],
                timestamp: 30,
                confidence: 0.92f,
                metadata: new Dictionary<string, string>
                {
                    ["chapter"] = "26",
                    ["kind"] = "retrieval"
                }),
            NewEntry(
                "evt-memory-forgetting",
                "Retention policy marked stale memories for controlled deletion.",
                [0.00f, 0.10f, 0.86f, 0.38f],
                ["memory", "safety"],
                timestamp: 35,
                confidence: 0.88f,
                metadata: new Dictionary<string, string>
                {
                    ["chapter"] = "26",
                    ["kind"] = "retention"
                }),
            NewEntry(
                "evt-deploy-api",
                "Inference service returned a versioned JSON response.",
                [0.02f, 0.02f, 0.18f, 0.97f],
                ["deployment", "api"],
                timestamp: 45,
                confidence: 0.87f,
                metadata: new Dictionary<string, string>
                {
                    ["chapter"] = "31",
                    ["kind"] = "deployment"
                })
        };
    }

    public void Upsert(EpisodicMemoryEntry entry)
    {
        ValidateEntry(entry, EmbeddingWidth);
        var existingIndex = _entries.FindIndex(item => string.Equals(item.Id, entry.Id, StringComparison.Ordinal));
        var insertedIndex = existingIndex >= 0
            ? _entries[existingIndex].InsertedIndex
            : _nextInsertedIndex++;
        var stored = CloneEntry(entry) with
        {
            InsertedIndex = insertedIndex,
            IsDeleted = false
        };
        if (existingIndex >= 0)
        {
            _entries[existingIndex] = stored;
            return;
        }

        _entries.Add(stored);
    }

    public bool Remove(string id)
    {
        if (string.IsNullOrWhiteSpace(id))
        {
            throw new ArgumentException("Entry id must not be empty.", nameof(id));
        }

        var index = _entries.FindIndex(entry => string.Equals(entry.Id, id, StringComparison.Ordinal));
        if (index < 0 || _entries[index].IsDeleted)
        {
            return false;
        }

        _entries[index] = _entries[index] with { IsDeleted = true };
        return true;
    }

    public int PruneOlderThan(int minTimestamp)
    {
        if (minTimestamp < 0)
        {
            throw new ArgumentOutOfRangeException(nameof(minTimestamp), "Minimum timestamp must be non-negative.");
        }

        var before = Count;
        for (var index = 0; index < _entries.Count; index++)
        {
            if (!_entries[index].IsDeleted && _entries[index].Timestamp < minTimestamp)
            {
                _entries[index] = _entries[index] with { IsDeleted = true };
            }
        }

        return before - Count;
    }

    public IReadOnlyList<EpisodicMemorySearchResult> Recall(EpisodicMemoryQuery query)
    {
        ValidateQuery(query, EmbeddingWidth);

        var candidates = _entries
            .Where(entry => !entry.IsDeleted)
            .Where(entry => HasRequiredTag(entry, query.RequiredTag))
            .Select(entry => new
            {
                Entry = entry,
                Similarity = CosineSimilarity(query.Embedding, entry.Embedding)
            })
            .Where(item => item.Similarity >= query.MinimumSimilarity)
            .OrderByDescending(item => item.Similarity)
            .ThenByDescending(item => item.Entry.Timestamp)
            .ThenBy(item => item.Entry.InsertedIndex)
            .ThenBy(item => item.Entry.Id, StringComparer.Ordinal)
            .Take(query.TopK)
            .ToArray();

        return candidates
            .Select((item, index) => new EpisodicMemorySearchResult(
                CloneEntry(item.Entry),
                item.Similarity,
                index + 1))
            .ToArray();
    }

    public torch.Tensor ToEmbeddingTensor()
    {
        if (Count == 0)
        {
            throw new InvalidOperationException("At least one memory entry is required.");
        }

        var active = _entries.Where(entry => !entry.IsDeleted).ToArray();
        return torch.tensor(active.SelectMany(entry => entry.Embedding).ToArray(), dtype: torch.float32)
            .reshape(active.Length, EmbeddingWidth);
    }

    public string ToEntryCsv()
    {
        var lines = new List<string>
        {
            "id,deleted,inserted_index,content,tags,metadata,vector"
        };

        foreach (var entry in _entries.OrderBy(entry => entry.InsertedIndex))
        {
            lines.Add(string.Create(
                CultureInfo.InvariantCulture,
                $"{Csv(entry.Id)},{(entry.IsDeleted ? "true" : "false")},{entry.InsertedIndex},{Csv(entry.Text)},{Csv(string.Join("|", entry.Tags))},{Csv(FormatMetadata(entry.Metadata))},{Csv(FormatVector(entry.Embedding))}"));
        }

        return string.Join(Environment.NewLine, lines) + Environment.NewLine;
    }

    public static string ToSearchCsv(IReadOnlyList<EpisodicMemorySearchResult> results)
    {
        var lines = new List<string>
        {
            "rank,id,score,content,tags,metadata"
        };

        foreach (var result in results)
        {
            lines.Add(string.Create(
                CultureInfo.InvariantCulture,
                $"{result.Rank},{Csv(result.Entry.Id)},{result.Similarity:0.######},{Csv(result.Entry.Text)},{Csv(string.Join("|", result.Entry.Tags))},{Csv(FormatMetadata(result.Entry.Metadata))}"));
        }

        return string.Join(Environment.NewLine, lines) + Environment.NewLine;
    }

    public static string FormatReport(EpisodicMemoryReport report)
    {
        var moeTop = report.MoeResults.Count == 0 ? "none" : report.MoeResults[0].Entry.Id;
        var memoryTop = report.MemoryResults.Count == 0 ? "none" : report.MemoryResults[0].Entry.Id;
        var moeSimilarity = report.MoeResults.Count == 0 ? 0.0f : report.MoeResults[0].Similarity;

        return string.Create(
            CultureInfo.InvariantCulture,
            $"episodic memory: entries={report.Entries}, active={report.Active}, deleted={report.Deleted}, width={report.EmbeddingWidth}, top_k={report.TopK}, pruned={report.PrunedCount}, best={moeTop}, score={moeSimilarity:0.###}, memory_top={memoryTop}");
    }

    public static float CosineSimilarity(
        IReadOnlyList<float> left,
        IReadOnlyList<float> right)
    {
        if (left.Count != right.Count)
        {
            throw new ArgumentException("Vectors must have the same length.", nameof(right));
        }

        var dot = 0.0f;
        var leftNormSquared = 0.0f;
        var rightNormSquared = 0.0f;

        for (var index = 0; index < left.Count; index++)
        {
            var leftValue = left[index];
            var rightValue = right[index];
            if (!IsFinite(leftValue) || !IsFinite(rightValue))
            {
                throw new ArgumentException("Vectors must contain finite values.");
            }

            dot += leftValue * rightValue;
            leftNormSquared += leftValue * leftValue;
            rightNormSquared += rightValue * rightValue;
        }

        if (leftNormSquared == 0.0f || rightNormSquared == 0.0f)
        {
            throw new ArgumentException("Cosine similarity is undefined for zero vectors.");
        }

        return dot / MathF.Sqrt(leftNormSquared * rightNormSquared);
    }

    private static EpisodicMemoryEntry NewEntry(
        string id,
        string text,
        float[] embedding,
        string[] tags,
        int timestamp,
        float confidence,
        IReadOnlyDictionary<string, string> metadata)
    {
        return new EpisodicMemoryEntry(id, text, embedding, tags, timestamp, confidence, metadata);
    }

    private static bool HasRequiredTag(
        EpisodicMemoryEntry entry,
        string? requiredTag)
    {
        return string.IsNullOrWhiteSpace(requiredTag)
            || entry.Tags.Any(tag => string.Equals(tag, requiredTag, StringComparison.OrdinalIgnoreCase));
    }

    private static EpisodicMemoryEntry CloneEntry(EpisodicMemoryEntry entry)
    {
        return entry with
        {
            Embedding = entry.Embedding.ToArray(),
            Tags = entry.Tags
                .Select(tag => tag.Trim())
                .ToArray(),
            Metadata = entry.Metadata is null
                ? new Dictionary<string, string>()
                : entry.Metadata.ToDictionary(pair => pair.Key, pair => pair.Value, StringComparer.Ordinal)
        };
    }

    private static void ValidateEntry(
        EpisodicMemoryEntry entry,
        int embeddingWidth)
    {
        if (string.IsNullOrWhiteSpace(entry.Id))
        {
            throw new ArgumentException("Entry id must not be empty.", nameof(entry));
        }

        if (string.IsNullOrWhiteSpace(entry.Text))
        {
            throw new ArgumentException("Entry text must not be empty.", nameof(entry));
        }

        if (entry.Timestamp < 0)
        {
            throw new ArgumentOutOfRangeException(nameof(entry), "Timestamp must be non-negative.");
        }

        if (entry.Confidence < 0.0f || entry.Confidence > 1.0f || !IsFinite(entry.Confidence))
        {
            throw new ArgumentOutOfRangeException(nameof(entry), "Confidence must be in [0, 1].");
        }

        ValidateEmbedding(entry.Embedding, embeddingWidth, nameof(entry));

        if (entry.Tags.Any(tag => string.IsNullOrWhiteSpace(tag)))
        {
            throw new ArgumentException("Tags must not contain empty values.", nameof(entry));
        }

        if (entry.Metadata is not null
            && entry.Metadata.Any(pair => string.IsNullOrWhiteSpace(pair.Key) || pair.Value is null))
        {
            throw new ArgumentException("Metadata keys must not be empty and values must not be null.", nameof(entry));
        }
    }

    private static void ValidateQuery(
        EpisodicMemoryQuery query,
        int embeddingWidth)
    {
        if (query.TopK <= 0)
        {
            throw new ArgumentOutOfRangeException(nameof(query), "TopK must be positive.");
        }

        if (!IsFinite(query.MinimumSimilarity))
        {
            throw new ArgumentOutOfRangeException(nameof(query), "Minimum similarity must be finite.");
        }

        ValidateEmbedding(query.Embedding, embeddingWidth, nameof(query));
    }

    private static void ValidateEmbedding(
        IReadOnlyList<float> embedding,
        int embeddingWidth,
        string parameterName)
    {
        if (embedding.Count != embeddingWidth)
        {
            throw new ArgumentException("Embedding width is inconsistent.", parameterName);
        }

        var squaredNorm = 0.0f;
        foreach (var value in embedding)
        {
            if (!IsFinite(value))
            {
                throw new ArgumentException("Embedding values must be finite.", parameterName);
            }

            squaredNorm += value * value;
        }

        if (squaredNorm == 0.0f)
        {
            throw new ArgumentException("Embedding must not be the zero vector.", parameterName);
        }
    }

    private static bool IsFinite(float value)
    {
        return !float.IsNaN(value) && !float.IsInfinity(value);
    }

    private static string FormatMetadata(IReadOnlyDictionary<string, string>? metadata)
    {
        if (metadata is null || metadata.Count == 0)
        {
            return "";
        }

        return string.Join(
            "|",
            metadata
                .OrderBy(pair => pair.Key, StringComparer.Ordinal)
                .Select(pair => $"{pair.Key}={pair.Value}"));
    }

    private static string FormatVector(IReadOnlyList<float> vector)
    {
        return string.Join(
            "|",
            vector.Select(value => value.ToString("0.######", CultureInfo.InvariantCulture)));
    }

    private static string Csv(string value)
    {
        if (!value.Contains(',') && !value.Contains('"') && !value.Contains('\n') && !value.Contains('\r'))
        {
            return value;
        }

        return "\"" + value.Replace("\"", "\"\"", StringComparison.Ordinal) + "\"";
    }
}
