using LnnMoeBook.Examples.Memory;

namespace LnnMoeBook.Tests.Memory;

public sealed class EpisodicMemoryStoreTests
{
    [Fact]
    public void GenerateSyntheticEpisodesIsDeterministic()
    {
        var first = EpisodicMemoryStore.GenerateSyntheticEpisodes();
        var second = EpisodicMemoryStore.GenerateSyntheticEpisodes();

        Assert.Equal(6, first.Count);
        Assert.Equal(first.Select(entry => entry.Id), second.Select(entry => entry.Id));
        Assert.Equal(first.SelectMany(entry => entry.Embedding), second.SelectMany(entry => entry.Embedding));
        Assert.Equal(first.Select(entry => entry.Timestamp), second.Select(entry => entry.Timestamp));
    }

    [Fact]
    public void FromEntriesBuildsStoreWithExpectedWidth()
    {
        var store = EpisodicMemoryStore.FromEntries(EpisodicMemoryStore.GenerateSyntheticEpisodes());

        Assert.Equal(6, store.Count);
        Assert.Equal(6, store.TotalCount);
        Assert.Equal(0, store.DeletedCount);
        Assert.Equal(4, store.EmbeddingWidth);
        Assert.Equal(6, store.Entries.Count);
        Assert.Equal(6, store.AuditEntries.Count);
    }

    [Fact]
    public void EntriesAreReturnedAsCopies()
    {
        var source = EpisodicMemoryStore.GenerateSyntheticEpisodes()[0];
        var store = EpisodicMemoryStore.FromEntries(new[] { source });

        source.Embedding[0] = -10.0f;
        var stored = store.Entries[0];
        stored.Embedding[0] = -20.0f;

        Assert.NotEqual(-10.0f, store.Entries[0].Embedding[0]);
        Assert.NotEqual(-20.0f, store.Entries[0].Embedding[0]);
    }

    [Fact]
    public void RecallReturnsTopKByCosineSimilarity()
    {
        var store = EpisodicMemoryStore.FromEntries(EpisodicMemoryStore.GenerateSyntheticEpisodes());

        var results = store.Recall(new EpisodicMemoryQuery(
            Embedding: new[] { 1.0f, 0.12f, 0.02f, 0.0f },
            TopK: 2));

        Assert.Equal(2, results.Count);
        Assert.Equal("evt-moe-routing", results[0].Entry.Id);
        Assert.Equal(1, results[0].Rank);
        Assert.True(results[0].Similarity >= results[1].Similarity);
    }

    [Fact]
    public void RecallFiltersByRequiredTag()
    {
        var store = EpisodicMemoryStore.FromEntries(EpisodicMemoryStore.GenerateSyntheticEpisodes());

        var results = store.Recall(new EpisodicMemoryQuery(
            Embedding: new[] { 1.0f, 0.12f, 0.02f, 0.0f },
            TopK: 4,
            RequiredTag: "MEMORY"));

        Assert.Equal(2, results.Count);
        Assert.All(results, result => Assert.Contains(result.Entry.Tags, tag => string.Equals(tag, "memory", StringComparison.OrdinalIgnoreCase)));
    }

    [Fact]
    public void RecallAppliesMinimumSimilarity()
    {
        var store = EpisodicMemoryStore.FromEntries(EpisodicMemoryStore.GenerateSyntheticEpisodes());

        var results = store.Recall(new EpisodicMemoryQuery(
            Embedding: new[] { 1.0f, 0.12f, 0.02f, 0.0f },
            TopK: 6,
            MinimumSimilarity: 0.95f));

        Assert.True(results.Count < store.Count);
        Assert.All(results, result => Assert.True(result.Similarity >= 0.95f));
    }

    [Fact]
    public void RecallReturnsAtMostAvailableEntriesWhenTopKIsLarge()
    {
        var store = EpisodicMemoryStore.FromEntries(EpisodicMemoryStore.GenerateSyntheticEpisodes());

        var results = store.Recall(new EpisodicMemoryQuery(
            Embedding: new[] { 0.0f, 0.08f, 0.95f, 0.22f },
            TopK: 100,
            RequiredTag: "memory"));

        Assert.Equal(2, results.Count);
    }

    [Fact]
    public void RecallTieBreaksByRecencyThenId()
    {
        var store = new EpisodicMemoryStore(2);
        store.Upsert(new EpisodicMemoryEntry("b", "same vector newer b", [1.0f, 0.0f], ["tie"], 10, 0.9f));
        store.Upsert(new EpisodicMemoryEntry("a", "same vector newer a", [1.0f, 0.0f], ["tie"], 10, 0.9f));
        store.Upsert(new EpisodicMemoryEntry("old", "same vector older", [1.0f, 0.0f], ["tie"], 1, 0.9f));

        var results = store.Recall(new EpisodicMemoryQuery(
            Embedding: [1.0f, 0.0f],
            TopK: 3));

        Assert.Equal(new[] { "b", "a", "old" }, results.Select(result => result.Entry.Id).ToArray());
    }

    [Fact]
    public void UpsertReplacesEntryWithSameId()
    {
        var store = EpisodicMemoryStore.FromEntries(EpisodicMemoryStore.GenerateSyntheticEpisodes());

        store.Upsert(new EpisodicMemoryEntry(
            "evt-moe-routing",
            "Updated routing note",
            [1.0f, 0.0f, 0.0f, 0.0f],
            ["moe", "updated"],
            Timestamp: 50,
            Confidence: 0.99f));

        Assert.Equal(6, store.Count);
        var updated = store.Entries.Single(entry => entry.Id == "evt-moe-routing");
        Assert.Equal("Updated routing note", updated.Text);
        Assert.Equal(50, updated.Timestamp);
        Assert.Contains("updated", updated.Tags);
    }

    [Fact]
    public void RemoveDeletesKnownEntryOnly()
    {
        var store = EpisodicMemoryStore.FromEntries(EpisodicMemoryStore.GenerateSyntheticEpisodes());

        Assert.True(store.Remove("evt-moe-balance"));
        Assert.False(store.Remove("missing"));
        Assert.Equal(5, store.Count);
        Assert.Equal(6, store.TotalCount);
        Assert.Equal(1, store.DeletedCount);
        Assert.DoesNotContain(store.Entries, entry => entry.Id == "evt-moe-balance");
        Assert.Contains(store.AuditEntries, entry => entry.Id == "evt-moe-balance" && entry.IsDeleted);
    }

    [Fact]
    public void RemoveIsIdempotentForDeletedEntry()
    {
        var store = EpisodicMemoryStore.FromEntries(EpisodicMemoryStore.GenerateSyntheticEpisodes());

        Assert.True(store.Remove("evt-moe-balance"));
        Assert.False(store.Remove("evt-moe-balance"));
        Assert.Equal(5, store.Count);
        Assert.Equal(1, store.DeletedCount);
    }

    [Fact]
    public void PruneOlderThanRemovesStaleEpisodes()
    {
        var store = EpisodicMemoryStore.FromEntries(EpisodicMemoryStore.GenerateSyntheticEpisodes());

        var removed = store.PruneOlderThan(minTimestamp: 15);

        Assert.Equal(2, removed);
        Assert.Equal(4, store.Count);
        Assert.Equal(2, store.DeletedCount);
        Assert.DoesNotContain(store.Entries, entry => entry.Timestamp < 15);
        Assert.Contains(store.AuditEntries, entry => entry.Id == "evt-moe-routing" && entry.IsDeleted);
    }

    [Fact]
    public void EmbeddingsCanBeViewedAsTorchSharpTensor()
    {
        var store = EpisodicMemoryStore.FromEntries(EpisodicMemoryStore.GenerateSyntheticEpisodes());

        using var tensor = store.ToEmbeddingTensor();

        Assert.Equal(new long[] { 6, 4 }, tensor.shape.ToArray());
    }

    [Fact]
    public void CosineSimilarityComputesExpectedValues()
    {
        var identical = EpisodicMemoryStore.CosineSimilarity([1.0f, 0.0f], [1.0f, 0.0f]);
        var orthogonal = EpisodicMemoryStore.CosineSimilarity([1.0f, 0.0f], [0.0f, 1.0f]);

        Assert.Equal(1.0f, identical);
        Assert.Equal(0.0f, orthogonal);
    }

    [Fact]
    public void EntryCsvContainsStableHeaderAndRows()
    {
        var store = EpisodicMemoryStore.FromEntries(EpisodicMemoryStore.GenerateSyntheticEpisodes());

        var csv = store.ToEntryCsv();
        var lines = csv
            .Split('\n', StringSplitOptions.RemoveEmptyEntries)
            .Select(line => line.TrimEnd('\r'))
            .ToArray();

        Assert.Equal(7, lines.Length);
        Assert.Equal("id,deleted,inserted_index,content,tags,metadata,vector", lines[0]);
        Assert.StartsWith("evt-moe-routing,false,0,Top-k router", lines[1], StringComparison.Ordinal);
        Assert.Contains("chapter=19|kind=routing", lines[1], StringComparison.Ordinal);
        Assert.EndsWith(",0.96|0.12|0.03|0", lines[1], StringComparison.Ordinal);
    }

    [Fact]
    public void EntryCsvContainsDeletedEntriesForAudit()
    {
        var store = EpisodicMemoryStore.FromEntries(EpisodicMemoryStore.GenerateSyntheticEpisodes());

        store.Remove("evt-moe-balance");
        var csv = store.ToEntryCsv();

        Assert.Contains("evt-moe-balance,true,1,", csv, StringComparison.Ordinal);
    }

    [Fact]
    public void SearchCsvContainsStableHeaderAndRows()
    {
        var store = EpisodicMemoryStore.FromEntries(EpisodicMemoryStore.GenerateSyntheticEpisodes());
        var results = store.Recall(new EpisodicMemoryQuery(
            Embedding: new[] { 1.0f, 0.12f, 0.02f, 0.0f },
            TopK: 2));

        var csv = EpisodicMemoryStore.ToSearchCsv(results);
        var lines = csv
            .Split('\n', StringSplitOptions.RemoveEmptyEntries)
            .Select(line => line.TrimEnd('\r'))
            .ToArray();

        Assert.Equal(3, lines.Length);
        Assert.Equal("rank,id,score,content,tags,metadata", lines[0]);
        Assert.StartsWith("1,evt-moe-routing,", lines[1], StringComparison.Ordinal);
        Assert.Contains("chapter=19|kind=routing", lines[1], StringComparison.Ordinal);
    }

    [Fact]
    public void EmptySearchCsvContainsOnlyHeader()
    {
        var csv = EpisodicMemoryStore.ToSearchCsv(Array.Empty<EpisodicMemorySearchResult>());

        Assert.Equal("rank,id,score,content,tags,metadata" + Environment.NewLine, csv);
    }

    [Fact]
    public void RunDefaultPrunesOldEntriesAfterRecall()
    {
        var report = EpisodicMemoryStore.RunDefault();

        Assert.Equal(6, report.Entries);
        Assert.Equal(4, report.Active);
        Assert.Equal(2, report.Deleted);
        Assert.Equal(4, report.EmbeddingWidth);
        Assert.Equal(3, report.TopK);
        Assert.Equal(2, report.PrunedCount);
        Assert.Equal("evt-moe-routing", report.MoeResults[0].Entry.Id);
        Assert.Equal("evt-rag-source", report.MemoryResults[0].Entry.Id);
    }

    [Fact]
    public void FormatReportContainsStableFields()
    {
        var text = EpisodicMemoryStore.FormatReport(EpisodicMemoryStore.RunDefault());

        Assert.Contains("episodic memory", text);
        Assert.Contains("entries=6", text);
        Assert.Contains("active=4", text);
        Assert.Contains("deleted=2", text);
        Assert.Contains("width=4", text);
        Assert.Contains("top_k=3", text);
        Assert.Contains("pruned=2", text);
        Assert.Contains("best=evt-moe-routing", text);
        Assert.Contains("score=", text);
        Assert.Contains("memory_top=evt-rag-source", text);
    }

    [Fact]
    public void MetadataIsPreservedInResultsAndCsv()
    {
        var store = EpisodicMemoryStore.FromEntries(EpisodicMemoryStore.GenerateSyntheticEpisodes());

        var result = store.Recall(new EpisodicMemoryQuery(
            Embedding: new[] { 1.0f, 0.12f, 0.02f, 0.0f },
            TopK: 1))[0];

        Assert.Equal("19", result.Entry.Metadata?["chapter"]);
        Assert.Contains("kind=routing", EpisodicMemoryStore.ToSearchCsv(new[] { result }), StringComparison.Ordinal);
    }

    [Fact]
    public void FromEntriesRejectsEmptyInput()
    {
        Assert.Throws<ArgumentException>(() =>
            EpisodicMemoryStore.FromEntries(Array.Empty<EpisodicMemoryEntry>()));
    }

    [Fact]
    public void UpsertRejectsInvalidEntry()
    {
        var store = new EpisodicMemoryStore(2);

        Assert.Throws<ArgumentException>(() =>
            store.Upsert(new EpisodicMemoryEntry("", "text", [1.0f, 0.0f], ["tag"], 0, 0.5f)));
        Assert.Throws<ArgumentException>(() =>
            store.Upsert(new EpisodicMemoryEntry("id", "", [1.0f, 0.0f], ["tag"], 0, 0.5f)));
        Assert.Throws<ArgumentException>(() =>
            store.Upsert(new EpisodicMemoryEntry("id", "text", [1.0f], ["tag"], 0, 0.5f)));
        Assert.Throws<ArgumentException>(() =>
            store.Upsert(new EpisodicMemoryEntry("id", "text", [0.0f, 0.0f], ["tag"], 0, 0.5f)));
    }

    [Fact]
    public void UpsertRejectsInvalidMetadata()
    {
        var store = new EpisodicMemoryStore(2);

        Assert.Throws<ArgumentOutOfRangeException>(() =>
            store.Upsert(new EpisodicMemoryEntry("id", "text", [1.0f, 0.0f], ["tag"], -1, 0.5f)));
        Assert.Throws<ArgumentOutOfRangeException>(() =>
            store.Upsert(new EpisodicMemoryEntry("id", "text", [1.0f, 0.0f], ["tag"], 1, 1.1f)));
        Assert.Throws<ArgumentException>(() =>
            store.Upsert(new EpisodicMemoryEntry("id", "text", [1.0f, 0.0f], [""], 1, 0.5f)));
    }

    [Fact]
    public void RecallRejectsInvalidQuery()
    {
        var store = EpisodicMemoryStore.FromEntries(EpisodicMemoryStore.GenerateSyntheticEpisodes());

        Assert.Throws<ArgumentOutOfRangeException>(() =>
            store.Recall(new EpisodicMemoryQuery([1.0f, 0.0f, 0.0f, 0.0f], TopK: 0)));
        Assert.Throws<ArgumentException>(() =>
            store.Recall(new EpisodicMemoryQuery([1.0f, 0.0f], TopK: 1)));
        Assert.Throws<ArgumentException>(() =>
            store.Recall(new EpisodicMemoryQuery([0.0f, 0.0f, 0.0f, 0.0f], TopK: 1)));
        Assert.Throws<ArgumentException>(() =>
            store.Recall(new EpisodicMemoryQuery([float.NaN, 0.0f, 0.0f, 0.0f], TopK: 1)));
    }

    [Fact]
    public void RemoveRejectsEmptyId()
    {
        var store = new EpisodicMemoryStore(2);

        Assert.Throws<ArgumentException>(() => store.Remove(""));
    }

    [Fact]
    public void PruneRejectsNegativeTimestamp()
    {
        var store = new EpisodicMemoryStore(2);

        Assert.Throws<ArgumentOutOfRangeException>(() =>
            store.PruneOlderThan(minTimestamp: -1));
    }

    [Fact]
    public void EmbeddingTensorRejectsEmptyStore()
    {
        var store = new EpisodicMemoryStore(2);

        Assert.Throws<InvalidOperationException>(() => store.ToEmbeddingTensor());
    }

    [Fact]
    public void CosineSimilarityRejectsInvalidVectors()
    {
        Assert.Throws<ArgumentException>(() =>
            EpisodicMemoryStore.CosineSimilarity([1.0f], [1.0f, 0.0f]));
        Assert.Throws<ArgumentException>(() =>
            EpisodicMemoryStore.CosineSimilarity([0.0f, 0.0f], [1.0f, 0.0f]));
        Assert.Throws<ArgumentException>(() =>
            EpisodicMemoryStore.CosineSimilarity([float.PositiveInfinity, 0.0f], [1.0f, 0.0f]));
    }
}
