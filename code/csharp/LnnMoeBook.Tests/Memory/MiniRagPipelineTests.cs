using LnnMoeBook.Examples.Memory;

namespace LnnMoeBook.Tests.Memory;

public sealed class MiniRagPipelineTests
{
    [Fact]
    public void GenerateToyDocumentsIsDeterministic()
    {
        var first = MiniRagPipeline.GenerateToyDocuments();
        var second = MiniRagPipeline.GenerateToyDocuments();

        Assert.Equal(10, first.Count);
        Assert.Equal(first.Select(document => document.Id), second.Select(document => document.Id));
        Assert.Equal(first.SelectMany(document => document.Embedding), second.SelectMany(document => document.Embedding));
        Assert.All(first, document =>
        {
            Assert.Equal(MiniRagPipeline.EmbeddingWidth, document.Embedding.Length);
            Assert.False(string.IsNullOrWhiteSpace(document.Title));
            Assert.False(string.IsNullOrWhiteSpace(document.Content));
        });
    }

    [Fact]
    public void EmbedTextBuildsStableKeywordVector()
    {
        var embedding = MiniRagPipeline.EmbedText("RAG memory source context");

        Assert.Equal(4, embedding.Length);
        Assert.Equal(0.0f, embedding[0]);
        Assert.Equal(0.0f, embedding[1]);
        Assert.True(embedding[2] > 0.0f);
        Assert.Equal(0.0f, embedding[3]);
        Assert.Equal(embedding, MiniRagPipeline.EmbedText("RAG memory source context"));
    }

    [Fact]
    public void BuildMemoryStoreContainsAllDocuments()
    {
        var documents = MiniRagPipeline.GenerateToyDocuments();

        var store = MiniRagPipeline.BuildMemoryStore(documents);

        Assert.Equal(10, store.Count);
        Assert.Equal(4, store.EmbeddingWidth);
        Assert.Equal(documents.Select(document => document.Id), store.Entries.Select(entry => entry.Id));
    }

    [Fact]
    public void AnswerRetrievesSourcesForMemoryQuestion()
    {
        var store = MiniRagPipeline.BuildMemoryStore(MiniRagPipeline.GenerateToyDocuments());

        var response = MiniRagPipeline.Answer(
            store,
            new MiniRagRequest(
                Question: "Comment RAG recupere des sources de memoire pour construire un contexte ?",
                TopK: 3,
                RequiredTag: "memory",
                MinimumSimilarity: 0.10f));

        Assert.False(response.UsedFallback);
        Assert.Equal(3, response.Sources.Count);
        Assert.Equal("doc-rag-sources", response.Sources[0].Id);
        Assert.Contains("source(s)", response.Answer);
        Assert.Contains("doc-rag-sources", response.Answer);
        Assert.Contains("[S1] doc-rag-sources:", response.Context);
    }

    [Fact]
    public void AnswerCanFilterByTag()
    {
        var store = MiniRagPipeline.BuildMemoryStore(MiniRagPipeline.GenerateToyDocuments());

        var response = MiniRagPipeline.Answer(
            store,
            new MiniRagRequest(
                Question: "Comment fonctionne le routage MoE top-k ?",
                TopK: 2,
                RequiredTag: "moe"));

        Assert.Equal(2, response.Sources.Count);
        Assert.All(response.Sources, source => Assert.Contains(source.Tags, tag => string.Equals(tag, "moe", StringComparison.OrdinalIgnoreCase)));
    }

    [Fact]
    public void AnswerReturnsFallbackWhenNoSourceMatches()
    {
        var store = MiniRagPipeline.BuildMemoryStore(MiniRagPipeline.GenerateToyDocuments());

        var response = MiniRagPipeline.Answer(
            store,
            new MiniRagRequest(
                Question: "Question hors corpus",
                TopK: 2,
                RequiredTag: "absent"));

        Assert.True(response.UsedFallback);
        Assert.Empty(response.Sources);
        Assert.Equal("", response.Context);
        Assert.Contains("Aucune source locale", response.Answer);
    }

    [Fact]
    public void AnswerReturnsAtMostAvailableSources()
    {
        var store = MiniRagPipeline.BuildMemoryStore(MiniRagPipeline.GenerateToyDocuments());

        var response = MiniRagPipeline.Answer(
            store,
            new MiniRagRequest(
                Question: "memoire rag source contexte",
                TopK: 20,
                RequiredTag: "rag"));

        Assert.Equal(2, response.Sources.Count);
    }

    [Fact]
    public void MinimumSimilarityCanSuppressWeakSources()
    {
        var store = MiniRagPipeline.BuildMemoryStore(MiniRagPipeline.GenerateToyDocuments());

        var response = MiniRagPipeline.Answer(
            store,
            new MiniRagRequest(
                Question: "memoire rag source contexte",
                TopK: 10,
                RequiredTag: "memory",
                MinimumSimilarity: 1.01f));

        Assert.True(response.UsedFallback);
        Assert.Empty(response.Sources);
    }

    [Fact]
    public void DeletedMemoryEntriesAreNotReturned()
    {
        var store = MiniRagPipeline.BuildMemoryStore(MiniRagPipeline.GenerateToyDocuments());
        store.Remove("doc-rag-sources");

        var response = MiniRagPipeline.Answer(
            store,
            new MiniRagRequest(
                Question: "rag source contexte memoire",
                TopK: 3,
                RequiredTag: "memory"));

        Assert.DoesNotContain(response.Sources, source => source.Id == "doc-rag-sources");
        Assert.Contains(store.AuditEntries, entry => entry.Id == "doc-rag-sources" && entry.IsDeleted);
    }

    [Fact]
    public void ContextUsesStableRankedLines()
    {
        var store = MiniRagPipeline.BuildMemoryStore(MiniRagPipeline.GenerateToyDocuments());
        var response = MiniRagPipeline.Answer(
            store,
            new MiniRagRequest("rag source contexte memoire", TopK: 2, RequiredTag: "memory"));

        var lines = response.Context.Split(Environment.NewLine, StringSplitOptions.RemoveEmptyEntries);

        Assert.Equal(2, lines.Length);
        Assert.StartsWith("[S1] ", lines[0], StringComparison.Ordinal);
        Assert.StartsWith("[S2] ", lines[1], StringComparison.Ordinal);
    }

    [Fact]
    public void SourceScoresCanBeViewedAsTorchSharpTensor()
    {
        var store = MiniRagPipeline.BuildMemoryStore(MiniRagPipeline.GenerateToyDocuments());
        var response = MiniRagPipeline.Answer(
            store,
            new MiniRagRequest("rag source contexte memoire", TopK: 3, RequiredTag: "memory"));

        using var tensor = response.ToSourceScoreTensor();

        Assert.Equal(new long[] { 3 }, tensor.shape.ToArray());
    }

    [Fact]
    public void DocumentEmbeddingsCanBeViewedAsTorchSharpTensor()
    {
        var documents = MiniRagPipeline.GenerateToyDocuments();

        using var tensor = MiniRagPipeline.ToDocumentEmbeddingTensor(documents);

        Assert.Equal(new long[] { 10, 4 }, tensor.shape.ToArray());
    }

    [Fact]
    public void SourcesCsvContainsStableHeaderAndRows()
    {
        var store = MiniRagPipeline.BuildMemoryStore(MiniRagPipeline.GenerateToyDocuments());
        var response = MiniRagPipeline.Answer(
            store,
            new MiniRagRequest("rag source contexte memoire", TopK: 2, RequiredTag: "memory"));

        var csv = MiniRagPipeline.ToSourcesCsv(response);
        var lines = csv
            .Split('\n', StringSplitOptions.RemoveEmptyEntries)
            .Select(line => line.TrimEnd('\r'))
            .ToArray();

        Assert.Equal(3, lines.Length);
        Assert.Equal("rank,id,score,content,tags,metadata", lines[0]);
        Assert.StartsWith("1,doc-rag-sources,", lines[1], StringComparison.Ordinal);
        Assert.Contains("chapter=26|title=Sources RAG", lines[1]);
    }

    [Fact]
    public void AnswerCsvContainsStableHeaderAndRows()
    {
        var store = MiniRagPipeline.BuildMemoryStore(MiniRagPipeline.GenerateToyDocuments());
        var response = MiniRagPipeline.Answer(
            store,
            new MiniRagRequest("rag source contexte memoire", TopK: 2, RequiredTag: "memory"));

        var csv = MiniRagPipeline.ToAnswerCsv(response);
        var lines = csv
            .Split('\n', StringSplitOptions.RemoveEmptyEntries)
            .Select(line => line.TrimEnd('\r'))
            .ToArray();

        Assert.Equal(2, lines.Length);
        Assert.Equal("query,answer,source_count,sources,context", lines[0]);
        Assert.Contains("doc-rag-sources|", lines[1]);
        Assert.Contains("[S1] doc-rag-sources:", lines[1]);
    }

    [Fact]
    public void ContextCsvContainsStableHeaderAndRows()
    {
        var store = MiniRagPipeline.BuildMemoryStore(MiniRagPipeline.GenerateToyDocuments());
        var response = MiniRagPipeline.Answer(
            store,
            new MiniRagRequest("rag source contexte memoire", TopK: 2, RequiredTag: "memory"));

        var csv = MiniRagPipeline.ToContextCsv(response);
        var lines = csv
            .Split('\n', StringSplitOptions.RemoveEmptyEntries)
            .Select(line => line.TrimEnd('\r'))
            .ToArray();

        Assert.Equal(3, lines.Length);
        Assert.Equal("rank,context_line", lines[0]);
        Assert.StartsWith("1,", lines[1], StringComparison.Ordinal);
    }

    [Fact]
    public void EmptyContextCsvContainsOnlyHeader()
    {
        var response = new MiniRagResponse(
            new MiniRagRequest("question", TopK: 1),
            "fallback",
            "",
            Array.Empty<MiniRagSource>(),
            UsedFallback: true);

        Assert.Equal("rank,context_line" + Environment.NewLine, MiniRagPipeline.ToContextCsv(response));
    }

    [Fact]
    public void RunDefaultBuildsSourceBackedAndFallbackResponses()
    {
        var report = MiniRagPipeline.RunDefault();

        Assert.Equal(10, report.DocumentCount);
        Assert.Equal(4, report.EmbeddingWidth);
        Assert.False(report.RagResponse.UsedFallback);
        Assert.True(report.RagResponse.Sources.Count > 0);
        Assert.True(report.EmptyResponse.UsedFallback);
    }

    [Fact]
    public void FormatReportContainsStableFields()
    {
        var text = MiniRagPipeline.FormatReport(MiniRagPipeline.RunDefault());

        Assert.Contains("mini rag", text);
        Assert.Contains("documents=10", text);
        Assert.Contains("width=4", text);
        Assert.Contains("top_k=3", text);
        Assert.Contains("retrieved=", text);
        Assert.Contains("tag=memory", text);
        Assert.Contains("best=", text);
        Assert.Contains("has_sources=True", text);
        Assert.Contains("source_ids=", text);
        Assert.Contains("fallback=True", text);
    }

    [Fact]
    public void BuildMemoryStoreRejectsInvalidDocuments()
    {
        var documents = MiniRagPipeline.GenerateToyDocuments().ToArray();
        var duplicate = documents.Append(documents[0]).ToArray();

        Assert.Throws<ArgumentException>(() => MiniRagPipeline.BuildMemoryStore(Array.Empty<MiniRagDocument>()));
        Assert.Throws<ArgumentException>(() => MiniRagPipeline.BuildMemoryStore(duplicate));
        Assert.Throws<ArgumentException>(() =>
            MiniRagPipeline.BuildMemoryStore(new[]
            {
                documents[0] with { Id = "" }
            }));
        Assert.Throws<ArgumentException>(() =>
            MiniRagPipeline.BuildMemoryStore(new[]
            {
                documents[0] with { Embedding = [0.0f, 0.0f, 0.0f, 0.0f] }
            }));
    }

    [Fact]
    public void AnswerRejectsInvalidRequest()
    {
        var store = MiniRagPipeline.BuildMemoryStore(MiniRagPipeline.GenerateToyDocuments());

        Assert.Throws<ArgumentException>(() =>
            MiniRagPipeline.Answer(store, new MiniRagRequest("", TopK: 1)));
        Assert.Throws<ArgumentOutOfRangeException>(() =>
            MiniRagPipeline.Answer(store, new MiniRagRequest("rag", TopK: 0)));
        Assert.Throws<ArgumentOutOfRangeException>(() =>
            MiniRagPipeline.Answer(store, new MiniRagRequest("rag", TopK: 1, MinimumSimilarity: float.NaN)));
    }

    [Fact]
    public void EmbedTextRejectsEmptyText()
    {
        Assert.Throws<ArgumentException>(() => MiniRagPipeline.EmbedText(""));
    }
}
