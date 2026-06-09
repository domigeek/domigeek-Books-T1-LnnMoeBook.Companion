using LnnMoeBook.Examples.Concepts;

namespace LnnMoeBook.Tests.Concepts;

public sealed class ConceptGraphTests
{
    [Fact]
    public void GenerateSyntheticGraphIsDeterministic()
    {
        var first = ConceptGraph.GenerateSyntheticGraph();
        var second = ConceptGraph.GenerateSyntheticGraph();

        Assert.Equal(11, first.NodeCount);
        Assert.Equal(14, first.RelationCount);
        Assert.Equal(first.Nodes.Select(node => node.Id), second.Nodes.Select(node => node.Id));
        Assert.Equal(first.Relations.Select(relation => relation.Id), second.Relations.Select(relation => relation.Id));
        Assert.Equal(first.Relations.Select(relation => relation.Weight), second.Relations.Select(relation => relation.Weight));
    }

    [Fact]
    public void ConceptIdsAreUniqueAndStable()
    {
        var graph = ConceptGraph.GenerateSyntheticGraph();
        var ids = graph.Nodes.Select(node => node.Id).ToArray();

        Assert.Equal(ids.Length, ids.Distinct(StringComparer.Ordinal).Count());
        Assert.Equal("embedding", ids[0]);
        Assert.Equal("evolving-memory", ids[^2]);
        Assert.Equal("rag-context", ids[^1]);
    }

    [Fact]
    public void RelationsReferenceExistingConcepts()
    {
        var graph = ConceptGraph.GenerateSyntheticGraph();
        var ids = graph.Nodes.Select(node => node.Id).ToHashSet(StringComparer.Ordinal);

        Assert.All(graph.Relations, relation =>
        {
            Assert.Contains(relation.SourceId, ids);
            Assert.Contains(relation.TargetId, ids);
            Assert.NotEqual(relation.SourceId, relation.TargetId);
            Assert.InRange(relation.Weight, 0.0001f, 1.0f);
        });
    }

    [Fact]
    public void GetNodeReturnsDefensiveCopies()
    {
        var graph = ConceptGraph.GenerateSyntheticGraph();
        var node = graph.GetNode("concept-graph");
        var tags = Assert.IsType<string[]>(node.Tags);

        tags[0] = "mutated";

        Assert.DoesNotContain("mutated", graph.GetNode("concept-graph").Tags);
    }

    [Fact]
    public void NodesWithTagIsCaseInsensitive()
    {
        var graph = ConceptGraph.GenerateSyntheticGraph();

        var memoryNodes = graph.NodesWithTag("MEMORY");

        Assert.Equal(5, memoryNodes.Count);
        Assert.Contains(memoryNodes, node => node.Id == "semantic-memory");
        Assert.All(memoryNodes, node => Assert.Contains(node.Tags, tag => string.Equals(tag, "memory", StringComparison.OrdinalIgnoreCase)));
    }

    [Fact]
    public void SubgraphByTagKeepsOnlyInternalRelations()
    {
        var graph = ConceptGraph.GenerateSyntheticGraph();

        var subgraph = graph.SubgraphByTag("memory");
        var ids = subgraph.Nodes.Select(node => node.Id).ToHashSet(StringComparer.Ordinal);

        Assert.Equal("memory", subgraph.Tag);
        Assert.Equal(5, subgraph.Nodes.Count);
        Assert.Equal(5, subgraph.Relations.Count);
        Assert.All(subgraph.Relations, relation =>
        {
            Assert.Contains(relation.SourceId, ids);
            Assert.Contains(relation.TargetId, ids);
        });
    }

    [Fact]
    public void OutgoingNeighborsReturnDirectRelationsInInsertionOrder()
    {
        var graph = ConceptGraph.GenerateSyntheticGraph();

        var neighbors = graph.Neighbors("concept-graph", ConceptTraversalDirection.Outgoing);

        Assert.Equal(2, neighbors.Count);
        Assert.Equal("relation", neighbors[0].Node.Id);
        Assert.Equal("abstract-concept", neighbors[1].Node.Id);
        Assert.All(neighbors, neighbor => Assert.Equal("outgoing", neighbor.Direction));
    }

    [Fact]
    public void AnyDirectionNeighborsIncludeIncomingAndOutgoingRelations()
    {
        var graph = ConceptGraph.GenerateSyntheticGraph();

        var neighbors = graph.Neighbors("concept-graph");

        Assert.Contains(neighbors, neighbor => neighbor.Node.Id == "latent-space" && neighbor.Direction == "incoming");
        Assert.Contains(neighbors, neighbor => neighbor.Node.Id == "relation" && neighbor.Direction == "outgoing");
        Assert.Contains(neighbors, neighbor => neighbor.Node.Id == "evolving-memory" && neighbor.Direction == "incoming");
    }

    [Fact]
    public void ShortestPathFindsReachableConcepts()
    {
        var graph = ConceptGraph.GenerateSyntheticGraph();

        var path = graph.ShortestPath("embedding", "evolving-memory", maxDepth: 6);

        Assert.True(path.Found);
        Assert.Equal(5, path.HopCount);
        Assert.InRange(path.Score, 0.34f, 0.35f);
        Assert.Equal(
            "embedding->latent-space->concept-graph->abstract-concept->semantic-memory->evolving-memory",
            path.NodeIdSequence);
        Assert.Equal("maps-to->supports->models->stored-as->versioned-by", path.RelationTypeSequence);
    }

    [Fact]
    public void ShortestPathRespectsMaxDepth()
    {
        var graph = ConceptGraph.GenerateSyntheticGraph();

        var path = graph.ShortestPath("embedding", "evolving-memory", maxDepth: 4);

        Assert.False(path.Found);
        Assert.Equal("none", path.NodeIdSequence);
        Assert.Empty(path.Relations);
    }

    [Fact]
    public void ShortestPathCanFilterRelationTypes()
    {
        var graph = ConceptGraph.GenerateSyntheticGraph();

        var allowed = new[] { "maps-to", "supports", "models", "stored-as", "versioned-by" };
        var path = graph.ShortestPath(
            "embedding",
            "evolving-memory",
            maxDepth: 6,
            allowedRelationTypes: allowed);
        var blocked = graph.ShortestPath(
            "embedding",
            "evolving-memory",
            maxDepth: 6,
            allowedRelationTypes: new[] { "retrieves" });

        Assert.True(path.Found);
        Assert.False(blocked.Found);
    }

    [Fact]
    public void ShortestPathHandlesSourceEqualTarget()
    {
        var graph = ConceptGraph.GenerateSyntheticGraph();

        var path = graph.ShortestPath("embedding", "embedding", maxDepth: 0);

        Assert.True(path.Found);
        Assert.Equal(0, path.HopCount);
        Assert.Equal("embedding", path.NodeIdSequence);
    }

    [Fact]
    public void ExpandReturnsReachableConceptsWithoutRevisitingCycles()
    {
        var graph = ConceptGraph.GenerateSyntheticGraph();

        var expansions = graph.Expand("concept-graph", maxDepth: 3);
        var ids = expansions.Select(expansion => expansion.Node.Id).ToArray();

        Assert.Contains("evolving-memory", ids);
        Assert.Equal(ids.Length, ids.Distinct(StringComparer.Ordinal).Count());
        Assert.All(expansions, expansion => Assert.InRange(expansion.Depth, 1, 3));
    }

    [Fact]
    public void NodeCsvContainsStableHeaderAndRows()
    {
        var graph = ConceptGraph.GenerateSyntheticGraph();

        var csv = graph.ToNodeCsv();
        var lines = csv
            .Split('\n', StringSplitOptions.RemoveEmptyEntries)
            .Select(line => line.TrimEnd('\r'))
            .ToArray();

        Assert.Equal(12, lines.Length);
        Assert.Equal("id,label,kind,version,tags,description,metadata", lines[0]);
        Assert.StartsWith("embedding,Embedding,representation,1,vector|representation,", lines[1], StringComparison.Ordinal);
        Assert.Contains("chapter=27.1", lines[1]);
    }

    [Fact]
    public void RelationCsvContainsStableHeaderAndRows()
    {
        var graph = ConceptGraph.GenerateSyntheticGraph();

        var csv = graph.ToRelationCsv();
        var lines = csv
            .Split('\n', StringSplitOptions.RemoveEmptyEntries)
            .Select(line => line.TrimEnd('\r'))
            .ToArray();

        Assert.Equal(15, lines.Length);
        Assert.Equal("id,source,target,type,weight,directed,evidence,timestamp,metadata", lines[0]);
        Assert.StartsWith("rel-embedding-latent,embedding,latent-space,maps-to,0.92,true,", lines[1], StringComparison.Ordinal);
        Assert.Contains("source=synthetic", lines[1]);
    }

    [Fact]
    public void PathCsvContainsStableHeaderAndRows()
    {
        var graph = ConceptGraph.GenerateSyntheticGraph();
        var path = graph.ShortestPath("embedding", "evolving-memory", maxDepth: 6);

        var csv = ConceptGraph.ToPathCsv(path);
        var lines = csv
            .Split('\n', StringSplitOptions.RemoveEmptyEntries)
            .Select(line => line.TrimEnd('\r'))
            .ToArray();

        Assert.Equal(2, lines.Length);
        Assert.Equal("rank,source,target,hops,score,path_nodes,path_relations", lines[0]);
        Assert.StartsWith("1,embedding,evolving-memory,5,", lines[1], StringComparison.Ordinal);
        Assert.Contains("maps-to->supports->models->stored-as->versioned-by", lines[1]);
    }

    [Fact]
    public void EmptyPathCsvContainsOnlyHeader()
    {
        var graph = ConceptGraph.GenerateSyntheticGraph();
        var path = graph.ShortestPath("embedding", "evolving-memory", maxDepth: 2);

        Assert.Equal(
            "rank,source,target,hops,score,path_nodes,path_relations" + Environment.NewLine,
            ConceptGraph.ToPathCsv(path));
    }

    [Fact]
    public void MermaidExportContainsStableNodesAndEdges()
    {
        var graph = ConceptGraph.GenerateSyntheticGraph();

        var mermaid = graph.ToMermaid();

        Assert.StartsWith("graph LR", mermaid, StringComparison.Ordinal);
        Assert.Contains("concept_graph[\"Graphe conceptuel\"]", mermaid);
        Assert.Contains("embedding -->|\"maps-to\"| latent_space", mermaid);
    }

    [Fact]
    public void RunDefaultBuildsExpectedReportData()
    {
        var report = ConceptGraph.RunDefault();

        Assert.Equal(11, report.Graph.NodeCount);
        Assert.Equal(14, report.Graph.RelationCount);
        Assert.True(report.EvolutionPath.Found);
        Assert.Equal(5, report.MemorySubgraph.Nodes.Count);
        Assert.NotEmpty(report.Neighborhood);
    }

    [Fact]
    public void FormatReportContainsStableFields()
    {
        var text = ConceptGraph.FormatReport(ConceptGraph.RunDefault());

        Assert.Contains("concept graph", text);
        Assert.Contains("concepts=11", text);
        Assert.Contains("relations=14", text);
        Assert.Contains("path=embedding->latent-space->concept-graph", text);
        Assert.Contains("hops=5", text);
        Assert.Contains("score=", text);
        Assert.Contains("memory_concepts=5", text);
        Assert.Contains("memory_relations=5", text);
        Assert.Contains("neighborhood=", text);
    }

    [Fact]
    public void FromRejectsInvalidNodes()
    {
        var graph = ConceptGraph.GenerateSyntheticGraph();
        var nodes = graph.Nodes.ToArray();

        Assert.Throws<ArgumentException>(() =>
            ConceptGraph.From(Array.Empty<ConceptNode>(), graph.Relations));
        Assert.Throws<ArgumentException>(() =>
            ConceptGraph.From(nodes.Append(nodes[0]).ToArray(), graph.Relations));
        Assert.Throws<ArgumentException>(() =>
            ConceptGraph.From(new[] { nodes[0] with { Id = "" } }, graph.Relations));
        Assert.Throws<ArgumentException>(() =>
            ConceptGraph.From(new[] { nodes[0] with { Tags = Array.Empty<string>() } }, graph.Relations));
        Assert.Throws<ArgumentOutOfRangeException>(() =>
            ConceptGraph.From(new[] { nodes[0] with { Version = 0 } }, graph.Relations));
    }

    [Fact]
    public void FromRejectsInvalidRelations()
    {
        var graph = ConceptGraph.GenerateSyntheticGraph();
        var nodes = graph.Nodes;
        var relations = graph.Relations.ToArray();

        Assert.Throws<ArgumentException>(() =>
            ConceptGraph.From(nodes, Array.Empty<ConceptRelation>()));
        Assert.Throws<ArgumentException>(() =>
            ConceptGraph.From(nodes, relations.Append(relations[0]).ToArray()));
        Assert.Throws<ArgumentException>(() =>
            ConceptGraph.From(nodes, new[] { relations[0] with { TargetId = "missing" } }));
        Assert.Throws<ArgumentException>(() =>
            ConceptGraph.From(nodes, new[] { relations[0] with { Type = "" } }));
        Assert.Throws<ArgumentOutOfRangeException>(() =>
            ConceptGraph.From(nodes, new[] { relations[0] with { Weight = float.NaN } }));
        Assert.Throws<ArgumentException>(() =>
            ConceptGraph.From(nodes, new[] { relations[0] with { SourceId = relations[0].TargetId } }));
    }

    [Fact]
    public void TraversalRejectsInvalidArguments()
    {
        var graph = ConceptGraph.GenerateSyntheticGraph();

        Assert.Throws<KeyNotFoundException>(() => graph.GetNode("missing"));
        Assert.Throws<ArgumentException>(() => graph.NodesWithTag(""));
        Assert.Throws<ArgumentOutOfRangeException>(() => graph.Expand("concept-graph", maxDepth: 0));
        Assert.Throws<ArgumentOutOfRangeException>(() => graph.ShortestPath("embedding", "evolving-memory", maxDepth: -1));
        Assert.Throws<ArgumentException>(() =>
            graph.ShortestPath("embedding", "evolving-memory", allowedRelationTypes: Array.Empty<string>()));
    }
}
