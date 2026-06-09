using System.Globalization;

namespace LnnMoeBook.Examples.Concepts;

public enum ConceptTraversalDirection
{
    Outgoing,
    Incoming,
    Any
}

public sealed record ConceptNode(
    string Id,
    string Label,
    string Kind,
    IReadOnlyList<string> Tags,
    int Version,
    string Description,
    IReadOnlyDictionary<string, string> Metadata)
{
    public bool HasTag(string tag)
    {
        if (string.IsNullOrWhiteSpace(tag))
        {
            throw new ArgumentException("Tag must not be empty.", nameof(tag));
        }

        return Tags.Any(candidate => string.Equals(candidate, tag, StringComparison.OrdinalIgnoreCase));
    }
}

public sealed record ConceptRelation(
    string Id,
    string SourceId,
    string TargetId,
    string Type,
    float Weight,
    bool IsDirected,
    string Evidence,
    int Timestamp,
    IReadOnlyDictionary<string, string> Metadata);

public sealed record ConceptNeighbor(
    ConceptNode Node,
    ConceptRelation Relation,
    string Direction);

public sealed record ConceptExpansion(
    int Depth,
    ConceptNode Node,
    ConceptRelation ViaRelation);

public sealed record ConceptSubgraph(
    string Tag,
    IReadOnlyList<ConceptNode> Nodes,
    IReadOnlyList<ConceptRelation> Relations);

public sealed record ConceptPath(
    IReadOnlyList<ConceptNode> Nodes,
    IReadOnlyList<ConceptRelation> Relations)
{
    public bool Found => Nodes.Count > 0;

    public int HopCount => Relations.Count;

    public float Score => !Found
        ? 0.0f
        : Relations.Count == 0
            ? 1.0f
            : Relations.Aggregate(1.0f, (score, relation) => score * relation.Weight);

    public string NodeIdSequence => Found
        ? string.Join("->", Nodes.Select(node => node.Id))
        : "none";

    public string RelationTypeSequence => Relations.Count == 0
        ? "none"
        : string.Join("->", Relations.Select(relation => relation.Type));
}

public sealed record ConceptGraphReport(
    ConceptGraph Graph,
    ConceptPath EvolutionPath,
    ConceptSubgraph MemorySubgraph,
    IReadOnlyList<ConceptExpansion> Neighborhood);

public sealed class ConceptGraph
{
    private readonly Dictionary<string, ConceptNode> _nodeById;
    private readonly List<ConceptNode> _nodes;
    private readonly List<ConceptRelation> _relations;

    private ConceptGraph(
        IReadOnlyList<ConceptNode> nodes,
        IReadOnlyList<ConceptRelation> relations)
    {
        _nodes = nodes.Select(CloneNode).ToList();
        _nodeById = _nodes.ToDictionary(node => node.Id, StringComparer.Ordinal);
        _relations = relations.Select(CloneRelation).ToList();
    }

    public int NodeCount => _nodes.Count;

    public int RelationCount => _relations.Count;

    public IReadOnlyList<ConceptNode> Nodes => _nodes.Select(CloneNode).ToArray();

    public IReadOnlyList<ConceptRelation> Relations => _relations.Select(CloneRelation).ToArray();

    public static ConceptGraphReport RunDefault()
    {
        var graph = GenerateSyntheticGraph();
        var path = graph.ShortestPath("embedding", "evolving-memory", maxDepth: 6);
        var memorySubgraph = graph.SubgraphByTag("memory");
        var neighborhood = graph.Expand("concept-graph", maxDepth: 2);

        return new ConceptGraphReport(
            graph,
            path,
            memorySubgraph,
            neighborhood);
    }

    public static ConceptGraph GenerateSyntheticGraph()
    {
        var nodes = new[]
        {
            NewNode(
                "embedding",
                "Embedding",
                "representation",
                ["vector", "representation"],
                version: 1,
                "Vecteur numerique utilise pour placer un item dans un espace latent.",
                chapter: "27.1"),
            NewNode(
                "latent-space",
                "Espace latent",
                "representation",
                ["vector", "representation"],
                version: 1,
                "Espace ou des points proches peuvent signaler une proximite utile mais non garantie.",
                chapter: "27.2"),
            NewNode(
                "vector-store",
                "Vector store",
                "memory-index",
                ["memory", "retrieval", "vector"],
                version: 1,
                "Index local qui compare des vecteurs pour recuperer des candidats.",
                chapter: "26.6"),
            NewNode(
                "relation",
                "Relation",
                "graph-edge",
                ["graph", "relation"],
                version: 1,
                "Lien type entre deux concepts, avec poids et justification.",
                chapter: "27.4"),
            NewNode(
                "concept-graph",
                "Graphe conceptuel",
                "graph",
                ["graph", "representation"],
                version: 2,
                "Representation explicite de concepts et de relations parcourables.",
                chapter: "27.3"),
            NewNode(
                "contextualization",
                "Contextualisation",
                "process",
                ["context", "representation"],
                version: 1,
                "Operation qui interprete un concept selon une situation ou une question.",
                chapter: "27.5"),
            NewNode(
                "abstract-concept",
                "Concept abstrait",
                "concept",
                ["abstraction", "representation"],
                version: 1,
                "Concept qui ne se reduit pas a un exemple observe unique.",
                chapter: "27.6"),
            NewNode(
                "episodic-memory",
                "Memoire episodique",
                "memory",
                ["memory", "episode"],
                version: 1,
                "Trace d'un evenement ou d'une interaction localisee.",
                chapter: "26.2"),
            NewNode(
                "semantic-memory",
                "Memoire semantique",
                "memory",
                ["memory", "concept"],
                version: 1,
                "Memoire de faits, relations et regularites abstraites.",
                chapter: "26.3"),
            NewNode(
                "evolving-memory",
                "Memoire evolutive",
                "memory",
                ["memory", "versioning"],
                version: 3,
                "Memoire qui garde des revisions explicites au lieu d'ecraser silencieusement les concepts.",
                chapter: "27.7"),
            NewNode(
                "rag-context",
                "Contexte RAG",
                "retrieval-context",
                ["memory", "retrieval", "context"],
                version: 1,
                "Contexte assemble a partir de sources recuperees.",
                chapter: "26.5")
        };

        var relations = new[]
        {
            NewRelation(
                "rel-embedding-latent",
                "embedding",
                "latent-space",
                "maps-to",
                0.92f,
                "Un embedding est interprete comme un point de l'espace latent.",
                timestamp: 10),
            NewRelation(
                "rel-latent-graph",
                "latent-space",
                "concept-graph",
                "supports",
                0.78f,
                "Des voisinages latents peuvent suggerer des liens a inspecter.",
                timestamp: 20),
            NewRelation(
                "rel-vector-embedding",
                "vector-store",
                "embedding",
                "indexes",
                0.83f,
                "Un index vectoriel stocke les embeddings pour le retrieval.",
                timestamp: 30),
            NewRelation(
                "rel-rag-vector",
                "rag-context",
                "vector-store",
                "queries",
                0.82f,
                "Le contexte RAG depend d'une etape de retrieval.",
                timestamp: 40),
            NewRelation(
                "rel-rag-episodic",
                "rag-context",
                "episodic-memory",
                "retrieves",
                0.74f,
                "Une source RAG peut etre issue d'une memoire episodique.",
                timestamp: 50),
            NewRelation(
                "rel-rag-semantic",
                "rag-context",
                "semantic-memory",
                "retrieves",
                0.72f,
                "Une source RAG peut aussi provenir d'une memoire semantique.",
                timestamp: 60),
            NewRelation(
                "rel-graph-relation",
                "concept-graph",
                "relation",
                "contains",
                0.95f,
                "Un graphe conceptuel est fait de concepts et de relations.",
                timestamp: 70),
            NewRelation(
                "rel-relation-context",
                "relation",
                "contextualization",
                "qualifies",
                0.65f,
                "Le sens d'une relation depend souvent du contexte.",
                timestamp: 80),
            NewRelation(
                "rel-graph-abstract",
                "concept-graph",
                "abstract-concept",
                "models",
                0.81f,
                "Le graphe donne une forme explicite a des concepts abstraits.",
                timestamp: 90),
            NewRelation(
                "rel-abstract-semantic",
                "abstract-concept",
                "semantic-memory",
                "stored-as",
                0.76f,
                "Un concept abstrait stabilise peut rejoindre une memoire semantique.",
                timestamp: 100),
            NewRelation(
                "rel-episodic-evolving",
                "episodic-memory",
                "evolving-memory",
                "updates",
                0.70f,
                "Des episodes peuvent proposer une revision de memoire.",
                timestamp: 110),
            NewRelation(
                "rel-semantic-evolving",
                "semantic-memory",
                "evolving-memory",
                "versioned-by",
                0.79f,
                "La memoire semantique evolue par revisions explicites.",
                timestamp: 120),
            NewRelation(
                "rel-evolving-graph",
                "evolving-memory",
                "concept-graph",
                "revises",
                0.68f,
                "Une revision de memoire peut modifier le graphe conceptuel.",
                timestamp: 130),
            NewRelation(
                "rel-context-rag",
                "contextualization",
                "rag-context",
                "conditions",
                0.66f,
                "La question courante conditionne le contexte RAG assemble.",
                timestamp: 140)
        };

        return From(nodes, relations);
    }

    public static ConceptGraph From(
        IReadOnlyList<ConceptNode> nodes,
        IReadOnlyList<ConceptRelation> relations)
    {
        ValidateNodes(nodes);
        ValidateRelations(relations, nodes.Select(node => node.Id).ToHashSet(StringComparer.Ordinal));

        return new ConceptGraph(nodes, relations);
    }

    public ConceptNode GetNode(string id)
    {
        ValidateKnownNode(id);
        return CloneNode(_nodeById[id]);
    }

    public IReadOnlyList<ConceptNode> NodesWithTag(string tag)
    {
        if (string.IsNullOrWhiteSpace(tag))
        {
            throw new ArgumentException("Tag must not be empty.", nameof(tag));
        }

        return _nodes
            .Where(node => node.HasTag(tag))
            .Select(CloneNode)
            .ToArray();
    }

    public ConceptSubgraph SubgraphByTag(string tag)
    {
        var nodes = NodesWithTag(tag);
        var selectedIds = nodes.Select(node => node.Id).ToHashSet(StringComparer.Ordinal);
        var relations = _relations
            .Where(relation => selectedIds.Contains(relation.SourceId) && selectedIds.Contains(relation.TargetId))
            .Select(CloneRelation)
            .ToArray();

        return new ConceptSubgraph(tag, nodes, relations);
    }

    public IReadOnlyList<ConceptNeighbor> Neighbors(
        string id,
        ConceptTraversalDirection direction = ConceptTraversalDirection.Any)
    {
        ValidateKnownNode(id);

        var neighbors = new List<ConceptNeighbor>();
        foreach (var relation in _relations)
        {
            if (relation.SourceId == id && direction != ConceptTraversalDirection.Incoming)
            {
                neighbors.Add(new ConceptNeighbor(
                    CloneNode(_nodeById[relation.TargetId]),
                    CloneRelation(relation),
                    relation.IsDirected ? "outgoing" : "undirected"));
            }

            if (relation.TargetId == id && direction != ConceptTraversalDirection.Outgoing)
            {
                neighbors.Add(new ConceptNeighbor(
                    CloneNode(_nodeById[relation.SourceId]),
                    CloneRelation(relation),
                    relation.IsDirected ? "incoming" : "undirected"));
            }
        }

        return neighbors;
    }

    public IReadOnlyList<ConceptExpansion> Expand(
        string startId,
        int maxDepth,
        ConceptTraversalDirection direction = ConceptTraversalDirection.Any)
    {
        ValidateKnownNode(startId);
        if (maxDepth <= 0)
        {
            throw new ArgumentOutOfRangeException(nameof(maxDepth), "Max depth must be positive.");
        }

        var visited = new HashSet<string>(StringComparer.Ordinal)
        {
            startId
        };
        var queue = new Queue<(string Id, int Depth)>();
        var expansions = new List<ConceptExpansion>();
        queue.Enqueue((startId, 0));

        while (queue.Count > 0)
        {
            var current = queue.Dequeue();
            if (current.Depth >= maxDepth)
            {
                continue;
            }

            foreach (var neighbor in Neighbors(current.Id, direction))
            {
                if (!visited.Add(neighbor.Node.Id))
                {
                    continue;
                }

                var depth = current.Depth + 1;
                expansions.Add(new ConceptExpansion(
                    depth,
                    neighbor.Node,
                    neighbor.Relation));
                queue.Enqueue((neighbor.Node.Id, depth));
            }
        }

        return expansions;
    }

    public ConceptPath ShortestPath(
        string sourceId,
        string targetId,
        int maxDepth = 8,
        ConceptTraversalDirection direction = ConceptTraversalDirection.Outgoing,
        IReadOnlyCollection<string>? allowedRelationTypes = null)
    {
        ValidateKnownNode(sourceId);
        ValidateKnownNode(targetId);
        if (maxDepth < 0)
        {
            throw new ArgumentOutOfRangeException(nameof(maxDepth), "Max depth must be non-negative.");
        }

        var allowedTypes = BuildAllowedTypeSet(allowedRelationTypes);

        if (sourceId == targetId)
        {
            return new ConceptPath(
                new[] { GetNode(sourceId) },
                Array.Empty<ConceptRelation>());
        }

        var visited = new HashSet<string>(StringComparer.Ordinal)
        {
            sourceId
        };
        var queue = new Queue<(string Id, int Depth)>();
        var previous = new Dictionary<string, (string PreviousId, ConceptRelation Relation)>(StringComparer.Ordinal);
        queue.Enqueue((sourceId, 0));

        while (queue.Count > 0)
        {
            var current = queue.Dequeue();
            if (current.Depth >= maxDepth)
            {
                continue;
            }

            foreach (var neighbor in Neighbors(current.Id, direction))
            {
                if (allowedTypes is not null && !allowedTypes.Contains(neighbor.Relation.Type))
                {
                    continue;
                }

                if (!visited.Add(neighbor.Node.Id))
                {
                    continue;
                }

                previous[neighbor.Node.Id] = (current.Id, neighbor.Relation);
                if (neighbor.Node.Id == targetId)
                {
                    return ReconstructPath(sourceId, targetId, previous);
                }

                queue.Enqueue((neighbor.Node.Id, current.Depth + 1));
            }
        }

        return new ConceptPath(Array.Empty<ConceptNode>(), Array.Empty<ConceptRelation>());
    }

    public string ToNodeCsv()
    {
        var lines = new List<string>
        {
            "id,label,kind,version,tags,description,metadata"
        };

        foreach (var node in _nodes)
        {
            lines.Add(string.Create(
                CultureInfo.InvariantCulture,
                $"{Csv(node.Id)},{Csv(node.Label)},{Csv(node.Kind)},{node.Version},{Csv(string.Join("|", node.Tags))},{Csv(node.Description)},{Csv(FormatMetadata(node.Metadata))}"));
        }

        return string.Join(Environment.NewLine, lines) + Environment.NewLine;
    }

    public string ToRelationCsv()
    {
        var lines = new List<string>
        {
            "id,source,target,type,weight,directed,evidence,timestamp,metadata"
        };

        foreach (var relation in _relations)
        {
            lines.Add(string.Create(
                CultureInfo.InvariantCulture,
                $"{Csv(relation.Id)},{Csv(relation.SourceId)},{Csv(relation.TargetId)},{Csv(relation.Type)},{relation.Weight:0.######},{relation.IsDirected.ToString().ToLowerInvariant()},{Csv(relation.Evidence)},{relation.Timestamp},{Csv(FormatMetadata(relation.Metadata))}"));
        }

        return string.Join(Environment.NewLine, lines) + Environment.NewLine;
    }

    public static string ToPathCsv(ConceptPath path)
    {
        var lines = new List<string>
        {
            "rank,source,target,hops,score,path_nodes,path_relations"
        };

        if (path.Found)
        {
            lines.Add(string.Create(
                CultureInfo.InvariantCulture,
                $"1,{Csv(path.Nodes[0].Id)},{Csv(path.Nodes[^1].Id)},{path.HopCount},{path.Score:0.######},{Csv(path.NodeIdSequence)},{Csv(path.RelationTypeSequence)}"));
        }

        return string.Join(Environment.NewLine, lines) + Environment.NewLine;
    }

    public string ToMermaid()
    {
        var lines = new List<string>
        {
            "graph LR"
        };

        foreach (var node in _nodes)
        {
            lines.Add($"    {MermaidId(node.Id)}[\"{EscapeMermaid(node.Label)}\"]");
        }

        foreach (var relation in _relations)
        {
            var arrow = relation.IsDirected ? "-->" : "---";
            lines.Add($"    {MermaidId(relation.SourceId)} {arrow}|\"{EscapeMermaid(relation.Type)}\"| {MermaidId(relation.TargetId)}");
        }

        return string.Join(Environment.NewLine, lines) + Environment.NewLine;
    }

    public static string FormatReport(ConceptGraphReport report)
    {
        return string.Create(
            CultureInfo.InvariantCulture,
            $"concept graph: concepts={report.Graph.NodeCount}, relations={report.Graph.RelationCount}, path={report.EvolutionPath.NodeIdSequence}, hops={report.EvolutionPath.HopCount}, score={report.EvolutionPath.Score:0.###}, memory_concepts={report.MemorySubgraph.Nodes.Count}, memory_relations={report.MemorySubgraph.Relations.Count}, neighborhood={report.Neighborhood.Count}");
    }

    private ConceptPath ReconstructPath(
        string sourceId,
        string targetId,
        IReadOnlyDictionary<string, (string PreviousId, ConceptRelation Relation)> previous)
    {
        var nodeIds = new Stack<string>();
        var relations = new Stack<ConceptRelation>();
        var current = targetId;
        nodeIds.Push(current);

        while (!string.Equals(current, sourceId, StringComparison.Ordinal))
        {
            var step = previous[current];
            relations.Push(CloneRelation(step.Relation));
            current = step.PreviousId;
            nodeIds.Push(current);
        }

        return new ConceptPath(
            nodeIds.Select(GetNode).ToArray(),
            relations.ToArray());
    }

    private void ValidateKnownNode(string id)
    {
        if (string.IsNullOrWhiteSpace(id))
        {
            throw new ArgumentException("Concept id must not be empty.", nameof(id));
        }

        if (!_nodeById.ContainsKey(id))
        {
            throw new KeyNotFoundException($"Concept '{id}' does not exist.");
        }
    }

    private static HashSet<string>? BuildAllowedTypeSet(IReadOnlyCollection<string>? allowedRelationTypes)
    {
        if (allowedRelationTypes is null)
        {
            return null;
        }

        if (allowedRelationTypes.Count == 0)
        {
            throw new ArgumentException("Allowed relation types must not be empty.", nameof(allowedRelationTypes));
        }

        if (allowedRelationTypes.Any(string.IsNullOrWhiteSpace))
        {
            throw new ArgumentException("Allowed relation types must not contain empty values.", nameof(allowedRelationTypes));
        }

        return allowedRelationTypes.ToHashSet(StringComparer.OrdinalIgnoreCase);
    }

    private static ConceptNode NewNode(
        string id,
        string label,
        string kind,
        string[] tags,
        int version,
        string description,
        string chapter)
    {
        return new ConceptNode(
            id,
            label,
            kind,
            tags,
            version,
            description,
            new Dictionary<string, string>
            {
                ["chapter"] = chapter
            });
    }

    private static ConceptRelation NewRelation(
        string id,
        string sourceId,
        string targetId,
        string type,
        float weight,
        string evidence,
        int timestamp,
        bool isDirected = true)
    {
        return new ConceptRelation(
            id,
            sourceId,
            targetId,
            type,
            weight,
            isDirected,
            evidence,
            timestamp,
            new Dictionary<string, string>
            {
                ["source"] = "synthetic"
            });
    }

    private static void ValidateNodes(IReadOnlyList<ConceptNode> nodes)
    {
        if (nodes is null || nodes.Count == 0)
        {
            throw new ArgumentException("At least one concept node is required.", nameof(nodes));
        }

        var ids = new HashSet<string>(StringComparer.Ordinal);
        foreach (var node in nodes)
        {
            if (node is null)
            {
                throw new ArgumentException("Concept nodes must not contain null values.", nameof(nodes));
            }

            if (string.IsNullOrWhiteSpace(node.Id))
            {
                throw new ArgumentException("Concept node id must not be empty.", nameof(nodes));
            }

            if (!ids.Add(node.Id))
            {
                throw new ArgumentException($"Duplicate concept node id '{node.Id}'.", nameof(nodes));
            }

            if (string.IsNullOrWhiteSpace(node.Label))
            {
                throw new ArgumentException("Concept node label must not be empty.", nameof(nodes));
            }

            if (string.IsNullOrWhiteSpace(node.Kind))
            {
                throw new ArgumentException("Concept node kind must not be empty.", nameof(nodes));
            }

            if (node.Tags is null || node.Tags.Count == 0 || node.Tags.Any(string.IsNullOrWhiteSpace))
            {
                throw new ArgumentException("Concept node tags must be non-empty.", nameof(nodes));
            }

            if (node.Version <= 0)
            {
                throw new ArgumentOutOfRangeException(nameof(nodes), "Concept node version must be positive.");
            }

            if (string.IsNullOrWhiteSpace(node.Description))
            {
                throw new ArgumentException("Concept node description must not be empty.", nameof(nodes));
            }

            ValidateMetadata(node.Metadata, nameof(nodes));
        }
    }

    private static void ValidateRelations(
        IReadOnlyList<ConceptRelation> relations,
        IReadOnlySet<string> nodeIds)
    {
        if (relations is null || relations.Count == 0)
        {
            throw new ArgumentException("At least one concept relation is required.", nameof(relations));
        }

        var ids = new HashSet<string>(StringComparer.Ordinal);
        foreach (var relation in relations)
        {
            if (relation is null)
            {
                throw new ArgumentException("Concept relations must not contain null values.", nameof(relations));
            }

            if (string.IsNullOrWhiteSpace(relation.Id))
            {
                throw new ArgumentException("Concept relation id must not be empty.", nameof(relations));
            }

            if (!ids.Add(relation.Id))
            {
                throw new ArgumentException($"Duplicate concept relation id '{relation.Id}'.", nameof(relations));
            }

            if (!nodeIds.Contains(relation.SourceId) || !nodeIds.Contains(relation.TargetId))
            {
                throw new ArgumentException($"Relation '{relation.Id}' references an unknown concept.", nameof(relations));
            }

            if (string.Equals(relation.SourceId, relation.TargetId, StringComparison.Ordinal))
            {
                throw new ArgumentException("Self-relations are not used in this pedagogical graph.", nameof(relations));
            }

            if (string.IsNullOrWhiteSpace(relation.Type))
            {
                throw new ArgumentException("Concept relation type must not be empty.", nameof(relations));
            }

            if (!float.IsFinite(relation.Weight) || relation.Weight <= 0.0f || relation.Weight > 1.0f)
            {
                throw new ArgumentOutOfRangeException(nameof(relations), "Concept relation weight must be finite and in (0, 1].");
            }

            if (string.IsNullOrWhiteSpace(relation.Evidence))
            {
                throw new ArgumentException("Concept relation evidence must not be empty.", nameof(relations));
            }

            if (relation.Timestamp < 0)
            {
                throw new ArgumentOutOfRangeException(nameof(relations), "Concept relation timestamp must be non-negative.");
            }

            ValidateMetadata(relation.Metadata, nameof(relations));
        }
    }

    private static void ValidateMetadata(
        IReadOnlyDictionary<string, string> metadata,
        string parameterName)
    {
        if (metadata is null)
        {
            throw new ArgumentException("Metadata must not be null.", parameterName);
        }

        foreach (var pair in metadata)
        {
            if (string.IsNullOrWhiteSpace(pair.Key) || string.IsNullOrWhiteSpace(pair.Value))
            {
                throw new ArgumentException("Metadata keys and values must be non-empty.", parameterName);
            }
        }
    }

    private static ConceptNode CloneNode(ConceptNode node)
    {
        return node with
        {
            Tags = node.Tags.ToArray(),
            Metadata = node.Metadata.ToDictionary(pair => pair.Key, pair => pair.Value, StringComparer.Ordinal)
        };
    }

    private static ConceptRelation CloneRelation(ConceptRelation relation)
    {
        return relation with
        {
            Metadata = relation.Metadata.ToDictionary(pair => pair.Key, pair => pair.Value, StringComparer.Ordinal)
        };
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

    private static string MermaidId(string id)
    {
        return id.Replace("-", "_", StringComparison.Ordinal);
    }

    private static string EscapeMermaid(string text)
    {
        return text.Replace("\"", "\\\"", StringComparison.Ordinal);
    }
}
