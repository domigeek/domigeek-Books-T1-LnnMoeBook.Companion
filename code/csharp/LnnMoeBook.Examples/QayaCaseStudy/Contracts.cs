using System.Globalization;
using static LnnMoeBook.Examples.QayaCaseStudy.QayaCaseStudyContracts;

namespace LnnMoeBook.Examples.QayaCaseStudy;

public enum QayaMessageRole
{
    System,
    User,
    Orchestrator,
    Expert
}

public enum QayaClaimStatus
{
    CaseStudyObservation,
    EngineeringChoice,
    ExperimentalHypothesis,
    KnownLimitation
}

public enum QayaTelemetrySeverity
{
    Debug,
    Info,
    Warning,
    Error
}

public sealed record QayaMessage(
    string Id,
    string TraceId,
    QayaMessageRole Role,
    string Content,
    int Timestamp,
    IReadOnlyDictionary<string, string> Metadata);

public sealed record QayaMemoryQuery(
    string TraceId,
    string Text,
    int TopK,
    IReadOnlyList<string> RequiredTags);

public sealed record QayaContextItem(
    string Id,
    string Source,
    string Content,
    float Score,
    int Timestamp,
    IReadOnlyList<string> Tags,
    IReadOnlyDictionary<string, string> Metadata);

public sealed record QayaExpertDescriptor(
    string Id,
    string Specialty,
    string Version,
    bool IsReplaceable,
    IReadOnlyList<string> SupportedIntents,
    IReadOnlyDictionary<string, string> Metadata);

public sealed record QayaPreparedContext(
    string TraceId,
    string Text,
    int TokenBudget,
    IReadOnlyList<QayaContextItem> Sources,
    IReadOnlyList<QayaExpertDescriptor> CandidateExperts);

public sealed record QayaRoutingCandidate(
    int Rank,
    string ExpertId,
    float Score,
    float Weight,
    string Rationale);

public sealed record QayaRoutingDecision(
    string TraceId,
    IReadOnlyList<QayaRoutingCandidate> Candidates)
{
    public IReadOnlyList<string> SelectedExpertIds => Candidates.Select(candidate => candidate.ExpertId).ToArray();
}

public sealed record QayaExpertRequest(
    string TraceId,
    QayaMessage UserMessage,
    QayaPreparedContext Context,
    QayaExpertDescriptor Descriptor,
    QayaRoutingCandidate Route);

public sealed record QayaExpertResponse(
    string TraceId,
    string ExpertId,
    string Content,
    float Confidence,
    IReadOnlyList<string> EvidenceIds,
    IReadOnlyDictionary<string, string> Metadata);

public sealed record QayaTelemetryEvent(
    int Timestamp,
    string TraceId,
    string Component,
    string EventName,
    QayaTelemetrySeverity Severity,
    IReadOnlyDictionary<string, string> Metadata);

public sealed record QayaCaseStudyClaim(
    string Id,
    QayaClaimStatus Status,
    string Statement,
    string Evidence,
    IReadOnlyList<string> Sections);

public sealed record QayaContractValidationResult(IReadOnlyList<string> Errors)
{
    public bool IsValid => Errors.Count == 0;
}

public sealed record QayaCaseStudyRun(
    IReadOnlyList<QayaMessage> Messages,
    IReadOnlyList<QayaExpertDescriptor> Experts,
    IReadOnlyList<QayaContextItem> RetrievedContext,
    QayaPreparedContext PreparedContext,
    QayaRoutingDecision Routing,
    IReadOnlyList<QayaExpertResponse> Responses,
    IReadOnlyList<QayaTelemetryEvent> Telemetry,
    IReadOnlyList<QayaCaseStudyClaim> Claims)
{
    public int ReplaceableExpertCount => Experts.Count(expert => expert.IsReplaceable);

    public string CaseStudyStatus => "experimental-case-study";
}

public interface IQayaMemoryStore
{
    IReadOnlyList<QayaContextItem> Retrieve(QayaMemoryQuery query);
}

public interface IQayaContextAssembler
{
    QayaPreparedContext Assemble(
        QayaMessage userMessage,
        IReadOnlyList<QayaContextItem> sources,
        IReadOnlyList<QayaExpertDescriptor> candidateExperts,
        int tokenBudget);
}

public interface IQayaOrchestrator
{
    QayaRoutingDecision Route(
        QayaMessage userMessage,
        IReadOnlyList<QayaExpertDescriptor> experts,
        IReadOnlyList<QayaContextItem> context);
}

public interface IQayaExpert
{
    QayaExpertDescriptor Descriptor { get; }

    QayaExpertResponse Respond(QayaExpertRequest request);
}

public interface IQayaTelemetrySink
{
    void Track(QayaTelemetryEvent telemetryEvent);
}

public sealed class QayaKeywordMemoryStore : IQayaMemoryStore
{
    private readonly List<QayaContextItem> _items;

    public QayaKeywordMemoryStore(IReadOnlyList<QayaContextItem> items)
    {
        ValidateContextItems(items, nameof(items));
        _items = items.Select(CloneContextItem).ToList();
    }

    public IReadOnlyList<QayaContextItem> Retrieve(QayaMemoryQuery query)
    {
        ValidateMemoryQuery(query);

        return _items
            .Where(item => MatchesRequiredTags(item, query.RequiredTags))
            .Select(item => item with
            {
                Score = ScoreContextItem(query.Text, item),
                Tags = item.Tags.ToArray(),
                Metadata = CloneMetadata(item.Metadata)
            })
            .Where(item => item.Score > 0.0f)
            .OrderByDescending(item => item.Score)
            .ThenByDescending(item => item.Timestamp)
            .ThenBy(item => item.Id, StringComparer.Ordinal)
            .Take(query.TopK)
            .Select(CloneContextItem)
            .ToArray();
    }
}

public sealed class QayaContextWindowAssembler : IQayaContextAssembler
{
    public QayaPreparedContext Assemble(
        QayaMessage userMessage,
        IReadOnlyList<QayaContextItem> sources,
        IReadOnlyList<QayaExpertDescriptor> candidateExperts,
        int tokenBudget)
    {
        ValidateMessage(userMessage, nameof(userMessage));
        ValidateContextItems(sources, nameof(sources));
        ValidateExpertDescriptors(candidateExperts, nameof(candidateExperts));
        if (tokenBudget <= 0)
        {
            throw new ArgumentOutOfRangeException(nameof(tokenBudget), "Token budget must be positive.");
        }

        var lines = new List<string>
        {
            $"[M] {userMessage.Id}: {userMessage.Content}"
        };
        lines.AddRange(sources.Select((source, index) => $"[C{index + 1}] {source.Id}: {source.Content}"));
        lines.AddRange(candidateExperts.Select((expert, index) => $"[E{index + 1}] {expert.Id}: {expert.Specialty}"));

        return new QayaPreparedContext(
            userMessage.TraceId,
            string.Join(Environment.NewLine, lines),
            tokenBudget,
            sources.Select(CloneContextItem).ToArray(),
            candidateExperts.Select(CloneExpertDescriptor).ToArray());
    }
}

public sealed class QayaKeywordOrchestrator : IQayaOrchestrator
{
    public QayaKeywordOrchestrator(int topK)
    {
        if (topK <= 0)
        {
            throw new ArgumentOutOfRangeException(nameof(topK), "TopK must be positive.");
        }

        TopK = topK;
    }

    public int TopK { get; }

    public QayaRoutingDecision Route(
        QayaMessage userMessage,
        IReadOnlyList<QayaExpertDescriptor> experts,
        IReadOnlyList<QayaContextItem> context)
    {
        ValidateMessage(userMessage, nameof(userMessage));
        ValidateExpertDescriptors(experts, nameof(experts));
        ValidateContextItems(context, nameof(context));

        var routingText = userMessage.Content + " " + string.Join(" ", context.Select(item => item.Content));
        var selected = experts
            .Select(expert => new
            {
                Expert = expert,
                MatchedIntents = expert.SupportedIntents
                    .Where(intent => ContainsKeyword(routingText, intent))
                    .OrderBy(intent => intent, StringComparer.Ordinal)
                    .ToArray()
            })
            .Select(item => new
            {
                item.Expert,
                item.MatchedIntents,
                Score = item.MatchedIntents.Length == 0
                    ? 0.0f
                    : MathF.Min(1.0f, item.MatchedIntents.Length / (float)item.Expert.SupportedIntents.Count)
            })
            .OrderByDescending(item => item.Score)
            .ThenBy(item => item.Expert.Id, StringComparer.Ordinal)
            .Take(Math.Min(TopK, experts.Count))
            .ToArray();
        var totalScore = selected.Sum(item => item.Score);
        var fallbackWeight = selected.Length == 0 ? 0.0f : 1.0f / selected.Length;
        var candidates = selected
            .Select((item, index) => new QayaRoutingCandidate(
                index + 1,
                item.Expert.Id,
                item.Score,
                totalScore > 0.0f ? item.Score / totalScore : fallbackWeight,
                item.MatchedIntents.Length == 0
                    ? "fallback deterministic order"
                    : "matched intents: " + string.Join("|", item.MatchedIntents)))
            .ToArray();

        return new QayaRoutingDecision(userMessage.TraceId, candidates);
    }
}

public sealed class QayaTemplateExpert : IQayaExpert
{
    public QayaTemplateExpert(QayaExpertDescriptor descriptor)
    {
        ValidateExpertDescriptor(descriptor, nameof(descriptor));
        Descriptor = CloneExpertDescriptor(descriptor);
    }

    public QayaExpertDescriptor Descriptor { get; }

    public QayaExpertResponse Respond(QayaExpertRequest request)
    {
        ValidateExpertRequest(request);
        if (!string.Equals(request.Descriptor.Id, Descriptor.Id, StringComparison.Ordinal))
        {
            throw new ArgumentException("Request descriptor must match the expert descriptor.", nameof(request));
        }

        var evidenceIds = request.Context.Sources
            .Where(source => source.Tags.Any(tag => Descriptor.SupportedIntents.Any(intent => string.Equals(intent, tag, StringComparison.OrdinalIgnoreCase))))
            .Select(source => source.Id)
            .DefaultIfEmpty(request.Context.Sources.Count == 0 ? "none" : request.Context.Sources[0].Id)
            .Take(2)
            .ToArray();
        var confidence = MathF.Min(0.95f, 0.55f + (0.10f * evidenceIds.Count(id => id != "none")) + (0.10f * request.Route.Score));
        var evidenceText = string.Join("|", evidenceIds);

        return new QayaExpertResponse(
            request.TraceId,
            Descriptor.Id,
            $"Expert {Descriptor.Id}: proposition locale pour '{request.UserMessage.Id}'. Evidence={evidenceText}. Statut=etude de cas experimentale.",
            confidence,
            evidenceIds,
            new Dictionary<string, string>
            {
                ["specialty"] = Descriptor.Specialty,
                ["status"] = "experimental"
            });
    }
}

public sealed class QayaInMemoryTelemetrySink : IQayaTelemetrySink
{
    private readonly List<QayaTelemetryEvent> _events = new();

    public IReadOnlyList<QayaTelemetryEvent> Events => _events.Select(CloneTelemetryEvent).ToArray();

    public void Track(QayaTelemetryEvent telemetryEvent)
    {
        ValidateTelemetryEvent(telemetryEvent, nameof(telemetryEvent));
        _events.Add(CloneTelemetryEvent(telemetryEvent));
    }
}

public static class QayaCaseStudyContracts
{
    public static QayaCaseStudyRun RunDefault()
    {
        var traceId = "trace-qaya-0001";
        var messages = GenerateSimulatedMessages(traceId);
        var userMessage = messages.First(message => message.Role == QayaMessageRole.User);
        var memoryStore = new QayaKeywordMemoryStore(GenerateSyntheticContextItems());
        var retrieved = memoryStore.Retrieve(new QayaMemoryQuery(
            traceId,
            userMessage.Content,
            TopK: 3,
            RequiredTags: Array.Empty<string>()));
        var experts = GenerateExpertDescriptors();
        var router = new QayaKeywordOrchestrator(topK: 2);
        var routing = router.Route(userMessage, experts, retrieved);
        var selectedExperts = routing.SelectedExpertIds
            .Select(id => experts.Single(expert => expert.Id == id))
            .ToArray();
        var assembler = new QayaContextWindowAssembler();
        var preparedContext = assembler.Assemble(
            userMessage,
            retrieved,
            selectedExperts,
            tokenBudget: 512);
        var responses = selectedExperts
            .Select(expert => new QayaTemplateExpert(expert))
            .Select(expert =>
            {
                var route = routing.Candidates.Single(candidate => candidate.ExpertId == expert.Descriptor.Id);
                return expert.Respond(new QayaExpertRequest(
                    traceId,
                    userMessage,
                    preparedContext,
                    expert.Descriptor,
                    route));
            })
            .ToArray();
        var telemetry = BuildTelemetry(traceId, retrieved, routing, responses);

        return new QayaCaseStudyRun(
            messages,
            experts,
            retrieved,
            preparedContext,
            routing,
            responses,
            telemetry,
            GenerateCaseStudyClaims());
    }

    public static IReadOnlyList<QayaMessage> GenerateSimulatedMessages(string traceId)
    {
        if (string.IsNullOrWhiteSpace(traceId))
        {
            throw new ArgumentException("Trace id must not be empty.", nameof(traceId));
        }

        return new[]
        {
            new QayaMessage(
                "msg-system-001",
                traceId,
                QayaMessageRole.System,
                "Qaya est traitee comme une etude de cas experimentale, pas comme une preuve scientifique.",
                Timestamp: 0,
                new Dictionary<string, string>
                {
                    ["claim_policy"] = "case-study"
                }),
            new QayaMessage(
                "msg-user-001",
                traceId,
                QayaMessageRole.User,
                "Comment separer memoire, routage, experts remplacables et observabilite dans Qaya ?",
                Timestamp: 10,
                new Dictionary<string, string>
                {
                    ["chapter"] = "28",
                    ["scenario"] = "architecture"
                })
        };
    }

    public static IReadOnlyList<QayaContextItem> GenerateSyntheticContextItems()
    {
        return new[]
        {
            NewContextItem(
                "ctx-architecture",
                "docs/qaya/architecture",
                "L'etude de cas separe orchestrateur, memoire, contexte, experts et observabilite.",
                0.80f,
                timestamp: 20,
                ["architecture", "orchestration", "expert"],
                section: "35.2"),
            NewContextItem(
                "ctx-memory",
                "docs/qaya/memory",
                "La memoire fournit des sources auditees au contexte au lieu de remplacer le raisonnement.",
                0.78f,
                timestamp: 30,
                ["memory", "memoire", "context", "rag"],
                section: "35.5"),
            NewContextItem(
                "ctx-observability",
                "docs/qaya/observability",
                "Les traces rendent visibles routage, contexte, experts selectionnes, erreurs et limites.",
                0.76f,
                timestamp: 40,
                ["observability", "observabilite", "trace", "monitoring"],
                section: "35.8"),
            NewContextItem(
                "ctx-limits",
                "docs/qaya/limits",
                "Ces contrats decrivent une proposition architecturale et ne valident pas une superiorite generale.",
                0.70f,
                timestamp: 50,
                ["limitation", "hypothesis", "science"],
                section: "35.9")
        };
    }

    public static IReadOnlyList<QayaExpertDescriptor> GenerateExpertDescriptors()
    {
        return new[]
        {
            NewExpert(
                "expert-memory",
                "memoire et contexte",
                "0.1.0",
                ["memory", "memoire", "contexte"],
                section: "35.5"),
            NewExpert(
                "expert-routing",
                "routage et experts",
                "0.1.0",
                ["routing", "routage", "expert", "experts", "orchestration"],
                section: "35.3"),
            NewExpert(
                "expert-observability",
                "observabilite",
                "0.1.0",
                ["observability", "observabilite", "trace", "monitoring"],
                section: "35.8")
        };
    }

    public static IReadOnlyList<QayaCaseStudyClaim> GenerateCaseStudyClaims()
    {
        return new[]
        {
            NewClaim(
                "claim-qaya-boundary",
                QayaClaimStatus.CaseStudyObservation,
                "Qaya est utilisee comme etude de cas experimentale.",
                "Chapitre 35 et avertissement scientifique",
                ["35.1", "35.9"]),
            NewClaim(
                "claim-qaya-modularity",
                QayaClaimStatus.EngineeringChoice,
                "Les contrats separent orchestrateur, experts, memoire, contexte et observabilite.",
                "CODE-035-001",
                ["35.2", "35.7"]),
            NewClaim(
                "claim-qaya-liquid-routing",
                QayaClaimStatus.ExperimentalHypothesis,
                "Un orchestrateur dynamique pourrait moduler le routage selon un etat temporel.",
                "Chapitres 14 et 25; non demontre par cette etude de cas seule.",
                ["35.3", "35.10"]),
            NewClaim(
                "claim-qaya-limits",
                QayaClaimStatus.KnownLimitation,
                "Ce prototype ne demontre ni robustesse generale ni superiorite scientifique.",
                "Chapitre 35 et avertissement scientifique",
                ["35.9", "35.10"])
        };
    }

    public static QayaContractValidationResult Validate(QayaCaseStudyRun run)
    {
        var errors = new List<string>();
        if (run.Messages.Count == 0)
        {
            errors.Add("At least one message is required.");
        }

        if (run.Experts.Count == 0)
        {
            errors.Add("At least one expert is required.");
        }

        if (run.Routing.Candidates.Count == 0)
        {
            errors.Add("At least one routing candidate is required.");
        }

        var traceIds = run.Messages.Select(message => message.TraceId).Distinct(StringComparer.Ordinal).ToArray();
        if (traceIds.Length != 1)
        {
            errors.Add("Messages must share a single trace id.");
        }

        var traceId = traceIds.Length == 1 ? traceIds[0] : "";
        if (!string.Equals(run.PreparedContext.TraceId, traceId, StringComparison.Ordinal)
            || !string.Equals(run.Routing.TraceId, traceId, StringComparison.Ordinal))
        {
            errors.Add("Prepared context and routing must use the message trace id.");
        }

        var expertIds = run.Experts.Select(expert => expert.Id).ToHashSet(StringComparer.Ordinal);
        foreach (var candidate in run.Routing.Candidates)
        {
            if (!expertIds.Contains(candidate.ExpertId))
            {
                errors.Add($"Routing candidate '{candidate.ExpertId}' is not a known expert.");
            }
        }

        var sourceIds = run.RetrievedContext.Select(source => source.Id).ToHashSet(StringComparer.Ordinal);
        foreach (var response in run.Responses)
        {
            if (!expertIds.Contains(response.ExpertId))
            {
                errors.Add($"Response expert '{response.ExpertId}' is not a known expert.");
            }

            foreach (var evidenceId in response.EvidenceIds.Where(id => id != "none"))
            {
                if (!sourceIds.Contains(evidenceId))
                {
                    errors.Add($"Response evidence '{evidenceId}' was not retrieved.");
                }
            }
        }

        if (run.Telemetry.Any(telemetry => !string.Equals(telemetry.TraceId, traceId, StringComparison.Ordinal)))
        {
            errors.Add("Telemetry events must use the message trace id.");
        }

        if (run.Claims.Count == 0)
        {
            errors.Add("At least one case-study claim is required.");
        }

        if (run.Claims.Any(claim => claim.Status == QayaClaimStatus.ExperimentalHypothesis && !ContainsKeyword(claim.Evidence, "non demontre")))
        {
            errors.Add("Experimental hypotheses must explicitly indicate non-demonstration.");
        }

        return new QayaContractValidationResult(errors);
    }

    public static string ToMessagesCsv(IReadOnlyList<QayaMessage> messages)
    {
        ValidateMessages(messages, nameof(messages));
        var lines = new List<string>
        {
            "id,trace_id,role,timestamp,content,metadata"
        };

        foreach (var message in messages)
        {
            lines.Add(string.Create(
                CultureInfo.InvariantCulture,
                $"{Csv(message.Id)},{Csv(message.TraceId)},{RoleToText(message.Role)},{message.Timestamp},{Csv(message.Content)},{Csv(FormatMetadata(message.Metadata))}"));
        }

        return string.Join(Environment.NewLine, lines) + Environment.NewLine;
    }

    public static string ToExpertsCsv(IReadOnlyList<QayaExpertDescriptor> experts)
    {
        ValidateExpertDescriptors(experts, nameof(experts));
        var lines = new List<string>
        {
            "id,specialty,version,replaceable,intents,metadata"
        };

        foreach (var expert in experts)
        {
            lines.Add(string.Create(
                CultureInfo.InvariantCulture,
                $"{Csv(expert.Id)},{Csv(expert.Specialty)},{Csv(expert.Version)},{expert.IsReplaceable.ToString().ToLowerInvariant()},{Csv(string.Join("|", expert.SupportedIntents))},{Csv(FormatMetadata(expert.Metadata))}"));
        }

        return string.Join(Environment.NewLine, lines) + Environment.NewLine;
    }

    public static string ToContextCsv(IReadOnlyList<QayaContextItem> context)
    {
        ValidateContextItems(context, nameof(context));
        var lines = new List<string>
        {
            "id,source,score,timestamp,tags,content,metadata"
        };

        foreach (var item in context)
        {
            lines.Add(string.Create(
                CultureInfo.InvariantCulture,
                $"{Csv(item.Id)},{Csv(item.Source)},{item.Score:0.######},{item.Timestamp},{Csv(string.Join("|", item.Tags))},{Csv(item.Content)},{Csv(FormatMetadata(item.Metadata))}"));
        }

        return string.Join(Environment.NewLine, lines) + Environment.NewLine;
    }

    public static string ToRoutingCsv(QayaRoutingDecision routing)
    {
        ValidateRouting(routing, nameof(routing));
        var lines = new List<string>
        {
            "trace_id,rank,expert_id,score,weight,rationale"
        };

        foreach (var candidate in routing.Candidates)
        {
            lines.Add(string.Create(
                CultureInfo.InvariantCulture,
                $"{Csv(routing.TraceId)},{candidate.Rank},{Csv(candidate.ExpertId)},{candidate.Score:0.######},{candidate.Weight:0.######},{Csv(candidate.Rationale)}"));
        }

        return string.Join(Environment.NewLine, lines) + Environment.NewLine;
    }

    public static string ToResponsesCsv(IReadOnlyList<QayaExpertResponse> responses)
    {
        ValidateResponses(responses, nameof(responses));
        var lines = new List<string>
        {
            "trace_id,expert_id,confidence,evidence,content,metadata"
        };

        foreach (var response in responses)
        {
            lines.Add(string.Create(
                CultureInfo.InvariantCulture,
                $"{Csv(response.TraceId)},{Csv(response.ExpertId)},{response.Confidence:0.######},{Csv(string.Join("|", response.EvidenceIds))},{Csv(response.Content)},{Csv(FormatMetadata(response.Metadata))}"));
        }

        return string.Join(Environment.NewLine, lines) + Environment.NewLine;
    }

    public static string ToTelemetryCsv(IReadOnlyList<QayaTelemetryEvent> telemetry)
    {
        ValidateTelemetry(telemetry, nameof(telemetry));
        var lines = new List<string>
        {
            "timestamp,trace_id,component,event,severity,metadata"
        };

        foreach (var telemetryEvent in telemetry)
        {
            lines.Add(string.Create(
                CultureInfo.InvariantCulture,
                $"{telemetryEvent.Timestamp},{Csv(telemetryEvent.TraceId)},{Csv(telemetryEvent.Component)},{Csv(telemetryEvent.EventName)},{SeverityToText(telemetryEvent.Severity)},{Csv(FormatMetadata(telemetryEvent.Metadata))}"));
        }

        return string.Join(Environment.NewLine, lines) + Environment.NewLine;
    }

    public static string ToClaimsCsv(IReadOnlyList<QayaCaseStudyClaim> claims)
    {
        ValidateClaims(claims, nameof(claims));
        var lines = new List<string>
        {
            "id,status,statement,evidence,sections"
        };

        foreach (var claim in claims)
        {
            lines.Add($"{Csv(claim.Id)},{StatusToText(claim.Status)},{Csv(claim.Statement)},{Csv(claim.Evidence)},{Csv(string.Join("|", claim.Sections))}");
        }

        return string.Join(Environment.NewLine, lines) + Environment.NewLine;
    }

    public static string FormatReport(QayaCaseStudyRun run)
    {
        var validation = Validate(run);
        return string.Create(
            CultureInfo.InvariantCulture,
            $"qaya contracts: messages={run.Messages.Count}, experts={run.Experts.Count}, selected={run.Routing.Candidates.Count}, context={run.RetrievedContext.Count}, responses={run.Responses.Count}, telemetry={run.Telemetry.Count}, claims={run.Claims.Count}, replaceable={run.ReplaceableExpertCount}, valid={validation.IsValid}, status={run.CaseStudyStatus}");
    }

    internal static IReadOnlyList<QayaTelemetryEvent> BuildTelemetry(
        string traceId,
        IReadOnlyList<QayaContextItem> retrieved,
        QayaRoutingDecision routing,
        IReadOnlyList<QayaExpertResponse> responses)
    {
        var sink = new QayaInMemoryTelemetrySink();
        sink.Track(NewTelemetry(10, traceId, "api", "message.received", QayaTelemetrySeverity.Info, ("messages", "1")));
        sink.Track(NewTelemetry(20, traceId, "memory", "context.retrieved", QayaTelemetrySeverity.Info, ("sources", retrieved.Count.ToString(CultureInfo.InvariantCulture))));
        sink.Track(NewTelemetry(30, traceId, "orchestrator", "experts.routed", QayaTelemetrySeverity.Info, ("selected", routing.Candidates.Count.ToString(CultureInfo.InvariantCulture))));

        var timestamp = 40;
        foreach (var response in responses)
        {
            sink.Track(NewTelemetry(timestamp, traceId, response.ExpertId, "expert.responded", QayaTelemetrySeverity.Info, ("confidence", response.Confidence.ToString("0.###", CultureInfo.InvariantCulture))));
            timestamp += 10;
        }

        sink.Track(NewTelemetry(timestamp, traceId, "observability", "trace.completed", QayaTelemetrySeverity.Debug, ("status", "experimental")));
        return sink.Events;
    }

    internal static QayaContextItem NewContextItem(
        string id,
        string source,
        string content,
        float score,
        int timestamp,
        string[] tags,
        string section)
    {
        return new QayaContextItem(
            id,
            source,
            content,
            score,
            timestamp,
            tags,
            new Dictionary<string, string>
            {
                ["section"] = section,
                ["status"] = "case-study"
            });
    }

    internal static QayaExpertDescriptor NewExpert(
        string id,
        string specialty,
        string version,
        string[] supportedIntents,
        string section)
    {
        return new QayaExpertDescriptor(
            id,
            specialty,
            version,
            IsReplaceable: true,
            supportedIntents,
            new Dictionary<string, string>
            {
                ["section"] = section,
                ["status"] = "experimental"
            });
    }

    internal static QayaCaseStudyClaim NewClaim(
        string id,
        QayaClaimStatus status,
        string statement,
        string evidence,
        string[] sections)
    {
        return new QayaCaseStudyClaim(
            id,
            status,
            statement,
            evidence,
            sections);
    }

    internal static QayaTelemetryEvent NewTelemetry(
        int timestamp,
        string traceId,
        string component,
        string eventName,
        QayaTelemetrySeverity severity,
        (string Key, string Value) metadata)
    {
        return new QayaTelemetryEvent(
            timestamp,
            traceId,
            component,
            eventName,
            severity,
            new Dictionary<string, string>
            {
                [metadata.Key] = metadata.Value
            });
    }

    internal static float ScoreContextItem(string queryText, QayaContextItem item)
    {
        var text = item.Content + " " + item.Source + " " + string.Join(" ", item.Tags);
        var matches = CountMatches(queryText, item.Tags) + CountKeywords(queryText + " " + text, "qaya");
        return MathF.Min(1.0f, matches * 0.25f);
    }

    internal static bool MatchesRequiredTags(QayaContextItem item, IReadOnlyList<string> requiredTags)
    {
        if (requiredTags.Count == 0)
        {
            return true;
        }

        return requiredTags.All(requiredTag =>
            item.Tags.Any(tag => string.Equals(tag, requiredTag, StringComparison.OrdinalIgnoreCase)));
    }

    internal static int CountMatches(string text, IReadOnlyList<string> keywords)
    {
        return keywords.Count(keyword => ContainsKeyword(text, keyword));
    }

    internal static int CountKeywords(string text, params string[] keywords)
    {
        return keywords.Count(keyword => ContainsKeyword(text, keyword));
    }

    internal static bool ContainsKeyword(string text, string keyword)
    {
        return text.Contains(keyword, StringComparison.OrdinalIgnoreCase);
    }

    internal static void ValidateMemoryQuery(QayaMemoryQuery query)
    {
        if (string.IsNullOrWhiteSpace(query.TraceId))
        {
            throw new ArgumentException("Trace id must not be empty.", nameof(query));
        }

        if (string.IsNullOrWhiteSpace(query.Text))
        {
            throw new ArgumentException("Query text must not be empty.", nameof(query));
        }

        if (query.TopK <= 0)
        {
            throw new ArgumentOutOfRangeException(nameof(query), "TopK must be positive.");
        }

        if (query.RequiredTags is null || query.RequiredTags.Any(string.IsNullOrWhiteSpace))
        {
            throw new ArgumentException("Required tags must not contain empty values.", nameof(query));
        }
    }

    internal static void ValidateMessages(IReadOnlyList<QayaMessage> messages, string parameterName)
    {
        if (messages is null || messages.Count == 0)
        {
            throw new ArgumentException("At least one message is required.", parameterName);
        }

        var ids = new HashSet<string>(StringComparer.Ordinal);
        foreach (var message in messages)
        {
            ValidateMessage(message, parameterName);
            if (!ids.Add(message.Id))
            {
                throw new ArgumentException($"Duplicate message id '{message.Id}'.", parameterName);
            }
        }
    }

    internal static void ValidateMessage(QayaMessage message, string parameterName)
    {
        if (message is null)
        {
            throw new ArgumentException("Message must not be null.", parameterName);
        }

        if (string.IsNullOrWhiteSpace(message.Id) || string.IsNullOrWhiteSpace(message.TraceId) || string.IsNullOrWhiteSpace(message.Content))
        {
            throw new ArgumentException("Message id, trace id and content must be non-empty.", parameterName);
        }

        if (message.Timestamp < 0)
        {
            throw new ArgumentOutOfRangeException(parameterName, "Message timestamp must be non-negative.");
        }

        ValidateMetadata(message.Metadata, parameterName);
    }

    internal static void ValidateContextItems(IReadOnlyList<QayaContextItem> items, string parameterName)
    {
        if (items is null)
        {
            throw new ArgumentException("Context items must not be null.", parameterName);
        }

        var ids = new HashSet<string>(StringComparer.Ordinal);
        foreach (var item in items)
        {
            if (item is null)
            {
                throw new ArgumentException("Context items must not contain null values.", parameterName);
            }

            if (string.IsNullOrWhiteSpace(item.Id) || string.IsNullOrWhiteSpace(item.Source) || string.IsNullOrWhiteSpace(item.Content))
            {
                throw new ArgumentException("Context id, source and content must be non-empty.", parameterName);
            }

            if (!ids.Add(item.Id))
            {
                throw new ArgumentException($"Duplicate context item id '{item.Id}'.", parameterName);
            }

            if (!float.IsFinite(item.Score) || item.Score < 0.0f || item.Score > 1.0f)
            {
                throw new ArgumentOutOfRangeException(parameterName, "Context score must be finite and in [0, 1].");
            }

            if (item.Timestamp < 0)
            {
                throw new ArgumentOutOfRangeException(parameterName, "Context timestamp must be non-negative.");
            }

            ValidateTags(item.Tags, parameterName);
            ValidateMetadata(item.Metadata, parameterName);
        }
    }

    internal static void ValidateExpertDescriptors(IReadOnlyList<QayaExpertDescriptor> experts, string parameterName)
    {
        if (experts is null || experts.Count == 0)
        {
            throw new ArgumentException("At least one expert descriptor is required.", parameterName);
        }

        var ids = new HashSet<string>(StringComparer.Ordinal);
        foreach (var expert in experts)
        {
            ValidateExpertDescriptor(expert, parameterName);
            if (!ids.Add(expert.Id))
            {
                throw new ArgumentException($"Duplicate expert id '{expert.Id}'.", parameterName);
            }
        }
    }

    internal static void ValidateExpertDescriptor(QayaExpertDescriptor expert, string parameterName)
    {
        if (expert is null)
        {
            throw new ArgumentException("Expert descriptor must not be null.", parameterName);
        }

        if (string.IsNullOrWhiteSpace(expert.Id) || string.IsNullOrWhiteSpace(expert.Specialty) || string.IsNullOrWhiteSpace(expert.Version))
        {
            throw new ArgumentException("Expert id, specialty and version must be non-empty.", parameterName);
        }

        ValidateTags(expert.SupportedIntents, parameterName);
        ValidateMetadata(expert.Metadata, parameterName);
    }

    internal static void ValidateExpertRequest(QayaExpertRequest request)
    {
        if (request is null)
        {
            throw new ArgumentException("Expert request must not be null.", nameof(request));
        }

        if (string.IsNullOrWhiteSpace(request.TraceId))
        {
            throw new ArgumentException("Trace id must not be empty.", nameof(request));
        }

        ValidateMessage(request.UserMessage, nameof(request));
        ValidateExpertDescriptor(request.Descriptor, nameof(request));
        ValidateRouting(new QayaRoutingDecision(request.TraceId, new[] { request.Route }), nameof(request));
        if (!string.Equals(request.TraceId, request.UserMessage.TraceId, StringComparison.Ordinal)
            || !string.Equals(request.TraceId, request.Context.TraceId, StringComparison.Ordinal))
        {
            throw new ArgumentException("Request trace id must match message and context trace ids.", nameof(request));
        }
    }

    internal static void ValidateRouting(QayaRoutingDecision routing, string parameterName)
    {
        if (routing is null)
        {
            throw new ArgumentException("Routing must not be null.", parameterName);
        }

        if (string.IsNullOrWhiteSpace(routing.TraceId))
        {
            throw new ArgumentException("Routing trace id must not be empty.", parameterName);
        }

        if (routing.Candidates is null || routing.Candidates.Count == 0)
        {
            throw new ArgumentException("Routing must contain at least one candidate.", parameterName);
        }

        var ranks = new HashSet<int>();
        foreach (var candidate in routing.Candidates)
        {
            if (candidate.Rank <= 0 || !ranks.Add(candidate.Rank))
            {
                throw new ArgumentException("Routing ranks must be positive and unique.", parameterName);
            }

            if (string.IsNullOrWhiteSpace(candidate.ExpertId) || string.IsNullOrWhiteSpace(candidate.Rationale))
            {
                throw new ArgumentException("Routing candidate expert id and rationale must be non-empty.", parameterName);
            }

            if (!float.IsFinite(candidate.Score) || candidate.Score < 0.0f || candidate.Score > 1.0f)
            {
                throw new ArgumentOutOfRangeException(parameterName, "Routing score must be finite and in [0, 1].");
            }

            if (!float.IsFinite(candidate.Weight) || candidate.Weight < 0.0f || candidate.Weight > 1.0f)
            {
                throw new ArgumentOutOfRangeException(parameterName, "Routing weight must be finite and in [0, 1].");
            }
        }
    }

    internal static void ValidateResponses(IReadOnlyList<QayaExpertResponse> responses, string parameterName)
    {
        if (responses is null || responses.Count == 0)
        {
            throw new ArgumentException("At least one expert response is required.", parameterName);
        }

        foreach (var response in responses)
        {
            if (response is null)
            {
                throw new ArgumentException("Expert responses must not contain null values.", parameterName);
            }

            if (string.IsNullOrWhiteSpace(response.TraceId) || string.IsNullOrWhiteSpace(response.ExpertId) || string.IsNullOrWhiteSpace(response.Content))
            {
                throw new ArgumentException("Response trace id, expert id and content must be non-empty.", parameterName);
            }

            if (!float.IsFinite(response.Confidence) || response.Confidence < 0.0f || response.Confidence > 1.0f)
            {
                throw new ArgumentOutOfRangeException(parameterName, "Response confidence must be finite and in [0, 1].");
            }

            ValidateTags(response.EvidenceIds, parameterName);
            ValidateMetadata(response.Metadata, parameterName);
        }
    }

    internal static void ValidateTelemetry(IReadOnlyList<QayaTelemetryEvent> telemetry, string parameterName)
    {
        if (telemetry is null || telemetry.Count == 0)
        {
            throw new ArgumentException("At least one telemetry event is required.", parameterName);
        }

        foreach (var telemetryEvent in telemetry)
        {
            ValidateTelemetryEvent(telemetryEvent, parameterName);
        }
    }

    internal static void ValidateTelemetryEvent(QayaTelemetryEvent telemetryEvent, string parameterName)
    {
        if (telemetryEvent is null)
        {
            throw new ArgumentException("Telemetry event must not be null.", parameterName);
        }

        if (telemetryEvent.Timestamp < 0)
        {
            throw new ArgumentOutOfRangeException(parameterName, "Telemetry timestamp must be non-negative.");
        }

        if (string.IsNullOrWhiteSpace(telemetryEvent.TraceId) || string.IsNullOrWhiteSpace(telemetryEvent.Component) || string.IsNullOrWhiteSpace(telemetryEvent.EventName))
        {
            throw new ArgumentException("Telemetry trace id, component and event name must be non-empty.", parameterName);
        }

        ValidateMetadata(telemetryEvent.Metadata, parameterName);
    }

    internal static void ValidateClaims(IReadOnlyList<QayaCaseStudyClaim> claims, string parameterName)
    {
        if (claims is null || claims.Count == 0)
        {
            throw new ArgumentException("At least one claim is required.", parameterName);
        }

        var ids = new HashSet<string>(StringComparer.Ordinal);
        foreach (var claim in claims)
        {
            if (claim is null)
            {
                throw new ArgumentException("Claims must not contain null values.", parameterName);
            }

            if (string.IsNullOrWhiteSpace(claim.Id) || string.IsNullOrWhiteSpace(claim.Statement) || string.IsNullOrWhiteSpace(claim.Evidence))
            {
                throw new ArgumentException("Claim id, statement and evidence must be non-empty.", parameterName);
            }

            if (!ids.Add(claim.Id))
            {
                throw new ArgumentException($"Duplicate claim id '{claim.Id}'.", parameterName);
            }

            ValidateTags(claim.Sections, parameterName);
        }
    }

    internal static void ValidateTags(IReadOnlyList<string> tags, string parameterName)
    {
        if (tags is null || tags.Count == 0 || tags.Any(string.IsNullOrWhiteSpace))
        {
            throw new ArgumentException("Tags must be non-empty.", parameterName);
        }
    }

    internal static void ValidateMetadata(IReadOnlyDictionary<string, string> metadata, string parameterName)
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

    internal static QayaMessage CloneMessage(QayaMessage message)
    {
        return message with
        {
            Metadata = CloneMetadata(message.Metadata)
        };
    }

    internal static QayaContextItem CloneContextItem(QayaContextItem item)
    {
        return item with
        {
            Tags = item.Tags.ToArray(),
            Metadata = CloneMetadata(item.Metadata)
        };
    }

    internal static QayaExpertDescriptor CloneExpertDescriptor(QayaExpertDescriptor expert)
    {
        return expert with
        {
            SupportedIntents = expert.SupportedIntents.ToArray(),
            Metadata = CloneMetadata(expert.Metadata)
        };
    }

    internal static QayaTelemetryEvent CloneTelemetryEvent(QayaTelemetryEvent telemetryEvent)
    {
        return telemetryEvent with
        {
            Metadata = CloneMetadata(telemetryEvent.Metadata)
        };
    }

    internal static Dictionary<string, string> CloneMetadata(IReadOnlyDictionary<string, string> metadata)
    {
        return metadata.ToDictionary(pair => pair.Key, pair => pair.Value, StringComparer.Ordinal);
    }

    internal static string Csv(string value)
    {
        if (!value.Contains(',') && !value.Contains('"') && !value.Contains('\n') && !value.Contains('\r'))
        {
            return value;
        }

        return "\"" + value.Replace("\"", "\"\"", StringComparison.Ordinal) + "\"";
    }

    internal static string FormatMetadata(IReadOnlyDictionary<string, string> metadata)
    {
        return string.Join(
            "|",
            metadata
                .OrderBy(pair => pair.Key, StringComparer.Ordinal)
                .Select(pair => $"{pair.Key}={pair.Value}"));
    }

    internal static string RoleToText(QayaMessageRole role)
    {
        return role.ToString().ToLowerInvariant();
    }

    internal static string SeverityToText(QayaTelemetrySeverity severity)
    {
        return severity.ToString().ToLowerInvariant();
    }

    internal static string StatusToText(QayaClaimStatus status)
    {
        return status switch
        {
            QayaClaimStatus.CaseStudyObservation => "case-study-observation",
            QayaClaimStatus.EngineeringChoice => "engineering-choice",
            QayaClaimStatus.ExperimentalHypothesis => "experimental-hypothesis",
            QayaClaimStatus.KnownLimitation => "known-limitation",
            _ => status.ToString().ToLowerInvariant()
        };
    }
}
