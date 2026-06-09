using LnnMoeBook.Examples.QayaCaseStudy;

namespace LnnMoeBook.Tests.QayaCaseStudy;

public sealed class ContractsTests
{
    [Fact]
    public void GenerateSimulatedMessagesIsDeterministic()
    {
        var first = QayaCaseStudyContracts.GenerateSimulatedMessages("trace-test");
        var second = QayaCaseStudyContracts.GenerateSimulatedMessages("trace-test");

        Assert.Equal(2, first.Count);
        Assert.Equal(first.Select(message => message.Id), second.Select(message => message.Id));
        Assert.Equal(first.Select(message => message.Content), second.Select(message => message.Content));
        Assert.All(first, message => Assert.Equal("trace-test", message.TraceId));
        Assert.Equal(QayaMessageRole.System, first[0].Role);
        Assert.Equal(QayaMessageRole.User, first[1].Role);
    }

    [Fact]
    public void GenerateExpertDescriptorsBuildsReplaceableExperts()
    {
        var experts = QayaCaseStudyContracts.GenerateExpertDescriptors();

        Assert.Equal(3, experts.Count);
        Assert.All(experts, expert =>
        {
            Assert.True(expert.IsReplaceable);
            Assert.NotEmpty(expert.SupportedIntents);
            Assert.Contains("status", expert.Metadata.Keys);
        });
        Assert.Contains(experts, expert => expert.Id == "expert-memory");
        Assert.Contains(experts, expert => expert.Id == "expert-routing");
        Assert.Contains(experts, expert => expert.Id == "expert-observability");
    }

    [Fact]
    public void MemoryRecallIsDeterministicAndBounded()
    {
        var store = new QayaKeywordMemoryStore(QayaCaseStudyContracts.GenerateSyntheticContextItems());
        var query = new QayaMemoryQuery(
            "trace-test",
            "qaya memoire contexte observabilite",
            TopK: 2,
            RequiredTags: Array.Empty<string>());

        var first = store.Retrieve(query);
        var second = store.Retrieve(query);

        Assert.Equal(2, first.Count);
        Assert.Equal(first.Select(item => item.Id), second.Select(item => item.Id));
        Assert.All(first, item => Assert.InRange(item.Score, 0.0001f, 1.0f));
        Assert.True(first[0].Score >= first[1].Score);
    }

    [Fact]
    public void MemoryRecallCanFilterByTag()
    {
        var store = new QayaKeywordMemoryStore(QayaCaseStudyContracts.GenerateSyntheticContextItems());

        var observability = store.Retrieve(new QayaMemoryQuery(
            "trace-test",
            "qaya observabilite trace",
            TopK: 3,
            RequiredTags: new[] { "OBSERVABILITY" }));
        var absent = store.Retrieve(new QayaMemoryQuery(
            "trace-test",
            "qaya memoire",
            TopK: 3,
            RequiredTags: new[] { "absent" }));

        Assert.Single(observability);
        Assert.Equal("ctx-observability", observability[0].Id);
        Assert.Empty(absent);
    }

    [Fact]
    public void RouterSelectsTopKExpertsWithNormalizedWeights()
    {
        var run = QayaCaseStudyContracts.RunDefault();

        Assert.Equal(2, run.Routing.Candidates.Count);
        Assert.Equal(new[] { 1, 2 }, run.Routing.Candidates.Select(candidate => candidate.Rank));
        Assert.Equal("expert-memory", run.Routing.Candidates[0].ExpertId);
        Assert.Equal("expert-routing", run.Routing.Candidates[1].ExpertId);
        Assert.Equal(2, run.Routing.SelectedExpertIds.Distinct(StringComparer.Ordinal).Count());
        Assert.InRange(run.Routing.Candidates.Sum(candidate => candidate.Weight), 0.99999f, 1.00001f);
        Assert.All(run.Routing.Candidates, candidate =>
        {
            Assert.InRange(candidate.Score, 0.0f, 1.0f);
            Assert.InRange(candidate.Weight, 0.0f, 1.0f);
        });
    }

    [Fact]
    public void RouterTieBreaksDeterministicallyByExpertId()
    {
        var message = NewUserMessage("trace-tie", "alpha");
        var experts = new[]
        {
            NewExpert("expert-b", "beta", ["alpha"]),
            NewExpert("expert-a", "alpha", ["alpha"])
        };
        var router = new QayaKeywordOrchestrator(topK: 2);

        var decision = router.Route(message, experts, Array.Empty<QayaContextItem>());

        Assert.Equal("expert-a", decision.Candidates[0].ExpertId);
        Assert.Equal("expert-b", decision.Candidates[1].ExpertId);
    }

    [Fact]
    public void ContextAssemblerIncludesMessageSourcesAndExperts()
    {
        var run = QayaCaseStudyContracts.RunDefault();

        Assert.Equal("trace-qaya-0001", run.PreparedContext.TraceId);
        Assert.Equal(512, run.PreparedContext.TokenBudget);
        Assert.Contains("[M] msg-user-001:", run.PreparedContext.Text);
        Assert.Contains("[C1]", run.PreparedContext.Text);
        Assert.Contains("[E1] expert-memory:", run.PreparedContext.Text);
        Assert.Equal(run.RetrievedContext.Count, run.PreparedContext.Sources.Count);
        Assert.Equal(run.Routing.Candidates.Count, run.PreparedContext.CandidateExperts.Count);
    }

    [Fact]
    public void TemplateExpertUsesOnlyContextEvidence()
    {
        var run = QayaCaseStudyContracts.RunDefault();
        var response = run.Responses.Single(response => response.ExpertId == "expert-memory");
        var sourceIds = run.RetrievedContext.Select(item => item.Id).ToHashSet(StringComparer.Ordinal);

        Assert.InRange(response.Confidence, 0.0f, 1.0f);
        Assert.NotEmpty(response.EvidenceIds);
        Assert.All(response.EvidenceIds.Where(id => id != "none"), id => Assert.Contains(id, sourceIds));
        Assert.Contains("Statut=etude de cas experimentale", response.Content);
    }

    [Fact]
    public void RunDefaultBuildsValidCaseStudyRun()
    {
        var run = QayaCaseStudyContracts.RunDefault();
        var validation = QayaCaseStudyContracts.Validate(run);

        Assert.True(validation.IsValid);
        Assert.Empty(validation.Errors);
        Assert.Equal("experimental-case-study", run.CaseStudyStatus);
        Assert.Equal(3, run.Experts.Count);
        Assert.Equal(3, run.ReplaceableExpertCount);
        Assert.Equal(2, run.Responses.Count);
        Assert.Equal(run.Routing.SelectedExpertIds.OrderBy(id => id), run.Responses.Select(response => response.ExpertId).OrderBy(id => id));
    }

    [Fact]
    public void TelemetryEventsAreCorrelatedAndOrdered()
    {
        var run = QayaCaseStudyContracts.RunDefault();

        Assert.Equal(6, run.Telemetry.Count);
        Assert.All(run.Telemetry, telemetry => Assert.Equal("trace-qaya-0001", telemetry.TraceId));
        Assert.Equal("message.received", run.Telemetry[0].EventName);
        Assert.Equal("context.retrieved", run.Telemetry[1].EventName);
        Assert.Equal("experts.routed", run.Telemetry[2].EventName);
        Assert.Equal("trace.completed", run.Telemetry[^1].EventName);
        Assert.True(run.Telemetry.Zip(run.Telemetry.Skip(1)).All(pair => pair.First.Timestamp <= pair.Second.Timestamp));
    }

    [Fact]
    public void ClaimsSeparateObservationChoiceHypothesisAndLimitation()
    {
        var claims = QayaCaseStudyContracts.GenerateCaseStudyClaims();

        Assert.Contains(claims, claim => claim.Status == QayaClaimStatus.CaseStudyObservation);
        Assert.Contains(claims, claim => claim.Status == QayaClaimStatus.EngineeringChoice);
        Assert.Contains(claims, claim => claim.Status == QayaClaimStatus.ExperimentalHypothesis);
        Assert.Contains(claims, claim => claim.Status == QayaClaimStatus.KnownLimitation);
        Assert.All(claims.Where(claim => claim.Status == QayaClaimStatus.ExperimentalHypothesis), claim =>
            Assert.Contains("non demontre", claim.Evidence, StringComparison.OrdinalIgnoreCase));
    }

    [Fact]
    public void ValidationDetectsUnknownRoutingExpertAndEvidence()
    {
        var run = QayaCaseStudyContracts.RunDefault();
        var invalid = run with
        {
            Routing = run.Routing with
            {
                Candidates = new[]
                {
                    run.Routing.Candidates[0] with { ExpertId = "expert-missing" }
                }
            },
            Responses = new[]
            {
                run.Responses[0] with { EvidenceIds = new[] { "ctx-missing" } }
            }
        };

        var validation = QayaCaseStudyContracts.Validate(invalid);

        Assert.False(validation.IsValid);
        Assert.Contains(validation.Errors, error => error.Contains("not a known expert", StringComparison.Ordinal));
        Assert.Contains(validation.Errors, error => error.Contains("was not retrieved", StringComparison.Ordinal));
    }

    [Fact]
    public void MessagesCsvContainsStableHeaderAndRows()
    {
        var messages = QayaCaseStudyContracts.GenerateSimulatedMessages("trace-test");

        var lines = Lines(QayaCaseStudyContracts.ToMessagesCsv(messages));

        Assert.Equal(3, lines.Length);
        Assert.Equal("id,trace_id,role,timestamp,content,metadata", lines[0]);
        Assert.StartsWith("msg-system-001,trace-test,system,0,", lines[1], StringComparison.Ordinal);
        Assert.Contains("claim_policy=case-study", lines[1]);
    }

    [Fact]
    public void ExpertsCsvContainsStableHeaderAndRows()
    {
        var experts = QayaCaseStudyContracts.GenerateExpertDescriptors();

        var lines = Lines(QayaCaseStudyContracts.ToExpertsCsv(experts));

        Assert.Equal(4, lines.Length);
        Assert.Equal("id,specialty,version,replaceable,intents,metadata", lines[0]);
        Assert.StartsWith("expert-memory,memoire et contexte,0.1.0,true,memory|memoire|contexte,", lines[1], StringComparison.Ordinal);
    }

    [Fact]
    public void ContextCsvContainsStableHeaderAndRows()
    {
        var run = QayaCaseStudyContracts.RunDefault();

        var lines = Lines(QayaCaseStudyContracts.ToContextCsv(run.RetrievedContext));

        Assert.Equal(4, lines.Length);
        Assert.Equal("id,source,score,timestamp,tags,content,metadata", lines[0]);
        Assert.StartsWith("ctx-observability,docs/qaya/observability,", lines[1], StringComparison.Ordinal);
    }

    [Fact]
    public void RoutingCsvContainsStableHeaderAndRows()
    {
        var run = QayaCaseStudyContracts.RunDefault();

        var lines = Lines(QayaCaseStudyContracts.ToRoutingCsv(run.Routing));

        Assert.Equal(3, lines.Length);
        Assert.Equal("trace_id,rank,expert_id,score,weight,rationale", lines[0]);
        Assert.StartsWith("trace-qaya-0001,1,expert-memory,", lines[1], StringComparison.Ordinal);
        Assert.Contains("matched intents:", lines[1]);
    }

    [Fact]
    public void ResponsesCsvContainsStableHeaderAndRows()
    {
        var run = QayaCaseStudyContracts.RunDefault();

        var lines = Lines(QayaCaseStudyContracts.ToResponsesCsv(run.Responses));

        Assert.Equal(3, lines.Length);
        Assert.Equal("trace_id,expert_id,confidence,evidence,content,metadata", lines[0]);
        Assert.StartsWith("trace-qaya-0001,expert-memory,", lines[1], StringComparison.Ordinal);
        Assert.Contains("status=experimental", lines[1]);
    }

    [Fact]
    public void TelemetryCsvContainsStableHeaderAndRows()
    {
        var run = QayaCaseStudyContracts.RunDefault();

        var lines = Lines(QayaCaseStudyContracts.ToTelemetryCsv(run.Telemetry));

        Assert.Equal(7, lines.Length);
        Assert.Equal("timestamp,trace_id,component,event,severity,metadata", lines[0]);
        Assert.StartsWith("10,trace-qaya-0001,api,message.received,info,", lines[1], StringComparison.Ordinal);
    }

    [Fact]
    public void ClaimsCsvContainsStableHeaderAndRows()
    {
        var claims = QayaCaseStudyContracts.GenerateCaseStudyClaims();

        var lines = Lines(QayaCaseStudyContracts.ToClaimsCsv(claims));

        Assert.Equal(5, lines.Length);
        Assert.Equal("id,status,statement,evidence,sections", lines[0]);
        Assert.StartsWith("claim-qaya-boundary,case-study-observation,", lines[1], StringComparison.Ordinal);
        Assert.Contains("35.1|35.9", lines[1]);
    }

    [Fact]
    public void FormatReportContainsExperimentalStatusAndStableCounters()
    {
        var text = QayaCaseStudyContracts.FormatReport(QayaCaseStudyContracts.RunDefault());

        Assert.Contains("qaya contracts", text);
        Assert.Contains("messages=2", text);
        Assert.Contains("experts=3", text);
        Assert.Contains("selected=2", text);
        Assert.Contains("context=3", text);
        Assert.Contains("responses=2", text);
        Assert.Contains("telemetry=6", text);
        Assert.Contains("claims=4", text);
        Assert.Contains("replaceable=3", text);
        Assert.Contains("valid=True", text);
        Assert.Contains("status=experimental-case-study", text);
    }

    [Fact]
    public void FormatReportDoesNotUsePromotionOrProofLanguage()
    {
        var text = QayaCaseStudyContracts.FormatReport(QayaCaseStudyContracts.RunDefault());

        Assert.DoesNotContain("proof", text, StringComparison.OrdinalIgnoreCase);
        Assert.DoesNotContain("validated", text, StringComparison.OrdinalIgnoreCase);
        Assert.DoesNotContain("superior", text, StringComparison.OrdinalIgnoreCase);
        Assert.DoesNotContain("production-ready", text, StringComparison.OrdinalIgnoreCase);
    }

    [Fact]
    public void ContractsRejectInvalidInputs()
    {
        Assert.Throws<ArgumentException>(() =>
            QayaCaseStudyContracts.GenerateSimulatedMessages(""));
        Assert.Throws<ArgumentOutOfRangeException>(() =>
            new QayaKeywordOrchestrator(topK: 0));
        Assert.Throws<ArgumentException>(() =>
            new QayaKeywordMemoryStore(new[]
            {
                QayaCaseStudyContracts.GenerateSyntheticContextItems()[0] with { Id = "" }
            }));

        var store = new QayaKeywordMemoryStore(QayaCaseStudyContracts.GenerateSyntheticContextItems());
        Assert.Throws<ArgumentException>(() =>
            store.Retrieve(new QayaMemoryQuery("trace", "", TopK: 1, RequiredTags: Array.Empty<string>())));
        Assert.Throws<ArgumentOutOfRangeException>(() =>
            store.Retrieve(new QayaMemoryQuery("trace", "qaya", TopK: 0, RequiredTags: Array.Empty<string>())));

        var assembler = new QayaContextWindowAssembler();
        Assert.Throws<ArgumentOutOfRangeException>(() =>
            assembler.Assemble(
                NewUserMessage("trace", "qaya"),
                QayaCaseStudyContracts.GenerateSyntheticContextItems(),
                QayaCaseStudyContracts.GenerateExpertDescriptors(),
                tokenBudget: 0));
    }

    private static QayaMessage NewUserMessage(string traceId, string content)
    {
        return new QayaMessage(
            "msg-test",
            traceId,
            QayaMessageRole.User,
            content,
            Timestamp: 1,
            new Dictionary<string, string>
            {
                ["source"] = "test"
            });
    }

    private static QayaExpertDescriptor NewExpert(
        string id,
        string specialty,
        string[] intents)
    {
        return new QayaExpertDescriptor(
            id,
            specialty,
            "test",
            IsReplaceable: true,
            intents,
            new Dictionary<string, string>
            {
                ["source"] = "test"
            });
    }

    private static string[] Lines(string csv)
    {
        return csv
            .Split('\n', StringSplitOptions.RemoveEmptyEntries)
            .Select(line => line.TrimEnd('\r'))
            .ToArray();
    }
}
