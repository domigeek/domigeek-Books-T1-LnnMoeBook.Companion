using System;
using System.Text.Json;
using System.Text.Json.Serialization;

namespace LnnMoeBook.Solutions.Ch038;

public static class Ex038002
{
    public static void Main()
    {
        var manifest = CreateDemoManifest();
        var json = JsonSerializer.Serialize(
            manifest,
            new JsonSerializerOptions
            {
                WriteIndented = true,
                DefaultIgnoreCondition = JsonIgnoreCondition.WhenWritingNull
            });

        Console.WriteLine(json);
    }

    public static ModelManifest CreateDemoManifest()
    {
        return new ModelManifest(
            Schema: "lnn-moe-model-manifest/v1",
            Model: new ModelInfo(
                Name: "simple-ltc-synthetic",
                Version: "0.3.0",
                Status: "candidate",
                CreatedAtUtc: DateTimeOffset.Parse("2026-06-11T18:30:00Z")),
            Artifact: new ArtifactInfo(
                Path: "models/simple-ltc-synthetic/0.3.0/model.bin",
                Sha256: "replace-with-real-sha256",
                Format: "custom-demo",
                SizeBytes: 18342),
            Code: new CodeInfo(
                Repository: "LnnMoeBook.Companion",
                Commit: "replace-with-git-commit",
                Branch: "main",
                Dirty: false),
            Data: new DataInfo(
                Kind: "synthetic",
                Generator: "SimpleLtcCell.GenerateSyntheticSequences",
                SequenceCount: 32,
                SequenceLength: 7,
                Seed: 42,
                DataHash: "replace-with-data-hash"),
            Configuration: new TrainingConfiguration(
                Solver: "rk4",
                InternalStepSize: 0.02f,
                Iterations: 60,
                LearningRate: 0.22f,
                Epsilon: 0.001f),
            Runtime: new RuntimeInfo(
                DotNet: "8.0",
                TorchSharp: "replace-with-version",
                Os: "windows",
                Device: "cpu"),
            Metrics: new MetricInfo(
                InitialLoss: 1.2345f,
                FinalLoss: 0.1234f,
                LossReductionRatio: 0.90f),
            Approval: new ApprovalInfo(
                ApprovedForDemo: true,
                ApprovedForProduction: false,
                Notes: "Modèle pédagogique entraîné sur données synthétiques."));
    }
}

public sealed record ModelManifest(
    string Schema,
    ModelInfo Model,
    ArtifactInfo Artifact,
    CodeInfo Code,
    DataInfo Data,
    TrainingConfiguration Configuration,
    RuntimeInfo Runtime,
    MetricInfo Metrics,
    ApprovalInfo Approval);

public sealed record ModelInfo(string Name, string Version, string Status, DateTimeOffset CreatedAtUtc);

public sealed record ArtifactInfo(string Path, string Sha256, string Format, long SizeBytes);

public sealed record CodeInfo(string Repository, string Commit, string Branch, bool Dirty);

public sealed record DataInfo(
    string Kind,
    string Generator,
    int SequenceCount,
    int SequenceLength,
    int Seed,
    string DataHash);

public sealed record TrainingConfiguration(
    string Solver,
    float InternalStepSize,
    int Iterations,
    float LearningRate,
    float Epsilon);

public sealed record RuntimeInfo(string DotNet, string TorchSharp, string Os, string Device);

public sealed record MetricInfo(float InitialLoss, float FinalLoss, float LossReductionRatio);

public sealed record ApprovalInfo(bool ApprovedForDemo, bool ApprovedForProduction, string Notes);
