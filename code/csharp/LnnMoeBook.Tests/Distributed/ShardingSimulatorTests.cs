using LnnMoeBook.Examples.Distributed;
using LnnMoeBook.Examples.MoE;

namespace LnnMoeBook.Tests.Distributed;

public sealed class ShardingSimulatorTests
{
    [Fact]
    public void DefaultDevicesAreDeterministic()
    {
        var first = ShardingSimulator.DefaultDevices(deviceCount: 2);
        var second = ShardingSimulator.DefaultDevices(deviceCount: 2);

        Assert.Equal(2, first.Count);
        Assert.Equal(first, second);
        Assert.Equal("device-0", first[0].Name);
        Assert.Equal("device-1", first[1].Name);
        Assert.All(first, device =>
        {
            Assert.True(device.ComputeCostPerExpertToken > 0.0f);
            Assert.True(device.TransferCostPerRemoteExpertToken >= 0.0f);
            Assert.True(device.ExpertCapacity > 0);
        });
    }

    [Fact]
    public void RoundRobinTokenOriginsAreDeterministic()
    {
        var origins = ShardingSimulator.GenerateRoundRobinTokenOrigins(
            tokenCount: 6,
            new[] { 10, 20 });

        Assert.Equal(new[] { 10, 20, 10, 20, 10, 20 }, origins);
    }

    [Fact]
    public void RoundRobinConfigurationSpreadsExpertsAcrossDevices()
    {
        var devices = ShardingSimulator.DefaultDevices(deviceCount: 2);
        var origins = ShardingSimulator.GenerateRoundRobinTokenOrigins(8, devices.Select(device => device.Id).ToArray());

        var configuration = ShardingSimulator.CreateRoundRobinConfiguration(
            expertCount: 4,
            devices,
            origins);

        Assert.Equal(new[] { 0, 1, 0, 1 }, configuration.Placements.Select(placement => placement.DeviceId));
    }

    [Fact]
    public void ConcentratedConfigurationPlacesAllExpertsOnOneDevice()
    {
        var devices = ShardingSimulator.DefaultDevices(deviceCount: 2);
        var origins = ShardingSimulator.GenerateRoundRobinTokenOrigins(8, devices.Select(device => device.Id).ToArray());

        var configuration = ShardingSimulator.CreateConcentratedConfiguration(
            expertCount: 4,
            devices,
            origins,
            deviceId: 0);

        Assert.All(configuration.Placements, placement => Assert.Equal(0, placement.DeviceId));
    }

    [Fact]
    public void SimulateComputesActiveDenseAndRemoteDispatchCounts()
    {
        var routing = BuildDefaultRouting();
        var devices = ShardingSimulator.DefaultDevices(deviceCount: 2);
        var origins = ShardingSimulator.GenerateRoundRobinTokenOrigins(
            routing.Input.TokenCount,
            devices.Select(device => device.Id).ToArray());
        var configuration = ShardingSimulator.CreateRoundRobinConfiguration(
            routing.Options.ExpertCount,
            devices,
            origins);

        var result = ShardingSimulator.Simulate(routing, configuration);

        Assert.Equal(routing.Input.TokenCount * routing.Options.TopK, result.ActiveExpertEvaluations);
        Assert.Equal(routing.Input.TokenCount * routing.Options.ExpertCount, result.DenseExpertEvaluations);
        Assert.InRange(result.RemoteDispatchFraction, 0.0f, 1.0f);
        Assert.Equal(result.RemoteDispatchCount, result.DeviceLoads.Sum(load => load.RemoteExpertEvaluationCount));
    }

    [Fact]
    public void DeviceLoadsAndExpertTrafficAreCoherent()
    {
        var report = ShardingSimulator.RunDefault();
        var result = report.RoundRobin;

        Assert.Equal(2, result.DeviceLoads.Count);
        Assert.Equal(result.Routing.Options.ExpertCount, result.ExpertTraffic.Count);
        Assert.Equal(result.ActiveExpertEvaluations, result.DeviceLoads.Sum(load => load.ExpertEvaluationCount));
        Assert.Equal(result.ActiveExpertEvaluations, result.ExpertTraffic.Sum(traffic => traffic.EvaluationCount));
        Assert.InRange(result.ExpertTraffic.Sum(traffic => traffic.RoutingMass), result.Routing.Input.TokenCount - 0.0001f, result.Routing.Input.TokenCount + 0.0001f);
        Assert.InRange(result.DeviceLoads.Sum(load => load.CostShare), 0.99999f, 1.00001f);
    }

    [Fact]
    public void TotalCostIsSumOfComputeAndCommunication()
    {
        var report = ShardingSimulator.RunDefault();
        var result = report.RoundRobin;

        Assert.Equal(result.TotalComputeCost + result.TotalCommunicationCost, result.TotalCost, precision: 5);
        Assert.Equal(result.TotalComputeCost, result.DeviceLoads.Sum(load => load.ComputeCost), precision: 5);
        Assert.Equal(result.TotalCommunicationCost, result.DeviceLoads.Sum(load => load.CommunicationCost), precision: 5);
    }

    [Fact]
    public void RoundRobinPlacementHasLowerImbalanceThanConcentratedPlacement()
    {
        var report = ShardingSimulator.RunDefault();

        Assert.True(report.RoundRobin.LoadImbalance < report.Concentrated.LoadImbalance);
        Assert.True(report.ImbalanceDelta > 0.0f);
    }

    [Fact]
    public void ConcentratedPlacementExceedsExpertCapacityOnFirstDevice()
    {
        var report = ShardingSimulator.RunDefault();

        var first = report.Concentrated.DeviceLoads.Single(load => load.DeviceId == 0);

        Assert.True(first.ExpertCapacityRatio > 1.0f);
    }

    [Fact]
    public void SingleDeviceConfigurationHasNoRemoteDispatches()
    {
        var routing = BuildDefaultRouting();
        var devices = ShardingSimulator.DefaultDevices(deviceCount: 1);
        var origins = ShardingSimulator.GenerateRoundRobinTokenOrigins(
            routing.Input.TokenCount,
            devices.Select(device => device.Id).ToArray());
        var configuration = ShardingSimulator.CreateRoundRobinConfiguration(
            routing.Options.ExpertCount,
            devices,
            origins);

        var result = ShardingSimulator.Simulate(routing, configuration);

        Assert.Single(result.DeviceLoads);
        Assert.Equal(0, result.RemoteDispatchCount);
        Assert.Equal(0.0f, result.RemoteDispatchFraction);
        Assert.Equal(1.0f, result.LoadImbalance);
    }

    [Fact]
    public void MoreDevicesThanExpertsKeepsUnusedDevicesVisible()
    {
        var batch = SparseMoeLayer.GenerateSyntheticBatch(tokensPerClass: 2);
        var routing = SparseMoeLayer.Forward(batch, SparseMoeLayerOptions.Default with { TopK = 1 }).Routing;
        var devices = ShardingSimulator.DefaultDevices(deviceCount: 6);
        var origins = ShardingSimulator.GenerateRoundRobinTokenOrigins(
            routing.Input.TokenCount,
            devices.Select(device => device.Id).ToArray());
        var configuration = ShardingSimulator.CreateRoundRobinConfiguration(
            routing.Options.ExpertCount,
            devices,
            origins);

        var result = ShardingSimulator.Simulate(routing, configuration);

        Assert.Equal(6, result.DeviceLoads.Count);
        Assert.Equal(2, result.DeviceLoads.Count(load => load.ExpertCount == 0));
    }

    [Fact]
    public void RoundRobinConfigurationSupportsNonDivisibleExpertCounts()
    {
        var devices = ShardingSimulator.DefaultDevices(deviceCount: 3);
        var origins = ShardingSimulator.GenerateRoundRobinTokenOrigins(10, devices.Select(device => device.Id).ToArray());

        var configuration = ShardingSimulator.CreateRoundRobinConfiguration(
            expertCount: 5,
            devices,
            origins);

        Assert.Equal(new[] { 0, 1, 2, 0, 1 }, configuration.Placements.Select(placement => placement.DeviceId));
    }

    [Fact]
    public void CsvContainsStableHeaderAndOneLinePerDevice()
    {
        var report = ShardingSimulator.RunDefault();

        var csv = ShardingSimulator.ToDeviceCsv(report.RoundRobin);
        var lines = csv
            .Split('\n', StringSplitOptions.RemoveEmptyEntries)
            .Select(line => line.TrimEnd('\r'))
            .ToArray();

        Assert.Equal(report.RoundRobin.DeviceLoads.Count + 1, lines.Length);
        Assert.Equal("device,experts,origin_tokens,expert_evaluations,remote_evaluations,routing_mass,compute_cost,communication_cost,total_cost,cost_share,capacity_ratio", lines[0]);
        Assert.StartsWith("device-0,", lines[1], StringComparison.Ordinal);
    }

    [Fact]
    public void SimulateRejectsPlacementCountMismatch()
    {
        var routing = BuildDefaultRouting();
        var devices = ShardingSimulator.DefaultDevices(deviceCount: 2);
        var origins = ShardingSimulator.GenerateRoundRobinTokenOrigins(
            routing.Input.TokenCount,
            devices.Select(device => device.Id).ToArray());
        var configuration = new ShardingConfiguration(
            devices,
            new[] { new ExpertPlacement(0, 0) },
            origins);

        Assert.Throws<ArgumentException>(() =>
            ShardingSimulator.Simulate(routing, configuration));
    }

    [Fact]
    public void SimulateRejectsUnknownDevicePlacement()
    {
        var routing = BuildDefaultRouting();
        var devices = ShardingSimulator.DefaultDevices(deviceCount: 2);
        var origins = ShardingSimulator.GenerateRoundRobinTokenOrigins(
            routing.Input.TokenCount,
            devices.Select(device => device.Id).ToArray());
        var placements = Enumerable
            .Range(0, routing.Options.ExpertCount)
            .Select(expert => new ExpertPlacement(expert, expert == 0 ? 99 : 0))
            .ToArray();
        var configuration = new ShardingConfiguration(devices, placements, origins);

        Assert.Throws<ArgumentException>(() =>
            ShardingSimulator.Simulate(routing, configuration));
    }

    [Fact]
    public void SimulateRejectsInvalidTokenOrigins()
    {
        var routing = BuildDefaultRouting();
        var devices = ShardingSimulator.DefaultDevices(deviceCount: 2);
        var configuration = ShardingSimulator.CreateRoundRobinConfiguration(
            routing.Options.ExpertCount,
            devices,
            new[] { 0, 1 });

        Assert.Throws<ArgumentException>(() =>
            ShardingSimulator.Simulate(routing, configuration));
    }

    [Fact]
    public void DeviceValidationRejectsDuplicateIds()
    {
        var routing = BuildDefaultRouting();
        var devices = new[]
        {
            new DeviceSpec(0, "a", 1.0f, 0.1f, 2),
            new DeviceSpec(0, "b", 1.0f, 0.1f, 2)
        };
        var origins = Enumerable.Repeat(0, routing.Input.TokenCount).ToArray();
        var configuration = ShardingSimulator.CreateRoundRobinConfiguration(
            routing.Options.ExpertCount,
            devices,
            origins);

        Assert.Throws<ArgumentException>(() =>
            ShardingSimulator.Simulate(routing, configuration));
    }

    [Fact]
    public void SimulateRejectsOutOfRangeRoutedExpert()
    {
        var input = new TokenRoutingInput(new[] { 1.0f, 0.0f }, TokenCount: 1, ExpertCount: 2);
        var route = new TopKTokenRoute(
            Token: 0,
            ExpertIndices: new[] { 3 },
            ExpertWeights: new[] { 1.0f },
            SparseWeights: new[] { 1.0f, 0.0f });
        var routing = new TopKRoutingResult(
            input,
            new TopKRoutingOptions(ExpertCount: 2, TopK: 1, Temperature: 1.0f),
            new[] { route });
        var devices = ShardingSimulator.DefaultDevices(deviceCount: 1);
        var configuration = ShardingSimulator.CreateRoundRobinConfiguration(
            expertCount: 2,
            devices,
            new[] { 0 });

        Assert.Throws<ArgumentException>(() =>
            ShardingSimulator.Simulate(routing, configuration));
    }

    [Fact]
    public void FormatReportContainsStableFields()
    {
        var text = ShardingSimulator.FormatReport(ShardingSimulator.RunDefault());

        Assert.Contains("sharding", text);
        Assert.Contains("devices=2", text);
        Assert.Contains("experts=4", text);
        Assert.Contains("active=", text);
        Assert.Contains("rr_cost=", text);
        Assert.Contains("rr_imbalance=", text);
        Assert.Contains("concentrated_imbalance=", text);
        Assert.Contains("remote=", text);
    }

    private static TopKRoutingResult BuildDefaultRouting()
    {
        var batch = SparseMoeLayer.GenerateSyntheticBatch(tokensPerClass: 4);
        return SparseMoeLayer.Forward(batch, SparseMoeLayerOptions.Default).Routing;
    }
}
