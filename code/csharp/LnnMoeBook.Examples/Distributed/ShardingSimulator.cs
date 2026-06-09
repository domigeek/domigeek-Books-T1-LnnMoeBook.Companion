using System.Globalization;
using LnnMoeBook.Examples.MoE;

namespace LnnMoeBook.Examples.Distributed;

public sealed record DeviceSpec(
    int Id,
    string Name,
    float ComputeCostPerExpertToken,
    float TransferCostPerRemoteExpertToken,
    int ExpertCapacity);

public sealed record ExpertPlacement(
    int Expert,
    int DeviceId);

public sealed record ShardingConfiguration(
    IReadOnlyList<DeviceSpec> Devices,
    IReadOnlyList<ExpertPlacement> Placements,
    IReadOnlyList<int> TokenOriginDevices);

public sealed record ExpertTraffic(
    int Expert,
    int DeviceId,
    int EvaluationCount,
    int RemoteEvaluationCount,
    float RoutingMass);

public sealed record DeviceLoad(
    int DeviceId,
    string DeviceName,
    int ExpertCount,
    int OriginTokenCount,
    int ExpertEvaluationCount,
    int RemoteExpertEvaluationCount,
    float RoutingMass,
    float ComputeCost,
    float CommunicationCost,
    float TotalCost,
    float CostShare,
    float ExpertCapacityRatio);

public sealed record ShardingSimulationResult(
    ShardingConfiguration Configuration,
    TopKRoutingResult Routing,
    IReadOnlyList<DeviceLoad> DeviceLoads,
    IReadOnlyList<ExpertTraffic> ExpertTraffic,
    int ActiveExpertEvaluations,
    int DenseExpertEvaluations,
    int RemoteDispatchCount,
    float RemoteDispatchFraction,
    float TotalComputeCost,
    float TotalCommunicationCost,
    float TotalCost,
    float LoadImbalance);

public sealed record ShardingSimulatorReport(
    TopKRoutingResult Routing,
    ShardingSimulationResult RoundRobin,
    ShardingSimulationResult Concentrated)
{
    public float ImbalanceDelta => Concentrated.LoadImbalance - RoundRobin.LoadImbalance;
}

public static class ShardingSimulator
{
    public static ShardingSimulatorReport RunDefault()
    {
        var batch = SparseMoeLayer.GenerateSyntheticBatch(tokensPerClass: 6);
        var forward = SparseMoeLayer.Forward(batch, SparseMoeLayerOptions.Default);
        var devices = DefaultDevices(deviceCount: 2);
        var tokenOrigins = GenerateRoundRobinTokenOrigins(
            forward.Routing.Input.TokenCount,
            devices.Select(device => device.Id).ToArray());
        var roundRobin = CreateRoundRobinConfiguration(
            expertCount: forward.Routing.Options.ExpertCount,
            devices,
            tokenOrigins);
        var concentrated = CreateConcentratedConfiguration(
            expertCount: forward.Routing.Options.ExpertCount,
            devices,
            tokenOrigins,
            deviceId: devices[0].Id);

        return new ShardingSimulatorReport(
            forward.Routing,
            Simulate(forward.Routing, roundRobin),
            Simulate(forward.Routing, concentrated));
    }

    public static IReadOnlyList<DeviceSpec> DefaultDevices(int deviceCount)
    {
        if (deviceCount <= 0)
        {
            throw new ArgumentOutOfRangeException(nameof(deviceCount), "Device count must be positive.");
        }

        var devices = new DeviceSpec[deviceCount];
        for (var device = 0; device < deviceCount; device++)
        {
            devices[device] = new DeviceSpec(
                device,
                string.Create(CultureInfo.InvariantCulture, $"device-{device}"),
                ComputeCostPerExpertToken: 1.0f + (0.05f * device),
                TransferCostPerRemoteExpertToken: 0.25f,
                ExpertCapacity: 2);
        }

        return devices;
    }

    public static IReadOnlyList<int> GenerateRoundRobinTokenOrigins(
        int tokenCount,
        IReadOnlyList<int> deviceIds)
    {
        if (tokenCount <= 0)
        {
            throw new ArgumentOutOfRangeException(nameof(tokenCount), "Token count must be positive.");
        }

        if (deviceIds.Count == 0)
        {
            throw new ArgumentException("At least one device id is required.", nameof(deviceIds));
        }

        var origins = new int[tokenCount];
        for (var token = 0; token < tokenCount; token++)
        {
            origins[token] = deviceIds[token % deviceIds.Count];
        }

        return origins;
    }

    public static ShardingConfiguration CreateRoundRobinConfiguration(
        int expertCount,
        IReadOnlyList<DeviceSpec> devices,
        IReadOnlyList<int> tokenOriginDevices)
    {
        if (expertCount <= 0)
        {
            throw new ArgumentOutOfRangeException(nameof(expertCount), "Expert count must be positive.");
        }

        if (devices.Count == 0)
        {
            throw new ArgumentException("At least one device is required.", nameof(devices));
        }

        var placements = new ExpertPlacement[expertCount];
        for (var expert = 0; expert < expertCount; expert++)
        {
            placements[expert] = new ExpertPlacement(
                expert,
                devices[expert % devices.Count].Id);
        }

        return new ShardingConfiguration(devices, placements, tokenOriginDevices);
    }

    public static ShardingConfiguration CreateConcentratedConfiguration(
        int expertCount,
        IReadOnlyList<DeviceSpec> devices,
        IReadOnlyList<int> tokenOriginDevices,
        int deviceId)
    {
        if (expertCount <= 0)
        {
            throw new ArgumentOutOfRangeException(nameof(expertCount), "Expert count must be positive.");
        }

        if (devices.All(device => device.Id != deviceId))
        {
            throw new ArgumentException("Target device must exist.", nameof(deviceId));
        }

        var placements = Enumerable
            .Range(0, expertCount)
            .Select(expert => new ExpertPlacement(expert, deviceId))
            .ToArray();

        return new ShardingConfiguration(devices, placements, tokenOriginDevices);
    }

    public static ShardingSimulationResult Simulate(
        TopKRoutingResult routing,
        ShardingConfiguration configuration)
    {
        ValidateRouting(routing);
        ValidateConfiguration(configuration, routing);

        var devicesById = configuration.Devices.ToDictionary(device => device.Id);
        var placementsByExpert = configuration.Placements.ToDictionary(placement => placement.Expert);
        var deviceLoads = configuration.Devices.ToDictionary(
            device => device.Id,
            device => new MutableDeviceLoad(device));
        var expertTraffic = configuration.Placements.ToDictionary(
            placement => placement.Expert,
            placement => new MutableExpertTraffic(placement.Expert, placement.DeviceId));

        foreach (var placement in configuration.Placements)
        {
            deviceLoads[placement.DeviceId].ExpertCount++;
        }

        foreach (var origin in configuration.TokenOriginDevices)
        {
            deviceLoads[origin].OriginTokenCount++;
        }

        var remoteDispatchCount = 0;
        foreach (var route in routing.Routes)
        {
            var originDevice = configuration.TokenOriginDevices[route.Token];
            foreach (var expert in route.ExpertIndices)
            {
                var placement = placementsByExpert[expert];
                var targetDevice = devicesById[placement.DeviceId];
                var weight = route.SparseWeights[expert];
                var isRemote = placement.DeviceId != originDevice;

                var load = deviceLoads[placement.DeviceId];
                load.ExpertEvaluationCount++;
                load.RoutingMass += weight;
                load.ComputeCost += targetDevice.ComputeCostPerExpertToken;

                var traffic = expertTraffic[expert];
                traffic.EvaluationCount++;
                traffic.RoutingMass += weight;

                if (isRemote)
                {
                    remoteDispatchCount++;
                    load.RemoteExpertEvaluationCount++;
                    load.CommunicationCost += targetDevice.TransferCostPerRemoteExpertToken;
                    traffic.RemoteEvaluationCount++;
                }
            }
        }

        var totalComputeCost = deviceLoads.Values.Sum(load => load.ComputeCost);
        var totalCommunicationCost = deviceLoads.Values.Sum(load => load.CommunicationCost);
        var totalCost = totalComputeCost + totalCommunicationCost;
        var loads = deviceLoads.Values
            .OrderBy(load => load.Device.Id)
            .Select(load => load.ToDeviceLoad(totalCost))
            .ToArray();
        var averageLoad = totalCost / loads.Length;
        var loadImbalance = averageLoad > 0.0f
            ? loads.Max(load => load.TotalCost) / averageLoad
            : 0.0f;
        var activeEvaluations = routing.Input.TokenCount * routing.Options.TopK;
        var denseEvaluations = routing.Input.TokenCount * routing.Options.ExpertCount;

        return new ShardingSimulationResult(
            configuration,
            routing,
            loads,
            expertTraffic.Values
                .OrderBy(traffic => traffic.Expert)
                .Select(traffic => traffic.ToExpertTraffic())
                .ToArray(),
            activeEvaluations,
            denseEvaluations,
            remoteDispatchCount,
            (float)remoteDispatchCount / activeEvaluations,
            totalComputeCost,
            totalCommunicationCost,
            totalCost,
            loadImbalance);
    }

    public static string ToDeviceCsv(ShardingSimulationResult result)
    {
        var lines = new List<string>
        {
            "device,experts,origin_tokens,expert_evaluations,remote_evaluations,routing_mass,compute_cost,communication_cost,total_cost,cost_share,capacity_ratio"
        };

        foreach (var load in result.DeviceLoads)
        {
            lines.Add(string.Create(
                CultureInfo.InvariantCulture,
                $"{load.DeviceName},{load.ExpertCount},{load.OriginTokenCount},{load.ExpertEvaluationCount},{load.RemoteExpertEvaluationCount},{load.RoutingMass:0.######},{load.ComputeCost:0.######},{load.CommunicationCost:0.######},{load.TotalCost:0.######},{load.CostShare:0.######},{load.ExpertCapacityRatio:0.######}"));
        }

        return string.Join(Environment.NewLine, lines) + Environment.NewLine;
    }

    public static string FormatReport(ShardingSimulatorReport report)
    {
        return string.Create(
            CultureInfo.InvariantCulture,
            $"sharding: devices={report.RoundRobin.DeviceLoads.Count}, experts={report.Routing.Options.ExpertCount}, active={report.RoundRobin.ActiveExpertEvaluations}/{report.RoundRobin.DenseExpertEvaluations}, rr_cost={report.RoundRobin.TotalCost:0.###}, rr_imbalance={report.RoundRobin.LoadImbalance:0.###}, concentrated_imbalance={report.Concentrated.LoadImbalance:0.###}, remote={report.RoundRobin.RemoteDispatchFraction:0.###}");
    }

    private static void ValidateRouting(TopKRoutingResult routing)
    {
        if (routing.Input.TokenCount <= 0)
        {
            throw new ArgumentOutOfRangeException(nameof(routing), "Routing must contain at least one token.");
        }

        if (routing.Options.ExpertCount <= 0)
        {
            throw new ArgumentOutOfRangeException(nameof(routing), "Routing must contain at least one expert.");
        }

        if (routing.Routes.Count != routing.Input.TokenCount)
        {
            throw new ArgumentException("Route count must match token count.", nameof(routing));
        }

        foreach (var route in routing.Routes)
        {
            foreach (var expert in route.ExpertIndices)
            {
                if (expert < 0 || expert >= routing.Options.ExpertCount)
                {
                    throw new ArgumentException("Routed expert index is out of range.", nameof(routing));
                }
            }
        }
    }

    private static void ValidateConfiguration(
        ShardingConfiguration configuration,
        TopKRoutingResult routing)
    {
        if (configuration.Devices.Count == 0)
        {
            throw new ArgumentException("At least one device is required.", nameof(configuration));
        }

        var deviceIds = new HashSet<int>();
        foreach (var device in configuration.Devices)
        {
            if (!deviceIds.Add(device.Id))
            {
                throw new ArgumentException("Device ids must be unique.", nameof(configuration));
            }

            if (string.IsNullOrWhiteSpace(device.Name))
            {
                throw new ArgumentException("Device names must not be empty.", nameof(configuration));
            }

            if (device.ComputeCostPerExpertToken <= 0.0f)
            {
                throw new ArgumentOutOfRangeException(nameof(configuration), "Compute cost must be positive.");
            }

            if (device.TransferCostPerRemoteExpertToken < 0.0f)
            {
                throw new ArgumentOutOfRangeException(nameof(configuration), "Transfer cost must be non-negative.");
            }

            if (device.ExpertCapacity <= 0)
            {
                throw new ArgumentOutOfRangeException(nameof(configuration), "Expert capacity must be positive.");
            }
        }

        if (configuration.Placements.Count != routing.Options.ExpertCount)
        {
            throw new ArgumentException("Placement count must match expert count.", nameof(configuration));
        }

        var experts = new HashSet<int>();
        foreach (var placement in configuration.Placements)
        {
            if (placement.Expert < 0 || placement.Expert >= routing.Options.ExpertCount)
            {
                throw new ArgumentException("Expert placement index is out of range.", nameof(configuration));
            }

            if (!experts.Add(placement.Expert))
            {
                throw new ArgumentException("Each expert must be placed exactly once.", nameof(configuration));
            }

            if (!deviceIds.Contains(placement.DeviceId))
            {
                throw new ArgumentException("Expert placement references an unknown device.", nameof(configuration));
            }
        }

        if (configuration.TokenOriginDevices.Count != routing.Input.TokenCount)
        {
            throw new ArgumentException("Token origin count must match token count.", nameof(configuration));
        }

        foreach (var origin in configuration.TokenOriginDevices)
        {
            if (!deviceIds.Contains(origin))
            {
                throw new ArgumentException("Token origin references an unknown device.", nameof(configuration));
            }
        }
    }

    private sealed class MutableDeviceLoad
    {
        public MutableDeviceLoad(DeviceSpec device)
        {
            Device = device;
        }

        public DeviceSpec Device { get; }
        public int ExpertCount { get; set; }
        public int OriginTokenCount { get; set; }
        public int ExpertEvaluationCount { get; set; }
        public int RemoteExpertEvaluationCount { get; set; }
        public float RoutingMass { get; set; }
        public float ComputeCost { get; set; }
        public float CommunicationCost { get; set; }

        public DeviceLoad ToDeviceLoad(float totalCost)
        {
            var deviceTotal = ComputeCost + CommunicationCost;
            return new DeviceLoad(
                Device.Id,
                Device.Name,
                ExpertCount,
                OriginTokenCount,
                ExpertEvaluationCount,
                RemoteExpertEvaluationCount,
                RoutingMass,
                ComputeCost,
                CommunicationCost,
                deviceTotal,
                totalCost > 0.0f ? deviceTotal / totalCost : 0.0f,
                (float)ExpertCount / Device.ExpertCapacity);
        }
    }

    private sealed class MutableExpertTraffic
    {
        public MutableExpertTraffic(int expert, int deviceId)
        {
            Expert = expert;
            DeviceId = deviceId;
        }

        public int Expert { get; }
        public int DeviceId { get; }
        public int EvaluationCount { get; set; }
        public int RemoteEvaluationCount { get; set; }
        public float RoutingMass { get; set; }

        public ExpertTraffic ToExpertTraffic()
        {
            return new ExpertTraffic(
                Expert,
                DeviceId,
                EvaluationCount,
                RemoteEvaluationCount,
                RoutingMass);
        }
    }
}
