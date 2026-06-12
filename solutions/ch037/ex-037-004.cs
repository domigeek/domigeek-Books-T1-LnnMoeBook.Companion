using System;
using LnnMoeBook.Examples.Distributed;

namespace LnnMoeBook.Solutions.Ch037;

public static class Ex037004
{
    public static void Main()
    {
        var report = ShardingSimulator.RunDefault();

        Console.WriteLine($"round-robin imbalance: {report.RoundRobin.LoadImbalance:0.###}");
        Console.WriteLine($"concentrated imbalance: {report.Concentrated.LoadImbalance:0.###}");
        Console.WriteLine($"delta: {report.ImbalanceDelta:0.###}");

        if (report.Concentrated.LoadImbalance <= report.RoundRobin.LoadImbalance)
        {
            throw new InvalidOperationException(
                "Dans ce scénario pédagogique, le placement concentré devrait augmenter le déséquilibre.");
        }

        foreach (var load in report.Concentrated.DeviceLoads)
        {
            Console.WriteLine(
                $"device={load.DeviceId}, experts={load.ExpertCount}, evals={load.ExpertEvaluationCount}, total={load.TotalCost:0.###}");
        }
    }
}
