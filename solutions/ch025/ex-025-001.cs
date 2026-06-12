using System;
using System.Collections.Generic;
using System.Linq;

namespace LnnMoeBook.Solutions.Ch025;

public static class Ex025001
{
    public static void Main()
    {
        var expertToDevice = AssignExpertsRoundRobin(expertCount: 8, deviceCount: 4);
        var tokenExpertAssignments = new[] { 0, 0, 1, 2, 2, 2, 4, 7, 7, 7, 7 };
        var deviceLoad = ComputeDeviceLoad(expertToDevice, tokenExpertAssignments);

        foreach (var pair in deviceLoad.OrderBy(pair => pair.Key))
        {
            Console.WriteLine($"device {pair.Key}: {pair.Value}");
        }

        if (deviceLoad[3] != 4)
        {
            throw new InvalidOperationException("Le device 3 devrait être le plus chargé dans cet exemple.");
        }
    }

    public static IReadOnlyDictionary<int, int> AssignExpertsRoundRobin(int expertCount, int deviceCount)
    {
        if (expertCount <= 0 || deviceCount <= 0)
        {
            throw new ArgumentOutOfRangeException("Le nombre d'experts et de devices doit être positif.");
        }

        return Enumerable
            .Range(0, expertCount)
            .ToDictionary(expert => expert, expert => expert % deviceCount);
    }

    public static IReadOnlyDictionary<int, int> ComputeDeviceLoad(
        IReadOnlyDictionary<int, int> expertToDevice,
        IReadOnlyList<int> tokenExpertAssignments)
    {
        var deviceLoad = expertToDevice.Values
            .Distinct()
            .ToDictionary(device => device, _ => 0);

        foreach (var expert in tokenExpertAssignments)
        {
            var device = expertToDevice[expert];
            deviceLoad[device]++;
        }

        return deviceLoad;
    }
}
