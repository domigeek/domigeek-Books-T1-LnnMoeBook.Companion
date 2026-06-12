using System.Collections.Generic;
using System.Globalization;
using System.IO;
using LnnMoeBook.Examples.LTC;

namespace LnnMoeBook.Solutions.Ch036;

public static class Ex036003
{
    public static void Main()
    {
        var report = SimpleLtcCell.RunDefault();
        WriteLossHistoryCsv(report.Training.LossHistory, "loss-ltc.csv");
    }

    public static void WriteLossHistoryCsv(IReadOnlyList<float> lossHistory, string path)
    {
        var rows = new List<string> { "epoch,loss" };

        for (var epoch = 0; epoch < lossHistory.Count; epoch++)
        {
            rows.Add(string.Create(
                CultureInfo.InvariantCulture,
                $"{epoch},{lossHistory[epoch]}"));
        }

        File.WriteAllLines(path, rows);
    }
}
