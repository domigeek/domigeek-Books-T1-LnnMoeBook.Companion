using LnnMoeBook.Core;

Console.WriteLine(BookProjectInfo.GetStatus());
Console.WriteLine(TorchSharpDiagnostics.RunSmokeTest().ToDiagnosticLine());
Console.WriteLine("CLI project initialized. Diagnostics and inference commands will be added in later tasks.");
