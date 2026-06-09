namespace LnnMoeBook.Core;

public static class BookProjectInfo
{
    public const string Title = "Tome I - Reseaux de Neurones Liquides, Mixture-of-Experts et Architectures Cognitives Modulaires";
    public const string PrimaryLanguage = "C#";
    public const string PlannedFramework = "TorchSharp";

    public static string GetStatus()
    {
        return $"{Title} | {PrimaryLanguage} | planned framework: {PlannedFramework}";
    }
}
