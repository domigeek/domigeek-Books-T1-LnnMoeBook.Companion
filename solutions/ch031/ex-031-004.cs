using System;
using System.Collections.Generic;
using System.Linq;

namespace LnnMoeBook.Solutions.Ch031;

public static class Ex031004
{
    public static RagDiagnostic Diagnose(string query, IReadOnlyList<RetrievedDocument> documents)
    {
        var visibleDocuments = documents
            .Where(document => document.UserCanRead)
            .OrderByDescending(document => document.Score)
            .ToArray();

        var weakSources = visibleDocuments
            .Where(document => document.Score < 0.70f || document.IsExpired)
            .ToArray();

        var canAnswer = visibleDocuments.Length > 0 &&
            visibleDocuments[0].Score >= 0.78f &&
            !visibleDocuments[0].IsExpired;

        var recommendations = new List<string>();

        if (visibleDocuments.Length == 0)
        {
            recommendations.Add("Aucune source accessible : refuser la réponse ou demander plus de contexte.");
        }

        if (weakSources.Length > 0)
        {
            recommendations.Add("Sources faibles ou périmées : ajouter un seuil, un reranker ou une vérification de fraîcheur.");
        }

        if (!canAnswer)
        {
            recommendations.Add("Réponse non suffisamment supportée par les documents récupérés.");
        }

        return new RagDiagnostic(
            Query: query,
            RetrievedCount: documents.Count,
            VisibleCount: visibleDocuments.Length,
            BestScore: visibleDocuments.FirstOrDefault()?.Score ?? 0.0f,
            CanAnswer: canAnswer,
            Recommendations: recommendations);
    }

    public sealed record RetrievedDocument(
        string Id,
        float Score,
        bool IsExpired,
        bool UserCanRead,
        string Source);

    public sealed record RagDiagnostic(
        string Query,
        int RetrievedCount,
        int VisibleCount,
        float BestScore,
        bool CanAnswer,
        IReadOnlyList<string> Recommendations);
}
