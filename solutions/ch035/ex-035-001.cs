using System;
using System.Collections.Generic;

namespace LnnMoeBook.Solutions.Ch035;

public static class Ex035001
{
    public static IEnumerable<EventEnvelope> ConvertVideoWindowToEvents(
        string streamId,
        int window,
        int firstFrame,
        int chunkCount,
        DateTimeOffset timestamp)
    {
        var correlationId = $"{streamId}_window_{window:000}";

        for (var chunk = 0; chunk < chunkCount; chunk++)
        {
            var startFrame = firstFrame + (chunk * 20);
            var endFrame = startFrame + 19;

            yield return new EventEnvelope(
                EventId: $"evt_{window:000}_{chunk:000}",
                CorrelationId: correlationId,
                Sequence: chunk,
                Timestamp: timestamp.AddMilliseconds(chunk * 666),
                Type: "VisionFrameChunk",
                Payload: new Dictionary<string, object>
                {
                    ["frameStart"] = startFrame,
                    ["frameEnd"] = endFrame,
                    ["embeddingRef"] = $"vec_{window:000}_{chunk:000}",
                    ["source"] = streamId,
                    ["targetPipeline"] = new[] { "ED", "OAD", "WoV", "ECAD" }
                });
        }
    }

    public sealed record EventEnvelope(
        string EventId,
        string CorrelationId,
        int Sequence,
        DateTimeOffset Timestamp,
        string Type,
        IReadOnlyDictionary<string, object> Payload);
}
