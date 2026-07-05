/**
 * Regression test for Bug #3: res.json() called before res.ok check.
 *
 * The current implementation parses the JSON body unconditionally, then
 * throws a generic SyntaxError when the server returns a 5xx with a
 * non-JSON body (e.g. plain text "Internal server error"). The error
 * message then masks the real cause and is shown to the clinician as
 * "Server connection failed" or similar — mis-classifying a backend
 * failure as a client/network problem.
 *
 * Correct behavior: only call res.json() after confirming res.ok; surface
 * the server-provided error text when available.
 */
import { renderHook, act } from "@testing-library/react";
import { usePrediction } from "../hooks/usePrediction";

// jsdom doesn't ship URL.createObjectURL — needed by analyzeFile.
if (typeof window !== "undefined" && !window.URL.createObjectURL) {
  window.URL.createObjectURL = jest.fn(
    (file: any) => `blob:http://localhost/${file?.name || "x"}`
  );
  window.URL.revokeObjectURL = jest.fn();
}

function makeFile() {
  return new File(["dummy"], "test.png", { type: "image/png" });
}

describe("usePrediction error handling — non-JSON 500 response", () => {
  let mockFetch: jest.Mock;

  beforeEach(() => {
    mockFetch = jest.fn();
    global.fetch = mockFetch;
    jest.spyOn(console, "error").mockImplementation(() => {});
  });

  afterEach(() => {
    jest.restoreAllMocks();
  });

  it("does not call res.json() when response is not ok with non-JSON body", async () => {
    // Server returns 500 with plain-text body. res.json() would throw.
    const res = {
      ok: false,
      status: 500,
      json: jest.fn(() => {
        throw new SyntaxError("Unexpected token I in JSON at position 0");
      }),
      text: jest.fn(async () => "Internal server error"),
    };
    mockFetch.mockResolvedValueOnce(res);

    const mockSetResults = jest.fn();
    const mockSetSelectedIdx = jest.fn();
    const { result } = renderHook(() =>
      usePrediction(
        [makeFile()],
        [{ filename: "test.png", status: "pending" }],
        mockSetResults,
        mockSetSelectedIdx
      )
    );

    await act(async () => {
      await result.current.analyzeFile(0);
    });

    // We must NOT have tried to parse the non-JSON body as JSON.
    expect(res.json).not.toHaveBeenCalled();

    // The error message surfaced to the user must NOT be the JSON parse error.
    const lastCall =
      mockSetResults.mock.calls[mockSetResults.mock.calls.length - 1][0];
    const updater = lastCall;
    const finalState = updater([{ filename: "test.png", status: "loading" }]);
    expect(finalState[0].status).toBe("error");
    expect(finalState[0].errorMsg).toBe("Internal server error");
  });
});
