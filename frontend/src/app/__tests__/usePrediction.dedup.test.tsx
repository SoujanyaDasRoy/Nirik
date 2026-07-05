/**
 * Regression test for Bug #9: analyzeFile effect loop in ScreeningTab.
 *
 * The ScreeningTab useEffect fires on [selectedIdx, activeResult?.status,
 * analyzeFile]. The parent re-renders caused analyzeFile to receive a fresh
 * closure each render; if the status lingered in "pending" between effect
 * ticks, multiple POSTs would fire for the same file.
 *
 * Fix: analyzeFile is wrapped in useCallback AND guarded by an in-flight
 * Set so concurrent calls for the same index short-circuit.
 */
import { renderHook, act } from "@testing-library/react";
import { usePrediction } from "../hooks/usePrediction";

if (typeof window !== "undefined" && !window.URL.createObjectURL) {
  window.URL.createObjectURL = jest.fn(
    (file: any) => `blob:http://localhost/${file?.name || "x"}`
  );
  window.URL.revokeObjectURL = jest.fn();
}

function makeFile(name = "x.png") {
  return new File(["dummy"], name, { type: "image/png" });
}

describe("usePrediction analyzeFile dedup", () => {
  let mockFetch: jest.Mock;

  beforeEach(() => {
    mockFetch = jest.fn();
    global.fetch = mockFetch;
    jest.spyOn(console, "error").mockImplementation(() => {});
  });

  afterEach(() => {
    jest.restoreAllMocks();
  });

  it("does not call fetch twice for the same index when analyzeFile is invoked concurrently", async () => {
    let resolveFetch: (v: any) => void = () => {};
    mockFetch.mockImplementation(
      () =>
        new Promise((resolve) => {
          resolveFetch = resolve;
        })
    );

    const mockSetResults = jest.fn();
    const mockSetSelectedIdx = jest.fn();
    const file = makeFile();
    const { result } = renderHook(() =>
      usePrediction(
        [file],
        [{ filename: file.name, status: "pending" }],
        mockSetResults,
        mockSetSelectedIdx
      )
    );

    // Kick off both calls back-to-back inside a single act() so the second
    // call enters while the first is awaiting the unresolved fetch.
    let firstPromise: Promise<void> | undefined;
    let secondPromise: Promise<void> | undefined;
    act(() => {
      firstPromise = result.current.analyzeFile(0);
      secondPromise = result.current.analyzeFile(0);
    });

    // Let the microtask queue run so both calls have entered analyzeFile.
    await new Promise((resolve) => setTimeout(resolve, 0));

    // Even though we entered twice, the in-flight Set should have allowed
    // only one network request.
    expect(mockFetch).toHaveBeenCalledTimes(1);

    // Resolve the single in-flight request so promises settle and we can
    // safely complete the test without leaking.
    await act(async () => {
      resolveFetch({
        ok: true,
        text: async () =>
          JSON.stringify({ prediction: "Normal", confidence: 0.9, is_tb: false }),
      });
      await Promise.all([firstPromise, secondPromise]);
    });

    expect(mockFetch).toHaveBeenCalledTimes(1);
  });

  it("allows sequential calls for the same index once the previous one finishes", async () => {
    mockFetch
      .mockResolvedValueOnce({
        ok: true,
        text: async () =>
          JSON.stringify({ prediction: "Normal", confidence: 0.9, is_tb: false }),
      })
      .mockResolvedValueOnce({
        ok: true,
        text: async () =>
          JSON.stringify({ prediction: "Normal", confidence: 0.9, is_tb: false }),
      });

    const mockSetResults = jest.fn();
    const mockSetSelectedIdx = jest.fn();
    const file = makeFile();
    const { result } = renderHook(() =>
      usePrediction(
        [file],
        [{ filename: file.name, status: "pending" }],
        mockSetResults,
        mockSetSelectedIdx
      )
    );

    await act(async () => {
      await result.current.analyzeFile(0);
    });
    await act(async () => {
      await result.current.analyzeFile(0);
    });

    expect(mockFetch).toHaveBeenCalledTimes(2);
  });
});