import { renderHook, act } from "@testing-library/react";
import { render, screen, fireEvent, waitFor } from "@testing-library/react";
import React from "react";
import "@testing-library/jest-dom";
import { useFileUpload } from "../hooks/useFileUpload";
import { usePrediction } from "../hooks/usePrediction";
import { ScreeningTab } from "../components/ScreeningTab";
import Home from "../page";

// Mock URL methods for Node/JSDOM environment
if (typeof window !== "undefined") {
  window.URL.createObjectURL = jest.fn((file) => `blob:http://localhost:3000/${(file as any)?.name || "mock-blob"}`);
  window.URL.revokeObjectURL = jest.fn();
}

// Mock the next/dynamic import
jest.mock("next/dynamic", () => {
  return function mockDynamic(importFn: any) {
    const DummyComponent = ({
      imageBase64,
      heatmapBase64,
      label,
      pixelSpacing,
      viewMode,
      heatmapOpacity,
      lungSegmentationActive,
      boxes,
      setBoxes,
      activeZone,
      annotateMode,
      annotationCanvasRef,
      setViewMode,
      setAnnotateMode,
      observationFocusRegion,
      setHeatmapOpacity,
      setLungSegmentationActive,
      setActiveZone,
      ...props
    }: any) => <div data-testid="mock-dynamic" {...props} />;
    return DummyComponent;
  };
});

// Mock next-themes
jest.mock("next-themes", () => ({
  useTheme: () => ({ theme: "light", setTheme: jest.fn() }),
}));

// Mock next/navigation
jest.mock("next/navigation", () => ({
  useRouter: () => ({
    push: jest.fn(),
    replace: jest.fn(),
    prefetch: jest.fn(),
    back: jest.fn(),
  }),
}));


// Mock individual dynamic components to avoid rendering complications
jest.mock("../components/DicomViewer", () => {
  return function MockDicomViewer() {
    return <div data-testid="mock-dicom-viewer">Mock DicomViewer</div>;
  };
});

jest.mock("../components/AnnotationCanvas", () => {
  return function MockAnnotationCanvas() {
    return <div data-testid="mock-annotation-canvas">Mock AnnotationCanvas</div>;
  };
});



// Setup mock files
const file1 = new File(["dummy content 1"], "chest_xray_1.png", { type: "image/png" });
const file2 = new File(["dummy content 2"], "chest_xray_2.png", { type: "image/png" });
const file3 = new File(["dummy content 3"], "chest_xray_3.png", { type: "image/png" });
const file4 = new File(["dummy content 4"], "chest_xray_4.png", { type: "image/png" });
const file5 = new File(["dummy content 5"], "chest_xray_5.png", { type: "image/png" });

describe("useFileUpload Hook", () => {
  it("should initialize with correct default states", () => {
    const { result } = renderHook(() => useFileUpload());
    expect(result.current.files).toEqual([]);
    expect(result.current.results).toEqual([]);
    expect(result.current.selectedIdx).toBeNull();
    expect(result.current.isDragActive).toBe(false);
  });

  it("should add files and initialize pending results", () => {
    const { result } = renderHook(() => useFileUpload());
    act(() => {
      result.current.addFiles([file1, file2]);
    });
    expect(result.current.files).toEqual([file1, file2]);
    expect(result.current.results).toEqual([
      { filename: "chest_xray_1.png", status: "pending" },
      { filename: "chest_xray_2.png", status: "pending" },
    ]);
  });

  it("should remove correct file and update state", () => {
    const { result } = renderHook(() => useFileUpload());
    act(() => {
      result.current.addFiles([file1, file2, file3]);
      result.current.setSelectedIdx(1);
    });

    expect(result.current.selectedIdx).toBe(1);

    // Remove first file (index 0) -> selected index should adjust to 0
    act(() => {
      result.current.removeFile(0);
    });
    expect(result.current.files).toEqual([file2, file3]);
    expect(result.current.selectedIdx).toBe(0);

    // Select index 1 (which is file3) and remove it -> selected index should set to null since we removed the selected one
    act(() => {
      result.current.setSelectedIdx(1);
    });
    act(() => {
      result.current.removeFile(1);
    });
    expect(result.current.files).toEqual([file2]);
    expect(result.current.selectedIdx).toBeNull();
  });

  it("should clear all files and results", () => {
    const { result } = renderHook(() => useFileUpload());
    act(() => {
      result.current.addFiles([file1, file2]);
      result.current.setSelectedIdx(0);
    });

    act(() => {
      result.current.clearAll();
    });
    expect(result.current.files).toEqual([]);
    expect(result.current.results).toEqual([]);
    expect(result.current.selectedIdx).toBeNull();
  });
});

describe("usePrediction Hook", () => {
  let mockFetch: jest.Mock;

  beforeEach(() => {
    mockFetch = jest.fn();
    global.fetch = mockFetch;
    jest.spyOn(console, "error").mockImplementation(() => { });
  });

  afterEach(() => {
    jest.restoreAllMocks();
  });

  it("should analyze a single file successfully", async () => {
    const mockResponseData = {
      prediction: "Tuberculosis",
      confidence: 0.95,
      is_tb: true,
      metadata: { patient_id: "PX-12345", patient_name: "John Doe" },
      original_image: "original_b64",
      heatmap_image: "heatmap_b64",
    };

    mockFetch.mockResolvedValueOnce({
      ok: true,
      json: async () => mockResponseData,
    });

    const mockSetResults = jest.fn();
    const mockSetSelectedIdx = jest.fn();

    const { result } = renderHook(() =>
      usePrediction([file1], [{ filename: "chest_xray_1.png", status: "pending" }], mockSetResults, mockSetSelectedIdx)
    );

    await act(async () => {
      await result.current.analyzeFile(0);
    });

    expect(mockFetch).toHaveBeenCalledTimes(1);
    // Check loading state was set
    expect(mockSetResults).toHaveBeenCalledWith(expect.any(Function));
    // Verify that the final success update was dispatched
    expect(mockSetSelectedIdx).toHaveBeenCalledWith(0);
  });

  it("should handle single file analysis error", async () => {
    mockFetch.mockResolvedValueOnce({
      ok: false,
      json: async () => ({ error: "Inference endpoint failed" }),
    });

    const mockSetResults = jest.fn();
    const mockSetSelectedIdx = jest.fn();

    const { result } = renderHook(() =>
      usePrediction([file1], [{ filename: "chest_xray_1.png", status: "pending" }], mockSetResults, mockSetSelectedIdx)
    );

    await act(async () => {
      await result.current.analyzeFile(0);
    });

    expect(mockFetch).toHaveBeenCalledTimes(1);
    expect(mockSetResults).toHaveBeenCalled();
    // selectedIdx shouldn't be updated on error
    expect(mockSetSelectedIdx).not.toHaveBeenCalled();
  });

  it("should process batch runs with a concurrency limit of 3", async () => {
    // We have 5 files. Only 3 should run concurrently.
    const fileList = [file1, file2, file3, file4, file5];
    const initialResults = fileList.map(f => ({ filename: f.name, status: "pending" as const }));

    // Create custom resolvers to inspect the concurrent states
    let resolves: Array<(value: any) => void> = [];
    mockFetch.mockImplementation(() => {
      return new Promise((resolve) => {
        resolves.push(resolve);
      });
    });

    let currentResults = [...initialResults];
    const mockSetResults = jest.fn().mockImplementation((updater) => {
      if (typeof updater === "function") {
        currentResults = updater(currentResults);
      }
    });
    const mockSetSelectedIdx = jest.fn();

    const { result } = renderHook(() =>
      usePrediction(fileList, currentResults, mockSetResults, mockSetSelectedIdx)
    );

    // Start batch analysis
    let batchPromise: Promise<void> | undefined;
    act(() => {
      batchPromise = result.current.analyzeAll();
    });

    // Let the event loop run so workers start
    await new Promise((resolve) => setTimeout(resolve, 0));

    // Concurrency limit is 3, so fetch should be called exactly 3 times initially
    expect(mockFetch).toHaveBeenCalledTimes(3);
    expect(resolves.length).toBe(3);

    // Resolve the first one to free up a worker slot
    await act(async () => {
      resolves[0]({
        ok: true,
        json: async () => ({
          prediction: "Normal",
          confidence: 0.98,
          is_tb: false,
        }),
      });
      // Allow worker to proceed
      await new Promise((resolve) => setTimeout(resolve, 0));
    });

    // The 4th task should have started
    expect(mockFetch).toHaveBeenCalledTimes(4);
    expect(resolves.length).toBe(4);

    // Resolve the second one
    await act(async () => {
      resolves[1]({
        ok: true,
        json: async () => ({
          prediction: "Normal",
          confidence: 0.97,
          is_tb: false,
        }),
      });
      await new Promise((resolve) => setTimeout(resolve, 0));
    });

    // The 5th task should have started
    expect(mockFetch).toHaveBeenCalledTimes(5);
    expect(resolves.length).toBe(5);

    // Resolve remaining tasks
    await act(async () => {
      resolves[2]({ ok: true, json: async () => ({ prediction: "Normal", confidence: 0.96 }) });
      resolves[3]({ ok: true, json: async () => ({ prediction: "Normal", confidence: 0.95 }) });
      resolves[4]({ ok: true, json: async () => ({ prediction: "Normal", confidence: 0.94 }) });
      await new Promise((resolve) => setTimeout(resolve, 0));
    });

    await act(async () => {
      await batchPromise;
    });

    expect(result.current.isBatchProcessing).toBe(false);
  });
});

describe("ScreeningTab Queue Component & UI Actions", () => {
  const mockAnalyzeFile = jest.fn();
  const mockRemoveFile = jest.fn();
  const mockClearAll = jest.fn();
  const mockSetSelectedIdx = jest.fn();
  const mockSetGlobalNote = jest.fn();
  const mockDownloadReport = jest.fn();
  const mockFeedbackSaved = jest.fn();
  const fileInputRef = React.createRef<HTMLInputElement>();

  const defaultProps = {
    files: [],
    results: [],
    setResults: jest.fn(),
    selectedIdx: null,
    setSelectedIdx: mockSetSelectedIdx,
    isDragActive: false,
    isBatchProcessing: false,
    fileInputRef,
    handleDrag: jest.fn(),
    handleDrop: jest.fn(),
    handleFileInput: jest.fn(),
    analyzeFile: mockAnalyzeFile,
    removeFile: mockRemoveFile,
    clearAll: mockClearAll,
    globalNote: "",
    setGlobalNote: mockSetGlobalNote,
    reportRef: React.createRef<HTMLDivElement>(),
    downloadReport: mockDownloadReport,
    handleFeedbackSaved: mockFeedbackSaved,
  };

  it("renders empty state correctly with upload triggers", () => {
    render(<ScreeningTab {...defaultProps} />);
    expect(screen.getByText("No study loaded")).toBeInTheDocument();
    expect(screen.getByText(/Go to/i)).toBeInTheDocument();
  });

  it("renders pending case card correctly", () => {
    const props = {
      ...defaultProps,
      selectedIdx: 0,
      files: [file1],
      results: [{ filename: "chest_xray_1.png", status: "pending" as const }],
    };

    render(<ScreeningTab {...props} />);
    expect(screen.getByText("Awaiting Model Evaluation")).toBeInTheDocument();
    expect(screen.getByText("chest_xray_1.png")).toBeInTheDocument();
  });

  it("renders successful case workspace details correctly", () => {
    const props = {
      ...defaultProps,
      selectedIdx: 0,
      files: [file1],
      results: [{
        filename: "chest_xray_1.png",
        status: "success" as const,
        prediction: "Tuberculosis",
        confidence: 0.95,
        is_tb: true,
        original_image: "img_b64",
        heatmap_image: "heat_b64",
        metadata: { patient_id: "PX-11", patient_name: "Jane Doe" }
      }],
    };

    render(<ScreeningTab {...props} />);
    expect(screen.getByText("Tuberculosis")).toBeInTheDocument();
    expect(screen.getAllByTestId("mock-dynamic").length).toBeGreaterThan(0);
  });

  it("triggers analyzeFile on run button click", () => {
    const props = {
      ...defaultProps,
      selectedIdx: 0,
      files: [file1],
      results: [{ filename: "chest_xray_1.png", status: "pending" as const }],
    };

    render(<ScreeningTab {...props} />);
    const runBtn = screen.getByText("Run AI Diagnostic Analysis");
    fireEvent.click(runBtn);
    expect(mockAnalyzeFile).toHaveBeenCalledWith(0);
  });

  it("renders no-case-selected placeholder correctly", () => {
    const props = {
      ...defaultProps,
      selectedIdx: null,
      files: [file1],
      results: [{ filename: "chest_xray_1.png", status: "pending" as const }],
    };

    render(<ScreeningTab {...props} />);
    expect(screen.getByText("Select a radiograph Study")).toBeInTheDocument();
  });
});

describe("Home Page Component & UI Actions", () => {
  it("renders Landing Portal page by default", () => {
    render(<Home />);
    expect(screen.getByText("AI-Assisted Pulmonary Tuberculosis Screening Workstation")).toBeInTheDocument();
    expect(screen.getByText("Academic Prototype Notice")).toBeInTheDocument();
  });
});
