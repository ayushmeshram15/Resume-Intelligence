import React, { useState, useRef } from 'react';
import {
  Upload,
  Download,
  FileText,
  Trash2,
  AlertCircle,
  CheckCircle2,
  Play,
  ArrowRight,
  Wand2,
  Copy,
  Check,
} from 'lucide-react';
import {
  extractTextFromUploadedFile,
  extractJdSkills,
  parseResumeDocument,
  rankCandidates,
  exportCandidatesToCsv,
  generateGapAnalysisTips,
  applyClientSideLatexSurgery,
  RankedCandidate,
} from './nlp/engine';
import { SkillOverlapHeatmap } from './components/SkillOverlapHeatmap';

type PageSelection = 'Project Intro' | 'ATS Dashboard' | 'Resume Optimizer';

export default function App() {
  // Three-Page Sidebar Radio State
  const [selectedPage, setSelectedPage] = useState<PageSelection>('Project Intro');

  // ATS Dashboard State — Dynamic File Uploads
  const [jdTextInput, setJdTextInput] = useState<string>('');
  const [jdUploadedFile, setJdUploadedFile] = useState<File | null>(null);
  const [uploadedResumeFiles, setUploadedResumeFiles] = useState<File[]>([]);

  // Processing & Parsing Error Diagnostics State
  const [isProcessing, setIsProcessing] = useState<boolean>(false);
  const [activeJdText, setActiveJdText] = useState<string>('');
  const [fileParsingErrors, setFileParsingErrors] = useState<string[]>([]);
  const [entityParsingDiagnostics, setEntityParsingDiagnostics] = useState<string[]>([]);
  const [extractedJdSkills, setExtractedJdSkills] = useState<string[]>([]);
  const [rankedResults, setRankedResults] = useState<RankedCandidate[]>([]);
  const [hasProcessed, setHasProcessed] = useState<boolean>(false);

  // Page 3: Resume Optimizer (LaTeX-Only Surgical Editor) State
  const [selectedCandidateId, setSelectedCandidateId] = useState<string>('');
  const [customLatexOverrides, setCustomLatexOverrides] = useState<Record<string, string>>({});
  const [isOptimizing, setIsOptimizing] = useState<boolean>(false);
  const [rewrittenLatexMap, setRewrittenLatexMap] = useState<Record<string, string>>({});
  const [changesSummaryMap, setChangesSummaryMap] = useState<Record<string, string[]>>({});
  const [aiGeneratedTips, setAiGeneratedTips] = useState<Record<string, string[]>>({});
  const [copiedLatexId, setCopiedLatexId] = useState<string>('');

  const jdFileInputRef = useRef<HTMLInputElement | null>(null);
  const resumeFilesInputRef = useRef<HTMLInputElement | null>(null);
  const customTexUploadRef = useRef<HTMLInputElement | null>(null);

  const handleJdFileChange = (e: React.ChangeEvent<HTMLInputElement>) => {
    const file = e.target.files?.[0] || null;
    setJdUploadedFile(file);
  };

  const handleResumeFilesChange = (e: React.ChangeEvent<HTMLInputElement>) => {
    const files = e.target.files;
    if (!files || files.length === 0) return;
    const newFiles = Array.from(files);
    setUploadedResumeFiles((prev) => [...prev, ...newFiles]);
    e.target.value = '';
  };

  const handleRemoveQueuedResume = (index: number) => {
    setUploadedResumeFiles((prev) => prev.filter((_, idx) => idx !== index));
  };

  // Upload a custom .tex source file directly on Page 3 for the selected candidate
  const handleCustomTexUpload = async (
    e: React.ChangeEvent<HTMLInputElement>,
    candidateId: string
  ) => {
    const file = e.target.files?.[0];
    if (!file) return;
    try {
      const texContent = await file.text();
      if (texContent.trim()) {
        setCustomLatexOverrides((prev) => ({
          ...prev,
          [candidateId]: texContent,
        }));
      }
    } finally {
      e.target.value = '';
    }
  };

  // Page 2: "Process Candidates" Handler with Parsing Error Diagnostics
  const handleProcessCandidates = async () => {
    setFileParsingErrors([]);
    setEntityParsingDiagnostics([]);
    setRankedResults([]);
    setHasProcessed(false);

    let resolvedJdText = jdTextInput.trim();

    if (jdUploadedFile) {
      try {
        resolvedJdText = await extractTextFromUploadedFile(jdUploadedFile);
      } catch (err) {
        setFileParsingErrors([
          `JD Parsing Error ("${jdUploadedFile.name}"): ${
            err instanceof Error ? err.message : 'Corrupted or unreadable file.'
          }`,
        ]);
        return;
      }
    }

    if (!resolvedJdText) {
      setFileParsingErrors([
        'Parsing Error: Please paste a Job Description or upload a valid JD file before processing.',
      ]);
      return;
    }

    if (uploadedResumeFiles.length === 0) {
      setFileParsingErrors([
        'Parsing Error: Please upload at least one candidate resume (PDF, DOCX, or .tex) using the file uploader.',
      ]);
      return;
    }

    setIsProcessing(true);
    setActiveJdText(resolvedJdText);

    const currentFileErrors: string[] = [];
    const currentEntityDiagnostics: string[] = [];
    const parsedDocs = [];

    const jdSkills = extractJdSkills(resolvedJdText);
    setExtractedJdSkills(jdSkills);

    for (let i = 0; i < uploadedResumeFiles.length; i++) {
      const file = uploadedResumeFiles[i];
      try {
        const rawText = await extractTextFromUploadedFile(file);
        if (!rawText || rawText.trim().length < 5) {
          throw new Error('File is empty or contains no extractable text.');
        }
        const parsed = parseResumeDocument(
          `upload-${Date.now()}-${i}`,
          file.name,
          rawText,
          jdSkills
        );
        parsedDocs.push(parsed);

        if (parsed.parsingWarnings.length > 0) {
          for (const warn of parsed.parsingWarnings) {
            currentEntityDiagnostics.push(`[${file.name}] ${warn}`);
          }
        }
      } catch (err) {
        currentFileErrors.push(
          `File Parsing Error — Skipping "${file.name}": ${
            err instanceof Error ? err.message : 'Corrupted file'
          }`
        );
      }
    }

    if (parsedDocs.length === 0) {
      currentFileErrors.push('No valid resumes could be processed from the uploaded batch.');
      setFileParsingErrors(currentFileErrors);
      setEntityParsingDiagnostics(currentEntityDiagnostics);
      setIsProcessing(false);
      return;
    }

    const ranked = rankCandidates(parsedDocs, resolvedJdText, jdSkills, 60);
    setRankedResults(ranked);
    setSelectedCandidateId(ranked[0].id);
    setFileParsingErrors(currentFileErrors);
    setEntityParsingDiagnostics(currentEntityDiagnostics);
    setHasProcessed(true);
    setIsProcessing(false);
  };

  // Page 3: Run Surgical LaTeX-Only Auto-Editor (Preserves 100% of style, font, preamble, and macros)
  const handleRunLatexOptimizer = async (candidate: RankedCandidate) => {
    setIsOptimizing(true);

    const sourceLatex =
      customLatexOverrides[candidate.id] !== undefined
        ? customLatexOverrides[candidate.id]
        : candidate.rawLatex;

    try {
      const response = await fetch('/api/optimize-resume', {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({
          candidateName: candidate.name,
          resumeText: candidate.rawText,
          latexSource: sourceLatex,
          jdText: activeJdText || jdTextInput,
          foundSkills: candidate.matchingSkills,
          missingSkills: candidate.missingSkills,
        }),
      });

      if (response.ok) {
        const data = await response.json();
        if (Array.isArray(data.tips) && data.tips.length > 0) {
          setAiGeneratedTips((prev) => ({
            ...prev,
            [candidate.id]: data.tips,
          }));
        }
        if (typeof data.rewrittenLatex === 'string' && data.rewrittenLatex.trim()) {
          setRewrittenLatexMap((prev) => ({
            ...prev,
            [candidate.id]: data.rewrittenLatex,
          }));
          if (Array.isArray(data.changesSummary)) {
            setChangesSummaryMap((prev) => ({
              ...prev,
              [candidate.id]: data.changesSummary,
            }));
          }
          setIsOptimizing(false);
          return;
        }
      }
    } catch {
      // Fall through to deterministic client-side surgical LaTeX editor
    }

    // Guaranteed Client-Side Surgical LaTeX Editor (never fails, preserves 100% of preamble/styles)
    const fallbackResult = applyClientSideLatexSurgery(
      sourceLatex,
      candidate.missingSkills
    );
    setRewrittenLatexMap((prev) => ({
      ...prev,
      [candidate.id]: fallbackResult.rewrittenLatex,
    }));
    setChangesSummaryMap((prev) => ({
      ...prev,
      [candidate.id]: fallbackResult.changesSummary,
    }));
    setIsOptimizing(false);
  };

  const handleDownloadCsv = () => {
    if (rankedResults.length === 0) return;
    const csv = exportCandidatesToCsv(rankedResults);
    const blob = new Blob([csv], { type: 'text/csv;charset=utf-8;' });
    const url = URL.createObjectURL(blob);
    const a = document.createElement('a');
    a.href = url;
    a.download = 'ranked_candidates_ats_report.csv';
    a.click();
    URL.revokeObjectURL(url);
  };

  const handleCopyLatex = (candidateId: string, latexContent: string) => {
    navigator.clipboard.writeText(latexContent);
    setCopiedLatexId(candidateId);
    setTimeout(() => setCopiedLatexId(''), 2000);
  };

  const handleDownloadTexFile = (candidateName: string, latexContent: string) => {
    const blob = new Blob([latexContent], { type: 'application/x-tex;charset=utf-8;' });
    const url = URL.createObjectURL(blob);
    const a = document.createElement('a');
    a.href = url;
    a.download = `${candidateName.replace(/\s+/g, '_')}_Optimized_Resume.tex`;
    a.click();
    URL.revokeObjectURL(url);
  };

  const activeCandidateForOptimizer =
    rankedResults.find((c) => c.id === selectedCandidateId) || rankedResults[0];

  const activeSourceLatex = activeCandidateForOptimizer
    ? customLatexOverrides[activeCandidateForOptimizer.id] !== undefined
      ? customLatexOverrides[activeCandidateForOptimizer.id]
      : activeCandidateForOptimizer.rawLatex
    : '';

  const activeTips = activeCandidateForOptimizer
    ? aiGeneratedTips[activeCandidateForOptimizer.id] ||
      generateGapAnalysisTips(
        activeCandidateForOptimizer.name,
        activeCandidateForOptimizer.matchingSkills,
        activeCandidateForOptimizer.missingSkills
      )
    : [];

  return (
    <div className="min-h-screen flex flex-col bg-[#F8FAFC] text-[#0F172A]">
      {/* Strict 3-Zone Top Bar Contract */}
      <header className="sticky top-0 z-30 bg-white border-b border-slate-200 px-6 py-3.5 flex items-center justify-between">
        {/* Zone 1: Single text element wordmark */}
        <a
          href="#top"
          onClick={(e) => {
            e.preventDefault();
            setSelectedPage('Project Intro');
          }}
          className="text-lg font-bold tracking-tight text-slate-900 whitespace-nowrap"
        >
          Resume Intelligence
        </a>

        {/* Zone 2: Three-Page Navigation Links */}
        <nav className="hidden md:flex items-center gap-7 text-sm font-medium text-slate-600">
          {(['Project Intro', 'ATS Dashboard', 'Resume Optimizer'] as PageSelection[]).map(
            (page) => (
              <button
                key={page}
                type="button"
                onClick={() => setSelectedPage(page)}
                className={`py-1 transition-colors whitespace-nowrap border-b-2 ${
                  selectedPage === page
                    ? 'border-slate-900 text-slate-900 font-semibold'
                    : 'border-transparent hover:text-slate-900'
                }`}
              >
                {page}
              </button>
            )
          )}
        </nav>

        {/* Zone 3: Primary Action */}
        <div className="flex items-center gap-2.5">
          <button
            type="button"
            onClick={() =>
              setSelectedPage(
                selectedPage === 'Project Intro'
                  ? 'ATS Dashboard'
                  : selectedPage === 'ATS Dashboard'
                  ? 'Resume Optimizer'
                  : 'ATS Dashboard'
              )
            }
            className="px-4 py-2 text-xs font-semibold text-white bg-slate-900 rounded-lg hover:bg-slate-800 transition-colors whitespace-nowrap flex items-center gap-1.5"
          >
            {selectedPage === 'Project Intro'
              ? 'Open ATS Dashboard'
              : selectedPage === 'ATS Dashboard'
              ? 'Open Resume Optimizer'
              : 'Back to ATS Dashboard'}
            <ArrowRight className="w-3.5 h-3.5" />
          </button>
        </div>
      </header>

      {/* Main Workspace Layout: Sidebar Radio Selector + Main Content Viewport */}
      <div className="flex-1 flex flex-col lg:flex-row max-w-[1440px] w-full mx-auto">
        {/* Three-Page Sidebar Radio Selector (`st.sidebar.radio`) */}
        <aside className="w-full lg:w-64 shrink-0 bg-white border-b lg:border-b-0 lg:border-r border-slate-200 p-6 flex flex-col justify-between gap-6">
          <div className="space-y-6">
            <div>
              <div className="text-sm font-bold text-slate-900">Navigation</div>
              <p className="text-xs text-slate-500 mt-0.5">
                3-Page Pipeline Workspace
              </p>
            </div>

            <fieldset className="space-y-2">
              <legend className="text-xs font-semibold text-slate-700 mb-2">
                Select Page
              </legend>
              {(
                ['Project Intro', 'ATS Dashboard', 'Resume Optimizer'] as PageSelection[]
              ).map((page) => {
                const isSelected = selectedPage === page;
                return (
                  <label
                    key={page}
                    className={`flex items-center gap-3 px-3.5 py-2.5 rounded-lg border text-xs font-medium cursor-pointer transition-colors ${
                      isSelected
                        ? 'bg-slate-900 text-white border-slate-900'
                        : 'bg-slate-50 text-slate-700 border-slate-200 hover:bg-slate-100'
                    }`}
                  >
                    <input
                      type="radio"
                      name="streamlit_page_radio"
                      value={page}
                      checked={isSelected}
                      onChange={() => setSelectedPage(page)}
                      className="sr-only"
                    />
                    <span
                      className={`w-3.5 h-3.5 rounded-full border flex items-center justify-center ${
                        isSelected ? 'border-white' : 'border-slate-400'
                      }`}
                    >
                      {isSelected && <span className="w-1.5 h-1.5 rounded-full bg-white" />}
                    </span>
                    <span className="whitespace-nowrap">{page}</span>
                  </label>
                );
              })}
            </fieldset>
          </div>

          {/* Developer Attribution in Sidebar */}
          <div className="pt-4 border-t border-slate-200 space-y-1 text-xs">
            <div className="text-slate-500">Developer</div>
            <div className="font-semibold text-slate-900">
              Ayush Harshwardhan Meshram
            </div>
          </div>
        </aside>

        {/* Main Content Viewport */}
        <main className="flex-1 p-6 sm:p-8 space-y-8 overflow-x-hidden">
          {selectedPage === 'Project Intro' && (
            /* =========================================================
               PAGE 1: PROJECT INTRO
               ========================================================= */
            <div className="space-y-8">
              <section className="bg-white border border-slate-200 rounded-xl p-6 sm:p-8 space-y-6">
                <div className="space-y-2 border-b border-slate-200 pb-6">
                  <p className="text-xs font-medium text-slate-500">
                    End-to-End NLP Parsing · Hybrid Semantic Ranking · Surgical LaTeX Resume Auto-Editor
                  </p>
                  <h1 className="text-2xl sm:text-3xl font-bold tracking-tight text-slate-900">
                    Resume Intelligence & LaTeX Auto-Editor Pipeline
                  </h1>
                  <p className="text-sm text-slate-600 leading-relaxed max-w-3xl">
                    An end-to-end Applicant Tracking, Candidate Ranking, and Surgical LaTeX Resume Optimization application built with <code className="font-mono text-xs">spaCy</code> Named Entity Recognition, deterministic Regex fallbacks, <code className="font-mono text-xs">sentence-transformers</code> semantic embeddings, and style-preserving LaTeX code editing.
                  </p>
                </div>

                {/* Attribution & Formula Summary */}
                <div className="grid grid-cols-1 md:grid-cols-3 gap-6 py-2">
                  <div className="p-4 bg-slate-50 border border-slate-200 rounded-lg">
                    <div className="text-xs text-slate-500">Developer</div>
                    <div className="text-sm font-bold text-slate-900 mt-1">
                      Ayush Harshwardhan Meshram
                    </div>
                    <div className="text-xs text-slate-600 mt-0.5">
                      AI / NLP Engineer
                    </div>
                  </div>

                  <div className="p-4 bg-slate-50 border border-slate-200 rounded-lg">
                    <div className="text-xs text-slate-500">Hybrid Scoring Formula</div>
                    <div className="text-sm font-bold font-mono tabular-nums text-slate-900 mt-1">
                      0.60 × Rule + 0.40 × Semantic
                    </div>
                    <div className="text-xs text-slate-600 mt-0.5">
                      Normalized 0–100 Candidate Score
                    </div>
                  </div>

                  <div className="p-4 bg-slate-50 border border-slate-200 rounded-lg">
                    <div className="text-xs text-slate-500">LaTeX Preservation Mode</div>
                    <div className="text-sm font-bold text-slate-900 mt-1">
                      100% Style & Font Locked
                    </div>
                    <div className="text-xs text-slate-600 mt-0.5">
                      Surgical Content Additions/Subtractions Only
                    </div>
                  </div>
                </div>

                {/* Core Architecture Explanation */}
                <div className="grid grid-cols-1 md:grid-cols-2 gap-6 pt-2">
                  <div className="space-y-3">
                    <h2 className="text-base font-semibold text-slate-900">
                      01. Robust In-Memory Parsing & Diagnostics
                    </h2>
                    <ul className="space-y-2.5 text-xs text-slate-600 leading-relaxed">
                      <li>
                        <strong className="text-slate-900">Dynamic In-Memory Uploads:</strong> Processes uploaded PDF, DOCX, and LaTeX (<code className="font-mono">.tex</code>) files directly in memory via <code className="font-mono">io.BytesIO</code> without hardcoded file paths.
                      </li>
                      <li>
                        <strong className="text-slate-900">Unicode & Newline Sanitization:</strong> Strips PDF unicode ligatures, private-use bullet symbols, and broken newline wraps (<code className="font-mono">\n</code>), matching all skills via strict lowercase normalization.
                      </li>
                      <li>
                        <strong className="text-slate-900">Parsing Error Diagnostics:</strong> Surfaces both corrupted file errors and entity-level parsing warnings (e.g., missing email/phone or Regex fallback activations) directly in the ATS Dashboard.
                      </li>
                    </ul>
                  </div>

                  <div className="space-y-3">
                    <h2 className="text-base font-semibold text-slate-900">
                      02. Surgical LaTeX Resume Optimizer (Zero Style Change)
                    </h2>
                    <ul className="space-y-2.5 text-xs text-slate-600 leading-relaxed">
                      <li>
                        <strong className="text-slate-900">60% Rule-Based + 40% Semantic Scoring:</strong> Combines exact JD skill overlap with <code className="font-mono">all-MiniLM-L6-v2</code> dense embedding cosine similarity.
                      </li>
                      <li>
                        <strong className="text-slate-900">Strict Preamble & Font Lock:</strong> Locks 100% of your original <code className="font-mono">\documentclass</code>, font packages, margins, custom macros, and layout presets untouched.
                      </li>
                      <li>
                        <strong className="text-slate-900">Surgical Content Edits Only:</strong> Only adds or refines text inside existing <code className="font-mono">\item</code> bullets and Skills lines so the compiled PDF looks identical in style and layout to your uploaded resume.
                      </li>
                    </ul>
                  </div>
                </div>

                <div className="pt-4 border-t border-slate-200 flex items-center justify-between">
                  <span className="text-xs text-slate-500">
                    Start by uploading a Job Description and candidate resumes on Page 2.
                  </span>
                  <button
                    type="button"
                    onClick={() => setSelectedPage('ATS Dashboard')}
                    className="px-4 py-2 text-xs font-semibold text-white bg-slate-900 rounded-lg hover:bg-slate-800 transition-colors flex items-center gap-1.5"
                  >
                    Go to ATS Dashboard
                    <ArrowRight className="w-3.5 h-3.5" />
                  </button>
                </div>
              </section>
            </div>
          )}

          {selectedPage === 'ATS Dashboard' && (
            /* =========================================================
               PAGE 2: ATS DASHBOARD (Processing, Parsing Errors & Ranking)
               ========================================================= */
            <div className="space-y-8">
              <section className="bg-white border border-slate-200 rounded-xl p-6 sm:p-8 space-y-6">
                <div className="border-b border-slate-200 pb-5">
                  <p className="text-xs font-medium text-slate-500">
                    Page 2 · Dynamic In-Memory Screening & Parsing Error Diagnostics
                  </p>
                  <h1 className="text-2xl font-bold tracking-tight text-slate-900 mt-0.5">
                    ATS Candidate Screening & Ranking Dashboard
                  </h1>
                  <p className="text-xs text-slate-600 mt-1">
                    Paste a Job Description (or upload a JD file), upload multiple candidate resumes (PDF, DOCX, or <code className="font-mono">.tex</code>), and click <strong>Process Candidates</strong>.
                  </p>
                </div>

                <div className="grid grid-cols-1 lg:grid-cols-2 gap-8">
                  {/* Column 1: Job Description Input */}
                  <div className="space-y-4">
                    <h2 className="text-sm font-semibold text-slate-900">
                      1. Job Description (Paste Text or Upload File)
                    </h2>

                    <div>
                      <label
                        htmlFor="jd-text-area"
                        className="block text-xs font-medium text-slate-700 mb-1.5"
                      >
                        Paste the Job Description Text
                      </label>
                      <textarea
                        id="jd-text-area"
                        rows={7}
                        value={jdTextInput}
                        onChange={(e) => setJdTextInput(e.target.value)}
                        placeholder="Paste Job Description requirements here (e.g., Looking for a Python AI/NLP Engineer experienced with spaCy, PyTorch, Transformers, Docker, AWS, FastAPI, and SQL)..."
                        className="w-full rounded-lg border border-slate-300 bg-slate-50/50 p-3 text-xs font-mono leading-relaxed text-slate-900 focus:bg-white focus:border-slate-900 focus:outline-none"
                      />
                    </div>

                    <div className="pt-1">
                      <label className="block text-xs font-medium text-slate-700 mb-1.5">
                        Or Upload a Job Description File (PDF, DOCX, TXT)
                      </label>
                      <input
                        ref={jdFileInputRef}
                        type="file"
                        accept=".pdf,.docx,.txt"
                        onChange={handleJdFileChange}
                        className="hidden"
                      />
                      <div className="flex items-center gap-3">
                        <button
                          type="button"
                          onClick={() => jdFileInputRef.current?.click()}
                          className="px-3.5 py-2 text-xs font-medium text-slate-800 bg-slate-100 hover:bg-slate-200 border border-slate-300 rounded-lg transition-colors flex items-center gap-1.5 whitespace-nowrap"
                        >
                          <Upload className="w-3.5 h-3.5" />
                          Choose JD File
                        </button>
                        {jdUploadedFile ? (
                          <div className="flex items-center gap-2 text-xs text-slate-700 font-mono truncate">
                            <span className="truncate">{jdUploadedFile.name}</span>
                            <button
                              type="button"
                              onClick={() => setJdUploadedFile(null)}
                              className="text-slate-400 hover:text-red-600"
                              title="Remove JD file"
                            >
                              <Trash2 className="w-3.5 h-3.5" />
                            </button>
                          </div>
                        ) : (
                          <span className="text-xs text-slate-400">
                            No JD file selected (using pasted text)
                          </span>
                        )}
                      </div>
                    </div>
                  </div>

                  {/* Column 2: Multiple Resume File Uploader */}
                  <div className="space-y-4 flex flex-col justify-between">
                    <div className="space-y-4">
                      <h2 className="text-sm font-semibold text-slate-900">
                        2. Upload Candidate Resumes (PDF, DOCX, or LaTeX .tex)
                      </h2>

                      <input
                        ref={resumeFilesInputRef}
                        type="file"
                        accept=".pdf,.docx,.tex"
                        multiple
                        onChange={handleResumeFilesChange}
                        className="hidden"
                      />

                      <div
                        onClick={() => resumeFilesInputRef.current?.click()}
                        className="border-2 border-dashed border-slate-300 hover:border-slate-900 bg-slate-50/60 rounded-xl p-6 text-center cursor-pointer transition-colors space-y-2"
                      >
                        <Upload className="w-6 h-6 text-slate-500 mx-auto" />
                        <div className="text-xs font-semibold text-slate-800">
                          Click to upload multiple resumes (<code className="font-mono">.pdf</code>, <code className="font-mono">.docx</code>, or <code className="font-mono">.tex</code>)
                        </div>
                        <p className="text-[11px] text-slate-500">
                          Processed directly in memory; uploading <code className="font-mono">.tex</code> preserves 100% of your exact LaTeX preamble and macros
                        </p>
                      </div>

                      {uploadedResumeFiles.length > 0 && (
                        <div className="space-y-1.5 max-h-40 overflow-y-auto border border-slate-200 rounded-lg p-2.5 bg-white divide-y divide-slate-100">
                          {uploadedResumeFiles.map((file, idx) => (
                            <div
                              key={`${file.name}-${idx}`}
                              className="py-1.5 flex items-center justify-between gap-2 text-xs"
                            >
                              <div className="flex items-center gap-2 truncate">
                                <FileText className="w-3.5 h-3.5 text-slate-400 shrink-0" />
                                <span className="font-mono text-slate-800 truncate">
                                  {file.name}
                                </span>
                                <span className="text-slate-400 font-mono text-[11px] shrink-0">
                                  ({(file.size / 1024).toFixed(1)} KB)
                                </span>
                              </div>
                              <button
                                type="button"
                                onClick={() => handleRemoveQueuedResume(idx)}
                                className="text-slate-400 hover:text-red-600 transition-colors"
                                title="Remove file"
                              >
                                <Trash2 className="w-3.5 h-3.5" />
                              </button>
                            </div>
                          ))}
                        </div>
                      )}
                    </div>

                    <div className="pt-2">
                      <button
                        type="button"
                        disabled={isProcessing}
                        onClick={handleProcessCandidates}
                        className="w-full py-3 px-5 text-xs font-semibold text-white bg-slate-900 hover:bg-slate-800 disabled:bg-slate-400 rounded-lg transition-colors flex items-center justify-center gap-2"
                      >
                        <Play className="w-3.5 h-3.5" />
                        {isProcessing ? 'Sanitizing & Processing Resumes...' : 'Process Candidates'}
                      </button>
                    </div>
                  </div>
                </div>

                {/* Parsing Errors & Sanitization Diagnostics Panel */}
                {(fileParsingErrors.length > 0 || entityParsingDiagnostics.length > 0) && (
                  <div className="pt-4 border-t border-slate-200 space-y-3">
                    <h3 className="text-xs font-semibold text-slate-900">
                      Parsing Errors & Document Sanitization Diagnostics
                    </h3>

                    {fileParsingErrors.map((err, idx) => (
                      <div
                        key={`file-err-${idx}`}
                        className="p-3.5 rounded-lg bg-red-50 border border-red-200 flex items-start gap-2.5 text-xs text-red-800"
                      >
                        <AlertCircle className="w-4 h-4 text-red-600 shrink-0 mt-0.5" />
                        <span className="font-mono">{err}</span>
                      </div>
                    ))}

                    {entityParsingDiagnostics.map((diag, idx) => {
                      const isError = diag.includes('Parsing Error');
                      return (
                        <div
                          key={`diag-${idx}`}
                          className={`p-3 rounded-lg border flex items-start gap-2.5 text-xs ${
                            isError
                              ? 'bg-red-50 border-red-200 text-red-800'
                              : 'bg-amber-50 border-amber-200 text-amber-900'
                          }`}
                        >
                          <AlertCircle
                            className={`w-4 h-4 shrink-0 mt-0.5 ${
                              isError ? 'text-red-600' : 'text-amber-600'
                            }`}
                          />
                          <span className="font-mono">{diag}</span>
                        </div>
                      );
                    })}
                  </div>
                )}

                {/* Extracted JD Skills Banner */}
                {hasProcessed && (
                  <div className="p-3.5 rounded-lg bg-slate-100 border border-slate-200 flex items-start gap-2.5 text-xs text-slate-800">
                    <CheckCircle2 className="w-4 h-4 text-emerald-700 shrink-0 mt-0.5" />
                    <div>
                      <span className="font-semibold">
                        Normalized JD Skills ({extractedJdSkills.length}):{' '}
                      </span>
                      <span className="font-mono">
                        {extractedJdSkills.length > 0
                          ? extractedJdSkills.join(', ')
                          : 'None matched from taxonomy (using semantic text similarity)'}
                      </span>
                    </div>
                  </div>
                )}
              </section>

              {/* Ranked Candidates DataFrame Output */}
              {hasProcessed && rankedResults.length > 0 && (
                <section className="bg-white border border-slate-200 rounded-xl overflow-hidden">
                  <div className="p-6 border-b border-slate-200 flex flex-col sm:flex-row sm:items-center justify-between gap-4">
                    <div>
                      <h2 className="text-base font-semibold text-slate-900">
                        3. Ranked Candidates DataFrame
                      </h2>
                      <p className="text-xs text-slate-500 mt-0.5">
                        Sorted highest-to-lowest by Final Score. Open Page 3 to run surgical LaTeX optimization on any candidate.
                      </p>
                    </div>

                    <div className="flex items-center gap-2.5">
                      <button
                        type="button"
                        onClick={() => setSelectedPage('Resume Optimizer')}
                        className="px-3.5 py-2 text-xs font-medium text-slate-800 bg-slate-100 hover:bg-slate-200 rounded-lg transition-colors flex items-center gap-1.5 whitespace-nowrap"
                      >
                        <Wand2 className="w-3.5 h-3.5" />
                        Open LaTeX Resume Optimizer
                      </button>
                      <button
                        type="button"
                        onClick={handleDownloadCsv}
                        className="px-4 py-2 text-xs font-semibold text-white bg-slate-900 hover:bg-slate-800 rounded-lg transition-colors flex items-center gap-1.5 whitespace-nowrap"
                      >
                        <Download className="w-3.5 h-3.5" />
                        Download Ranked Report (CSV)
                      </button>
                    </div>
                  </div>

                  {/* Exact Required Columns:
                      [Rank, Name, Email, Phone, Final Score, Found Skills, Missing Skills, Reason] */}
                  <div className="overflow-x-auto">
                    <table className="w-full text-left border-collapse">
                      <thead>
                        <tr className="border-b border-slate-200 bg-slate-50 text-[11px] font-semibold text-slate-600">
                          <th className="py-3 px-4 w-16">Rank</th>
                          <th className="py-3 px-4">Name</th>
                          <th className="py-3 px-4">Email</th>
                          <th className="py-3 px-4 whitespace-nowrap">Phone</th>
                          <th className="py-3 px-4 text-right whitespace-nowrap">Final Score</th>
                          <th className="py-3 px-4">Found Skills</th>
                          <th className="py-3 px-4">Missing Skills</th>
                          <th className="py-3 px-4">Reason</th>
                        </tr>
                      </thead>
                      <tbody className="divide-y divide-slate-200 text-xs">
                        {rankedResults.map((cand) => (
                          <tr key={cand.id} className="hover:bg-slate-50 transition-colors">
                            <td className="py-3 px-4 font-mono tabular-nums font-semibold text-slate-900">
                              {cand.rank}
                            </td>
                            <td className="py-3 px-4 font-semibold text-slate-900 whitespace-nowrap">
                              {cand.name}
                            </td>
                            <td className="py-3 px-4 font-mono text-slate-700 whitespace-nowrap">
                              {cand.email}
                            </td>
                            <td className="py-3 px-4 font-mono tabular-nums text-slate-700 whitespace-nowrap">
                              {cand.phone}
                            </td>
                            <td className="py-3 px-4 text-right font-mono tabular-nums font-bold text-slate-900 whitespace-nowrap">
                              {cand.finalScore.toFixed(2)}
                            </td>
                            <td className="py-3 px-4 text-slate-700 max-w-[220px]">
                              {cand.matchingSkills.length > 0
                                ? cand.matchingSkills.join(', ')
                                : 'None'}
                            </td>
                            <td className="py-3 px-4 text-slate-500 max-w-[220px]">
                              {cand.missingSkills.length > 0
                                ? cand.missingSkills.join(', ')
                                : 'None'}
                            </td>
                            <td className="py-3 px-4 font-mono text-[11px] text-slate-600 max-w-[280px]">
                              {cand.reason}
                            </td>
                          </tr>
                        ))}
                      </tbody>
                    </table>
                  </div>
                </section>
              )}

              {/* D3 Skill Overlap & Rarity Heatmap Visualization */}
              {hasProcessed && rankedResults.length > 0 && (
                <SkillOverlapHeatmap
                  candidates={rankedResults}
                  jdSkills={extractedJdSkills}
                />
              )}
            </div>
          )}

          {selectedPage === 'Resume Optimizer' && (
            /* =========================================================
               PAGE 3: RESUME OPTIMIZER & SURGICAL LATEX AUTO-EDITOR
               ========================================================= */
            <div className="space-y-8">
              <section className="bg-white border border-slate-200 rounded-xl p-6 sm:p-8 space-y-6">
                <div className="border-b border-slate-200 pb-5 flex flex-col sm:flex-row sm:items-center justify-between gap-4">
                  <div>
                    <p className="text-xs font-medium text-slate-500">
                      Page 3 · Gap Analysis & Style-Locked LaTeX (.tex) Auto-Editor
                    </p>
                    <h1 className="text-2xl font-bold tracking-tight text-slate-900 mt-0.5">
                      LaTeX Resume Optimizer & Surgical Auto-Editor
                    </h1>
                    <p className="text-xs text-slate-600 mt-1">
                      Outputs strictly <strong>LaTeX (<code className="font-mono">.tex</code>) code</strong>. 100% of your original style, font packages, margins, and custom commands are locked and preserved—only targeted keyword additions/subtractions are made inside the body.
                    </p>
                  </div>
                </div>

                {rankedResults.length === 0 ? (
                  <div className="p-8 rounded-xl bg-slate-50 border border-slate-200 text-center space-y-3">
                    <p className="text-sm font-medium text-slate-800">
                      No uploaded candidates have been processed yet.
                    </p>
                    <p className="text-xs text-slate-500">
                      Please navigate to <strong>Page 2: ATS Dashboard</strong>, upload a Job Description and candidate resumes, and click <strong>Process Candidates</strong> first.
                    </p>
                    <button
                      type="button"
                      onClick={() => setSelectedPage('ATS Dashboard')}
                      className="px-4 py-2 text-xs font-semibold text-white bg-slate-900 rounded-lg hover:bg-slate-800 transition-colors inline-flex items-center gap-1.5"
                    >
                      Go to ATS Dashboard
                      <ArrowRight className="w-3.5 h-3.5" />
                    </button>
                  </div>
                ) : (
                  activeCandidateForOptimizer && (
                    <div className="space-y-6">
                      {/* Candidate Dropdown Menu */}
                      <div className="flex flex-col sm:flex-row sm:items-center justify-between gap-4 p-4 bg-slate-50 border border-slate-200 rounded-lg">
                        <div className="flex items-center gap-3">
                          <label
                            htmlFor="optimizer-candidate-select"
                            className="text-xs font-semibold text-slate-800 whitespace-nowrap"
                          >
                            Select Candidate:
                          </label>
                          <select
                            id="optimizer-candidate-select"
                            value={activeCandidateForOptimizer.id}
                            onChange={(e) => {
                              setSelectedCandidateId(e.target.value);
                            }}
                            className="rounded-lg border border-slate-300 bg-white px-3 py-2 text-xs font-medium text-slate-900 focus:border-slate-900 focus:outline-none"
                          >
                            {rankedResults.map((cand) => (
                              <option key={cand.id} value={cand.id}>
                                Rank #{cand.rank} — {cand.name} ({cand.finalScore.toFixed(2)} pts)
                              </option>
                            ))}
                          </select>
                        </div>

                        <div className="text-xs font-mono tabular-nums text-slate-600">
                          Rule Score: {activeCandidateForOptimizer.ruleScore.toFixed(1)}% · Semantic Score: {activeCandidateForOptimizer.semanticScore.toFixed(1)}% · Final: {activeCandidateForOptimizer.finalScore.toFixed(2)}
                        </div>
                      </div>

                      {/* Side-by-Side Comparison (`st.columns(2)`) */}
                      <div className="grid grid-cols-1 lg:grid-cols-2 gap-8">
                        {/* Left Column: Original LaTeX (.tex) Code & "Missing Skills" List */}
                        <div className="space-y-5 border border-slate-200 rounded-xl p-5 bg-slate-50/40">
                          <div className="flex items-center justify-between gap-2">
                            <div>
                              <h2 className="text-sm font-semibold text-slate-900">
                                Original Resume LaTeX Code & Missing Skills
                              </h2>
                              <p className="text-xs text-slate-500 mt-0.5">
                                You can also paste or upload your exact Overleaf <code className="font-mono">.tex</code> code below to lock its exact preamble & fonts.
                              </p>
                            </div>

                            <div>
                              <input
                                ref={customTexUploadRef}
                                type="file"
                                accept=".tex,.txt"
                                onChange={(e) =>
                                  handleCustomTexUpload(e, activeCandidateForOptimizer.id)
                                }
                                className="hidden"
                              />
                              <button
                                type="button"
                                onClick={() => customTexUploadRef.current?.click()}
                                className="px-3 py-1.5 text-xs font-medium text-slate-800 bg-white border border-slate-300 hover:bg-slate-100 rounded-lg transition-colors flex items-center gap-1.5 whitespace-nowrap"
                              >
                                <Upload className="w-3.5 h-3.5" />
                                Upload .tex Source
                              </button>
                            </div>
                          </div>

                          <div className="space-y-1.5">
                            <div className="text-xs font-semibold text-slate-800">
                              Missing JD Skills ({activeCandidateForOptimizer.missingSkills.length})
                            </div>
                            <div className="p-3 rounded-lg bg-red-50 border border-red-200 text-xs font-mono text-red-900">
                              {activeCandidateForOptimizer.missingSkills.length > 0
                                ? activeCandidateForOptimizer.missingSkills.join(', ')
                                : 'None — All target JD skills were found in this resume!'}
                            </div>
                          </div>

                          <div className="space-y-1.5">
                            <div className="flex items-center justify-between">
                              <span className="text-xs font-semibold text-slate-800">
                                Source LaTeX (<code className="font-mono">.tex</code>) — Editable Input
                              </span>
                              <span className="text-[11px] font-mono text-slate-500">
                                Preamble & Fonts Locked
                              </span>
                            </div>
                            <textarea
                              value={activeSourceLatex}
                              onChange={(e) =>
                                setCustomLatexOverrides((prev) => ({
                                  ...prev,
                                  [activeCandidateForOptimizer.id]: e.target.value,
                                }))
                              }
                              rows={18}
                              className="w-full p-4 rounded-lg bg-white border border-slate-300 text-xs font-mono leading-relaxed text-slate-800 focus:border-slate-900 focus:outline-none"
                              placeholder="Paste your exact LaTeX (.tex) resume code here..."
                            />
                          </div>
                        </div>

                        {/* Right Column: Actionable Tips and Surgically Edited LaTeX (.tex) Output */}
                        <div className="space-y-5 border border-slate-200 rounded-xl p-5 bg-white flex flex-col justify-between">
                          <div className="space-y-5">
                            <div>
                              <h2 className="text-sm font-semibold text-slate-900">
                                Gap Analysis Tips & Surgical LaTeX (.tex) Output
                              </h2>
                              <p className="text-xs text-slate-500 mt-0.5">
                                Preserves 100% of your original LaTeX style, font packages, and layout presets—only adding/subtracting targeted text in the body.
                              </p>
                            </div>

                            {/* 3-5 Specific Actionable Bullet Points */}
                            <div className="space-y-2">
                              <div className="text-xs font-semibold text-slate-800">
                                Gap Analysis Tips (3–5 Actionable Bullets)
                              </div>
                              <ul className="space-y-2 p-4 rounded-lg bg-slate-50 border border-slate-200 text-xs text-slate-700 leading-relaxed list-disc pl-8">
                                {activeTips.map((tip, idx) => (
                                  <li key={idx}>{tip}</li>
                                ))}
                              </ul>
                            </div>

                            {/* Surgical LaTeX Auto-Editor Trigger & Output */}
                            <div className="space-y-3 pt-2 border-t border-slate-200">
                              <div className="flex items-center justify-between gap-2">
                                <span className="text-xs font-semibold text-slate-800">
                                  Surgically Edited LaTeX Code (<code className="font-mono">.tex</code>)
                                </span>
                                <button
                                  type="button"
                                  disabled={isOptimizing}
                                  onClick={() =>
                                    handleRunLatexOptimizer(activeCandidateForOptimizer)
                                  }
                                  className="px-3.5 py-2 text-xs font-semibold text-white bg-slate-900 hover:bg-slate-800 disabled:bg-slate-400 rounded-lg transition-colors flex items-center gap-1.5 whitespace-nowrap"
                                >
                                  <Wand2 className="w-3.5 h-3.5" />
                                  {isOptimizing
                                    ? 'Applying Surgical LaTeX Edits...'
                                    : 'Generate Optimized LaTeX (.tex) Code'}
                                </button>
                              </div>

                              {rewrittenLatexMap[activeCandidateForOptimizer.id] ? (
                                <div className="space-y-3">
                                  {/* Surgical Changes Log */}
                                  {changesSummaryMap[activeCandidateForOptimizer.id] && (
                                    <div className="p-3 rounded-lg bg-emerald-50 border border-emerald-200 space-y-1 text-xs text-emerald-900">
                                      <div className="font-semibold">
                                        Style-Locked Surgical Modifications Applied:
                                      </div>
                                      <ul className="list-disc pl-5 space-y-0.5">
                                        {changesSummaryMap[
                                          activeCandidateForOptimizer.id
                                        ].map((item, i) => (
                                          <li key={i}>{item}</li>
                                        ))}
                                      </ul>
                                    </div>
                                  )}

                                  <textarea
                                    readOnly
                                    rows={14}
                                    value={rewrittenLatexMap[activeCandidateForOptimizer.id]}
                                    className="w-full p-4 rounded-lg bg-[#0F172A] text-slate-100 border border-slate-800 text-xs font-mono leading-relaxed focus:outline-none"
                                  />

                                  <div className="flex items-center justify-end gap-2.5">
                                    <button
                                      type="button"
                                      onClick={() =>
                                        handleCopyLatex(
                                          activeCandidateForOptimizer.id,
                                          rewrittenLatexMap[activeCandidateForOptimizer.id]
                                        )
                                      }
                                      className="px-3.5 py-1.5 text-xs font-medium text-slate-800 bg-slate-100 hover:bg-slate-200 rounded-lg transition-colors flex items-center gap-1.5"
                                    >
                                      {copiedLatexId === activeCandidateForOptimizer.id ? (
                                        <>
                                          <Check className="w-3.5 h-3.5 text-emerald-600" />
                                          Copied LaTeX
                                        </>
                                      ) : (
                                        <>
                                          <Copy className="w-3.5 h-3.5" />
                                          Copy LaTeX (.tex)
                                        </>
                                      )}
                                    </button>
                                    <button
                                      type="button"
                                      onClick={() =>
                                        handleDownloadTexFile(
                                          activeCandidateForOptimizer.name,
                                          rewrittenLatexMap[activeCandidateForOptimizer.id]
                                        )
                                      }
                                      className="px-3.5 py-1.5 text-xs font-semibold text-white bg-slate-900 hover:bg-slate-800 rounded-lg transition-colors flex items-center gap-1.5"
                                    >
                                      <Download className="w-3.5 h-3.5" />
                                      Download .tex File
                                    </button>
                                  </div>
                                </div>
                              ) : (
                                <div className="h-60 rounded-lg border border-dashed border-slate-300 bg-slate-50/50 p-6 flex items-center justify-center text-center text-xs text-slate-500">
                                  Click &ldquo;Generate Optimized LaTeX (.tex) Code&rdquo; above to surgically inject missing JD keywords into {activeCandidateForOptimizer.name}&apos;s LaTeX code while keeping 100% of the original style, font, and layout untouched.
                                </div>
                              )}
                            </div>
                          </div>
                        </div>
                      </div>
                    </div>
                  )
                )}
              </section>
            </div>
          )}
        </main>
      </div>
    </div>
  );
}
