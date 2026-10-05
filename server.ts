import 'dotenv/config';
import express from 'express';
import path from 'path';
import { fileURLToPath } from 'url';
import { GoogleGenAI, Type } from '@google/genai';

const __filename = fileURLToPath(import.meta.url);
const __dirname = path.dirname(__filename);

const app = express();
const PORT = Number(process.env.PORT) || 3000;

app.use(express.json({ limit: '15mb' }));

/**
 * Generates 3-5 specific, actionable bullet points based on Missing Skills.
 */
function buildDeterministicGapTips(
  candidateName: string,
  foundSkills: string[],
  missingSkills: string[]
): string[] {
  if (!missingSkills || missingSkills.length === 0) {
    return [
      `All target Job Description skills (${foundSkills.slice(0, 5).join(', ') || 'core skills'}) are already present in ${candidateName}'s resume.`,
      'Keep your existing LaTeX formatting, font presets, and margins untouched; only tighten wordy "\\item" lines that wrap onto an orphan second line.',
      'Quantify production impact inside existing "\\item" bullet points with concrete metrics (e.g., latency reduction %, throughput scale, or dataset size).',
      'Ensure the technical skills section lists primary Job Description keywords first within their existing category row.',
    ];
  }

  const tips: string[] = [];
  const primaryMissing = missingSkills.slice(0, 3);
  const secondaryMissing = missingSkills.slice(3, 6);

  tips.push(
    `In your existing LaTeX Skills section, append missing core keywords (${primaryMissing.join(', ')}) to the matching category line without changing your "\\textbf{...}" or tabular structure.`
  );

  for (const skill of primaryMissing) {
    tips.push(
      `Inside an existing experience "\\item", surgically add "${skill}" where you performed adjacent work (alongside ${foundSkills.slice(0, 2).join(' or ') || 'your current stack'}) without altering any LaTeX macros or spacing.`
    );
  }

  if (secondaryMissing.length > 0) {
    tips.push(
      `Incorporate secondary JD keywords (${secondaryMissing.join(', ')}) into existing project or summary lines by replacing generic terms while keeping the exact line count and page layout.`
    );
  } else {
    tips.push(
      'Replace low-relevance generic verbs in existing "\\item" bullets with exact terminology from the Job Description while preserving all LaTeX commands.'
    );
  }

  return tips.slice(0, 5);
}

/**
 * Escapes special LaTeX characters in plain text strings when injecting into LaTeX.
 */
function escapeLatexText(str: string): string {
  return str
    .replace(/\\/g, '\\textbackslash{}')
    .replace(/([&%$#_{}])/g, '\\$1')
    .replace(/~/g, '\\textasciitilde{}')
    .replace(/\^/g, '\\textasciicircum{}');
}

/**
 * Converts plain extracted resume text (from PDF/DOCX) into a faithful LaTeX representation
 * preserving the candidate's exact lines, headings, and bullet points so surgical LaTeX
 * edits can be applied if the user did not upload a raw `.tex` file directly.
 */
function convertPlainResumeToLatex(rawText: string, candidateName: string): string {
  const trimmed = rawText.trim();
  if (trimmed.includes('\\documentclass') || trimmed.includes('\\begin{document}')) {
    return trimmed;
  }

  const lines = trimmed.split('\n').map((l) => l.trim()).filter(Boolean);
  const bodyLines: string[] = [];
  let inItemize = false;

  const sectionHeaders = new Set([
    'SUMMARY',
    'PROFESSIONAL SUMMARY',
    'PROFILE',
    'OBJECTIVE',
    'EXPERIENCE',
    'WORK EXPERIENCE',
    'PROFESSIONAL EXPERIENCE',
    'EMPLOYMENT',
    'PROJECTS',
    'KEY PROJECTS',
    'ACADEMIC PROJECTS',
    'EDUCATION',
    'SKILLS',
    'TECHNICAL SKILLS',
    'CERTIFICATIONS',
    'ACHIEVEMENTS',
    'PUBLICATIONS',
  ]);

  for (let i = 0; i < lines.length; i++) {
    const line = lines[i];
    const upper = line.replace(/[:\-]+$/, '').trim().toUpperCase();

    if (i === 0) {
      bodyLines.push(`\\begin{center}`);
      bodyLines.push(`  {\\LARGE \\textbf{${escapeLatexText(line || candidateName)}}} \\\\[4pt]`);
      if (lines[1] && (lines[1].includes('@') || /\d{3}/.test(lines[1]) || lines[1].includes('|'))) {
        bodyLines.push(`  \\small ${escapeLatexText(lines[1])}`);
        i++;
      }
      bodyLines.push(`\\end{center}`);
      continue;
    }

    if (sectionHeaders.has(upper)) {
      if (inItemize) {
        bodyLines.push('\\end{itemize}');
        inItemize = false;
      }
      bodyLines.push(`\n\\section*{${escapeLatexText(line.replace(/[:]+$/, ''))}}`);
      continue;
    }

    if (/^[-•*]\s+/.test(line)) {
      if (!inItemize) {
        bodyLines.push('\\begin{itemize}[leftmargin=*, itemsep=2pt, topsep=2pt]');
        inItemize = true;
      }
      const cleanBullet = line.replace(/^[-•*]\s+/, '').trim();
      bodyLines.push(`  \\item ${escapeLatexText(cleanBullet)}`);
    } else {
      if (inItemize) {
        bodyLines.push('\\end{itemize}');
        inItemize = false;
      }
      bodyLines.push(`${escapeLatexText(line)}\\\\[2pt]`);
    }
  }

  if (inItemize) {
    bodyLines.push('\\end{itemize}');
  }

  return [
    '\\documentclass[10pt,a4paper]{article}',
    '\\usepackage[utf8]{inputenc}',
    '\\usepackage[T1]{fontenc}',
    '\\usepackage[margin=0.6in]{geometry}',
    '\\usepackage{enumitem}',
    '\\usepackage{hyperref}',
    '\\usepackage{titlesec}',
    '\\pagestyle{empty}',
    '\\titleformat{\\section}{\\large\\bfseries\\uppercase}{}{0em}{}[\\titlerule]',
    '\\titlespacing*{\\section}{0pt}{8pt}{4pt}',
    '',
    '\\begin{document}',
    ...bodyLines,
    '\\end{document}',
  ].join('\n');
}

/**
 * Deterministic Surgical LaTeX Editor:
 * Keeps 100% of the LaTeX preamble, packages, fonts, macros, and structure untouched,
 * and ONLY injects missing JD skills into the existing Skills section and first relevant
 * `\item` bullet point.
 */
function performDeterministicLatexSurgery(
  baseLatex: string,
  missingSkills: string[]
): { updatedLatex: string; changesSummary: string[] } {
  if (!missingSkills || missingSkills.length === 0) {
    return {
      updatedLatex: baseLatex,
      changesSummary: [
        'Preserved 100% of original LaTeX preamble, font presets, macros, and layout.',
        'All required JD keywords are already present in the resume; no keyword additions needed.',
      ],
    };
  }

  let workingLatex = baseLatex;
  const changesSummary: string[] = [
    'Preserved 100% of original LaTeX preamble, font packages, geometry, and custom macros untouched.',
  ];

  const escapedMissing = missingSkills.map((s) => escapeLatexText(s));
  const skillsAdditionStr = escapedMissing.join(', ');

  // 1. Try to surgically append missing skills into an existing Skills line in the LaTeX body
  const docStartIdx = workingLatex.indexOf('\\begin{document}');
  const preamble = docStartIdx !== -1 ? workingLatex.slice(0, docStartIdx) : '';
  let body = docStartIdx !== -1 ? workingLatex.slice(docStartIdx) : workingLatex;

  // Look for a Skills section or \textbf{...Skills...} or \textbf{Languages/Technologies/Tools}
  const skillLineRegex = /(\\textbf\{[^}]*(?:Skills|Technologies|Tools|Frameworks|Languages|Stack)[^}]*\}\s*[:&]?\s*)([^\n\\]+)/i;
  if (skillLineRegex.test(body)) {
    body = body.replace(skillLineRegex, (_match, prefix, existingList) => {
      const trimmedExisting = existingList.trim().replace(/[,;.\s]+$/, '');
      return `${prefix}${trimmedExisting}, ${skillsAdditionStr} `;
    });
    changesSummary.push(
      `Surgically appended missing JD keywords (${missingSkills.join(', ')}) to the existing LaTeX Skills line without altering section commands.`
    );
  } else if (/\\section\*?\{[^}]*Skills[^}]*\}/i.test(body)) {
    body = body.replace(
      /(\\section\*?\{[^}]*Skills[^}]*\}\s*\n)([^\n]+)/i,
      (_match, secHeader, nextLine) => {
        const cleanNext = nextLine.replace(/\\\\.*$/, '').trim();
        return `${secHeader}${cleanNext}, ${skillsAdditionStr}\\\\[2pt]`;
      }
    );
    changesSummary.push(
      `Surgically added missing JD keywords (${missingSkills.join(', ')}) inside the existing Skills section.`
    );
  } else {
    // Insert right before \end{document} using existing itemize/text style
    body = body.replace(
      /\\end\{document\}/,
      `% Added Missing JD Keywords\n\\noindent\\textbf{Additional JD Technical Skills:} ${skillsAdditionStr}\n\\end{document}`
    );
    changesSummary.push(
      `Appended missing JD keywords (${missingSkills.join(', ')}) before \\end{document} while keeping all existing layout intact.`
    );
  }

  // 2. Surgically refine up to 2 existing \item lines to naturally include top missing keywords
  const topMissing = escapedMissing.slice(0, 3);
  let itemEditCount = 0;
  body = body.replace(/^(\s*\\item\s+)([^\n]+)$/gm, (fullLine, itemCmd, itemContent) => {
    if (itemEditCount >= 2 || itemContent.length < 35) {
      return fullLine;
    }
    const targetKeyword = topMissing[itemEditCount];
    if (!targetKeyword || itemContent.toLowerCase().includes(targetKeyword.toLowerCase())) {
      return fullLine;
    }
    itemEditCount++;
    const cleaned = itemContent.trim().replace(/\.\s*$/, '');
    return `${itemCmd}${cleaned}, utilizing ${targetKeyword} to align with production workflow requirements.`;
  });

  if (itemEditCount > 0) {
    changesSummary.push(
      `Surgically updated ${itemEditCount} existing "\\item" bullet point(s) in-place to incorporate ${missingSkills.slice(0, itemEditCount).join(', ')} without changing any LaTeX macros or spacing.`
    );
  }

  return {
    updatedLatex: preamble + body,
    changesSummary,
  };
}

app.post('/api/optimize-resume', async (req, res) => {
  try {
    const {
      candidateName = 'Candidate',
      resumeText = '',
      latexSource = '',
      jdText = '',
      foundSkills = [],
      missingSkills = [],
    } = req.body || {};

    const sourceInput = (latexSource && latexSource.trim()) ? latexSource.trim() : resumeText.trim();

    if (!sourceInput || !jdText.trim()) {
      res.status(400).json({
        error: 'Both Resume content and Job Description text are required for optimization.',
      });
      return;
    }

    const fallbackTips = buildDeterministicGapTips(candidateName, foundSkills, missingSkills);

    // Ensure we have a LaTeX representation to edit surgically
    const baseLatex = convertPlainResumeToLatex(sourceInput, candidateName);

    // Split at \begin{document} so the preamble (documentclass, packages, fonts, macros) is 100% locked
    const docSplitIndex = baseLatex.indexOf('\\begin{document}');
    const lockedPreamble = docSplitIndex !== -1 ? baseLatex.slice(0, docSplitIndex) : '';
    const originalBody = docSplitIndex !== -1 ? baseLatex.slice(docSplitIndex) : baseLatex;

    // Always prepare the deterministic surgical LaTeX output as a guaranteed baseline
    const deterministicResult = performDeterministicLatexSurgery(baseLatex, missingSkills);

    const apiKey = process.env.GEMINI_API_KEY;
    if (!apiKey || apiKey === 'MY_GEMINI_API_KEY') {
      res.json({
        tips: fallbackTips,
        rewrittenLatex: deterministicResult.updatedLatex,
        changesSummary: deterministicResult.changesSummary,
      });
      return;
    }

    const ai = new GoogleGenAI({
      apiKey,
      httpOptions: {
        headers: {
          'User-Agent': 'aistudio-build',
        },
      },
    });

    const prompt = `You are a Senior LaTeX Resume Editor and ATS NLP Engineer.

CRITICAL LATEX PRESERVATION RULES (NON-NEGOTIABLE):
1. Return ONLY valid LaTeX code for the document body starting with "\\begin{document}" and ending with "\\end{document}".
2. DO NOT change ANY LaTeX styling, font sizes, spacing commands ("\\vspace", "\\hspace"), custom macros, tabular column definitions, section headers, or layout presets.
3. The resume MUST look 100% identical in layout, font, and structure to the uploaded original.
4. ONLY make minimal, surgical text additions or subtractions inside the existing summary text, "\\item" bullet points, and skills list to naturally incorporate the Missing JD Skills (${missingSkills.join(', ') || 'None'}) and fix any broken word wraps, without fabricating jobs or degrees.

Candidate Name: ${candidateName}
Matched JD Skills Already Present: ${foundSkills.join(', ') || 'None'}
Missing JD Skills to Surgically Add: ${missingSkills.join(', ') || 'None'}

--- TARGET JOB DESCRIPTION ---
${jdText}

--- ORIGINAL LATEX DOCUMENT BODY (KEEP ALL COMMANDS/MACROS IDENTICAL) ---
${originalBody}`;

    let aiResultJson: {
      tips?: string[];
      editedBodyLatex?: string;
      changesSummary?: string[];
    } | null = null;

    const candidateModels = ['gemini-3.8-flash', 'gemini-flash-latest'];

    for (const modelName of candidateModels) {
      try {
        const response = await ai.models.generateContent({
          model: modelName,
          contents: prompt,
          config: {
            responseMimeType: 'application/json',
            responseSchema: {
              type: Type.OBJECT,
              properties: {
                tips: {
                  type: Type.ARRAY,
                  items: { type: Type.STRING },
                  description:
                    '3 to 5 specific, actionable bullet points on what text was added or should be rephrased in the LaTeX resume.',
                },
                editedBodyLatex: {
                  type: Type.STRING,
                  description:
                    'The LaTeX body starting with \\begin{document} and ending with \\end{document}, keeping all original macros, fonts, and layout commands 100% intact and only editing text content.',
                },
                changesSummary: {
                  type: Type.ARRAY,
                  items: { type: Type.STRING },
                  description:
                    'Short list of exact surgical text additions/subtractions made to the LaTeX body.',
                },
              },
              required: ['tips', 'editedBodyLatex', 'changesSummary'],
            },
          },
        });

        const rawText = response.text?.trim();
        if (rawText) {
          aiResultJson = JSON.parse(rawText);
          break;
        }
      } catch {
        // Try next model or fall back to deterministic LaTeX surgery
      }
    }

    if (
      aiResultJson &&
      typeof aiResultJson.editedBodyLatex === 'string' &&
      aiResultJson.editedBodyLatex.includes('\\begin{document}')
    ) {
      // Strip any accidental markdown fences and re-attach the locked original preamble verbatim
      const cleanBody = aiResultJson.editedBodyLatex
        .replace(/^```(?:latex|tex)?\s*/i, '')
        .replace(/\s*```$/i, '')
        .trim();

      const bodyStart = cleanBody.indexOf('\\begin{document}');
      const finalBody = bodyStart !== -1 ? cleanBody.slice(bodyStart) : cleanBody;

      res.json({
        tips:
          Array.isArray(aiResultJson.tips) && aiResultJson.tips.length > 0
            ? aiResultJson.tips.slice(0, 5)
            : fallbackTips,
        rewrittenLatex: lockedPreamble + finalBody,
        changesSummary:
          Array.isArray(aiResultJson.changesSummary) && aiResultJson.changesSummary.length > 0
            ? [
                'Locked 100% of original LaTeX preamble, font presets, and custom commands verbatim.',
                ...aiResultJson.changesSummary,
              ]
            : deterministicResult.changesSummary,
      });
      return;
    }

    // Fallback to deterministic surgical LaTeX output so generation NEVER fails
    res.json({
      tips: fallbackTips,
      rewrittenLatex: deterministicResult.updatedLatex,
      changesSummary: deterministicResult.changesSummary,
    });
  } catch {
    res.status(500).json({
      error: 'Unable to process LaTeX optimization request.',
    });
  }
});

async function startServer() {
  if (process.env.NODE_ENV !== 'production') {
    const { createServer: createViteServer } = await import('vite');
    const vite = await createViteServer({
      server: { middlewareMode: true },
      appType: 'spa',
    });
    app.use(vite.middlewares);
  } else {
    const distPath = path.resolve(__dirname, 'dist');
    app.use(express.static(distPath));
    app.get('*', (_req, res) => {
      res.sendFile(path.join(distPath, 'index.html'));
    });
  }

  app.listen(PORT, '0.0.0.0', () => {
    console.log(`Server listening on http://0.0.0.0:${PORT}`);
  });
}

startServer();
