import { getDocument, GlobalWorkerOptions } from "pdfjs-dist";
import pdfWorkerSrc from "pdfjs-dist/build/pdf.worker.min.mjs?url";

GlobalWorkerOptions.workerSrc = pdfWorkerSrc;

const MAX_PDF_PAGES = 20;

function normalizeExtractedText(text: string): string {
  return text
    .replace(/\u0000/g, "")
    .replace(/[\u2018\u2019\u0060\u00B4]/g, "'")
    .replace(/[\u201C\u201D]/g, '"')
    .replace(/\r\n/g, "\n")
    .replace(/[\t\f\v]+/g, " ")
    .replace(/\n{4,}/g, "\n\n\n")
    .trim();
}

function reconstructPageText(items: any[]): string {
  let output = "";
  let lastY: number | null = null;
  let lastXEnd: number | null = null;

  for (const raw of items) {
    if (!("str" in raw)) continue;

    const chunk = String(raw.str ?? "");
    if (!chunk) continue;

    const transform = Array.isArray(raw.transform) ? raw.transform : [];
    const x = Number(transform[4] ?? 0);
    const y = Number(transform[5] ?? 0);
    const width = Number(raw.width ?? Math.max(4, chunk.length * 4));

    if (lastY !== null) {
      if (Math.abs(y - lastY) > 2.5) {
        output += "\n";
        lastXEnd = null;
      } else if (lastXEnd !== null && x - lastXEnd > 1.5) {
        output += " ";
      }
    }

    output += chunk;

    if (raw.hasEOL) {
      output += "\n";
      lastY = y;
      lastXEnd = null;
      continue;
    }

    lastY = y;
    lastXEnd = x + width;
  }

  return output;
}

async function extractPdfText(file: File): Promise<string> {
  const buffer = await file.arrayBuffer();
  const pdf = await getDocument({ data: new Uint8Array(buffer) }).promise;

  try {
    const pagesToRead = Math.min(pdf.numPages, MAX_PDF_PAGES);
    const pageTexts: string[] = [];

    for (let pageNum = 1; pageNum <= pagesToRead; pageNum += 1) {
      const page = await pdf.getPage(pageNum);
      const textContent = await page.getTextContent({
        includeMarkedContent: true,
        disableNormalization: false,
      });
      const pageText = reconstructPageText(textContent.items as any[]);

      if (pageText.trim()) {
        pageTexts.push(pageText);
      }
    }

    return normalizeExtractedText(pageTexts.join("\n\n"));
  } finally {
    await pdf.destroy();
  }
}

export async function extractTextFromFile(file: File): Promise<string> {
  const isPdf = file.type === "application/pdf" || file.name.toLowerCase().endsWith(".pdf");

  if (isPdf) {
    const extracted = await extractPdfText(file);
    if (!extracted) {
      throw new Error("No readable text found in the PDF");
    }
    return extracted;
  }

  const raw = await file.text();
  return normalizeExtractedText(raw);
}
