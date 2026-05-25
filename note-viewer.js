function escapeHtml(text) {
  return text
    .replace(/&/g, "&amp;")
    .replace(/</g, "&lt;")
    .replace(/>/g, "&gt;")
    .replace(/"/g, "&quot;")
    .replace(/'/g, "&#39;");
}

function renderInline(text) {
  let html = escapeHtml(text);
  html = html.replace(/`([^`]+)`/g, "<code>$1</code>");
  html = html.replace(/\*\*([^*]+)\*\*/g, "<strong>$1</strong>");
  html = html.replace(/\*([^*]+)\*/g, "<em>$1</em>");
  html = html.replace(/\[([^\]]+)\]\(([^)]+)\)/g, '<a href="$2" target="_blank" rel="noopener">$1</a>');
  return html;
}

function renderMarkdown(markdown) {
  const lines = markdown.replace(/\r\n/g, "\n").split("\n");
  const html = [];
  let inCodeBlock = false;
  let codeBuffer = [];
  let listType = null;
  let paragraph = [];

  function flushParagraph() {
    if (paragraph.length) {
      html.push("<p>" + renderInline(paragraph.join(" ")) + "</p>");
      paragraph = [];
    }
  }

  function flushList() {
    if (listType) {
      html.push(`</${listType}>`);
      listType = null;
    }
  }

  function flushCode() {
    if (codeBuffer.length) {
      html.push("<pre><code>" + escapeHtml(codeBuffer.join("\n")) + "</code></pre>");
      codeBuffer = [];
    }
  }

  for (const rawLine of lines) {
    const line = rawLine.trimEnd();

    if (line.startsWith("```")) {
      flushParagraph();
      flushList();
      if (inCodeBlock) {
        flushCode();
      }
      inCodeBlock = !inCodeBlock;
      continue;
    }

    if (inCodeBlock) {
      codeBuffer.push(rawLine);
      continue;
    }

    if (!line.trim()) {
      flushParagraph();
      flushList();
      continue;
    }

    const heading = line.match(/^(#{1,4})\s+(.*)$/);
    if (heading) {
      flushParagraph();
      flushList();
      const level = heading[1].length;
      html.push(`<h${level}>${renderInline(heading[2])}</h${level}>`);
      continue;
    }

    if (/^---+$/.test(line)) {
      flushParagraph();
      flushList();
      html.push("<hr>");
      continue;
    }

    const blockquote = line.match(/^>\s?(.*)$/);
    if (blockquote) {
      flushParagraph();
      flushList();
      html.push(`<blockquote>${renderInline(blockquote[1])}</blockquote>`);
      continue;
    }

    const unordered = line.match(/^[-*+]\s+(.*)$/);
    if (unordered) {
      flushParagraph();
      if (listType !== "ul") {
        flushList();
        listType = "ul";
        html.push("<ul>");
      }
      html.push(`<li>${renderInline(unordered[1])}</li>`);
      continue;
    }

    const ordered = line.match(/^\d+\.\s+(.*)$/);
    if (ordered) {
      flushParagraph();
      if (listType !== "ol") {
        flushList();
        listType = "ol";
        html.push("<ol>");
      }
      html.push(`<li>${renderInline(ordered[1])}</li>`);
      continue;
    }

    paragraph.push(line.trim());
  }

  flushParagraph();
  flushList();
  flushCode();
  return html.join("\n");
}

async function main() {
  const params = new URLSearchParams(window.location.search);
  const file = params.get("file");
  const title = params.get("title") || "未命名笔记";
  const pdf = params.get("pdf");

  const titleNode = document.getElementById("note-title");
  const contentNode = document.getElementById("note-content");
  const actionsNode = document.getElementById("viewer-actions");

  titleNode.textContent = title;

  if (!file) {
    contentNode.innerHTML = '<p class="error-message">内容暂时无法加载。</p>';
    return;
  }

  if (pdf) {
    const pdfLink = document.createElement("a");
    pdfLink.className = "viewer-button primary";
    pdfLink.href = pdf;
    pdfLink.download = "";
    pdfLink.textContent = "下载 PDF";
    actionsNode.prepend(pdfLink);
  }

  try {
    const response = await fetch(file);
    if (!response.ok) {
      throw new Error("HTTP " + response.status);
    }

    const markdown = await response.text();
    contentNode.innerHTML = renderMarkdown(markdown);
  } catch (error) {
    contentNode.innerHTML =
      '<p class="error-message">内容暂时无法加载。</p>';
  }
}

main();
