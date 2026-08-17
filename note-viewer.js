function escapeHtml(text) {
  return text
    .replace(/&/g, "&amp;")
    .replace(/</g, "&lt;")
    .replace(/>/g, "&gt;")
    .replace(/"/g, "&quot;")
    .replace(/'/g, "&#39;");
}

function resolveSafeUrl(value, baseUrl, imageOnly = false) {
  try {
    const url = new URL(value, baseUrl);
    const allowedProtocols = imageOnly ? ["http:", "https:"] : ["http:", "https:", "mailto:"];
    return allowedProtocols.includes(url.protocol) ? url.href : "#";
  } catch (_error) {
    return "#";
  }
}

function renderInline(text, sourceUrl) {
  const tokens = [];
  const saveToken = (html) => {
    const token = `@@MDTOKEN${tokens.length}@@`;
    tokens.push(html);
    return token;
  };

  let prepared = text.replace(/`([^`]+)`/g, (_match, code) =>
    saveToken(`<code>${escapeHtml(code)}</code>`)
  );

  prepared = prepared.replace(/!\[([^\]]*)\]\(([^)]+)\)/g, (_match, alt, source) => {
    const url = resolveSafeUrl(source.trim(), sourceUrl, true);
    if (url === "#") return saveToken(escapeHtml(_match));
    return saveToken(`<img src="${escapeHtml(url)}" alt="${escapeHtml(alt)}" loading="lazy">`);
  });

  prepared = prepared.replace(/\[([^\]]+)\]\(([^)]+)\)/g, (_match, label, target) => {
    const url = resolveSafeUrl(target.trim(), sourceUrl);
    if (url === "#") return saveToken(escapeHtml(label));
    const isExternal = new URL(url).origin !== window.location.origin;
    const externalAttrs = isExternal ? ' target="_blank" rel="noopener noreferrer"' : "";
    return saveToken(`<a href="${escapeHtml(url)}"${externalAttrs}>${escapeHtml(label)}</a>`);
  });

  let html = escapeHtml(prepared);
  html = html.replace(/\*\*([^*]+)\*\*/g, "<strong>$1</strong>");
  html = html.replace(/\*([^*]+)\*/g, "<em>$1</em>");
  html = html.replace(/@@MDTOKEN(\d+)@@/g, (_match, index) => tokens[Number(index)]);
  return html;
}

function renderMarkdown(markdown, sourceUrl) {
  const lines = markdown.replace(/\r\n/g, "\n").split("\n");
  const html = [];
  let inCodeBlock = false;
  let codeBuffer = [];
  let listType = null;
  let paragraph = [];

  function flushParagraph() {
    if (paragraph.length) {
      html.push("<p>" + renderInline(paragraph.join(" "), sourceUrl) + "</p>");
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
      html.push(`<h${level}>${renderInline(heading[2], sourceUrl)}</h${level}>`);
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
      html.push(`<blockquote>${renderInline(blockquote[1], sourceUrl)}</blockquote>`);
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
      html.push(`<li>${renderInline(unordered[1], sourceUrl)}</li>`);
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
      html.push(`<li>${renderInline(ordered[1], sourceUrl)}</li>`);
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
    const fetchUrl = new URL(file, window.location.href);
    fetchUrl.searchParams.set("v", Date.now().toString());
    const response = await fetch(fetchUrl, { cache: "no-store" });
    if (!response.ok) {
      throw new Error("HTTP " + response.status);
    }

    const markdown = await response.text();
    contentNode.innerHTML = renderMarkdown(markdown, fetchUrl);
  } catch (error) {
    contentNode.innerHTML =
      '<p class="error-message">内容暂时无法加载。</p>';
  }
}

main();
