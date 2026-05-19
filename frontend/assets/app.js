const messages = document.getElementById("messages");
const form = document.getElementById("chatForm");
const input = document.getElementById("queryInput");
const sendBtn = document.getElementById("sendBtn");
const sourcesPanel = document.getElementById("sourcesPanel");
const sourcesBox = document.getElementById("sources");
const latency = document.getElementById("latency");
const sourceCount = document.getElementById("sourceCount");

function appendMessage(role, text) {
  const article = document.createElement("article");
  article.className = `message ${role}`;
  const bubble = document.createElement("div");
  bubble.className = "bubble";
  bubble.textContent = text;
  article.appendChild(bubble);
  messages.appendChild(article);
  messages.scrollTop = messages.scrollHeight;
  return bubble;
}

function renderSources(data) {
  const sources = data.sources || [];
  latency.textContent = `Latency: ${Math.round(data.latency_ms || 0)} ms`;
  sourceCount.textContent = `Sources: ${sources.length}`;
  sourcesPanel.hidden = false;

  if (!sources.length) {
    sourcesBox.innerHTML = "<p class=\"empty-source\">No source was returned for this answer.</p>";
    return;
  }

  sourcesBox.innerHTML = sources.map((source, index) => {
    const page = source.page === undefined || source.page === null ? "" : ` · page ${Number(source.page) + 1}`;
    const score = source.score ? ` · score ${Number(source.score).toFixed(2)}` : "";
    return `
      <article class="source">
        <strong>${index + 1}. ${escapeHtml(source.source_name || "unknown")}${page}${score}</strong>
        <p>${escapeHtml((source.text || "").slice(0, 360))}</p>
      </article>
    `;
  }).join("");
}

async function ask(query) {
  appendMessage("user", query);
  const pending = appendMessage("assistant", "Thinking...");
  sendBtn.disabled = true;

  try {
    const response = await fetch("/api/query", {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({ query }),
    });
    const data = await response.json();
    if (!response.ok) throw new Error(data.detail || "Request failed");
    pending.textContent = data.answer || "No answer returned.";
    renderSources(data);
  } catch (error) {
    pending.textContent = error.message;
  } finally {
    sendBtn.disabled = false;
    input.focus();
  }
}

function escapeHtml(value) {
  return String(value)
    .replaceAll("&", "&amp;")
    .replaceAll("<", "&lt;")
    .replaceAll(">", "&gt;")
    .replaceAll('"', "&quot;")
    .replaceAll("'", "&#039;");
}

form.addEventListener("submit", (event) => {
  event.preventDefault();
  const query = input.value.trim();
  if (!query) return;
  input.value = "";
  ask(query);
});
