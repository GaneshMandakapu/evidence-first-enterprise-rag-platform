const defaultApiKey = 'change-me-local-dev-key';
const questionInput = document.querySelector('#question');
const apiKeyInput = document.querySelector('#api-key');
const queryForm = document.querySelector('#query-form');
const askButton = document.querySelector('#ask-button');
const errorMessage = document.querySelector('#error-message');
const resultSection = document.querySelector('#result-section');
const emptyState = document.querySelector('#empty-state');
const answerText = document.querySelector('#answer-text');
const answerCard = document.querySelector('#answer-card');
const groundingBadge = document.querySelector('#grounding-badge');
const resultTitle = document.querySelector('#result-title');
const citationsBlock = document.querySelector('#citations-block');
const citationGrid = document.querySelector('#citation-grid');

apiKeyInput.value = localStorage.getItem('rag-api-key') || defaultApiKey;
apiKeyInput.addEventListener('change', () => localStorage.setItem('rag-api-key', apiKeyInput.value));

function setStatus(online, text) {
  document.querySelector('#status-dot').classList.toggle('online', online);
  document.querySelector('#status-text').textContent = text;
}

function showError(message) {
  errorMessage.textContent = message;
  errorMessage.hidden = false;
}

function clearError() {
  errorMessage.hidden = true;
  errorMessage.textContent = '';
}

async function request(path, options = {}) {
  const response = await fetch(path, {
    ...options,
    headers: { 'X-API-Key': apiKeyInput.value, ...(options.headers || {}) },
  });
  if (!response.ok) {
    const detail = await response.json().catch(() => ({}));
    throw new Error(detail.detail || `Request failed (${response.status})`);
  }
  return response.json();
}

function renderCitations(citations) {
  citationGrid.innerHTML = citations.map((citation) => `
    <article class="citation">
      <div class="citation-meta">
        <span class="citation-source" title="${citation.source}">${citation.source}</span>
        <span class="citation-score">${citation.score.toFixed(3)} score</span>
      </div>
      <p>${citation.snippet}</p>
    </article>
  `).join('');
  document.querySelector('#citation-count').textContent = `${citations.length} source${citations.length === 1 ? '' : 's'}`;
}

function renderResponse(response) {
  resultSection.hidden = false;
  emptyState.hidden = true;
  answerText.textContent = response.answer;
  answerCard.classList.toggle('abstained', !response.grounded);
  groundingBadge.classList.toggle('abstained', !response.grounded);
  groundingBadge.textContent = response.grounded ? 'Grounded' : 'Abstained';
  resultTitle.textContent = response.grounded ? 'Grounded answer' : 'No supported answer';
  citationsBlock.hidden = !response.citations.length;
  if (response.citations.length) renderCitations(response.citations);
  resultSection.scrollIntoView({ behavior: 'smooth', block: 'start' });
}

queryForm.addEventListener('submit', async (event) => {
  event.preventDefault();
  clearError();
  askButton.disabled = true;
  askButton.querySelector('span').textContent = 'Searching evidence...';
  try {
    const response = await request('/query', {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify({ question: questionInput.value }),
    });
    renderResponse(response);
  } catch (error) {
    showError(error.message);
  } finally {
    askButton.disabled = false;
    askButton.querySelector('span').textContent = 'Ask the evidence base';
  }
});

document.querySelectorAll('.example').forEach((button) => {
  button.addEventListener('click', () => {
    questionInput.value = button.dataset.question;
    questionInput.focus();
  });
});

document.querySelector('#toggle-key').addEventListener('click', (event) => {
  const visible = apiKeyInput.type === 'text';
  apiKeyInput.type = visible ? 'password' : 'text';
  event.currentTarget.setAttribute('aria-label', visible ? 'Show API key' : 'Hide API key');
});

document.querySelector('#ingest-button').addEventListener('click', async () => {
  const status = document.querySelector('#ingest-status');
  status.textContent = 'Refreshing index...';
  try {
    const result = await request('/ingest', { method: 'POST' });
    status.textContent = `${result.documents_ingested} documents · ${result.chunks_indexed} chunks indexed`;
    await loadHealth();
  } catch (error) {
    status.textContent = error.message;
  }
});

async function loadHealth() {
  try {
    const health = await fetch('/health').then((response) => response.json());
    setStatus(true, 'System online');
    document.querySelector('#index-size').textContent = health.index_size;
    document.querySelector('#vector-store').textContent = health.vector_store;
    document.querySelector('#llm-provider').textContent = health.llm_provider;
    document.querySelector('#model-chip').textContent = health.llm_provider === 'mock' ? 'Mock generation' : 'Live generation';
  } catch (error) {
    setStatus(false, 'System unavailable');
  }
}

loadHealth();