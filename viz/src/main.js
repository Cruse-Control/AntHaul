import Graph from "graphology";
import Sigma from "sigma";
import forceAtlas2 from "graphology-layout-forceatlas2";
import noverlap from "graphology-layout-noverlap";

// ── Color palette by entity type ────────────────────────────────
const TYPE_COLORS = {
  Person: "#4e79a7",
  Organization: "#e15759",
  Product: "#f28e2c",
  Concept: "#76b7b2",
  Location: "#59a14f",
  Event: "#edc949",
  Decision: "#af7aa1",
  MentalModel: "#ff9da7",
};
const DEFAULT_NODE_COLOR = "#888";
const EDGE_COLOR = "#333";
const EDGE_HIGHLIGHT_COLOR = "#667";

// ── State ────────────────────────────────────────────────────────
let renderer = null;
let graphInstance = null;
let selectedNode = null;
let hoveredNode = null;
let visibleNodeTypes = new Set();
let visibleEdgeTypes = new Set();
let allNodeTypes = new Set();
let allEdgeTypes = new Set();

// ── Fetch graph data ─────────────────────────────────────────────
async function fetchGraph() {
  const resp = await fetch("/api/graph/full");
  if (!resp.ok) throw new Error(`API error: ${resp.status}`);
  return resp.json();
}

async function searchEntities(query) {
  const resp = await fetch(`/api/graph/search?q=${encodeURIComponent(query)}&limit=15`);
  if (!resp.ok) return [];
  const data = await resp.json();
  return data.results || [];
}

// ── Build graphology graph ───────────────────────────────────────
function buildGraph(data) {
  const graph = new Graph();

  for (const node of data.nodes) {
    if (!node.id) continue;
    const type = node.entity_type || "Unknown";
    allNodeTypes.add(type);
    graph.addNode(node.id, {
      label: node.name || node.canonical_name || node.id,
      x: Math.random() * 1000,
      y: Math.random() * 1000,
      size: 5,
      color: TYPE_COLORS[type] || DEFAULT_NODE_COLOR,
      type: "circle",
      // Store metadata for detail panel
      _entityType: type,
      _description: node.description || "",
      _aliases: node.aliases || [],
      _canonicalName: node.canonical_name || "",
      _createdAt: node.created_at || "",
    });
  }

  const nodeIds = new Set(graph.nodes());
  for (const edge of data.edges) {
    if (!edge.source || !edge.target) continue;
    if (!nodeIds.has(edge.source) || !nodeIds.has(edge.target)) continue;
    // Skip self-loops
    if (edge.source === edge.target) continue;
    const edgeType = edge.type || "RELATED_TO";
    allEdgeTypes.add(edgeType);
    const edgeKey = `${edge.source}-${edgeType}-${edge.target}`;
    if (!graph.hasEdge(edgeKey)) {
      graph.addEdgeWithKey(edgeKey, edge.source, edge.target, {
        label: edgeType,
        color: EDGE_COLOR,
        size: 1,
        _type: edgeType,
        _description: edge.description || "",
        _confidence: edge.confidence || 0,
      });
    }
  }

  // Size nodes by degree
  graph.forEachNode((node) => {
    const degree = graph.degree(node);
    graph.setNodeAttribute(node, "size", Math.max(3, Math.min(25, 3 + degree * 1.5)));
  });

  // Initialize visible sets
  visibleNodeTypes = new Set(allNodeTypes);
  visibleEdgeTypes = new Set(allEdgeTypes);

  return graph;
}

// ── Layout ───────────────────────────────────────────────────────
function runLayout(graph) {
  const settings = forceAtlas2.inferSettings(graph);
  settings.gravity = 1;
  settings.scalingRatio = 5;
  forceAtlas2.assign(graph, { settings, iterations: 300 });
  noverlap.assign(graph, { maxIterations: 100, ratio: 1.5 });
}

// ── Render with sigma ────────────────────────────────────────────
function renderGraph(graph) {
  const container = document.getElementById("graph-container");
  renderer = new Sigma(graph, container, {
    renderEdgeLabels: false,
    enableEdgeEvents: true,
    defaultEdgeType: "arrow",
    edgeLabelSize: 10,
    labelRenderedSizeThreshold: 8,
    labelFont: "sans-serif",
    labelColor: { color: "#ccc" },
    labelSize: 12,
    zIndex: true,
    nodeReducer(node, data) {
      const res = { ...data };
      const nodeType = graph.getNodeAttribute(node, "_entityType");
      if (!visibleNodeTypes.has(nodeType)) {
        res.hidden = true;
        return res;
      }
      if (selectedNode && selectedNode !== node && !graph.areNeighbors(selectedNode, node)) {
        res.color = res.color + "33"; // dim non-neighbors
        res.label = "";
      }
      if (hoveredNode && hoveredNode !== node && !graph.areNeighbors(hoveredNode, node)) {
        res.color = res.color.length <= 7 ? res.color + "55" : res.color;
      }
      if (node === selectedNode || node === hoveredNode) {
        res.highlighted = true;
        res.zIndex = 10;
      }
      return res;
    },
    edgeReducer(edge, data) {
      const res = { ...data };
      const edgeType = graph.getEdgeAttribute(edge, "_type");
      if (!visibleEdgeTypes.has(edgeType)) {
        res.hidden = true;
        return res;
      }
      if (selectedNode) {
        const src = graph.source(edge);
        const tgt = graph.target(edge);
        if (src === selectedNode || tgt === selectedNode) {
          res.color = EDGE_HIGHLIGHT_COLOR;
          res.size = 2;
        } else {
          res.hidden = true;
        }
      }
      return res;
    },
  });
  return renderer;
}

// ── Detail panel ─────────────────────────────────────────────────
function showDetail(nodeId) {
  if (!graphInstance) return;
  const panel = document.getElementById("detail-panel");
  const type = graphInstance.getNodeAttribute(nodeId, "_entityType");
  const color = TYPE_COLORS[type] || DEFAULT_NODE_COLOR;

  document.getElementById("detail-name").textContent =
    graphInstance.getNodeAttribute(nodeId, "label");
  const badge = document.getElementById("detail-type");
  badge.textContent = type;
  badge.style.background = color + "33";
  badge.style.color = color;

  document.getElementById("detail-description").textContent =
    graphInstance.getNodeAttribute(nodeId, "_description") || "No description.";

  const aliases = graphInstance.getNodeAttribute(nodeId, "_aliases") || [];
  const aliasEl = document.getElementById("detail-aliases");
  aliasEl.textContent = aliases.length ? `Aliases: ${aliases.join(", ")}` : "";

  // Build relationship list
  const relDiv = document.getElementById("detail-relationships");
  relDiv.innerHTML = "";

  const outgoing = [];
  const incoming = [];
  graphInstance.forEachOutEdge(nodeId, (edge, attrs, source, target) => {
    outgoing.push({
      type: attrs._type,
      targetId: target,
      targetName: graphInstance.getNodeAttribute(target, "label"),
    });
  });
  graphInstance.forEachInEdge(nodeId, (edge, attrs, source) => {
    incoming.push({
      type: attrs._type,
      sourceId: source,
      sourceName: graphInstance.getNodeAttribute(source, "label"),
    });
  });

  if (outgoing.length) {
    const h4 = document.createElement("h4");
    h4.textContent = `Outgoing (${outgoing.length})`;
    relDiv.appendChild(h4);
    for (const rel of outgoing) {
      const div = document.createElement("div");
      div.className = "rel-item";
      div.innerHTML = `<span class="rel-type">${rel.type}</span> <span>${rel.targetName}</span>`;
      div.addEventListener("click", () => selectNode(rel.targetId));
      relDiv.appendChild(div);
    }
  }
  if (incoming.length) {
    const h4 = document.createElement("h4");
    h4.textContent = `Incoming (${incoming.length})`;
    relDiv.appendChild(h4);
    for (const rel of incoming) {
      const div = document.createElement("div");
      div.className = "rel-item";
      div.innerHTML = `<span class="rel-type">${rel.type}</span> <span>${rel.sourceName}</span>`;
      div.addEventListener("click", () => selectNode(rel.sourceId));
      relDiv.appendChild(div);
    }
  }

  panel.classList.remove("hidden");
}

function hideDetail() {
  document.getElementById("detail-panel").classList.add("hidden");
}

function selectNode(nodeId) {
  selectedNode = nodeId;
  showDetail(nodeId);
  // Animate camera to the node
  const pos = renderer.getNodeDisplayData(nodeId);
  if (pos) {
    renderer.getCamera().animate({ x: pos.x, y: pos.y, ratio: 0.3 }, { duration: 300 });
  }
  renderer.refresh();
}

// ── Search ───────────────────────────────────────────────────────
function setupSearch() {
  const input = document.getElementById("search-input");
  const resultsDiv = document.getElementById("search-results");
  let debounce = null;

  input.addEventListener("input", () => {
    clearTimeout(debounce);
    const q = input.value.trim();
    if (q.length < 2) {
      resultsDiv.classList.remove("open");
      return;
    }
    debounce = setTimeout(async () => {
      // Search locally first (fast), fall back to API for fulltext
      const localResults = [];
      const qLower = q.toLowerCase();
      graphInstance.forEachNode((node, attrs) => {
        if (attrs.label.toLowerCase().includes(qLower)) {
          localResults.push({
            id: node,
            name: attrs.label,
            entity_type: attrs._entityType,
          });
        }
      });
      localResults.sort((a, b) => a.name.localeCompare(b.name));
      const results = localResults.slice(0, 15);

      if (!results.length) {
        resultsDiv.classList.remove("open");
        return;
      }
      resultsDiv.innerHTML = "";
      for (const r of results) {
        const div = document.createElement("div");
        div.className = "search-result-item";
        const color = TYPE_COLORS[r.entity_type] || DEFAULT_NODE_COLOR;
        div.innerHTML = `
          <span class="dot" style="background:${color}"></span>
          <span class="name">${r.name}</span>
          <span class="type">${r.entity_type}</span>
        `;
        div.addEventListener("click", () => {
          selectNode(r.id);
          resultsDiv.classList.remove("open");
          input.value = r.name;
        });
        resultsDiv.appendChild(div);
      }
      resultsDiv.classList.add("open");
    }, 150);
  });

  // Close search on click outside
  document.addEventListener("click", (e) => {
    if (!e.target.closest("#search-box")) {
      resultsDiv.classList.remove("open");
    }
  });

  // Escape clears selection
  document.addEventListener("keydown", (e) => {
    if (e.key === "Escape") {
      selectedNode = null;
      hoveredNode = null;
      hideDetail();
      renderer?.refresh();
      input.value = "";
      resultsDiv.classList.remove("open");
    }
  });
}

// ── Filters ──────────────────────────────────────────────────────
function setupFilters() {
  const nodeDiv = document.getElementById("node-type-filters");
  const edgeDiv = document.getElementById("edge-type-filters");

  // Node type filters
  const nodeHeader = document.createElement("h4");
  nodeHeader.textContent = "Node types";
  nodeDiv.appendChild(nodeHeader);

  for (const type of [...allNodeTypes].sort()) {
    const color = TYPE_COLORS[type] || DEFAULT_NODE_COLOR;
    const count = countNodesByType(type);
    const label = document.createElement("label");
    label.className = "filter-item";
    label.innerHTML = `
      <input type="checkbox" checked data-type="${type}" />
      <span class="dot" style="background:${color}"></span>
      ${type} (${count})
    `;
    label.querySelector("input").addEventListener("change", (e) => {
      if (e.target.checked) visibleNodeTypes.add(type);
      else visibleNodeTypes.delete(type);
      renderer.refresh();
    });
    nodeDiv.appendChild(label);
  }

  // Edge type filters
  const edgeHeader = document.createElement("h4");
  edgeHeader.textContent = "Relationship types";
  edgeDiv.appendChild(edgeHeader);

  for (const type of [...allEdgeTypes].sort()) {
    const count = countEdgesByType(type);
    const label = document.createElement("label");
    label.className = "filter-item";
    label.innerHTML = `
      <input type="checkbox" checked data-edge-type="${type}" />
      ${type} (${count})
    `;
    label.querySelector("input").addEventListener("change", (e) => {
      if (e.target.checked) visibleEdgeTypes.add(type);
      else visibleEdgeTypes.delete(type);
      renderer.refresh();
    });
    edgeDiv.appendChild(label);
  }
}

function countNodesByType(type) {
  let count = 0;
  graphInstance.forEachNode((_, attrs) => {
    if (attrs._entityType === type) count++;
  });
  return count;
}

function countEdgesByType(type) {
  let count = 0;
  graphInstance.forEachEdge((_, attrs) => {
    if (attrs._type === type) count++;
  });
  return count;
}

// ── Stats ────────────────────────────────────────────────────────
function updateStats() {
  const stats = document.getElementById("stats-bar");
  const nodes = graphInstance.order;
  const edges = graphInstance.size;
  stats.textContent = `${nodes.toLocaleString()} nodes | ${edges.toLocaleString()} edges`;
}

// ── Main ─────────────────────────────────────────────────────────
async function main() {
  const loading = document.getElementById("loading");

  try {
    const data = await fetchGraph();

    graphInstance = buildGraph(data);
    runLayout(graphInstance);
    renderer = renderGraph(graphInstance);

    // Wire up interactions
    renderer.on("clickNode", ({ node }) => selectNode(node));
    renderer.on("clickStage", () => {
      selectedNode = null;
      hideDetail();
      renderer.refresh();
    });
    renderer.on("enterNode", ({ node }) => {
      hoveredNode = node;
      document.body.style.cursor = "pointer";
      renderer.refresh();
    });
    renderer.on("leaveNode", () => {
      hoveredNode = null;
      document.body.style.cursor = "default";
      renderer.refresh();
    });

    setupSearch();
    setupFilters();
    updateStats();

    // Hide loading
    loading.classList.add("done");
    setTimeout(() => loading.remove(), 300);
  } catch (err) {
    loading.querySelector("p").textContent = `Error: ${err.message}`;
    loading.querySelector(".spinner").style.display = "none";
    console.error(err);
  }
}

main();
