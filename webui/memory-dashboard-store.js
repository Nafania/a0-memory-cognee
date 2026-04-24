import { createStore } from "/js/AlpineStore.js";
import { getContext } from "/index.js";
import * as API from "/js/api.js";
import { openModal, closeModal } from "/js/modals.js";
import { store as notificationStore } from "/components/notifications/notification-store.js";
const MEMORY_DASHBOARD_API = "/plugins/memory_cognee/memory_dashboard";

// Helper function for toasts
function justToast(text, type = "info", timeout = 5000) {
  notificationStore.addFrontendToastOnly(type, text, "", timeout / 1000);
}

// Memory Dashboard Store
const memoryDashboardStore = {
  // Data
  memories: [],
  currentPage: 1,
  itemsPerPage: 10,

  // State
  loading: false,
  loadingSubdirs: false,
  initializingMemory: false,
  message: null,

  // Memory subdirectories
  memorySubdirs: [],
  selectedMemorySubdir: "default",
  memoryInitialized: {},

  // Search and filters
  searchQuery: "",
  areaFilter: "",
  limit: parseInt(localStorage.getItem("memoryDashboard_limit") || "1000"),

  // Stats
  totalCount: 0,
  totalDbCount: 0,
  knowledgeCount: 0,
  conversationCount: 0,
  availableAreas: {},
  areasCount: {},

  // Memory detail modal (standard modal approach)
  detailMemory: null,
  editMode: false,
  editMemoryBackup: null,

  // Polling
  pollingInterval: null,
  pollingEnabled: false,

  // Graph view
  viewMode: "table",
  graphLoading: false,
  graphNodes: [],
  graphEdges: [],
  graphNodeCount: 0,
  graphEdgeCount: 0,
  graphNodeTypes: [],
  graphEmpty: false,
  graphError: null,
  graphLayout: localStorage.getItem("memoryDashboard_graphLayout") || "cose",
  graphSelectedNode: null,
  graphSearchQuery: "",
  graphTypeFilters: {},
  graphNodeLimit: parseInt(localStorage.getItem("memoryDashboard_graphNodeLimit") || "300"),
  graphTruncated: false,
  graphTotalNodeCount: 0,
  graphTotalEdgeCount: 0,
  _cyInstance: null,
  _cyLoaded: false,

  async openModal() {
    await openModal("/plugins/memory_cognee/webui/memory-dashboard.html");
  },

  init() {
    this.initialize();
  },

  async onOpen() {
    await this.getCurrentMemorySubdir();
    await this.loadMemorySubdirs();
    await this.searchMemories();
    this.startPolling();
  },

  async initialize() {
    this.currentPage = 1;
    this.searchQuery = "";
    this.areaFilter = "";
  },

  async getCurrentMemorySubdir() {
    try {
      const response = await API.callJsonApi(MEMORY_DASHBOARD_API, {
        action: "get_current_memory_subdir",
        context_id: getContext(),
      });

      if (response.success && response.memory_subdir) {
        this.selectedMemorySubdir = response.memory_subdir;
      } else {
        this.selectedMemorySubdir = "default";
      }
    } catch (error) {
      console.error("Failed to get current memory subdirectory:", error);
      this.selectedMemorySubdir = "default";
    }
  },

  async loadMemorySubdirs() {
    this.loadingSubdirs = true;

    try {
      const response = await API.callJsonApi(MEMORY_DASHBOARD_API, {
        action: "get_memory_subdirs",
      });

      if (response.success) {
        let subdirs = response.subdirs || ["default"];

        subdirs = subdirs.filter((dir) => dir !== "default").sort();
        if (response.subdirs && response.subdirs.includes("default")) {
          subdirs.unshift("default");
        } else {
          subdirs.unshift("default");
        }

        this.memorySubdirs = subdirs;

        if (!this.memorySubdirs.includes(this.selectedMemorySubdir)) {
          this.selectedMemorySubdir = "default";
        }
      } else {
        justToast(response.error || "Failed to load memory subdirectories", "error");
        this.memorySubdirs = ["default"];
        this.selectedMemorySubdir = "default";
      }
    } catch (error) {
      justToast(error.message || "Failed to load memory subdirectories", "error");
      this.memorySubdirs = ["default"];
      if (!this.memorySubdirs.includes(this.selectedMemorySubdir)) {
        this.selectedMemorySubdir = "default";
      }
      console.error("Memory subdirectory loading error:", error);
    } finally {
      this.loadingSubdirs = false;
    }
  },

  async searchMemories(silent = false) {
    localStorage.setItem("memoryDashboard_limit", this.limit.toString());

    if (!silent) {
      this.loading = true;
      this.message = null;

      if (!this.memoryInitialized[this.selectedMemorySubdir]) {
        this.initializingMemory = true;
      }
    }

    try {
      const response = await API.callJsonApi(MEMORY_DASHBOARD_API, {
        action: "search",
        memory_subdir: this.selectedMemorySubdir,
        area: this.areaFilter,
        search: this.searchQuery,
        limit: this.limit,
      });

      if (response.success) {
        const existingSelections = {};
        if (silent && this.memories) {
          this.memories.forEach((memory) => {
            if (memory.selected) {
              existingSelections[memory.id] = true;
            }
          });
        }

        this.memories = (response.memories || []).map((memory) => ({
          ...memory,
          selected: existingSelections[memory.id] || false,
        }));
        this.totalCount = response.total_count || 0;
        this.totalDbCount = response.total_db_count || 0;
        this.knowledgeCount = response.knowledge_count || 0;
        this.conversationCount = response.conversation_count || 0;
        if (response.available_areas) {
          this.availableAreas = response.available_areas;
        }

        if (!silent) {
          this.message = response.message || null;
          this.currentPage = 1;
        } else {
          if (this.currentPage > this.totalPages && this.totalPages > 0) {
            this.currentPage = this.totalPages;
          }
        }

        this.memoryInitialized[this.selectedMemorySubdir] = true;
      } else {
        if (!silent) {
          justToast(response.error || "Failed to search memories", "error");
          this.memories = [];
          this.message = null;
        } else {
          console.warn("Memory dashboard polling failed:", response.error);
        }
      }
    } catch (error) {
      if (!silent) {
        justToast(error.message || "Failed to search memories", "error");
        this.memories = [];
        this.message = null;
        console.error("Memory search error:", error);
      } else {
        console.warn("Memory dashboard polling error:", error);
      }
    } finally {
      if (!silent) {
        this.loading = false;
        this.initializingMemory = false;
      }
    }
  },

  async clearSearch() {
    this.areaFilter = "";
    this.searchQuery = "";
    this.currentPage = 1;

    await this.searchMemories();
  },

  async onMemorySubdirChange() {
    if (this.viewMode === "graph") {
      this.graphNodes = [];
      this.graphEdges = [];
      this.graphSelectedNode = null;
      if (this._cyInstance) {
        this._cyInstance.destroy();
        this._cyInstance = null;
      }
      await this.loadGraphData();
    } else {
      await this.clearSearch();
    }
  },

  // Pagination
  get totalPages() {
    return Math.ceil(this.memories.length / this.itemsPerPage);
  },

  get sortedAreaKeys() {
    return Object.keys(this.availableAreas).sort();
  },

  get paginatedMemories() {
    const start = (this.currentPage - 1) * this.itemsPerPage;
    const end = start + this.itemsPerPage;
    return this.memories.slice(start, end);
  },

  goToPage(page) {
    if (page >= 1 && page <= this.totalPages) {
      this.currentPage = page;
    }
  },

  nextPage() {
    if (this.currentPage < this.totalPages) {
      this.currentPage++;
    }
  },

  prevPage() {
    if (this.currentPage > 1) {
      this.currentPage--;
    }
  },

  // Mass selection
  get selectedMemories() {
    return this.memories.filter((memory) => memory.selected);
  },

  get selectedCount() {
    return this.selectedMemories.length;
  },

  get allSelected() {
    return (
      this.memories.length > 0 &&
      this.memories.every((memory) => memory.selected)
    );
  },

  get someSelected() {
    return this.memories.some((memory) => memory.selected);
  },

  toggleSelectAll() {
    const shouldSelectAll = !this.allSelected;
    this.memories.forEach((memory) => {
      memory.selected = shouldSelectAll;
    });
  },

  clearSelection() {
    this.memories.forEach((memory) => {
      memory.selected = false;
    });
  },

  // Bulk operations
  async bulkDeleteMemories() {
    const selectedMemories = this.selectedMemories;
    if (selectedMemories.length === 0) return;

    try {
      this.loading = true;
      const response = await API.callJsonApi(MEMORY_DASHBOARD_API, {
        action: "bulk_delete",
        memory_subdir: this.selectedMemorySubdir,
        memory_ids: selectedMemories.map((memory) => memory.id),
      });

      if (response.success) {
        justToast(
          `Successfully deleted ${selectedMemories.length} memories`,
          "success"
        );

        await this.searchMemories(true);
      } else {
        justToast(
          response.error || "Failed to delete selected memories",
          "error"
        );
      }
    } catch (error) {
      justToast(error.message || "Failed to delete selected memories", "error");
    } finally {
      this.loading = false;
    }
  },

  formatMemoryForCopy(memory) {
    let formatted = `=== Memory ID: ${memory.id} ===
Area: ${memory.area}
Timestamp: ${this.formatTimestamp(memory.timestamp)}
Source: ${memory.knowledge_source ? "Knowledge" : "Conversation"}
${memory.source_file ? `File: ${memory.source_file}` : ""}
${
  memory.tags && memory.tags.length > 0 ? `Tags: ${memory.tags.join(", ")}` : ""
}`;

    if (
      memory.metadata &&
      typeof memory.metadata === "object" &&
      Object.keys(memory.metadata).length > 0
    ) {
      formatted += "\n\nMetadata:";
      for (const [key, value] of Object.entries(memory.metadata)) {
        const displayValue =
          typeof value === "object" ? JSON.stringify(value, null, 2) : value;
        formatted += `\n${key}: ${displayValue}`;
      }
    }

    formatted += `\n\nContent:
${memory.content_full}

`;
    return formatted;
  },

  bulkCopyMemories() {
    const selectedMemories = this.selectedMemories;
    if (selectedMemories.length === 0) return;

    const content = selectedMemories
      .map((memory) => this.formatMemoryForCopy(memory))
      .join("\n");

    this.copyToClipboard(content, false);
    justToast(
      `Copied ${selectedMemories.length} memories with metadata to clipboard`,
      "success"
    );
  },

  bulkExportMemories() {
    const selectedMemories = this.selectedMemories;
    if (selectedMemories.length === 0) return;

    const exportData = {
      export_timestamp: new Date().toISOString(),
      memory_subdir: this.selectedMemorySubdir,
      total_memories: selectedMemories.length,
      memories: selectedMemories.map((memory) => ({
        id: memory.id,
        area: memory.area,
        timestamp: memory.timestamp,
        content: memory.content_full,
        tags: memory.tags || [],
        knowledge_source: memory.knowledge_source,
        source_file: memory.source_file || null,
        metadata: memory.metadata || {},
      })),
    };

    const jsonString = JSON.stringify(exportData, null, 2);
    const blob = new Blob([jsonString], { type: "application/json" });
    const url = URL.createObjectURL(blob);

    const timestamp = new Date().toISOString().split("T")[0];
    const filename = `memories_${this.selectedMemorySubdir}_selected_${selectedMemories.length}_${timestamp}.json`;

    const a = document.createElement("a");
    a.href = url;
    a.download = filename;
    document.body.appendChild(a);
    a.click();
    document.body.removeChild(a);
    URL.revokeObjectURL(url);

    justToast(
      `Exported ${selectedMemories.length} selected memories to ${filename}`,
      "success"
    );
  },

  showMemoryDetails(memory) {
    this.detailMemory = memory;
    this.editMode = false;
    this.editMemoryBackup = null;
    openModal("/plugins/memory_cognee/webui/memory-detail-modal.html");
  },

  closeMemoryDetails() {
    this.detailMemory = null;
  },

  // Utilities
  formatTimestamp(timestamp, compact = false) {
    if (!timestamp || timestamp === "unknown") {
      return "Unknown";
    }

    const date = new Date(timestamp);
    if (isNaN(date.getTime())) {
      return "Invalid Date";
    }

    if (compact) {
      return (
        date.toLocaleDateString("en-US", {
          month: "2-digit",
          day: "2-digit",
        }) +
        " " +
        date.toLocaleTimeString("en-US", {
          hour12: false,
          hour: "2-digit",
          minute: "2-digit",
        })
      );
    } else {
      return (
        date.toLocaleDateString("en-US", {
          year: "numeric",
          month: "long",
          day: "numeric",
        }) +
        " at " +
        date.toLocaleTimeString("en-US", {
          hour12: true,
          hour: "numeric",
          minute: "2-digit",
        })
      );
    }
  },

  formatTags(tags) {
    if (!Array.isArray(tags) || tags.length === 0) return "None";
    return tags.join(", ");
  },

  getAreaColor(area) {
    const colors = {
      main: "#3b82f6",
      fragments: "#10b981",
      solutions: "#8b5cf6",
      skills: "#f59e0b",
    };
    return colors[area] || "#6c757d";
  },

  copyToClipboard(text, toastSuccess = true) {
    if (navigator.clipboard && window.isSecureContext) {
      navigator.clipboard
        .writeText(text)
        .then(() => {
          if(toastSuccess)
            justToast("Copied to clipboard!", "success");
        })
        .catch((err) => {
          console.error("Clipboard copy failed:", err);
          this.fallbackCopyToClipboard(text, toastSuccess);
        });
    } else {
      this.fallbackCopyToClipboard(text, toastSuccess);
    }
  },

  fallbackCopyToClipboard(text, toastSuccess = true) {
    const textArea = document.createElement("textarea");
    textArea.value = text;
    textArea.style.position = "fixed";
    textArea.style.left = "-999999px";
    textArea.style.top = "-999999px";
    document.body.appendChild(textArea);
    textArea.focus();
    textArea.select();
    try {
      document.execCommand("copy");
      if(toastSuccess)
        justToast("Copied to clipboard!", "success");
    } catch (err) {
      console.error("Fallback clipboard copy failed:", err);
      justToast("Failed to copy to clipboard", "error");
    }
    document.body.removeChild(textArea);
  },

  async deleteMemory(memory) {
    try {
      const isViewingThisMemory =
        this.detailMemory && this.detailMemory.id === memory.id;

      const response = await API.callJsonApi(MEMORY_DASHBOARD_API, {
        action: "delete",
        memory_subdir: this.selectedMemorySubdir,
        memory_id: memory.id,
      });

      if (response.success) {
        justToast("Memory deleted successfully", "success");

        if (isViewingThisMemory) {
          this.detailMemory = null;
          closeModal();
        }

        await this.searchMemories(true);
      } else {
        justToast(`Failed to delete memory: ${response.error}`, "error");
      }
    } catch (error) {
      console.error("Memory deletion error:", error);
      justToast("Failed to delete memory", "error");
    }
  },

  exportMemories() {
    if (this.memories.length === 0) {
      justToast("No memories to export", "warning");
      return;
    }

    try {
      const exportData = {
        memory_subdir: this.selectedMemorySubdir,
        export_timestamp: new Date().toISOString(),
        total_memories: this.memories.length,
        search_query: this.searchQuery,
        area_filter: this.areaFilter,
        memories: this.memories.map((memory) => ({
          id: memory.id,
          area: memory.area,
          timestamp: memory.timestamp,
          content: memory.content_full,
          metadata: memory.metadata,
        })),
      };

      const blob = new Blob([JSON.stringify(exportData, null, 2)], {
        type: "application/json",
      });
      const url = URL.createObjectURL(blob);
      const a = document.createElement("a");
      a.href = url;
      a.download = `memory-export-${this.selectedMemorySubdir}-${
        new Date().toISOString().split("T")[0]
      }.json`;
      document.body.appendChild(a);
      a.click();
      document.body.removeChild(a);
      URL.revokeObjectURL(url);

      justToast("Memory export completed", "success");
    } catch (error) {
      console.error("Memory export error:", error);
      justToast("Failed to export memories", "error");
    }
  },

  startPolling() {
    if (!this.pollingEnabled || this.pollingInterval) {
      return;
    }

    this.pollingInterval = setInterval(async () => {
      await this.searchMemories(true);
    }, 2000);
  },

  stopPolling() {
    if (this.pollingInterval) {
      clearInterval(this.pollingInterval);
      this.pollingInterval = null;
    }
  },

  // --- Graph view methods ---

  async switchView(mode) {
    this.viewMode = mode;
    if (mode === "graph" && this.graphNodes.length === 0 && !this.graphLoading) {
      await this.loadGraphData();
    }
  },

  async _ensureCytoscape() {
    if (this._cyLoaded) return true;
    if (window.cytoscape) {
      this._cyLoaded = true;
      return true;
    }
    return new Promise((resolve) => {
      const script = document.createElement("script");
      script.src = "/plugins/memory_cognee/webui/lib/cytoscape.min.js";
      script.onload = () => {
        this._cyLoaded = true;
        resolve(true);
      };
      script.onerror = () => {
        console.error("Failed to load cytoscape.js");
        resolve(false);
      };
      document.head.appendChild(script);
    });
  },

  async setGraphNodeLimit(limit) {
    this.graphNodeLimit = parseInt(limit) || 300;
    localStorage.setItem("memoryDashboard_graphNodeLimit", this.graphNodeLimit.toString());
    if (this.viewMode === "graph") {
      await this.loadGraphData();
    }
  },

  async loadGraphData() {
    this.graphLoading = true;
    this.graphError = null;

    try {
      localStorage.setItem("memoryDashboard_graphNodeLimit", this.graphNodeLimit.toString());
      const response = await API.callJsonApi(MEMORY_DASHBOARD_API, {
        action: "graph_data",
        memory_subdir: this.selectedMemorySubdir,
        node_limit: this.graphNodeLimit,
      });

      if (response.success) {
        this.graphNodes = response.nodes || [];
        this.graphEdges = response.edges || [];
        this.graphNodeCount = response.node_count || 0;
        this.graphEdgeCount = response.edge_count || 0;
        this.graphTotalNodeCount = response.total_node_count || 0;
        this.graphTotalEdgeCount = response.total_edge_count || 0;
        this.graphNodeTypes = response.node_types || [];
        this.graphEmpty = response.empty || false;
        this.graphTruncated = response.truncated || false;

        const filters = {};
        for (const t of this.graphNodeTypes) {
          filters[t] = this.graphTypeFilters[t] !== undefined
            ? this.graphTypeFilters[t]
            : true;
        }
        this.graphTypeFilters = filters;

        if (this.graphNodes.length > 0) {
          await this._ensureCytoscape();
          requestAnimationFrame(() => {
            requestAnimationFrame(() => this.initCytoscape());
          });
        }
      } else {
        this.graphError = response.error || "Failed to load graph data";
        justToast(this.graphError, "error");
      }
    } catch (error) {
      this.graphError = error.message || "Failed to load graph data";
      justToast(this.graphError, "error");
    } finally {
      this.graphLoading = false;
    }
  },

  _getNodeColor(type) {
    const palette = {
      Entity: "#3b82f6",
      EntityType: "#6366f1",
      DocumentChunk: "#10b981",
      TextChunk: "#10b981",
      Summary: "#f59e0b",
      Document: "#ef4444",
      NodeSet: "#8b5cf6",
      DataPoint: "#06b6d4",
      SourceCodeChunk: "#84cc16",
    };
    if (palette[type]) return palette[type];
    let hash = 0;
    for (let i = 0; i < type.length; i++) {
      hash = type.charCodeAt(i) + ((hash << 5) - hash);
    }
    const hue = Math.abs(hash) % 360;
    return `hsl(${hue}, 60%, 55%)`;
  },

  initCytoscape() {
    const container = document.getElementById("cy-graph-container");
    if (!container || !window.cytoscape) return;

    if (this._cyInstance) {
      this._cyInstance.destroy();
      this._cyInstance = null;
    }

    const elements = [];

    for (const node of this.graphNodes) {
      elements.push({
        group: "nodes",
        data: {
          id: node.id,
          label: node.label,
          type: node.type,
          properties: node.properties,
          color: this._getNodeColor(node.type),
        },
      });
    }

    const nodeIds = new Set(this.graphNodes.map((n) => n.id));
    const edgeIdSet = new Set();
    for (const edge of this.graphEdges) {
      if (nodeIds.has(edge.source) && nodeIds.has(edge.target)) {
        const eid = `${edge.source}-${edge.label}-${edge.target}`;
        if (edgeIdSet.has(eid)) continue;
        edgeIdSet.add(eid);
        elements.push({
          group: "edges",
          data: {
            id: eid,
            source: edge.source,
            target: edge.target,
            label: edge.label,
          },
        });
      }
    }

    const isLargeGraph = elements.length > 400;
    const nodeSize = isLargeGraph ? 20 : 28;

    const cy = window.cytoscape({
      container,
      elements,
      style: [
        {
          selector: "node",
          style: {
            label: "data(label)",
            "background-color": "data(color)",
            color: "#e0e0e0",
            "text-valign": "bottom",
            "text-halign": "center",
            "font-size": isLargeGraph ? "8px" : "10px",
            "text-margin-y": 4,
            "text-max-width": isLargeGraph ? "70px" : "100px",
            "text-wrap": "ellipsis",
            width: nodeSize,
            height: nodeSize,
            "border-width": 2,
            "border-color": "data(color)",
            "border-opacity": 0.4,
            "text-outline-width": 2,
            "text-outline-color": "#1a1a2e",
          },
        },
        {
          selector: "node:selected",
          style: {
            "border-width": 3,
            "border-color": "#ffffff",
            "border-opacity": 1,
            width: 36,
            height: 36,
          },
        },
        {
          selector: "node.highlighted",
          style: {
            "border-width": 3,
            "border-color": "#facc15",
            "border-opacity": 1,
            width: 34,
            height: 34,
          },
        },
        {
          selector: "node.dimmed",
          style: {
            opacity: 0.15,
          },
        },
        {
          selector: "node.neighbor-highlight",
          style: {
            "border-width": 3,
            "border-color": "#22d3ee",
            "border-opacity": 0.9,
            width: 32,
            height: 32,
          },
        },
        {
          selector: "edge",
          style: {
            width: isLargeGraph ? 1 : 1.5,
            "line-color": "#4a4a6a",
            "target-arrow-color": "#4a4a6a",
            "target-arrow-shape": "triangle",
            "arrow-scale": isLargeGraph ? 0.5 : 0.8,
            "curve-style": "bezier",
            label: isLargeGraph ? "" : "data(label)",
            "font-size": "8px",
            color: "#888",
            "text-rotation": "autorotate",
            "text-outline-width": 1.5,
            "text-outline-color": "#1a1a2e",
            "text-max-width": "80px",
            "text-wrap": "ellipsis",
          },
        },
        {
          selector: "edge.highlighted",
          style: {
            "line-color": "#facc15",
            "target-arrow-color": "#facc15",
            width: 2.5,
          },
        },
        {
          selector: "edge.dimmed",
          style: {
            opacity: 0.08,
          },
        },
        {
          selector: "edge.neighbor-highlight",
          style: {
            "line-color": "#22d3ee",
            "target-arrow-color": "#22d3ee",
            width: 2,
          },
        },
      ],
      layout: this._getLayoutOptions(isLargeGraph),
      minZoom: 0.1,
      maxZoom: 5,
      wheelSensitivity: 0.3,
      textureOnViewport: isLargeGraph,
      hideEdgesOnViewport: isLargeGraph,
      hideLabelsOnViewport: isLargeGraph,
    });

    cy.on("tap", "node", (evt) => {
      const node = evt.target;
      this.graphSelectedNode = {
        id: node.data("id"),
        label: node.data("label"),
        type: node.data("type"),
        properties: node.data("properties") || {},
        neighbors: this._getNodeNeighborSummary(cy, node),
        _showAllNeighbors: false,
      };
    });

    cy.on("tap", (evt) => {
      if (evt.target === cy) {
        this.graphSelectedNode = null;
        cy.elements().removeClass("neighbor-highlight dimmed");
      }
    });

    cy.on("dbltap", "node", (evt) => {
      this._expandNeighbors(cy, evt.target);
    });

    this._cyInstance = cy;
  },

  _getNodeNeighborSummary(cy, node) {
    const neighborhood = node.neighborhood();
    const neighbors = [];
    neighborhood.nodes().forEach((n) => {
      neighbors.push({ id: n.data("id"), label: n.data("label"), type: n.data("type") });
    });
    const edges = [];
    neighborhood.edges().forEach((e) => {
      edges.push({ source: e.data("source"), target: e.data("target"), label: e.data("label") });
    });
    return { neighbors, edges };
  },

  _expandNeighbors(cy, node) {
    cy.elements().removeClass("neighbor-highlight dimmed");
    const neighborhood = node.closedNeighborhood();
    cy.elements().not(neighborhood).addClass("dimmed");
    neighborhood.nodes().not(node).addClass("neighbor-highlight");
    neighborhood.edges().addClass("neighbor-highlight");
  },

  navigateToNode(nodeId) {
    if (!this._cyInstance) return;
    const cy = this._cyInstance;
    const node = cy.getElementById(nodeId);
    if (!node || node.empty()) return;

    cy.elements().removeClass("neighbor-highlight dimmed");

    cy.animate({
      center: { eles: node },
      zoom: Math.max(cy.zoom(), 1.5),
    }, { duration: 400 });

    this.graphSelectedNode = {
      id: node.data("id"),
      label: node.data("label"),
      type: node.data("type"),
      properties: node.data("properties") || {},
      neighbors: this._getNodeNeighborSummary(cy, node),
      _showAllNeighbors: false,
    };

    node.flashClass("search-match", 1500);
  },

  _getLayoutOptions(isLargeGraph) {
    const layoutName = isLargeGraph && this.graphLayout === "cose" ? "concentric" : this.graphLayout;
    const opts = { name: layoutName, animate: false };

    if (layoutName === "cose") {
      opts.nodeRepulsion = () => 8000;
      opts.idealEdgeLength = () => 80;
      opts.edgeElasticity = () => 100;
      opts.gravity = 0.25;
      opts.numIter = isLargeGraph ? 100 : 1000;
    } else if (layoutName === "breadthfirst") {
      opts.spacingFactor = 1.2;
    } else if (layoutName === "concentric") {
      opts.minNodeSpacing = isLargeGraph ? 20 : 40;
    }
    return opts;
  },

  applyGraphLayout(layoutName) {
    this.graphLayout = layoutName;
    localStorage.setItem("memoryDashboard_graphLayout", layoutName);
    if (!this._cyInstance) return;

    const isLarge = this._cyInstance.elements().length > 400;
    const opts = this._getLayoutOptions(isLarge);
    opts.name = layoutName;
    opts.animate = !isLarge;
    if (!isLarge) opts.animationDuration = 500;
    this._cyInstance.layout(opts).run();
  },

  filterGraphByType() {
    if (!this._cyInstance) return;
    this._cyInstance.batch(() => {
      this._cyInstance.nodes().forEach((node) => {
        const type = node.data("type");
        if (this.graphTypeFilters[type] === false) {
          node.style("display", "none");
        } else {
          node.style("display", "element");
        }
      });
      this._cyInstance.edges().forEach((edge) => {
        const src = this._cyInstance.getElementById(edge.data("source"));
        const tgt = this._cyInstance.getElementById(edge.data("target"));
        if (src.style("display") === "none" || tgt.style("display") === "none") {
          edge.style("display", "none");
        } else {
          edge.style("display", "element");
        }
      });
    });
  },

  searchGraph() {
    if (!this._cyInstance) return;
    const q = (this.graphSearchQuery || "").trim().toLowerCase();

    this._cyInstance.elements().removeClass("highlighted dimmed");

    if (!q) return;

    const matched = this._cyInstance.nodes().filter((node) => {
      const label = (node.data("label") || "").toLowerCase();
      const type = (node.data("type") || "").toLowerCase();
      const props = node.data("properties") || {};
      if (label.includes(q) || type.includes(q)) return true;
      for (const v of Object.values(props)) {
        if (String(v).toLowerCase().includes(q)) return true;
      }
      return false;
    });

    if (matched.length > 0) {
      this._cyInstance.elements().addClass("dimmed");
      matched.removeClass("dimmed").addClass("highlighted");
      matched.connectedEdges().removeClass("dimmed").addClass("highlighted");
      if (matched.length <= 20) {
        this._cyInstance.animate({ fit: { eles: matched, padding: 60 } }, { duration: 400 });
      }
    } else {
      justToast("No matching nodes found", "info", 2000);
    }
  },

  clearGraphSearch() {
    this.graphSearchQuery = "";
    if (!this._cyInstance) return;
    this._cyInstance.elements().removeClass("highlighted dimmed");
  },

  graphFitView() {
    if (!this._cyInstance) return;
    this._cyInstance.animate({ fit: { padding: 40 } }, { duration: 300 });
  },

  graphZoomIn() {
    if (!this._cyInstance) return;
    const lvl = this._cyInstance.zoom() * 1.3;
    this._cyInstance.zoom({ level: lvl, renderedPosition: { x: this._cyInstance.width() / 2, y: this._cyInstance.height() / 2 } });
  },

  graphZoomOut() {
    if (!this._cyInstance) return;
    const lvl = this._cyInstance.zoom() / 1.3;
    this._cyInstance.zoom({ level: lvl, renderedPosition: { x: this._cyInstance.width() / 2, y: this._cyInstance.height() / 2 } });
  },

  cleanup() {
    this.stopPolling();
    if (this._cyInstance) {
      this._cyInstance.destroy();
      this._cyInstance = null;
    }
    this.viewMode = "table";
    this.graphNodes = [];
    this.graphEdges = [];
    this.graphSelectedNode = null;
    this.graphSearchQuery = "";
    this.graphError = null;
    this.areaFilter = "";
    this.searchQuery = "";
    this.memories = [];
    this.totalCount = 0;
    this.totalDbCount = 0;
    this.knowledgeCount = 0;
    this.conversationCount = 0;
    this.areasCount = {};
    this.message = null;
    this.currentPage = 1;
    this.editMemoryBackup;
  },

  enableEditMode() {
    this.editMode = true;
    this.editMemoryBackup = JSON.stringify(this.detailMemory);
  },

  cancelEditMode() {
    this.editMode = false;
    this.detailMemory = JSON.parse(this.editMemoryBackup);
  },

  async confirmEditMode() {
    try {

      const response = await API.callJsonApi(MEMORY_DASHBOARD_API, {
        action: "update",
        memory_subdir: this.selectedMemorySubdir,
        original: JSON.parse(this.editMemoryBackup),
        edited: this.detailMemory,
      });

      if(response.success){
        justToast("Memory updated successfully", "success");
        await this.searchMemories(true);
      }else{
        justToast(`Failed to update memory: ${response.error}`, "error");
      }

      this.editMode = false;
      this.editMemoryBackup = null;
    } catch (error) {
      console.error("Error confirming edit mode:", error);
      justToast("Failed to save memory changes.", "error");
    }
  },
};

const store = createStore("memoryDashboardStore", memoryDashboardStore);

export { store };
