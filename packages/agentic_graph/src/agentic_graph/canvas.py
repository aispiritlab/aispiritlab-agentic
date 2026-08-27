"""Generates the HTML container and JS initialization for the PixiJS agent graph canvas."""

from __future__ import annotations

_HEX_BG = (
    "data:image/svg+xml,"
    "%3Csvg xmlns=%27http://www.w3.org/2000/svg%27 width=%2763%27 height=%2754%27%3E"
    "%3Cpath d=%27M31.5 0L63 13.5V40.5L31.5 54L0 40.5V13.5Z%27 fill=%27none%27 "
    "stroke=%27%231e293b%27 stroke-width=%270.8%27 opacity=%270.5%27/%3E"
    "%3Ccircle cx=%2731.5%27 cy=%2727%27 r=%271.2%27 fill=%27%23334155%27 opacity=%270.4%27/%3E"
    "%3C/svg%3E"
)


def build_canvas_html() -> str:
    """Return the HTML container for the canvas with CSS hex grid background."""
    return (
        '<div id="agent-canvas-container" style="'
        "width:100%;height:620px;position:relative;border-radius:12px;"
        "overflow:hidden;border:1px solid #1e293b;background-color:#0f172a;"
        f"background-image:url('{_HEX_BG}');background-size:63px 54px;"
        '">'
        '<canvas id="agent-canvas" style="width:100%;height:100%;display:block;"></canvas>'
        "</div>"
    )


CANVAS_LOAD_JS = r"""
async (json) => {
  if (!window._agentCanvasReady) {
    window._agentCanvasReady = true;

    const PIXI = await import('https://cdn.jsdelivr.net/npm/pixi.js@8.6.6/dist/pixi.min.mjs');

    // ── Color palette ──
    const COLORS = {
      bg:       0x0f172a,
      gridLine: 0x1e293b,
      gridDot:  0x334155,
    };
    const NODE_COLORS = {
      agent:             { fill: 0x10b981, glow: 0x34d399, bg: 0x064e3b },
      integration:       { fill: 0xf59e0b, glow: 0xfbbf24, bg: 0x78350f },
      structural_output: { fill: 0x8b5cf6, glow: 0xa78bfa, bg: 0x4c1d95 },
      provider:          { fill: 0x06b6d4, glow: 0x22d3ee, bg: 0x164e63 },
      entry_point:       { fill: 0x3b82f6, glow: 0x60a5fa, bg: 0x1e3a5f },
    };
    const NODE_W = 200, NODE_H = 88, PORT_R = 7, CONN_COL = 0x475569;

    class AgentCanvas {
      constructor(PIXI) {
        this.PIXI = PIXI; this.app = null; this.world = null;
        this.nodes = new Map(); this.connections = new Map();
        this.graphState = {graph_id:'',name:'',nodes:[],connections:[],entry_node_id:null};
        this.selectedNodeId = null; this.dragging = null;
        this.connecting = null; this.tempLine = null;
        this.connectionLayer = null; this.nodeLayer = null;
        this._animFrame = 0;
        this._panning = null; // {startX, startY, worldStartX, worldStartY}
        this._zoom = 1;
        this._minZoom = 0.3;
        this._maxZoom = 2.5;
      }

      async init() {
        const container = document.getElementById('agent-canvas-container');
        const canvas = document.getElementById('agent-canvas');
        if (!container || !canvas) return;

        this.app = new PIXI.Application();
        await this.app.init({
          canvas, width: container.clientWidth, height: container.clientHeight,
          backgroundAlpha: 0, antialias: true,
          resolution: window.devicePixelRatio || 1, autoDensity: true
        });

        this.world = new PIXI.Container();
        this.app.stage.addChild(this.world);
        this.connectionLayer = new PIXI.Container();
        this.nodeLayer = new PIXI.Container();
        this.world.addChild(this.connectionLayer);
        this.world.addChild(this.nodeLayer);

        this.app.stage.eventMode = 'static';
        this.app.stage.hitArea = this.app.screen;
        this.app.stage.on('pointerdown', (e) => {
          if (e.target === this.app.stage) {
            this.selectNode(null);
            if (this.connecting) this.cancelConnection();
            // Start panning (left-click on empty canvas)
            this._panning = { startX: e.global.x, startY: e.global.y, worldStartX: this.world.x, worldStartY: this.world.y };
          }
        });
        this.app.stage.on('pointermove', (e) => {
          if (this.connecting && this.tempLine) {
            // Convert global to world-local coords for temp line
            const lx = (e.global.x - this.world.x) / this._zoom;
            const ly = (e.global.y - this.world.y) / this._zoom;
            this.drawTempLine(this.connecting.startX, this.connecting.startY, lx, ly);
          }
          // Pan
          if (this._panning && !this.dragging) {
            this.world.x = this._panning.worldStartX + (e.global.x - this._panning.startX);
            this.world.y = this._panning.worldStartY + (e.global.y - this._panning.startY);
          }
        });
        this.app.stage.on('pointerup', () => { this._panning = null; });
        this.app.stage.on('pointerupoutside', () => { this._panning = null; });

        // Zoom (wheel)
        container.addEventListener('wheel', (e) => {
          e.preventDefault();
          const delta = e.deltaY > 0 ? 0.9 : 1.1;
          const newZoom = Math.min(this._maxZoom, Math.max(this._minZoom, this._zoom * delta));
          const rect = container.getBoundingClientRect();
          const mx = e.clientX - rect.left;
          const my = e.clientY - rect.top;
          // Zoom toward cursor
          const worldBefore = { x: (mx - this.world.x) / this._zoom, y: (my - this.world.y) / this._zoom };
          this._zoom = newZoom;
          this.world.scale.set(this._zoom);
          this.world.x = mx - worldBefore.x * this._zoom;
          this.world.y = my - worldBefore.y * this._zoom;
        }, { passive: false });

        new ResizeObserver(() => {
          if (this.app && container) {
            this.app.renderer.resize(container.clientWidth, container.clientHeight);
            this.app.stage.hitArea = this.app.screen;
          }
        }).observe(container);

        // Connection flow animation (throttled to ~30fps)
        this.app.ticker.add(() => { this._animFrame++; if (this._animFrame % 2 === 0) this._animateConnections(); });
      }

      loadGraph(state) {
        this.graphState = typeof state === 'string' ? JSON.parse(state) : state;
        this.rebuildScene();
      }
      getGraph() { return JSON.parse(JSON.stringify(this.graphState)); }

      rebuildScene() {
        this.nodeLayer.removeChildren(); this.connectionLayer.removeChildren();
        this.nodes.clear(); this.connections.clear();
        for (const n of this.graphState.nodes) this.createNodeSprite(n);
        for (const c of this.graphState.connections) this.createConnectionSprite(c);
      }

      createNodeSprite(nd) {
        const P = this.PIXI;
        const g = new P.Container(); g.x = nd.position.x; g.y = nd.position.y;
        g.eventMode = 'static'; g.cursor = 'pointer';

        const isEntry = this.graphState.entry_node_id === nd.node_id;
        const isSelected = this.selectedNodeId === nd.node_id;
        const palette = isEntry ? NODE_COLORS.entry_point : (NODE_COLORS[nd.node_type] || NODE_COLORS.agent);

        // Outer glow (selected or entry)
        if (isSelected || isEntry) {
          const glow = new P.Graphics();
          glow.roundRect(-4, -4, NODE_W + 8, NODE_H + 8, 16);
          glow.fill({ color: palette.glow, alpha: isSelected ? 0.15 : 0.08 });
          g.addChild(glow);
        }

        // Card body
        const body = new P.Graphics();
        body.roundRect(0, 0, NODE_W, NODE_H, 14);
        body.fill({ color: palette.bg, alpha: 0.92 });
        body.stroke({
          color: palette.fill,
          width: isSelected ? 2.5 : 1.5,
          alpha: isSelected ? 1 : 0.6,
        });
        g.addChild(body);

        // Top accent line
        const accent = new P.Graphics();
        accent.roundRect(0, 0, NODE_W, 4, 14);
        accent.fill({ color: palette.fill, alpha: 0.9 });
        g.addChild(accent);

        // Type badge
        const badgeText = isEntry ? 'ENTRY' : nd.node_type.replace('_', ' ').toUpperCase();
        const badge = new P.Text({ text: badgeText, style: {
          fontSize: 9, fill: palette.fill, fontFamily: 'monospace', fontWeight: 'bold', letterSpacing: 1
        } });
        badge.x = 10; badge.y = 10; g.addChild(badge);

        // Display name
        const label = new P.Text({ text: nd.display_name || nd.agent_name, style: {
          fontSize: 14, fill: 0xf1f5f9, fontFamily: 'Inter, Arial, sans-serif', fontWeight: 'bold'
        } });
        label.x = 10; label.y = 28; g.addChild(label);

        // Agent name subtitle
        if (nd.display_name && nd.display_name !== nd.agent_name) {
          const sub = new P.Text({ text: nd.agent_name, style: {
            fontSize: 10, fill: 0x94a3b8, fontFamily: 'JetBrains Mono, monospace'
          } });
          sub.x = 10; sub.y = 50; g.addChild(sub);
        }

        // Status indicator dot (bottom-right, for animation)
        const statusDot = new P.Graphics();
        statusDot.circle(NODE_W - 14, NODE_H - 14, 4);
        statusDot.fill({ color: palette.fill, alpha: 0.5 });
        g.addChild(statusDot);
        g._statusDot = statusDot;

        // Input port (left)
        const inP = new P.Graphics();
        inP.circle(0, NODE_H / 2, PORT_R);
        inP.fill({ color: 0x1e293b });
        inP.stroke({ color: palette.fill, width: 2, alpha: 0.8 });
        inP.eventMode = 'static'; inP.cursor = 'crosshair';
        inP.on('pointerdown', (e) => { e.stopPropagation(); this.finishConnection(nd.node_id); });
        g.addChild(inP);

        // Output port (right)
        const outP = new P.Graphics();
        outP.circle(NODE_W, NODE_H / 2, PORT_R);
        outP.fill({ color: palette.fill, alpha: 0.8 });
        outP.stroke({ color: palette.glow, width: 2, alpha: 0.6 });
        outP.eventMode = 'static'; outP.cursor = 'crosshair';
        outP.on('pointerdown', (e) => { e.stopPropagation(); this.startConnection(nd.node_id, g.x + NODE_W, g.y + NODE_H / 2); });
        g.addChild(outP);

        // Delete button
        const del = new P.Graphics();
        del.circle(NODE_W - 2, 2, 9);
        del.fill({ color: 0xef4444, alpha: 0 });
        del.eventMode = 'static'; del.cursor = 'pointer';
        del.on('pointerover', () => { del.clear(); del.circle(NODE_W - 2, 2, 9); del.fill({ color: 0xef4444, alpha: 0.9 }); });
        del.on('pointerout', () => { del.clear(); del.circle(NODE_W - 2, 2, 9); del.fill({ color: 0xef4444, alpha: 0 }); });
        del.on('pointerdown', (e) => { e.stopPropagation(); this.removeNode(nd.node_id); });
        g.addChild(del);
        const delX = new P.Text({ text: '\u00d7', style: { fontSize: 13, fill: 0xffffff, fontWeight: 'bold' } });
        delX.x = NODE_W - 8; delX.y = -6; delX.alpha = 0.6; g.addChild(delX);

        // Drag (accounts for world pan/zoom)
        g.on('pointerdown', (e) => {
          if (e.target === inP || e.target === outP || e.target === del) return;
          this.selectNode(nd.node_id);
          // Convert global coords to world-local coords
          const localX = (e.global.x - this.world.x) / this._zoom;
          const localY = (e.global.y - this.world.y) / this._zoom;
          this.dragging = { nodeId: nd.node_id, offX: localX - g.x, offY: localY - g.y };
          this._panning = null; // prevent pan while dragging
          const onMove = (ev) => {
            if (!this.dragging) return;
            const gr = this.nodes.get(this.dragging.nodeId); if (!gr) return;
            const lx = (ev.global.x - this.world.x) / this._zoom;
            const ly = (ev.global.y - this.world.y) / this._zoom;
            gr.x = lx - this.dragging.offX;
            gr.y = ly - this.dragging.offY;
            const d = this.graphState.nodes.find(n => n.node_id === this.dragging.nodeId);
            if (d) { d.position.x = gr.x; d.position.y = gr.y; }
            this.redrawConnections();
          };
          const onEnd = () => {
            this.dragging = null;
            this.app.stage.off('pointermove', onMove);
            this.app.stage.off('pointerup', onEnd);
            this.app.stage.off('pointerupoutside', onEnd);
            this.pushStateToGradio();
          };
          this.app.stage.on('pointermove', onMove);
          this.app.stage.on('pointerup', onEnd);
          this.app.stage.on('pointerupoutside', onEnd);
        });

        this.nodeLayer.addChild(g);
        this.nodes.set(nd.node_id, g);
      }

      selectNode(nodeId) {
        this.selectedNodeId = nodeId;
        this.rebuildScene();
        this._setBridge('#selected-node-bridge', nodeId || '');
      }

      // ── Connections ──
      startConnection(srcId, sx, sy) {
        this.connecting = { sourceId: srcId, startX: sx, startY: sy };
        this.tempLine = new this.PIXI.Graphics();
        this.connectionLayer.addChild(this.tempLine);
      }

      finishConnection(tgtId) {
        if (!this.connecting || this.connecting.sourceId === tgtId) { this.cancelConnection(); return; }
        const sourceNode = this.graphState.nodes.find(n => n.node_id === this.connecting.sourceId);
        const targetNode = this.graphState.nodes.find(n => n.node_id === tgtId);
        if (!this.isAllowedConnection(sourceNode, targetNode)) { this.cancelConnection(); return; }
        const exists = this.graphState.connections.some(
          c => c.source_node_id === this.connecting.sourceId && c.target_node_id === tgtId
        );
        if (!exists) {
          this.graphState.connections.push({
            connection_id: 'conn_' + Date.now().toString(36) + '_' + Math.random().toString(36).slice(2, 6),
            source_node_id: this.connecting.sourceId, target_node_id: tgtId,
            message_type: 'UserMessage', label: ''
          });
          this.rebuildScene(); this.pushStateToGradio();
        }
        this.cancelConnection();
      }

      isAllowedConnection(sourceNode, targetNode) {
        if (!sourceNode || !targetNode) return false;
        if (
          (sourceNode.node_type === 'integration' && targetNode.node_type === 'agent') ||
          (sourceNode.node_type === 'agent' && targetNode.node_type === 'integration')
        ) {
          const agentNode = sourceNode.node_type === 'agent' ? sourceNode : targetNode;
          return agentNode.agent_name === 'searcher';
        }
        if (
          (sourceNode.node_type === 'provider' && targetNode.node_type === 'agent') ||
          (sourceNode.node_type === 'agent' && targetNode.node_type === 'provider')
        ) return true;
        if (sourceNode.node_type === 'agent' && targetNode.node_type === 'structural_output') return true;
        if (sourceNode.node_type === 'agent' && targetNode.node_type === 'agent')
          return sourceNode.agent_name !== targetNode.agent_name;
        return false;
      }

      cancelConnection() {
        if (this.tempLine) { this.connectionLayer.removeChild(this.tempLine); this.tempLine.destroy(); this.tempLine = null; }
        this.connecting = null;
      }

      drawTempLine(x1, y1, x2, y2) {
        if (!this.tempLine) return;
        this.tempLine.clear();
        const cx1 = x1 + Math.abs(x2 - x1) * 0.3, cx2 = x2 - Math.abs(x2 - x1) * 0.3;
        this.tempLine.moveTo(x1, y1).bezierCurveTo(cx1, y1, cx2, y2, x2, y2);
        this.tempLine.stroke({ color: 0x10b981, width: 2, alpha: 0.7 });
      }

      createConnectionSprite(cd) {
        const P = this.PIXI;
        const s = this.nodes.get(cd.source_node_id), t = this.nodes.get(cd.target_node_id);
        if (!s || !t) return;
        const x1 = s.x + NODE_W, y1 = s.y + NODE_H / 2;
        const x2 = t.x, y2 = t.y + NODE_H / 2;
        const cx1 = x1 + Math.abs(x2 - x1) * 0.4;
        const cx2 = x2 - Math.abs(x2 - x1) * 0.4;

        // Main line
        const line = new P.Graphics();
        line.moveTo(x1, y1).bezierCurveTo(cx1, y1, cx2, y2, x2, y2);
        line.stroke({ color: CONN_COL, width: 2 });

        // Arrow
        const a = Math.atan2(y2 - (y1 * 0.3 + y2 * 0.7), x2 - (x1 * 0.3 + x2 * 0.7));
        const al = 10;
        line.moveTo(x2, y2).lineTo(x2 - al * Math.cos(a - 0.4), y2 - al * Math.sin(a - 0.4));
        line.moveTo(x2, y2).lineTo(x2 - al * Math.cos(a + 0.4), y2 - al * Math.sin(a + 0.4));
        line.stroke({ color: CONN_COL, width: 2 });

        // Flow dot (animated)
        const dot = new P.Graphics();
        dot.circle(0, 0, 3);
        dot.fill({ color: 0x10b981, alpha: 0.8 });
        dot._flowData = { x1, y1, x2, y2, cx1, cx2, offset: Math.random() };

        // Hit area
        const hit = new P.Graphics();
        hit.moveTo(x1, y1).bezierCurveTo(cx1, y1, cx2, y2, x2, y2);
        hit.stroke({ color: 0x000000, width: 14, alpha: 0.001 });
        hit.eventMode = 'static'; hit.cursor = 'pointer';
        hit.on('pointerdown', (e) => { e.stopPropagation(); this.removeConnection(cd.connection_id); });

        this.connectionLayer.addChild(line);
        this.connectionLayer.addChild(dot);
        this.connectionLayer.addChild(hit);
        this.connections.set(cd.connection_id, { line, dot });
      }

      _animateConnections() {
        const t = (this._animFrame % 180) / 180;
        for (const [, { dot }] of this.connections) {
          if (!dot._flowData) continue;
          const fd = dot._flowData;
          const p = (t + fd.offset) % 1;
          // Bezier interpolation
          const ip = 1 - p;
          const x = ip*ip*ip*fd.x1 + 3*ip*ip*p*fd.cx1 + 3*ip*p*p*fd.cx2 + p*p*p*fd.x2;
          const y = ip*ip*ip*fd.y1 + 3*ip*ip*p*fd.y1 + 3*ip*p*p*fd.y2 + p*p*p*fd.y2;
          dot.x = x; dot.y = y;
        }
      }

      redrawConnections() {
        this.connectionLayer.removeChildren(); this.connections.clear();
        if (this.tempLine) this.connectionLayer.addChild(this.tempLine);
        for (const c of this.graphState.connections) this.createConnectionSprite(c);
      }

      addNode(nd) { this.graphState.nodes.push(nd); this.rebuildScene(); this.pushStateToGradio(); }
      removeNode(id) {
        this.graphState.nodes = this.graphState.nodes.filter(n => n.node_id !== id);
        this.graphState.connections = this.graphState.connections.filter(c => c.source_node_id !== id && c.target_node_id !== id);
        if (this.graphState.entry_node_id === id) this.graphState.entry_node_id = null;
        if (this.selectedNodeId === id) this.selectedNodeId = null;
        this.rebuildScene(); this.pushStateToGradio();
      }
      removeConnection(id) {
        this.graphState.connections = this.graphState.connections.filter(c => c.connection_id !== id);
        this.rebuildScene(); this.pushStateToGradio();
      }
      setEntryNode(id) { this.graphState.entry_node_id = id; this.rebuildScene(); this.pushStateToGradio(); }

      pushStateToGradio() { this._setBridge('#graph-json-bridge', JSON.stringify(this.graphState)); }

      _setBridge(selector, value) {
        const el = document.querySelector(selector + ' textarea') || document.querySelector(selector + ' input');
        if (!el) return;
        const setter = Object.getOwnPropertyDescriptor(HTMLTextAreaElement.prototype, 'value')?.set ||
                       Object.getOwnPropertyDescriptor(HTMLInputElement.prototype, 'value')?.set;
        if (setter) setter.call(el, value); else el.value = value;
        el.dispatchEvent(new Event('input', { bubbles: true }));
        el.dispatchEvent(new Event('change', { bubbles: true }));
      }

      // ── Runtime animation API ──
      highlightNode(nodeId, status) {
        // status: 'active' | 'done' | 'error' | 'idle'
        const container = this.nodes.get(nodeId);
        if (!container || !container._statusDot) return;
        const dot = container._statusDot;
        const colors = { active: 0x10b981, done: 0x3b82f6, error: 0xef4444, idle: 0x475569 };
        dot.clear();
        dot.circle(NODE_W - 14, NODE_H - 14, status === 'active' ? 6 : 4);
        dot.fill({ color: colors[status] || colors.idle, alpha: status === 'active' ? 1 : 0.5 });
      }
    }

    const canvas = new AgentCanvas(PIXI);
    await canvas.init();
    window.agentBuilderCanvas = canvas;
  }

  if (window.agentBuilderCanvas && json) {
    try { window.agentBuilderCanvas.loadGraph(json); } catch(e) { console.error('loadGraph error:', e); }
  }
}
"""

CANVAS_GET_GRAPH_JS = r"""
(graphJson, ...rest) => {
  const currentGraphJson = window.agentBuilderCanvas
    ? JSON.stringify(window.agentBuilderCanvas.getGraph())
    : graphJson;
  if (rest.length > 0) {
    return [currentGraphJson, ...rest];
  }
  if (window.agentBuilderCanvas) {
    return currentGraphJson;
  }
  return graphJson;
}
"""

CANVAS_SELECT_JS = r"""
(nodeId) => {
  // Selection is handled by the canvas itself via _setBridge
}
"""

# JS to highlight all nodes as "active" before runtime runs
CANVAS_RUNTIME_START_JS = r"""
(graphJson, ...rest) => {
  const currentGraphJson = window.agentBuilderCanvas
    ? JSON.stringify(window.agentBuilderCanvas.getGraph())
    : graphJson;
  // Highlight all nodes as active
  if (window.agentBuilderCanvas) {
    const graph = window.agentBuilderCanvas.getGraph();
    for (const node of (graph.nodes || [])) {
      window.agentBuilderCanvas.highlightNode(node.node_id, 'active');
    }
  }
  if (rest.length > 0) return [currentGraphJson, ...rest];
  return currentGraphJson;
}
"""

# JS to mark all nodes as "done" after runtime completes
CANVAS_RUNTIME_DONE_JS = r"""
(status) => {
  if (window.agentBuilderCanvas) {
    const graph = window.agentBuilderCanvas.getGraph();
    const resolvedStatus = status === 'ok' ? 'done' : 'error';
    for (const node of (graph.nodes || [])) {
      window.agentBuilderCanvas.highlightNode(node.node_id, resolvedStatus);
    }
    // Reset to idle after 3 seconds
    setTimeout(() => {
      for (const node of (graph.nodes || [])) {
        window.agentBuilderCanvas.highlightNode(node.node_id, 'idle');
      }
    }, 3000);
  }
  return status;
}
"""
