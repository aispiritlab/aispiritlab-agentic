"""Generates the HTML container and JS initialization for the PixiJS agent graph canvas."""

from __future__ import annotations


def build_canvas_html() -> str:
    """Return the HTML container for the canvas (no scripts — scripts don't execute in gr.HTML)."""
    return """
<div id="agent-canvas-container" style="width:100%;height:600px;position:relative;background:#1a1a2e;border-radius:8px;overflow:hidden;">
  <canvas id="agent-canvas" style="width:100%;height:100%;display:block;"></canvas>
</div>
"""


# JS function that initializes PixiJS on first call, then loads the graph.
# Used via Gradio's js= parameter on events (the only reliable way to run JS in Gradio).
# Receives the graph JSON string as its single argument.
CANVAS_LOAD_JS = r"""
async (json) => {
  if (!window._agentCanvasReady) {
    window._agentCanvasReady = true;

    const PIXI = await import('https://cdn.jsdelivr.net/npm/pixi.js@8.6.6/dist/pixi.min.mjs');

    const NODE_COLORS = { agent: 0x2ecc71, integration: 0xe67e22, structural_output: 0xf1c40f, provider: 0x16a085, entry_point: 0x3498db };
    const NODE_W = 180, NODE_H = 80, PORT_R = 8, CONN_COL = 0xecf0f1;

    class AgentCanvas {
      constructor(PIXI) { this.PIXI = PIXI; this.app = null; this.world = null; this.nodes = new Map(); this.connections = new Map(); this.graphState = {graph_id:'',name:'',nodes:[],connections:[],entry_node_id:null}; this.selectedNodeId = null; this.dragging = null; this.connecting = null; this.tempLine = null; this.connectionLayer = null; this.nodeLayer = null; }

      async init() {
        const container = document.getElementById('agent-canvas-container');
        const canvas = document.getElementById('agent-canvas');
        if (!container || !canvas) { console.error('Canvas container not found'); return; }

        this.app = new PIXI.Application();
        await this.app.init({ canvas, width: container.clientWidth, height: container.clientHeight, backgroundColor: 0x1a1a2e, antialias: true, resolution: window.devicePixelRatio || 1, autoDensity: true });

        this.world = new PIXI.Container();
        this.app.stage.addChild(this.world);
        this.connectionLayer = new PIXI.Container();
        this.nodeLayer = new PIXI.Container();
        this.world.addChild(this.connectionLayer);
        this.world.addChild(this.nodeLayer);

        this.app.stage.eventMode = 'static';
        this.app.stage.hitArea = this.app.screen;
        this.app.stage.on('pointerdown', (e) => { if (e.target === this.app.stage) { this.selectNode(null); if (this.connecting) this.cancelConnection(); } });
        this.app.stage.on('pointermove', (e) => { if (this.connecting && this.tempLine) { const p = e.global; this.drawTempLine(this.connecting.startX, this.connecting.startY, p.x, p.y); } });

        new ResizeObserver(() => { if (this.app && container) { this.app.renderer.resize(container.clientWidth, container.clientHeight); this.app.stage.hitArea = this.app.screen; } }).observe(container);
      }

      loadGraph(state) { this.graphState = typeof state === 'string' ? JSON.parse(state) : state; this.rebuildScene(); }
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
        const baseCol = isEntry ? NODE_COLORS.entry_point : (NODE_COLORS[nd.node_type] || NODE_COLORS.agent);

        const body = new P.Graphics();
        body.roundRect(0, 0, NODE_W, NODE_H, 12);
        body.fill({ color: baseCol, alpha: 0.85 });
        body.stroke({ color: 0xffffff, width: this.selectedNodeId === nd.node_id ? 3 : 1, alpha: this.selectedNodeId === nd.node_id ? 1 : 0.3 });
        g.addChild(body);

        const badge = new P.Text({ text: nd.node_type.toUpperCase(), style: { fontSize: 9, fill: 0xffffff, fontFamily: 'monospace', fontWeight: 'bold' } });
        badge.x = 8; badge.y = 6; g.addChild(badge);

        const label = new P.Text({ text: nd.display_name, style: { fontSize: 14, fill: 0xffffff, fontFamily: 'Arial', fontWeight: 'bold' } });
        label.x = 8; label.y = 22; g.addChild(label);

        const sub = new P.Text({ text: nd.agent_name, style: { fontSize: 10, fill: 0xffffff, fontFamily: 'monospace' } });
        sub.x = 8; sub.y = 44; sub.alpha = 0.7; g.addChild(sub);

        // Input port (left)
        const inP = new P.Graphics(); inP.circle(0, NODE_H/2, PORT_R); inP.fill({color:0xecf0f1}); inP.stroke({color:0x2c3e50,width:2});
        inP.eventMode = 'static'; inP.cursor = 'crosshair';
        inP.on('pointerdown', (e) => { e.stopPropagation(); this.finishConnection(nd.node_id); });
        g.addChild(inP);

        // Output port (right)
        const outP = new P.Graphics(); outP.circle(NODE_W, NODE_H/2, PORT_R); outP.fill({color:0xecf0f1}); outP.stroke({color:0x2c3e50,width:2});
        outP.eventMode = 'static'; outP.cursor = 'crosshair';
        outP.on('pointerdown', (e) => { e.stopPropagation(); this.startConnection(nd.node_id, g.x+NODE_W, g.y+NODE_H/2); });
        g.addChild(outP);

        // Delete button
        const del = new P.Graphics(); del.circle(NODE_W-4, 4, 10); del.fill({color:0xe74c3c,alpha:0.8});
        del.eventMode = 'static'; del.cursor = 'pointer';
        del.on('pointerdown', (e) => { e.stopPropagation(); this.removeNode(nd.node_id); });
        g.addChild(del);
        const delX = new P.Text({text:'\u00d7',style:{fontSize:14,fill:0xffffff,fontWeight:'bold'}});
        delX.x = NODE_W-10; delX.y = -5; g.addChild(delX);

        // Drag
        g.on('pointerdown', (e) => {
          if (e.target === inP || e.target === outP || e.target === del) return;
          this.selectNode(nd.node_id);
          this.dragging = { nodeId: nd.node_id, offX: e.global.x - g.x, offY: e.global.y - g.y };
          const onMove = (ev) => {
            if (!this.dragging) return;
            const gr = this.nodes.get(this.dragging.nodeId); if (!gr) return;
            gr.x = ev.global.x - this.dragging.offX; gr.y = ev.global.y - this.dragging.offY;
            const d = this.graphState.nodes.find(n => n.node_id === this.dragging.nodeId);
            if (d) { d.position.x = gr.x; d.position.y = gr.y; }
            this.redrawConnections();
          };
          const onEnd = () => { this.dragging = null; this.app.stage.off('pointermove', onMove); this.app.stage.off('pointerup', onEnd); this.app.stage.off('pointerupoutside', onEnd); this.pushStateToGradio(); };
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

      startConnection(srcId, sx, sy) { this.connecting = {sourceId:srcId,startX:sx,startY:sy}; this.tempLine = new this.PIXI.Graphics(); this.connectionLayer.addChild(this.tempLine); }

      finishConnection(tgtId) {
        if (!this.connecting || this.connecting.sourceId === tgtId) { this.cancelConnection(); return; }
        const sourceNode = this.graphState.nodes.find(n => n.node_id === this.connecting.sourceId);
        const targetNode = this.graphState.nodes.find(n => n.node_id === tgtId);
        if (!this.isAllowedConnection(sourceNode, targetNode)) { this.cancelConnection(); return; }
        const exists = this.graphState.connections.some(c => c.source_node_id === this.connecting.sourceId && c.target_node_id === tgtId);
        if (!exists) {
          this.graphState.connections.push({ connection_id: 'conn_'+Date.now().toString(36)+'_'+Math.random().toString(36).slice(2,6), source_node_id: this.connecting.sourceId, target_node_id: tgtId, message_type: 'UserMessage', label: '' });
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
        ) {
          return true;
        }
        if (sourceNode.node_type === 'agent' && targetNode.node_type === 'structural_output') return true;
        if (sourceNode.node_type === 'agent' && targetNode.node_type === 'agent') return sourceNode.agent_name !== targetNode.agent_name;
        return false;
      }

      cancelConnection() { if (this.tempLine) { this.connectionLayer.removeChild(this.tempLine); this.tempLine.destroy(); this.tempLine = null; } this.connecting = null; }

      drawTempLine(x1,y1,x2,y2) { if (!this.tempLine) return; this.tempLine.clear(); this.tempLine.moveTo(x1,y1).lineTo(x2,y2).stroke({color:0xe74c3c,width:2,alpha:0.6}); }

      createConnectionSprite(cd) {
        const P = this.PIXI;
        const s = this.nodes.get(cd.source_node_id), t = this.nodes.get(cd.target_node_id);
        if (!s || !t) return;
        const x1=s.x+NODE_W, y1=s.y+NODE_H/2, x2=t.x, y2=t.y+NODE_H/2;
        const cx1=x1+Math.abs(x2-x1)*0.4, cx2=x2-Math.abs(x2-x1)*0.4;

        const line = new P.Graphics();
        line.moveTo(x1,y1).bezierCurveTo(cx1,y1,cx2,y2,x2,y2).stroke({color:CONN_COL,width:2.5});
        const a = Math.atan2(y2-y1,x2-x1), al=12;
        line.moveTo(x2,y2).lineTo(x2-al*Math.cos(a-0.4),y2-al*Math.sin(a-0.4)).moveTo(x2,y2).lineTo(x2-al*Math.cos(a+0.4),y2-al*Math.sin(a+0.4)).stroke({color:CONN_COL,width:2.5});

        const hit = new P.Graphics();
        hit.moveTo(x1,y1).bezierCurveTo(cx1,y1,cx2,y2,x2,y2).stroke({color:0x000000,width:12,alpha:0.001});
        hit.eventMode = 'static'; hit.cursor = 'pointer';
        hit.on('pointerdown', (e) => { e.stopPropagation(); this.removeConnection(cd.connection_id); });

        this.connectionLayer.addChild(line); this.connectionLayer.addChild(hit);
        this.connections.set(cd.connection_id, line);
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
      removeConnection(id) { this.graphState.connections = this.graphState.connections.filter(c => c.connection_id !== id); this.rebuildScene(); this.pushStateToGradio(); }
      setEntryNode(id) { this.graphState.entry_node_id = id; this.rebuildScene(); this.pushStateToGradio(); }

      pushStateToGradio() { this._setBridge('#graph-json-bridge', JSON.stringify(this.graphState)); }

      _setBridge(selector, value) {
        const el = document.querySelector(selector + ' textarea') || document.querySelector(selector + ' input');
        if (!el) return;
        const setter = Object.getOwnPropertyDescriptor(HTMLTextAreaElement.prototype, 'value')?.set || Object.getOwnPropertyDescriptor(HTMLInputElement.prototype, 'value')?.set;
        if (setter) setter.call(el, value); else el.value = value;
        el.dispatchEvent(new Event('input', {bubbles:true}));
        el.dispatchEvent(new Event('change', {bubbles:true}));
      }
    }

    const canvas = new AgentCanvas(PIXI);
    await canvas.init();
    window.agentBuilderCanvas = canvas;
  }

  // Load graph data
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

# Simpler JS for just pushing selection bridge (no PixiJS needed)
CANVAS_SELECT_JS = r"""
(nodeId) => {
  // Selection is handled by the canvas itself via _setBridge
}
"""
