/**
 * WebSocket + SSE connection manager with auto-reconnect.
 */
export class PipelineConnection {
  constructor(runId, wsUrl, onEvent) {
    this.runId = runId;
    this.wsUrl = wsUrl;
    this.onEvent = onEvent;
    this.ws = null;
    this.es = null;
    this.reconnectAttempts = 0;
    this.maxReconnects = 5;
  }

  connect() {
    try {
      this.ws = new WebSocket(`${this.wsUrl}/ws/pipeline/${this.runId}`);
      this.ws.onmessage = (event) => {
        const data = JSON.parse(event.data);
        if (data.type !== 'ping') {
          this.onEvent(data);
        }
      };
      this.ws.onclose = () => this._reconnect();
      this.ws.onerror = () => this._fallbackToSSE();
    } catch (e) {
      this._fallbackToSSE();
    }
  }

  _fallbackToSSE() {
    if (this.ws) { try { this.ws.close(); } catch(e) {} }
    this.es = new EventSource(`/api/pipeline/${this.runId}/events`);
    this.es.onmessage = (e) => {
      const data = JSON.parse(e.data);
      this.onEvent(data);
    };
    this.es.onerror = () => { if (this.es) this.es.close(); };
  }

  _reconnect() {
    if (this.reconnectAttempts < this.maxReconnects) {
      this.reconnectAttempts++;
      setTimeout(() => this.connect(), 2000 * this.reconnectAttempts);
    }
  }

  disconnect() {
    if (this.ws) { try { this.ws.close(); } catch(e) {} }
    if (this.es) { try { this.es.close(); } catch(e) {} }
  }
}
