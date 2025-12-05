import { Uuid } from './Uuid';

export class WebRTCSession {
  private readonly sessionId: string;
  private readonly sourceId: string;
  private readonly createdAt: Date;
  private lastHeartbeat: Date;

  constructor(sessionId: string, sourceId: string, createdAt?: Date, lastHeartbeat?: Date) {
    Uuid.validate(sessionId);
    
    if (!sourceId || sourceId.trim() === '') {
      throw new Error('Source ID cannot be empty');
    }

    this.sessionId = sessionId;
    this.sourceId = sourceId.trim();
    this.createdAt = createdAt || new Date();
    this.lastHeartbeat = lastHeartbeat || this.createdAt;

    if (this.lastHeartbeat < this.createdAt) {
      throw new Error('Last heartbeat cannot be before creation time');
    }
  }

  getId(): string {
    return this.sessionId;
  }

  getSourceId(): string {
    return this.sourceId;
  }

  getCreatedAt(): Date {
    return new Date(this.createdAt);
  }

  getLastHeartbeat(): Date {
    return new Date(this.lastHeartbeat);
  }

  updateHeartbeat(): WebRTCSession {
    const now = new Date();
    if (now < this.lastHeartbeat) {
      return this;
    }
    return new WebRTCSession(this.sessionId, this.sourceId, this.createdAt, now);
  }

  isExpired(timeoutMs: number): boolean {
    const now = new Date();
    const elapsed = now.getTime() - this.lastHeartbeat.getTime();
    return elapsed > timeoutMs;
  }

  equals(other: WebRTCSession): boolean {
    return this.sessionId === other.sessionId;
  }

  static create(sourceId: string): WebRTCSession {
    const sessionId = Uuid.generate();
    return new WebRTCSession(sessionId, sourceId);
  }
}

