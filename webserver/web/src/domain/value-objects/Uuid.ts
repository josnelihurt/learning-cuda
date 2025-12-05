import { v4 as uuidv4, validate as uuidValidate } from 'uuid';

export class Uuid {
  static generate(): string {
    return uuidv4();
  }

  static isValid(uuid: string): boolean {
    return uuidValidate(uuid);
  }

  static validate(uuid: string): void {
    if (!uuid || typeof uuid !== 'string') {
      throw new Error('UUID must be a non-empty string');
    }
    if (!this.isValid(uuid)) {
      throw new Error(`Invalid UUID v4 format: ${uuid}`);
    }
  }
}

