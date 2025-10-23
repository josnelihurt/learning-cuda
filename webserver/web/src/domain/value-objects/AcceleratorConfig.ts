import { AcceleratorType } from '../../gen/common_pb';

export type ValidAcceleratorType = 'cpu' | 'gpu';

export class AcceleratorConfig {
  private readonly type: ValidAcceleratorType;

  constructor(type: string) {
    this.validateAndNormalizeType(type);
  }

  private validateAndNormalizeType(type: string): void {
    if (!type || type.trim() === '') {
      throw new Error('Accelerator type cannot be empty');
    }

    const normalizedType = type.trim().toLowerCase();
    
    // Normalize 'cuda' to 'gpu'
    if (normalizedType === 'cuda') {
      this.type = 'gpu';
      return;
    }

    const validTypes: ValidAcceleratorType[] = ['cpu', 'gpu'];
    if (!validTypes.includes(normalizedType as ValidAcceleratorType)) {
      throw new Error(`Invalid accelerator type: ${type}. Valid types: ${validTypes.join(', ')}`);
    }

    this.type = normalizedType as ValidAcceleratorType;
  }

  getType(): ValidAcceleratorType {
    return this.type;
  }

  isCPU(): boolean {
    return this.type === 'cpu';
  }

  isGPU(): boolean {
    return this.type === 'gpu';
  }

  toProtocol(): AcceleratorType {
    switch (this.type) {
      case 'cpu':
        return AcceleratorType.CPU;
      case 'gpu':
        return AcceleratorType.CUDA;
      default:
        return AcceleratorType.CPU;
    }
  }

  equals(other: AcceleratorConfig): boolean {
    return this.type === other.type;
  }

  toString(): string {
    return this.type;
  }
}
