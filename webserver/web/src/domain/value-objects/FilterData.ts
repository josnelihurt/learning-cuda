import { FilterType } from '../../gen/common_pb';

export type ValidFilterType = 'none' | 'grayscale' | 'blur';

export class FilterData {
  private readonly type: ValidFilterType;
  private readonly parameters: Readonly<Record<string, any>>;

  constructor(type: ValidFilterType, parameters: Record<string, any> = {}) {
    this.validateType(type);
    this.validateParameters(type, parameters);
    
    this.type = type;
    this.parameters = Object.freeze({ ...parameters });
  }

  private validateType(type: ValidFilterType): void {
    if (!type || type.trim() === '') {
      throw new Error('Filter type cannot be empty');
    }
    
    const validTypes: ValidFilterType[] = ['none', 'grayscale', 'blur'];
    if (!validTypes.includes(type)) {
      throw new Error(`Invalid filter type: ${type}. Valid types: ${validTypes.join(', ')}`);
    }
  }

  private validateParameters(type: ValidFilterType, params: Record<string, any>): void {
    if (type === 'grayscale' && Object.keys(params).length > 0) {
      // Permitir parámetros pero no validar por ahora (futura extensión)
    }
    
    if (type === 'blur') {
      if (params.kernel_size !== undefined) {
        if (typeof params.kernel_size !== 'number' || params.kernel_size < 1 || params.kernel_size % 2 === 0) {
          throw new Error('kernel_size must be a positive odd number');
        }
      }
      
      if (params.sigma !== undefined) {
        if (typeof params.sigma !== 'number' || params.sigma < 0) {
          throw new Error('sigma must be a non-negative number');
        }
      }
      
      if (params.separable !== undefined && typeof params.separable !== 'boolean') {
        throw new Error('separable must be a boolean');
      }
    }
    
    if (params.radius !== undefined) {
      if (typeof params.radius !== 'number' || params.radius < 0 || params.radius > 100) {
        throw new Error('Radius must be a number between 0 and 100');
      }
    }
    
    if (params.intensity !== undefined) {
      if (typeof params.intensity !== 'number' || params.intensity < 0 || params.intensity > 1) {
        throw new Error('Intensity must be a number between 0 and 1');
      }
    }
  }

  getType(): ValidFilterType {
    return this.type;
  }

  getParameters(): Record<string, any> {
    return { ...this.parameters };
  }

  hasParameters(): boolean {
    return Object.keys(this.parameters).length > 0;
  }

  getParameter(key: string): any {
    return this.parameters[key];
  }

  isNone(): boolean {
    return this.type === 'none';
  }

  isGrayscale(): boolean {
    return this.type === 'grayscale';
  }

  isBlur(): boolean {
    return this.type === 'blur';
  }

  toProtocol(): FilterType {
    switch (this.type) {
      case 'none':
        return FilterType.NONE;
      case 'grayscale':
        return FilterType.GRAYSCALE;
      case 'blur':
        return FilterType.BLUR;
      default:
        return FilterType.NONE;
    }
  }

  equals(other: FilterData): boolean {
    if (this.type !== other.type) {
      return false;
    }
    
    return JSON.stringify(this.parameters) === JSON.stringify(other.parameters);
  }

  toString(): string {
    const params = this.hasParameters() 
      ? ` (${JSON.stringify(this.parameters)})`
      : '';
    return `${this.type}${params}`;
  }
}
