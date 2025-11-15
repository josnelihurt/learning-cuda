import { FilterType } from '../../gen/common_pb';

export type FilterTypeId = string;

export class FilterData {
  private readonly type: FilterTypeId;
  private readonly parameters: Readonly<Record<string, any>>;

  constructor(type: FilterTypeId, parameters: Record<string, any> = {}) {
    if (!type || type.trim() === '') {
      throw new Error('Filter type cannot be empty');
    }

    const normalizedType = type.trim();
    this.validateParameters(normalizedType, parameters);

    this.type = normalizedType;
    this.parameters = Object.freeze({ ...parameters });
  }

  private validateParameters(type: FilterTypeId, params: Record<string, any>): void {
    if (type === 'blur') {
      if (params.kernel_size !== undefined) {
        const kernel =
          typeof params.kernel_size === 'string'
            ? parseInt(params.kernel_size, 10)
            : params.kernel_size;
        if (typeof kernel !== 'number' || Number.isNaN(kernel) || kernel < 1 || kernel % 2 === 0) {
          throw new Error('kernel_size must be a positive odd number');
        }
      }

      if (params.sigma !== undefined) {
        const sigma =
          typeof params.sigma === 'string' ? parseFloat(params.sigma) : params.sigma;
        if (typeof sigma !== 'number' || Number.isNaN(sigma) || sigma < 0) {
          throw new Error('sigma must be a non-negative number');
        }
      }

      if (
        params.separable !== undefined &&
        typeof params.separable !== 'boolean' &&
        typeof params.separable !== 'string'
      ) {
        throw new Error('separable must be a boolean');
      }
    }
  }

  getId(): FilterTypeId {
    return this.type;
  }

  getType(): FilterTypeId {
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
    if (this.type === 'none') {
      return FilterType.NONE;
    }
    if (this.type === 'grayscale') {
      return FilterType.GRAYSCALE;
    }
    if (this.type === 'blur') {
      return FilterType.BLUR;
    }

    const normalized = this.type.replace(/ /g, '_').toUpperCase();
    if (normalized in FilterType) {
      return FilterType[normalized as keyof typeof FilterType];
    }

    return FilterType.UNSPECIFIED;
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
