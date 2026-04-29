import { FilterType } from '@/gen/common_pb';

type FilterTypeId = string;

export class FilterData {
  private readonly type: FilterTypeId;
  private readonly parameters: Readonly<Record<string, any>>;

  constructor(type: FilterTypeId, parameters: Record<string, any> = {}) {
    if (!type || type.trim() === '') {
      throw new Error('Filter type cannot be empty');
    }

    const normalizedType = type.trim();
    this.validateParameters(parameters);

    this.type = normalizedType;
    this.parameters = Object.freeze({ ...parameters });
  }

  private validateParameters(params: Record<string, any>): void {
    if (params === null || typeof params !== 'object' || Array.isArray(params)) {
      throw new Error('Filter parameters must be an object');
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

  toProtocol(): FilterType {
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
