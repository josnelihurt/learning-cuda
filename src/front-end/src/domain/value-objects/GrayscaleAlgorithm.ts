import { GrayscaleType } from '../../gen/common_pb';

export type ValidGrayscaleType = 'bt601' | 'bt709' | 'average' | 'lightness' | 'luminosity';

export class GrayscaleAlgorithm {
  private readonly type: ValidGrayscaleType;

  constructor(type: string) {
    this.validateType(type);
  }

  private validateType(type: string): void {
    if (!type || type.trim() === '') {
      throw new Error('Grayscale algorithm type cannot be empty');
    }

    const normalizedType = type.trim().toLowerCase();
    const validTypes: ValidGrayscaleType[] = ['bt601', 'bt709', 'average', 'lightness', 'luminosity'];
    
    if (!validTypes.includes(normalizedType as ValidGrayscaleType)) {
      throw new Error(`Invalid grayscale algorithm type: ${type}. Valid types: ${validTypes.join(', ')}`);
    }

    this.type = normalizedType as ValidGrayscaleType;
  }

  getType(): ValidGrayscaleType {
    return this.type;
  }

  isBT601(): boolean {
    return this.type === 'bt601';
  }

  isBT709(): boolean {
    return this.type === 'bt709';
  }

  isAverage(): boolean {
    return this.type === 'average';
  }

  isLightness(): boolean {
    return this.type === 'lightness';
  }

  isLuminosity(): boolean {
    return this.type === 'luminosity';
  }

  toProtocol(): GrayscaleType {
    switch (this.type) {
      case 'bt601':
        return GrayscaleType.BT601;
      case 'bt709':
        return GrayscaleType.BT709;
      case 'average':
        return GrayscaleType.AVERAGE;
      case 'lightness':
        return GrayscaleType.LIGHTNESS;
      case 'luminosity':
        return GrayscaleType.LUMINOSITY;
      default:
        return GrayscaleType.BT601;
    }
  }

  equals(other: GrayscaleAlgorithm): boolean {
    return this.type === other.type;
  }

  getDescription(): string {
    switch (this.type) {
      case 'bt601':
        return 'ITU-R BT.601 (SDTV): Y = 0.299R + 0.587G + 0.114B';
      case 'bt709':
        return 'ITU-R BT.709 (HDTV): Y = 0.2126R + 0.7152G + 0.0722B';
      case 'average':
        return 'Average: Y = (R + G + B) / 3';
      case 'lightness':
        return 'Lightness: Y = (max(R,G,B) + min(R,G,B)) / 2';
      case 'luminosity':
        return 'Luminosity: Y = 0.21R + 0.72G + 0.07B';
      default:
        return 'Unknown algorithm';
    }
  }

  toString(): string {
    return this.type;
  }
}
