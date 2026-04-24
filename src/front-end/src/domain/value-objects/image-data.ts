export class ImageData {
  private readonly base64: string;
  private readonly width: number;
  private readonly height: number;

  constructor(base64: string, width: number, height: number) {
    this.validateBase64(base64);
    this.validateDimensions(width, height);
    
    this.base64 = base64;
    this.width = width;
    this.height = height;
  }

  private validateBase64(base64: string): void {
    if (!base64 || base64.trim() === '') {
      throw new Error('Image data cannot be empty');
    }
    if (!base64.trim().startsWith('data:image/')) {
      throw new Error('Invalid image format: must start with data:image/');
    }
  }

  private validateDimensions(width: number, height: number): void {
    if (width <= 0 || height <= 0) {
      throw new Error('Image dimensions must be positive');
    }
    if (!Number.isInteger(width) || !Number.isInteger(height)) {
      throw new Error('Image dimensions must be integers');
    }
  }

  getBase64(): string {
    return this.base64;
  }

  getWidth(): number {
    return this.width;
  }

  getHeight(): number {
    return this.height;
  }

  getAspectRatio(): number {
    return this.width / this.height;
  }

  toDataUrl(): string {
    return this.base64;
  }

  equals(other: ImageData): boolean {
    return (
      this.base64 === other.base64 &&
      this.width === other.width &&
      this.height === other.height
    );
  }
}
