export interface IUIService {
  selectedInputSource: string;
  selectedAccelerator: string;
  currentState: 'static' | 'streaming';
  setInputSource(source: string): string;
  setAccelerator(type: string): string;
  applyFilter(): Promise<void>;
}
