export interface Filter {
    id: string;
    name: string;
    enabled: boolean;
    expanded: boolean;
    disabled?: boolean;
}

export const GRAYSCALE_ALGORITHMS = [
    { value: 'bt601', label: 'ITU-R BT.601 (SDTV)' },
    { value: 'bt709', label: 'ITU-R BT.709 (HDTV)' },
    { value: 'average', label: 'Average' },
    { value: 'lightness', label: 'Lightness' },
    { value: 'luminosity', label: 'Luminosity' }
] as const;

export const DEFAULT_FILTERS: Filter[] = [
    { id: 'grayscale', name: 'Grayscale', enabled: false, expanded: false },
    { id: 'blur', name: 'Blur', enabled: false, expanded: false, disabled: true },
    { id: 'edge', name: 'Edge Detect', enabled: false, expanded: false, disabled: true }
];

