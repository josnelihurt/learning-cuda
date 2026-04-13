import { describe, it, expect, vi, afterEach } from 'vitest';
import { render, screen, fireEvent, waitFor } from '@testing-library/react';
import userEvent from '@testing-library/user-event';
import { FilterPanel, type ActiveFilterState } from './FilterPanel';
import type {
  GenericFilterDefinition,
  GenericFilterParameter,
  GenericFilterParameterOption,
  GenericFilterParameterType,
} from '@/gen/image_processor_service_pb';

// Mock useFilters hook
const mockFilters: GenericFilterDefinition[] = [
  {
    id: 'blur',
    name: 'Gaussian Blur',
    type: 0,
    parameters: [
      {
        id: 'radius',
        name: 'Radius',
        type: 3, // NUMBER
        defaultValue: '5',
        metadata: { min: '1', max: '10', step: '1' },
        options: [],
      } as GenericFilterParameter,
      {
        id: 'mode',
        name: 'Edge Mode',
        type: 1, // SELECT
        defaultValue: 'clamp',
        metadata: {},
        options: [
          { value: 'clamp', label: 'Clamp' } as GenericFilterParameterOption,
          { value: 'reflect', label: 'Reflect' } as GenericFilterParameterOption,
        ],
      } as GenericFilterParameter,
    ],
  } as GenericFilterDefinition,
  {
    id: 'grayscale',
    name: 'Grayscale',
    type: 0,
    parameters: [
      {
        id: 'enabled',
        name: 'Enable',
        type: 4, // CHECKBOX
        defaultValue: 'true',
        metadata: {},
        options: [],
      } as GenericFilterParameter,
      {
        id: 'intensity',
        name: 'Intensity',
        type: 2, // RANGE
        defaultValue: '1.0',
        metadata: { min: '0', max: '2', step: '0.1' },
        options: [],
      } as GenericFilterParameter,
    ],
  } as GenericFilterDefinition,
];

vi.mock('../../hooks/useFilters', () => ({
  useFilters: vi.fn(() => ({
    filters: mockFilters,
    loading: false,
    error: null,
    refetch: vi.fn(),
  })),
}));

const mockToastError = vi.fn();
vi.mock('../../hooks/useToast', () => ({
  useToast: () => ({
    error: mockToastError,
  }),
}));

afterEach(() => {
  vi.clearAllMocks();
  document.body.replaceChildren();
});

describe('FilterPanel', () => {
  it('renders filter list with all filters', () => {
    const onFiltersChange = vi.fn();
    render(<FilterPanel filters={mockFilters} onFiltersChange={onFiltersChange} />);

    expect(screen.getByText('Filters')).toBeInTheDocument();
    expect(screen.getByText('(drag to reorder)')).toBeInTheDocument();
    expect(screen.getByText('Gaussian Blur')).toBeInTheDocument();
    expect(screen.getByText('Grayscale')).toBeInTheDocument();
  });

  it('shows no filters message when filters array is empty', () => {
    const onFiltersChange = vi.fn();
    render(<FilterPanel filters={[]} onFiltersChange={onFiltersChange} />);

    expect(screen.getByText('No filters available')).toBeInTheDocument();
  });

  it('toggles filter expansion on card click', () => {
    const onFiltersChange = vi.fn();
    render(<FilterPanel filters={mockFilters} onFiltersChange={onFiltersChange} />);

    // Find the filter card for blur
    const blurCard = screen.getByTestId('filter-checkbox-blur').closest('[data-filter-id="blur"]') as HTMLElement;
    const blurHeader = blurCard.querySelector('[class*="filterHeader"]') as HTMLElement;
    const filterBody = blurCard.querySelector('[class*="filterBody"]') as HTMLElement;

    expect(filterBody.className).not.toContain('expanded');

    fireEvent.click(blurHeader);

    expect(filterBody.className).toContain('expanded');
    expect(onFiltersChange).toHaveBeenCalled();
  });

  it('enables filter when card is clicked and was disabled', () => {
    const onFiltersChange = vi.fn();
    render(<FilterPanel filters={mockFilters} onFiltersChange={onFiltersChange} />);

    const blurCard = screen.getByTestId('filter-checkbox-blur').closest('[data-filter-id="blur"]') as HTMLElement;
    const blurHeader = blurCard.querySelector('[class*="filterHeader"]') as HTMLElement;
    const checkbox = screen.getByTestId('filter-checkbox-blur') as HTMLInputElement;

    expect(checkbox.checked).toBe(false);

    fireEvent.click(blurHeader);

    expect(checkbox.checked).toBe(true);

    // Check that filter body is now visible (expanded)
    const filterBody = blurCard.querySelector('[class*="filterBody"]') as HTMLElement;
    expect(filterBody.className).toContain('expanded');
  });

  it('toggles filter enable state via checkbox', () => {
    const onFiltersChange = vi.fn();
    render(<FilterPanel filters={mockFilters} onFiltersChange={onFiltersChange} />);

    const checkbox = screen.getByTestId('filter-checkbox-blur') as HTMLInputElement;

    expect(checkbox.checked).toBe(false);

    fireEvent.click(checkbox);

    expect(checkbox.checked).toBe(true);
    expect(onFiltersChange).toHaveBeenCalled();
  });

  it('collapses filter when checkbox is unchecked', () => {
    const onFiltersChange = vi.fn();
    render(<FilterPanel filters={mockFilters} onFiltersChange={onFiltersChange} />);

    const blurCard = screen.getByTestId('filter-checkbox-blur').closest('[data-filter-id="blur"]') as HTMLElement;
    const blurHeader = blurCard.querySelector('[class*="filterHeader"]') as HTMLElement;
    const checkbox = screen.getByTestId('filter-checkbox-blur') as HTMLInputElement;
    const filterBody = blurCard.querySelector('[class*="filterBody"]') as HTMLElement;

    // Enable first
    fireEvent.click(blurHeader);
    expect(filterBody.className).toContain('expanded');

    // Disable via checkbox
    fireEvent.click(checkbox);

    expect(checkbox.checked).toBe(false);
    expect(filterBody.className).not.toContain('expanded');
  });

  it('renders number input control', () => {
    const onFiltersChange = vi.fn();
    render(<FilterPanel filters={mockFilters} onFiltersChange={onFiltersChange} />);

    // Expand blur filter
    const blurCard = screen.getByTestId('filter-checkbox-blur').closest('[data-filter-id="blur"]') as HTMLElement;
    const blurHeader = blurCard.querySelector('[class*="filterHeader"]') as HTMLElement;
    fireEvent.click(blurHeader);

    expect(screen.getByText('Radius')).toBeInTheDocument();

    const numberInput = screen.getByTestId('filter-parameter-blur-radius') as HTMLInputElement;
    expect(numberInput.type).toBe('number');
    expect(numberInput.value).toBe('5');
    expect(numberInput.min).toBe('1');
    expect(numberInput.max).toBe('10');
    expect(numberInput.step).toBe('1');
  });

  it('renders radio group for select parameter', () => {
    const onFiltersChange = vi.fn();
    render(<FilterPanel filters={mockFilters} onFiltersChange={onFiltersChange} />);

    // Expand blur filter
    const blurCard = screen.getByTestId('filter-checkbox-blur').closest('[data-filter-id="blur"]') as HTMLElement;
    const blurHeader = blurCard.querySelector('[class*="filterHeader"]') as HTMLElement;
    fireEvent.click(blurHeader);

    expect(screen.getByText('Edge Mode')).toBeInTheDocument();

    const clampRadio = screen.getByTestId('filter-parameter-blur-mode-clamp') as HTMLInputElement;
    const reflectRadio = screen.getByTestId('filter-parameter-blur-mode-reflect') as HTMLInputElement;

    expect(clampRadio.type).toBe('radio');
    expect(reflectRadio.type).toBe('radio');
    expect(clampRadio.checked).toBe(true);
    expect(reflectRadio.checked).toBe(false);
    expect(screen.getByText('Clamp')).toBeInTheDocument();
    expect(screen.getByText('Reflect')).toBeInTheDocument();
  });

  it('renders slider for range parameter', () => {
    const onFiltersChange = vi.fn();
    render(<FilterPanel filters={mockFilters} onFiltersChange={onFiltersChange} />);

    // Expand grayscale filter
    const grayscaleCard = screen.getByTestId('filter-checkbox-grayscale').closest('[data-filter-id="grayscale"]') as HTMLElement;
    const grayscaleHeader = grayscaleCard.querySelector('[class*="filterHeader"]') as HTMLElement;
    fireEvent.click(grayscaleHeader);

    expect(screen.getByText('Intensity')).toBeInTheDocument();

    const slider = screen.getByTestId('filter-parameter-grayscale-intensity') as HTMLInputElement;
    expect(slider.type).toBe('range');
    expect(slider.value).toBe('1');
    expect(slider.min).toBe('0');
    expect(slider.max).toBe('2');
    expect(slider.step).toBe('0.1');

    expect(screen.getByText('1')).toBeInTheDocument();
  });

  it('renders checkbox for checkbox parameter', () => {
    const onFiltersChange = vi.fn();
    render(<FilterPanel filters={mockFilters} onFiltersChange={onFiltersChange} />);

    // Expand grayscale filter
    const grayscaleCard = screen.getByText('Grayscale');
    fireEvent.click(grayscaleCard);

    expect(screen.getByText('Enable')).toBeInTheDocument();

    const checkbox = screen.getByTestId('filter-parameter-grayscale-enabled') as HTMLInputElement;
    expect(checkbox.type).toBe('checkbox');
    expect(checkbox.checked).toBe(true);
  });

  it('updates parameter value and calls onFiltersChange', () => {
    const onFiltersChange = vi.fn();
    render(<FilterPanel filters={mockFilters} onFiltersChange={onFiltersChange} />);

    // Expand blur filter
    const blurCard = screen.getByText('Gaussian Blur');
    fireEvent.click(blurCard);

    const numberInput = screen.getByTestId('filter-parameter-blur-radius') as HTMLInputElement;
    fireEvent.change(numberInput, { target: { value: '7' } });

    expect(onFiltersChange).toHaveBeenCalled();

    const calls = onFiltersChange.mock.calls;
    const lastCall = calls[calls.length - 1][0] as ActiveFilterState[];
    const blurFilter = lastCall.find((f) => f.id === 'blur');
    expect(blurFilter?.parameters.radius).toBe('7');
  });

  it('clamps number input values to min/max', async () => {
    const onFiltersChange = vi.fn();

    render(<FilterPanel filters={mockFilters} onFiltersChange={onFiltersChange} />);

    // Expand blur filter
    const blurCard = screen.getByTestId('filter-checkbox-blur').closest('[data-filter-id="blur"]') as HTMLElement;
    const blurHeader = blurCard.querySelector('[class*="filterHeader"]') as HTMLElement;
    fireEvent.click(blurHeader);

    const numberInput = screen.getByTestId('filter-parameter-blur-radius') as HTMLInputElement;

    // Test exceeding max
    fireEvent.change(numberInput, { target: { value: '15' } });

    await waitFor(() => {
      expect(numberInput.value).toBe('10');
    });

    // Wait for debounced toast error (100ms)
    await waitFor(
      () => {
        expect(mockToastError).toHaveBeenCalled();
      },
      { timeout: 500 }
    );

    expect(mockToastError).toHaveBeenCalledWith('Invalid Value', expect.stringContaining('at most 10'));
  });

  it('supports drag and drop reordering', async () => {
    const onFiltersChange = vi.fn();
    render(<FilterPanel filters={mockFilters} onFiltersChange={onFiltersChange} />);

    const filtersList = screen.getByText('Filters').nextElementSibling as HTMLElement;
    const filterCards = filtersList.querySelectorAll('[data-filter-name]');

    const firstCard = filterCards[0] as HTMLElement;
    const secondCard = filterCards[1] as HTMLElement;

    expect(firstCard.getAttribute('data-filter-name')).toBe('Gaussian Blur');
    expect(secondCard.getAttribute('data-filter-name')).toBe('Grayscale');

    // Simulate drag and drop
    fireEvent.dragStart(firstCard);
    fireEvent.drop(secondCard);

    await waitFor(() => {
      expect(onFiltersChange).toHaveBeenCalled();
    });
  });

  it('applies initial active filters', () => {
    const onFiltersChange = vi.fn();
    const initialActiveFilters: ActiveFilterState[] = [
      {
        id: 'blur',
        parameters: { radius: '8', mode: 'reflect' },
      },
    ];

    render(
      <FilterPanel
        filters={mockFilters}
        onFiltersChange={onFiltersChange}
        initialActiveFilters={initialActiveFilters}
      />
    );

    const blurCheckbox = screen.getByTestId('filter-checkbox-blur') as HTMLInputElement;
    expect(blurCheckbox.checked).toBe(true);

    // Expand to see parameters
    const blurCard = screen.getByText('Gaussian Blur');
    fireEvent.click(blurCard);

    const radiusInput = screen.getByTestId('filter-parameter-blur-radius') as HTMLInputElement;
    expect(radiusInput.value).toBe('8');

    const reflectRadio = screen.getByTestId('filter-parameter-blur-mode-reflect') as HTMLInputElement;
    expect(reflectRadio.checked).toBe(true);
  });
});
