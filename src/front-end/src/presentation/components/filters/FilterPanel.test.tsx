import { describe, it, expect, vi, afterEach } from 'vitest';
import { render, screen, fireEvent, waitFor, createEvent } from '@testing-library/react';
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
        minValue: 1,
        maxValue: 10,
        step: 1,
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
        minValue: 0,
        maxValue: 2,
        step: 0.1,
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

// --- Helpers ---

/** Returns all filter cards in current render order. */
function getFilterCards(): HTMLElement[] {
  const filtersList = screen.getByText('Filters').nextElementSibling as HTMLElement;
  return Array.from(filtersList.querySelectorAll('[data-filter-name]')) as HTMLElement[];
}

/** Fires a dragStart event with a mocked dataTransfer to avoid JSDOM setDragImage errors. */
function fireDragStart(element: HTMLElement): void {
  const event = createEvent.dragStart(element);
  Object.defineProperty(event, 'dataTransfer', {
    value: {
      effectAllowed: 'move',
      dropEffect: 'move',
      setData: vi.fn(),
      getData: vi.fn(),
      setDragImage: vi.fn(),
    },
    writable: true,
  });
  fireEvent(element, event);
}

// --- Tests ---

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

  it('returns empty active list when no filter is enabled', () => {
    const onFiltersChange = vi.fn();
    render(<FilterPanel filters={mockFilters} onFiltersChange={onFiltersChange} />);
    expect(onFiltersChange).toHaveBeenLastCalledWith([]);
  });

  it('works with arbitrary backend filter ids', () => {
    const onFiltersChange = vi.fn();
    const customFilters = [
      {
        id: 'x_custom_filter',
        name: 'X Custom',
        type: 0,
        parameters: [],
      },
    ] as GenericFilterDefinition[];
    render(<FilterPanel filters={customFilters} onFiltersChange={onFiltersChange} />);
    fireEvent.click(screen.getByTestId('filter-checkbox-x_custom_filter'));
    const lastCall = onFiltersChange.mock.calls.at(-1)?.[0] as ActiveFilterState[];
    expect(lastCall[0]?.id).toBe('x_custom_filter');
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

  it('Success_CollapsingExpandedFilterKeepsItEnabled', () => {
    const onFiltersChange = vi.fn();
    render(<FilterPanel filters={mockFilters} onFiltersChange={onFiltersChange} />);

    const blurCard = screen.getByTestId('filter-checkbox-blur').closest('[data-filter-id="blur"]') as HTMLElement;
    const blurHeader = blurCard.querySelector('[class*="filterHeader"]') as HTMLElement;
    const checkbox = screen.getByTestId('filter-checkbox-blur') as HTMLInputElement;

    // Expand (auto-enables)
    fireEvent.click(blurHeader);
    expect(checkbox.checked).toBe(true);

    // Collapse — should remain enabled
    fireEvent.click(blurHeader);
    expect(checkbox.checked).toBe(true);
    const filterBody = blurCard.querySelector('[class*="filterBody"]') as HTMLElement;
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

  it('Success_SelectParameterChangesOnRadioClick', () => {
    const onFiltersChange = vi.fn();
    render(<FilterPanel filters={mockFilters} onFiltersChange={onFiltersChange} />);

    const blurCard = screen.getByTestId('filter-checkbox-blur').closest('[data-filter-id="blur"]') as HTMLElement;
    fireEvent.click(blurCard.querySelector('[class*="filterHeader"]') as HTMLElement);

    const reflectRadio = screen.getByTestId('filter-parameter-blur-mode-reflect') as HTMLInputElement;
    fireEvent.click(reflectRadio);

    const lastCall = onFiltersChange.mock.calls.at(-1)![0] as ActiveFilterState[];
    const blurFilter = lastCall.find((f) => f.id === 'blur');
    expect(blurFilter?.parameters.mode).toBe('reflect');
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
    expect(parseFloat(slider.value)).toBe(1);
    expect(slider.min).toBe('0');
    expect(slider.max).toBe('2');
    expect(slider.step).toBe('0.1');

    expect(screen.getByText('1.0')).toBeInTheDocument();
  });

  it('Success_RangeParameterUpdatesOnSliderChange', () => {
    const onFiltersChange = vi.fn();
    render(<FilterPanel filters={mockFilters} onFiltersChange={onFiltersChange} />);

    const grayscaleCard = screen.getByTestId('filter-checkbox-grayscale').closest('[data-filter-id="grayscale"]') as HTMLElement;
    fireEvent.click(grayscaleCard.querySelector('[class*="filterHeader"]') as HTMLElement);

    const slider = screen.getByTestId('filter-parameter-grayscale-intensity') as HTMLInputElement;
    fireEvent.change(slider, { target: { value: '1.5' } });

    const lastCall = onFiltersChange.mock.calls.at(-1)![0] as ActiveFilterState[];
    const grayscaleFilter = lastCall.find((f) => f.id === 'grayscale');
    expect(grayscaleFilter?.parameters.intensity).toBe('1.5');
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

  it('Success_CheckboxParameterTogglesOnChange', () => {
    const onFiltersChange = vi.fn();
    render(<FilterPanel filters={mockFilters} onFiltersChange={onFiltersChange} />);

    const grayscaleCard = screen.getByTestId('filter-checkbox-grayscale').closest('[data-filter-id="grayscale"]') as HTMLElement;
    fireEvent.click(grayscaleCard.querySelector('[class*="filterHeader"]') as HTMLElement);

    const paramCheckbox = screen.getByTestId('filter-parameter-grayscale-enabled') as HTMLInputElement;
    expect(paramCheckbox.checked).toBe(true);

    fireEvent.click(paramCheckbox);

    const lastCall = onFiltersChange.mock.calls.at(-1)![0] as ActiveFilterState[];
    const grayscaleFilter = lastCall.find((f) => f.id === 'grayscale');
    expect(grayscaleFilter?.parameters.enabled).toBe('false');
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

  it('Error_ClampsNumberInputBelowMin', async () => {
    const onFiltersChange = vi.fn();
    render(<FilterPanel filters={mockFilters} onFiltersChange={onFiltersChange} />);

    const blurCard = screen.getByTestId('filter-checkbox-blur').closest('[data-filter-id="blur"]') as HTMLElement;
    fireEvent.click(blurCard.querySelector('[class*="filterHeader"]') as HTMLElement);

    const numberInput = screen.getByTestId('filter-parameter-blur-radius') as HTMLInputElement;
    fireEvent.change(numberInput, { target: { value: '-5' } });

    await waitFor(() => {
      expect(numberInput.value).toBe('1');
    });

    await waitFor(() => expect(mockToastError).toHaveBeenCalled(), { timeout: 500 });
    expect(mockToastError).toHaveBeenCalledWith('Invalid Value', expect.stringContaining('at least 1'));
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

  // --- Drag and Drop ---

  describe('Drag and Drop', () => {
    it('Success_ReordersFiltersAfterDragAndDrop', () => {
      const onFiltersChange = vi.fn();
      render(<FilterPanel filters={mockFilters} onFiltersChange={onFiltersChange} />);

      const [firstCard, secondCard] = getFilterCards();
      expect(firstCard.getAttribute('data-filter-name')).toBe('Gaussian Blur');
      expect(secondCard.getAttribute('data-filter-name')).toBe('Grayscale');

      fireDragStart(firstCard);
      fireEvent.dragEnter(secondCard);
      fireEvent.drop(secondCard);

      const [newFirst, newSecond] = getFilterCards();
      expect(newFirst.getAttribute('data-filter-name')).toBe('Grayscale');
      expect(newSecond.getAttribute('data-filter-name')).toBe('Gaussian Blur');
    });

    it('Success_ReorderCallsOnFiltersChangeWithNewOrder', () => {
      const onFiltersChange = vi.fn();
      render(<FilterPanel filters={mockFilters} onFiltersChange={onFiltersChange} />);

      // Enable both filters first so they show up in active filters
      fireEvent.click(screen.getByTestId('filter-checkbox-blur'));
      fireEvent.click(screen.getByTestId('filter-checkbox-grayscale'));
      onFiltersChange.mockClear();

      const [firstCard, secondCard] = getFilterCards();
      fireDragStart(firstCard);
      fireEvent.dragEnter(secondCard);
      fireEvent.drop(secondCard);

      expect(onFiltersChange).toHaveBeenCalled();
      const lastCall = onFiltersChange.mock.calls.at(-1)![0] as ActiveFilterState[];
      expect(lastCall[0].id).toBe('grayscale');
      expect(lastCall[1].id).toBe('blur');
    });

    it('Success_AppliesDraggingClassToSourceCard', () => {
      render(<FilterPanel filters={mockFilters} onFiltersChange={vi.fn()} />);

      const [firstCard] = getFilterCards();
      expect(firstCard.className).not.toContain('dragging');

      fireDragStart(firstCard);

      expect(firstCard.className).toContain('dragging');
    });

    it('Success_AppliesDragOverHighlightToTargetCard', () => {
      render(<FilterPanel filters={mockFilters} onFiltersChange={vi.fn()} />);

      const [firstCard, secondCard] = getFilterCards();
      fireDragStart(firstCard);
      fireEvent.dragEnter(secondCard);

      expect(secondCard.className).toContain('dragOver');
      expect(firstCard.className).not.toContain('dragOver');
    });

    it('Success_SourceCardHasNoDragOverHighlight', () => {
      render(<FilterPanel filters={mockFilters} onFiltersChange={vi.fn()} />);

      const [firstCard] = getFilterCards();
      fireDragStart(firstCard);
      fireEvent.dragEnter(firstCard);

      // Source card should not get dragOver highlight (draggedIndex === dragOverIndex)
      expect(firstCard.className).not.toContain('dragOver');
    });

    it('Success_ClearsAllDragStateOnDragEnd', () => {
      render(<FilterPanel filters={mockFilters} onFiltersChange={vi.fn()} />);

      const [firstCard, secondCard] = getFilterCards();
      fireDragStart(firstCard);
      fireEvent.dragEnter(secondCard);

      expect(firstCard.className).toContain('dragging');
      expect(secondCard.className).toContain('dragOver');

      fireEvent.dragEnd(firstCard);

      expect(firstCard.className).not.toContain('dragging');
      expect(secondCard.className).not.toContain('dragOver');
    });

    it('Edge_NoReorderWhenDroppedOnSameCard', () => {
      const onFiltersChange = vi.fn();
      render(<FilterPanel filters={mockFilters} onFiltersChange={onFiltersChange} />);
      onFiltersChange.mockClear();

      const [firstCard] = getFilterCards();
      fireDragStart(firstCard);
      fireEvent.drop(firstCard);

      // Order unchanged
      const [newFirst, newSecond] = getFilterCards();
      expect(newFirst.getAttribute('data-filter-name')).toBe('Gaussian Blur');
      expect(newSecond.getAttribute('data-filter-name')).toBe('Grayscale');

      // No extra onFiltersChange call (localFilters did not change)
      expect(onFiltersChange).not.toHaveBeenCalled();
    });

    it('Edge_DragLeaveRemovesHighlightWhenCursorExitsCard', () => {
      render(<FilterPanel filters={mockFilters} onFiltersChange={vi.fn()} />);

      const [firstCard, secondCard] = getFilterCards();
      fireDragStart(firstCard);
      fireEvent.dragEnter(secondCard);
      expect(secondCard.className).toContain('dragOver');

      // Simulate leaving the card for an element outside it
      const leaveEvent = createEvent.dragLeave(secondCard);
      Object.defineProperty(leaveEvent, 'relatedTarget', { value: document.body, writable: true });
      fireEvent(secondCard, leaveEvent);

      expect(secondCard.className).not.toContain('dragOver');
    });

    it('Edge_DragLeaveDoesNotRemoveHighlightWhenMovingBetweenChildren', () => {
      render(<FilterPanel filters={mockFilters} onFiltersChange={vi.fn()} />);

      const [firstCard, secondCard] = getFilterCards();
      fireDragStart(firstCard);
      fireEvent.dragEnter(secondCard);
      expect(secondCard.className).toContain('dragOver');

      // Simulate leaving to a child element of the card (e.g. the header inside)
      const childEl = secondCard.querySelector('[class*="filterHeader"]') as HTMLElement;
      const leaveEvent = createEvent.dragLeave(secondCard);
      Object.defineProperty(leaveEvent, 'relatedTarget', { value: childEl, writable: true });
      fireEvent(secondCard, leaveEvent);

      // dragOver highlight should remain since relatedTarget is a child
      expect(secondCard.className).toContain('dragOver');
    });

    it('Edge_CanReorderFromSecondToFirst', () => {
      render(<FilterPanel filters={mockFilters} onFiltersChange={vi.fn()} />);

      const [firstCard, secondCard] = getFilterCards();
      fireDragStart(secondCard);
      fireEvent.dragEnter(firstCard);
      fireEvent.drop(firstCard);

      const [newFirst, newSecond] = getFilterCards();
      expect(newFirst.getAttribute('data-filter-name')).toBe('Grayscale');
      expect(newSecond.getAttribute('data-filter-name')).toBe('Gaussian Blur');
    });
  });
});
