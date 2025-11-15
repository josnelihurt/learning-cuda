import type { FilterDefinition } from '../../gen/common_pb';
import type {
  GenericFilterDefinition,
  GenericFilterParameter,
  GenericFilterParameterType,
  GenericFilterParameterOption,
} from '../../gen/image_processor_service_pb';

export type FilterParameterType = 'select' | 'range' | 'number' | 'checkbox' | 'text';

export interface FilterParameterConfig {
  id: string;
  name: string;
  type: FilterParameterType;
  options: FilterOption[];
  defaultValue?: string;
  min?: number;
  max?: number;
  step?: number;
  display?: string;
}

export interface Filter {
  id: string;
  name: string;
  enabled: boolean;
  expanded: boolean;
  parameters: FilterParameterConfig[];
  parameterValues: Record<string, string>;
}

export interface FilterOption {
  value: string;
  label: string;
}

export interface ActiveFilterState {
  id: string;
  parameters: Record<string, string>;
}

export function createFilterFromDefinition(def: FilterDefinition): Filter {
  const parameters = def.parameters.map((param) => ({
    id: param.id,
    name: param.name,
    type: mapLegacyParameterType(param.type),
    options: buildLegacyOptions(param.options),
    defaultValue: param.defaultValue || '',
  }));

  return buildFilter(def.id, def.name, parameters);
}

export function createFilterFromGenericDefinition(def: GenericFilterDefinition): Filter {
  const parameters = def.parameters.map((param) => mapGenericParameter(param));
  return buildFilter(def.id, def.name, parameters);
}

export function formatParameterLabel(param: FilterParameterConfig, value: string): string {
  const option = param.options.find((opt) => opt.value === value);
  if (option) {
    return option.label || option.value;
  }

  return legacyLabelMap[value] || value;
}

function buildFilter(id: string, name: string, parameters: FilterParameterConfig[]): Filter {
  const defaultValues: Record<string, string> = {};
  parameters.forEach((param) => {
    defaultValues[param.id] = param.defaultValue ?? '';
  });

  return {
    id,
    name,
    enabled: false,
    expanded: false,
    parameters,
    parameterValues: defaultValues,
  };
}

function mapLegacyParameterType(type: string): FilterParameterType {
  switch (type) {
    case 'select':
      return 'select';
    case 'range':
    case 'slider':
      return 'range';
    case 'number':
      return 'number';
    case 'checkbox':
      return 'checkbox';
    default:
      return 'text';
  }
}

function buildLegacyOptions(options: string[] = []): FilterOption[] {
  if (!options || options.length === 0) {
    return [];
  }
  return options.map((value) => ({
    value,
    label: legacyLabelMap[value] || value,
  }));
}

function mapGenericParameter(param: GenericFilterParameter): FilterParameterConfig {
  const metadata = param.metadata ?? {};
  return {
    id: param.id,
    name: param.name,
    type: mapGenericParameterType(param.type),
    options: mapGenericOptions(param.options),
    defaultValue: param.defaultValue || '',
    min: metadata.min !== undefined ? parseFloat(metadata.min) : undefined,
    max: metadata.max !== undefined ? parseFloat(metadata.max) : undefined,
    step: metadata.step !== undefined ? parseFloat(metadata.step) : undefined,
    display: metadata.display,
  };
}

function mapGenericParameterType(type: GenericFilterParameterType): FilterParameterType {
  switch (type) {
    case GenericFilterParameterType.GENERIC_FILTER_PARAMETER_TYPE_SELECT:
      return 'select';
    case GenericFilterParameterType.GENERIC_FILTER_PARAMETER_TYPE_RANGE:
      return 'range';
    case GenericFilterParameterType.GENERIC_FILTER_PARAMETER_TYPE_NUMBER:
      return 'number';
    case GenericFilterParameterType.GENERIC_FILTER_PARAMETER_TYPE_CHECKBOX:
      return 'checkbox';
    case GenericFilterParameterType.GENERIC_FILTER_PARAMETER_TYPE_TEXT:
      return 'text';
    default:
      return 'text';
  }
}

function mapGenericOptions(options: GenericFilterParameterOption[]): FilterOption[] {
  if (!options || options.length === 0) {
    return [];
  }

  return options.map((option) => {
    const value = option.value || option.label || '';
    return {
      value,
      label: option.label || legacyLabelMap[value] || value,
    };
  });
}

const legacyLabelMap: Record<string, string> = {
  bt601: 'ITU-R BT.601 (SDTV)',
  bt709: 'ITU-R BT.709 (HDTV)',
  average: 'Average',
  lightness: 'Lightness',
  luminosity: 'Luminosity',
  clamp: 'Clamp',
  reflect: 'Reflect',
  wrap: 'Wrap',
};
