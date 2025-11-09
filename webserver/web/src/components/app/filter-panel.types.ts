import type { FilterDefinition, FilterParameter } from '../../gen/common_pb';

export interface Filter {
  id: string;
  name: string;
  enabled: boolean;
  expanded: boolean;
  parameters: FilterParameter[];
  parameterValues: Record<string, string>;
}

export interface FilterOption {
  value: string;
  label: string;
}

export function createFilterFromDefinition(def: FilterDefinition): Filter {
  const defaultValues: Record<string, string> = {};

  def.parameters.forEach((param) => {
    defaultValues[param.id] = param.defaultValue;
  });

  return {
    id: def.id,
    name: def.name,
    enabled: false,
    expanded: false,
    parameters: def.parameters,
    parameterValues: defaultValues,
  };
}

export function formatParameterLabel(param: FilterParameter, value: string): string {
  if (param.type === 'select') {
    const optionLabels: Record<string, string> = {
      bt601: 'ITU-R BT.601 (SDTV)',
      bt709: 'ITU-R BT.709 (HDTV)',
      average: 'Average',
      lightness: 'Lightness',
      luminosity: 'Luminosity',
    };
    return optionLabels[value] || value;
  }
  return value;
}
