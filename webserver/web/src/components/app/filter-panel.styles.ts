import { css } from 'lit';

export const filterPanelStyles = css`
  :host {
    display: block;
  }

  .control-section {
    margin-bottom: 24px;
  }

  .control-label {
    display: block;
    font-weight: 600;
    font-size: 13px;
    text-transform: uppercase;
    letter-spacing: 0.5px;
    color: #333;
    margin-bottom: 12px;
  }

  .hint {
    font-weight: 400;
    font-size: 11px;
    color: #999;
    text-transform: none;
    letter-spacing: 0;
  }

  .filters-list {
    display: flex;
    flex-direction: column;
    gap: 12px;
  }

  .filter-card {
    background: white;
    border: 1px solid #e0e0e0;
    border-radius: 6px;
    overflow: hidden;
    transition: all 0.2s;
  }

  .filter-card:not(.disabled) {
    cursor: move;
  }

  .filter-card.dragging {
    opacity: 0.5;
  }

  .filter-card.drag-over {
    border-color: #ffa400;
    box-shadow: 0 0 0 2px rgba(255, 164, 0, 0.1);
  }

  .filter-card.disabled {
    opacity: 0.6;
    cursor: not-allowed;
  }

  .filter-header {
    display: flex;
    align-items: center;
    gap: 10px;
    padding: 12px 16px;
    cursor: pointer;
    user-select: none;
  }

  .filter-card.disabled .filter-header {
    cursor: not-allowed;
  }

  .drag-handle-container {
    padding: 4px;
    cursor: move;
    display: inline-flex;
    align-items: center;
    justify-content: flex-start;
    user-select: none;
  }

  .drag-handle {
    color: #999;
    font-size: 12px;
    cursor: move;
    pointer-events: none;
  }

  .drag-handle.disabled {
    cursor: not-allowed;
  }

  .filter-card.disabled .drag-handle-container {
    cursor: not-allowed;
  }

  .checkbox-container {
    padding: 8px;
    margin: -8px;
    cursor: pointer;
    display: inline-flex;
    align-items: center;
    justify-content: center;
  }

  input[type='checkbox'] {
    cursor: pointer;
  }

  .filter-header label {
    flex: 1;
    cursor: pointer;
    font-weight: 500;
  }

  .badge {
    background: #ffa400;
    color: white;
    font-size: 9px;
    padding: 2px 6px;
    border-radius: 3px;
    font-weight: 600;
    margin-left: 6px;
  }

  .chevron {
    color: #666;
    font-size: 10px;
    transition: transform 0.2s;
  }

  .filter-card .chevron.expanded {
    transform: rotate(90deg);
  }

  .filter-body {
    max-height: 0;
    overflow: hidden;
    transition: max-height 0.3s ease;
  }

  .filter-body.expanded {
    max-height: 400px;
    padding: 0 16px 16px 16px;
  }

  .radio-label {
    display: block;
    font-size: 12px;
    font-weight: 600;
    color: #666;
    margin-bottom: 8px;
  }

  .radio-group {
    display: flex;
    flex-direction: column;
    gap: 8px;
  }

  .radio-option {
    display: flex;
    align-items: center;
    gap: 8px;
    padding: 8px 12px;
    border-radius: 4px;
    cursor: pointer;
    transition: background 0.2s;
  }

  .radio-option:hover {
    background: #f5f5f5;
  }

  .radio-option input[type='radio'] {
    cursor: pointer;
  }

  .radio-option span {
    font-size: 13px;
    color: #333;
  }

  .param-label {
    display: block;
    font-size: 12px;
    font-weight: 600;
    color: #666;
    margin-bottom: 8px;
  }

  .param-control {
    margin-bottom: 16px;
  }

  .number-input-container {
    display: flex;
    flex-direction: column;
    gap: 4px;
  }

  .slider-container {
    display: flex;
    align-items: center;
    gap: 12px;
  }

  input[type='range'] {
    flex: 1;
    height: 4px;
    border-radius: 2px;
    background: #e0e0e0;
    outline: none;
    -webkit-appearance: none;
  }

  input[type='range']::-webkit-slider-thumb {
    -webkit-appearance: none;
    appearance: none;
    width: 16px;
    height: 16px;
    border-radius: 50%;
    background: #ffa400;
    cursor: pointer;
  }

  input[type='range']::-moz-range-thumb {
    width: 16px;
    height: 16px;
    border-radius: 50%;
    background: #ffa400;
    cursor: pointer;
    border: none;
  }

  .slider-value {
    font-size: 12px;
    font-weight: 600;
    color: #333;
    min-width: 30px;
    text-align: right;
  }

  input[type='number'] {
    width: 100%;
    padding: 8px 16px;
    border: 1px solid #e0e0e0;
    border-radius: 4px;
    font-size: 13px;
    color: #333;
    outline: none;
    transition: border-color 0.2s;
    box-sizing: border-box;
  }

  input[type='number']:focus {
    border-color: #ffa400;
  }

  input[type='number']::-webkit-inner-spin-button,
  input[type='number']::-webkit-outer-spin-button {
    opacity: 1;
    height: 20px;
    cursor: pointer;
  }

  input[type='number']:disabled {
    opacity: 0.5;
    cursor: not-allowed;
  }

  .checkbox-option {
    display: flex;
    align-items: center;
    gap: 8px;
    padding: 8px 12px;
    border-radius: 4px;
    cursor: pointer;
    transition: background 0.2s;
  }

  .checkbox-option:hover {
    background: #f5f5f5;
  }

  .checkbox-option input[type='checkbox'] {
    cursor: pointer;
  }

  .checkbox-option span {
    font-size: 13px;
    color: #333;
  }

  select {
    width: 100%;
    padding: 8px 12px;
    border: 1px solid #e0e0e0;
    border-radius: 4px;
    font-size: 13px;
    color: #333;
    background: white;
    outline: none;
    cursor: pointer;
    transition: border-color 0.2s;
  }

  select:focus {
    border-color: #ffa400;
  }
`;
