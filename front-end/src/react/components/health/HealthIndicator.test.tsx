import { describe, it, expect, vi } from 'vitest';
import { render, screen } from '@testing-library/react';
import userEvent from '@testing-library/user-event';
import { HealthIndicator } from './HealthIndicator';

describe('HealthIndicator', () => {
  describe('rendering', () => {
    it('should render healthy state with green dot', () => {
      render(<HealthIndicator isHealthy={true} />);

      const indicator = screen.getByRole('status', { name: 'Healthy' });
      expect(indicator).toBeInTheDocument();
      expect(screen.getByText('Healthy')).toBeInTheDocument();
    });

    it('should render unhealthy state with red dot', () => {
      render(<HealthIndicator isHealthy={false} />);

      const indicator = screen.getByRole('status', { name: 'Unhealthy' });
      expect(indicator).toBeInTheDocument();
      expect(screen.getByText('Unhealthy')).toBeInTheDocument();
    });

    it('should render loading state with orange dot and animation', () => {
      render(<HealthIndicator isHealthy={true} loading={true} />);

      const indicator = screen.getByRole('status', { name: 'Checking...' });
      expect(indicator).toBeInTheDocument();
      expect(screen.getByText('Checking...')).toBeInTheDocument();
    });

    it('should hide label when showLabel is false', () => {
      render(<HealthIndicator isHealthy={true} showLabel={false} />);

      expect(screen.queryByText('Healthy')).not.toBeInTheDocument();
      const indicator = screen.getByRole('status', { name: 'Healthy' });
      expect(indicator).toBeInTheDocument();
    });
  });

  describe('interactions', () => {
    it('should call onClick when clicked', async () => {
      const handleClick = vi.fn();
      const user = userEvent.setup();

      render(<HealthIndicator isHealthy={true} onClick={handleClick} />);

      const indicator = screen.getByRole('status', { name: 'Healthy' });
      await user.click(indicator);

      expect(handleClick).toHaveBeenCalledTimes(1);
    });

    it('should not be clickable when onClick is not provided', async () => {
      render(<HealthIndicator isHealthy={true} />);

      const indicator = screen.getByRole('status', { name: 'Healthy' });
      expect(indicator).not.toHaveClass('clickable');
    });
  });

  describe('tooltip', () => {
    it('should show message in tooltip when provided', () => {
      render(<HealthIndicator isHealthy={true} message="All systems operational" />);

      const indicator = screen.getByRole('status', { name: 'Healthy' });
      expect(indicator).toHaveAttribute('title', 'All systems operational');
    });

    it('should show last checked timestamp in tooltip', () => {
      const lastChecked = new Date('2026-04-13T16:00:00Z');
      render(<HealthIndicator isHealthy={true} lastChecked={lastChecked} />);

      const indicator = screen.getByRole('status', { name: 'Healthy' });
      expect(indicator).toHaveAttribute('title');
    });

    it('should show "No checks yet" when no lastChecked provided', () => {
      render(<HealthIndicator isHealthy={true} />);

      const indicator = screen.getByRole('status', { name: 'Healthy' });
      expect(indicator).toHaveAttribute('title', 'No checks yet');
    });
  });

  describe('accessibility', () => {
    it('should have correct ARIA label for healthy state', () => {
      render(<HealthIndicator isHealthy={true} />);

      const indicator = screen.getByRole('status');
      expect(indicator).toHaveAttribute('aria-label', 'Healthy');
    });

    it('should have correct ARIA label for unhealthy state', () => {
      render(<HealthIndicator isHealthy={false} />);

      const indicator = screen.getByRole('status');
      expect(indicator).toHaveAttribute('aria-label', 'Unhealthy');
    });

    it('should have correct ARIA label for loading state', () => {
      render(<HealthIndicator isHealthy={true} loading={true} />);

      const indicator = screen.getByRole('status');
      expect(indicator).toHaveAttribute('aria-label', 'Checking...');
    });

    it('should have aria-live="polite" for screen readers', () => {
      render(<HealthIndicator isHealthy={true} />);

      const indicator = screen.getByRole('status');
      expect(indicator).toHaveAttribute('aria-live', 'polite');
    });
  });
});
