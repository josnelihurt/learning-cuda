import { describe, it, expect, vi } from 'vitest';
import { render, screen, within } from '@testing-library/react';
import userEvent from '@testing-library/user-event';
import { HealthPanel } from './HealthPanel';

describe('HealthPanel', () => {
  const defaultProps = {
    isHealthy: true,
    loading: false,
    error: null,
    lastChecked: new Date('2026-04-13T16:00:00Z'),
  };

  describe('normal mode rendering', () => {
    it('should render healthy state with checkmark icon', () => {
      render(<HealthPanel {...defaultProps} />);

      expect(screen.getByText('Backend Health')).toBeInTheDocument();
      expect(screen.getByText('Healthy')).toBeInTheDocument();
      expect(screen.getByText('✓')).toBeInTheDocument();
    });

    it('should render unhealthy state with warning icon', () => {
      render(<HealthPanel {...defaultProps} isHealthy={false} />);

      expect(screen.getByText('Unhealthy')).toBeInTheDocument();
      expect(screen.getByText('⚠')).toBeInTheDocument();
    });

    it('should render loading state with spinner icon', () => {
      render(<HealthPanel {...defaultProps} loading={true} />);

      expect(screen.getByText('Checking...')).toBeInTheDocument();
      expect(screen.getByText('⏳')).toBeInTheDocument();
    });

    it('should display status message when provided', () => {
      render(<HealthPanel {...defaultProps} message="All systems operational" />);

      expect(screen.getByText('All systems operational')).toBeInTheDocument();
    });

    it('should display last checked timestamp', () => {
      render(<HealthPanel {...defaultProps} />);

      expect(screen.getByText(/Last checked:/)).toBeInTheDocument();
    });

    it('should hide details during loading', () => {
      render(<HealthPanel {...defaultProps} loading={true} message="Should not show" />);

      expect(screen.queryByText('Should not show')).not.toBeInTheDocument();
    });

    it('should display error details when error exists', () => {
      const error = { message: 'Connection refused', code: '14' };
      render(<HealthPanel {...defaultProps} error={error} />);

      expect(screen.getByText('Error Details:')).toBeInTheDocument();
      expect(screen.getByText('Connection refused')).toBeInTheDocument();
      expect(screen.getByText('Code: 14')).toBeInTheDocument();
    });
  });

  describe('compact mode', () => {
    it('should render single row in compact mode', () => {
      render(<HealthPanel {...defaultProps} compact={true} />);

      const panel = screen.getByRole('button');
      expect(panel).toBeInTheDocument();
      expect(screen.getByText('Healthy')).toBeInTheDocument();
      expect(screen.getByText(/\d+ min ago/)).toBeInTheDocument();
    });

    it('should not show details when collapsed', () => {
      render(<HealthPanel {...defaultProps} compact={true} message="Hidden details" />);

      expect(screen.queryByText('Hidden details')).not.toBeInTheDocument();
    });

    it('should expand on click and show details', async () => {
      const user = userEvent.setup();
      render(<HealthPanel {...defaultProps} compact={true} message="Visible details" />);

      const panel = screen.getByRole('button');
      expect(screen.queryByText('Visible details')).not.toBeInTheDocument();

      await user.click(panel);

      expect(screen.getByText('Visible details')).toBeInTheDocument();
    });

    it('should collapse when clicking expanded panel', async () => {
      const user = userEvent.setup();
      render(<HealthPanel {...defaultProps} compact={true} message="Test message" />);

      const panel = screen.getByRole('button');

      // Expand
      await user.click(panel);
      expect(screen.getByText('Test message')).toBeInTheDocument();

      // Collapse
      await user.click(panel);
      expect(screen.queryByText('Test message')).not.toBeInTheDocument();
    });

    it('should handle keyboard navigation', async () => {
      const user = userEvent.setup();
      render(<HealthPanel {...defaultProps} compact={true} message="Keyboard test" />);

      const panel = screen.getByRole('button');
      expect(screen.queryByText('Keyboard test')).not.toBeInTheDocument();

      // Enter key should toggle
      panel.focus();
      await user.keyboard('{Enter}');
      expect(screen.getByText('Keyboard test')).toBeInTheDocument();

      // Space key should toggle
      await user.keyboard(' ');
      expect(screen.queryByText('Keyboard test')).not.toBeInTheDocument();
    });

    it('should show error details in expanded compact mode', async () => {
      const user = userEvent.setup();
      const error = { message: 'Network error', code: '3' };
      render(<HealthPanel {...defaultProps} compact={true} isHealthy={false} error={error} />);

      const panel = screen.getByRole('button');
      await user.click(panel);

      expect(screen.getByText('Error:')).toBeInTheDocument();
      expect(screen.getByText('Network error')).toBeInTheDocument();
      // Error code is only shown in normal mode, not compact
    });
  });

  describe('timestamp formatting', () => {
    it('should format timestamp for today', () => {
      const now = new Date();
      render(<HealthPanel {...defaultProps} lastChecked={now} />);

      // Should show time format like "4:00 PM"
      expect(screen.getByText(/Last checked:/)).toBeInTheDocument();
    });

    it('should show "Never" when no lastChecked', () => {
      render(<HealthPanel {...defaultProps} lastChecked={undefined} />);

      expect(screen.getByText('Last checked: Never')).toBeInTheDocument();
    });
  });

  describe('accessibility', () => {
    it('should have correct ARIA attributes in compact mode', () => {
      render(<HealthPanel {...defaultProps} compact={true} />);

      const panel = screen.getByRole('button');
      expect(panel).toHaveAttribute('tabIndex', '0');
      expect(panel).toHaveAttribute('aria-expanded', 'false');
    });

    it('should update aria-expanded when toggled', async () => {
      const user = userEvent.setup();
      render(<HealthPanel {...defaultProps} compact={true} />);

      const panel = screen.getByRole('button');
      expect(panel).toHaveAttribute('aria-expanded', 'false');

      await user.click(panel);
      expect(panel).toHaveAttribute('aria-expanded', 'true');
    });
  });

  describe('styling', () => {
    it('should apply healthy styling', () => {
      const { container } = render(<HealthPanel {...defaultProps} />);

      const panel = container.firstChild as HTMLElement;
      expect(panel.className).toContain('healthy');
      expect(panel.className).not.toContain('unhealthy');
    });

    it('should apply unhealthy styling', () => {
      const { container } = render(<HealthPanel {...defaultProps} isHealthy={false} />);

      const panel = container.firstChild as HTMLElement;
      expect(panel.className).toContain('unhealthy');
      // Don't check for absence of 'healthy' since 'unhealthy' contains 'healthy' as a substring
    });

    it('should apply compact class in compact mode', () => {
      const { container } = render(<HealthPanel {...defaultProps} compact={true} />);

      const panel = container.firstChild as HTMLElement;
      expect(panel.className).toContain('compact');
    });
  });
});
