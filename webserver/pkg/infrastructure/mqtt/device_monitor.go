package mqtt

import (
	"context"
	"fmt"
	"sync"
	"time"

	"github.com/jrb/cuda-learning/webserver/pkg/config"
	"github.com/jrb/cuda-learning/webserver/pkg/domain"
)

type DeviceMonitor struct {
	client        *Client
	config        config.MQTTConfig
	status        *domain.DeviceStatus
	mu            sync.RWMutex
	subscribers   []func(*domain.DeviceStatus)
	subscribersMu sync.RWMutex
	ctx           context.Context
	cancel        context.CancelFunc
	started       bool
	startedMu     sync.RWMutex
}

func NewDeviceMonitor(cfg config.MQTTConfig) (*DeviceMonitor, error) {
	client, err := NewClient(cfg)
	if err != nil {
		return nil, fmt.Errorf("failed to create MQTT client: %w", err)
	}

	ctx, cancel := context.WithCancel(context.Background())

	return &DeviceMonitor{
		client:      client,
		config:      cfg,
		status:      domain.NewDeviceStatus(),
		subscribers: make([]func(*domain.DeviceStatus), 0),
		ctx:         ctx,
		cancel:      cancel,
	}, nil
}

func (dm *DeviceMonitor) Start(ctx context.Context) error {
	dm.startedMu.Lock()
	if dm.started {
		dm.startedMu.Unlock()
		return fmt.Errorf("monitor already started")
	}
	dm.started = true
	dm.startedMu.Unlock()

	dm.ctx, dm.cancel = context.WithCancel(ctx)

	sensorChan := make(chan SensorData, 10)
	info1Chan := make(chan Info1Data, 10)
	info2Chan := make(chan Info2Data, 10)
	lwtChan := make(chan string, 10)

	if err := dm.client.SubscribeToSensorWithRaw(func(data SensorData) error {
		select {
		case sensorChan <- data:
		default:
		}
		return nil
	}); err != nil {
		return fmt.Errorf("failed to subscribe to sensor: %w", err)
	}

	if err := dm.client.SubscribeToInfo1(func(data Info1Data) error {
		select {
		case info1Chan <- data:
		default:
		}
		return nil
	}); err != nil {
		return fmt.Errorf("failed to subscribe to INFO1: %w", err)
	}

	if err := dm.client.SubscribeToInfo2(func(data Info2Data) error {
		select {
		case info2Chan <- data:
		default:
		}
		return nil
	}); err != nil {
		return fmt.Errorf("failed to subscribe to INFO2: %w", err)
	}

	if err := dm.client.SubscribeToLWT(func(status string) error {
		select {
		case lwtChan <- status:
		default:
		}
		return nil
	}); err != nil {
		return fmt.Errorf("failed to subscribe to LWT: %w", err)
	}

	if err := dm.client.RestartDevice(); err != nil {
		return fmt.Errorf("failed to send restart command: %w", err)
	}

	time.Sleep(3 * time.Second)

	go dm.monitorLoop(sensorChan, info1Chan, info2Chan, lwtChan)

	return nil
}

func (dm *DeviceMonitor) monitorLoop(sensorChan <-chan SensorData, info1Chan <-chan Info1Data, info2Chan <-chan Info2Data, lwtChan <-chan string) {
	for {
		select {
		case <-dm.ctx.Done():
			return
		case sensorData := <-sensorChan:
			dm.handleSensorData(sensorData)
		case info1Data := <-info1Chan:
			dm.handleInfo1Data(info1Data)
		case info2Data := <-info2Chan:
			dm.handleInfo2Data(info2Data)
		case lwtStatus := <-lwtChan:
			dm.handleLWTStatus(lwtStatus)
		}
	}
}

func (dm *DeviceMonitor) handleSensorData(data SensorData) {
	timestamp, err := time.Parse("2006-01-02T15:04:05", data.Time)
	if err != nil {
		timestamp = time.Now()
	}

	dm.mu.Lock()
	dm.status.UpdatePower(data.ENERGY.Power, timestamp)
	dm.status.UpdateVoltage(data.ENERGY.Voltage)
	statusCopy := dm.status
	dm.mu.Unlock()

	dm.notify(statusCopy)
}

func (dm *DeviceMonitor) handleInfo1Data(data Info1Data) {
	dm.mu.Lock()
	dm.status.UpdateInfo1(data.Info1.Version, data.Info1.Module)
	statusCopy := dm.status
	dm.mu.Unlock()

	dm.notify(statusCopy)
}

func (dm *DeviceMonitor) handleInfo2Data(data Info2Data) {
	dm.mu.Lock()
	dm.status.UpdateInfo2(data.Info2.Hostname, data.Info2.IPAddress)
	statusCopy := dm.status
	dm.mu.Unlock()

	dm.notify(statusCopy)
}

func (dm *DeviceMonitor) handleLWTStatus(status string) {
	dm.mu.Lock()
	dm.status.UpdateLWTStatus(status)
	statusCopy := dm.status
	dm.mu.Unlock()

	dm.notify(statusCopy)
}

func (dm *DeviceMonitor) notify(status *domain.DeviceStatus) {
	dm.subscribersMu.RLock()
	subscribers := make([]func(*domain.DeviceStatus), len(dm.subscribers))
	copy(subscribers, dm.subscribers)
	dm.subscribersMu.RUnlock()

	for _, callback := range subscribers {
		callback(status)
	}
}

func (dm *DeviceMonitor) PowerOn() error {
	return dm.client.PublishPowerCommand(true)
}

func (dm *DeviceMonitor) PowerOff() error {
	return dm.client.PublishPowerCommand(false)
}

func (dm *DeviceMonitor) Subscribe(callback func(*domain.DeviceStatus)) func() {
	statusCopy := dm.status
	callback(statusCopy)
	dm.subscribersMu.Lock()
	dm.subscribers = append(dm.subscribers, callback)
	index := len(dm.subscribers) - 1
	dm.subscribersMu.Unlock()

	return func() {
		dm.subscribersMu.Lock()
		defer dm.subscribersMu.Unlock()
		if index < len(dm.subscribers) {
			dm.subscribers = append(dm.subscribers[:index], dm.subscribers[index+1:]...)
		}
	}
}

func (dm *DeviceMonitor) Stop() error {
	dm.startedMu.Lock()
	if !dm.started {
		dm.startedMu.Unlock()
		return nil
	}
	dm.started = false
	dm.startedMu.Unlock()

	dm.cancel()
	dm.client.Disconnect()
	return nil
}
