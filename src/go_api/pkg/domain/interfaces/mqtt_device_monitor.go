package interfaces

import (
	"context"

	"github.com/jrb/cuda-learning/src/go_api/pkg/domain"
)

type MQTTDeviceMonitor interface {
	Start(ctx context.Context) error
	Stop() error
	PowerOn() error
	PowerOff() error
	Subscribe(callback func(status *domain.DeviceStatus)) func()
}
