package mqtt

import (
	"encoding/json"
	"fmt"
	"time"

	mqtt "github.com/eclipse/paho.mqtt.golang"
	"github.com/jrb/cuda-learning/webserver/pkg/config"
)

type Client struct {
	client mqtt.Client
	config config.MQTTConfig
}

type SensorData struct {
	Time   string `json:"Time"`
	ENERGY struct {
		Power   float64 `json:"Power"`
		Voltage float64 `json:"Voltage"`
	} `json:"ENERGY"`
}

type Info1Data struct {
	Info1 struct {
		Module  string `json:"Module"`
		Version string `json:"Version"`
	} `json:"Info1"`
}

type Info2Data struct {
	Info2 struct {
		Hostname  string `json:"Hostname"`
		IPAddress string `json:"IPAddress"`
	} `json:"Info2"`
}

func NewClient(cfg config.MQTTConfig) (*Client, error) {
	brokerURL := fmt.Sprintf("tcp://%s:%d", cfg.Broker, cfg.Port)

	opts := mqtt.NewClientOptions()
	opts.AddBroker(brokerURL)
	opts.SetClientID(cfg.ClientID)
	opts.SetAutoReconnect(true)
	opts.SetConnectRetry(true)
	opts.SetConnectRetryInterval(5 * time.Second)
	opts.SetKeepAlive(30 * time.Second)
	opts.SetPingTimeout(10 * time.Second)

	client := mqtt.NewClient(opts)

	if token := client.Connect(); token.Wait() && token.Error() != nil {
		return nil, fmt.Errorf("failed to connect to MQTT broker: %w", token.Error())
	}

	return &Client{
		client: client,
		config: cfg,
	}, nil
}

func (c *Client) PublishPowerCommand(on bool) error {
	topic := fmt.Sprintf("cmnd/%s/POWER", c.config.Topic)
	payload := "OFF"
	if on {
		payload = "ON"
	}

	token := c.client.Publish(topic, 0, false, payload)
	token.Wait()

	if token.Error() != nil {
		return fmt.Errorf("failed to publish power command: %w", token.Error())
	}

	return nil
}

func (c *Client) SubscribeToSensor(callback func(power float64, timestamp string) error) error {
	topic := fmt.Sprintf("tele/%s/SENSOR", c.config.Topic)

	messageHandler := func(client mqtt.Client, msg mqtt.Message) {
		var sensorData SensorData
		if err := json.Unmarshal(msg.Payload(), &sensorData); err != nil {
			return
		}

		if err := callback(sensorData.ENERGY.Power, sensorData.Time); err != nil {
			return
		}
	}

	token := c.client.Subscribe(topic, 0, messageHandler)
	token.Wait()

	if token.Error() != nil {
		return fmt.Errorf("failed to subscribe to sensor topic: %w", token.Error())
	}

	return nil
}

func (c *Client) SubscribeToSensorWithRaw(callback func(data SensorData) error) error {
	topic := fmt.Sprintf("tele/%s/SENSOR", c.config.Topic)

	messageHandler := func(client mqtt.Client, msg mqtt.Message) {
		var sensorData SensorData
		if err := json.Unmarshal(msg.Payload(), &sensorData); err != nil {
			return
		}

		if err := callback(sensorData); err != nil {
			return
		}
	}

	token := c.client.Subscribe(topic, 0, messageHandler)
	token.Wait()

	if token.Error() != nil {
		return fmt.Errorf("failed to subscribe to sensor topic: %w", token.Error())
	}

	return nil
}

func (c *Client) SubscribeToInfo1(callback func(data Info1Data) error) error {
	topic := fmt.Sprintf("tele/%s/INFO1", c.config.Topic)

	messageHandler := func(client mqtt.Client, msg mqtt.Message) {
		var info1Data Info1Data
		if err := json.Unmarshal(msg.Payload(), &info1Data); err != nil {
			return
		}

		if err := callback(info1Data); err != nil {
			return
		}
	}

	token := c.client.Subscribe(topic, 0, messageHandler)
	token.Wait()

	if token.Error() != nil {
		return fmt.Errorf("failed to subscribe to INFO1 topic: %w", token.Error())
	}

	return nil
}

func (c *Client) SubscribeToInfo2(callback func(data Info2Data) error) error {
	topic := fmt.Sprintf("tele/%s/INFO2", c.config.Topic)

	messageHandler := func(client mqtt.Client, msg mqtt.Message) {
		var info2Data Info2Data
		if err := json.Unmarshal(msg.Payload(), &info2Data); err != nil {
			return
		}

		if err := callback(info2Data); err != nil {
			return
		}
	}

	token := c.client.Subscribe(topic, 0, messageHandler)
	token.Wait()

	if token.Error() != nil {
		return fmt.Errorf("failed to subscribe to INFO2 topic: %w", token.Error())
	}

	return nil
}

func (c *Client) SubscribeToLWT(callback func(status string) error) error {
	topic := fmt.Sprintf("tele/%s/LWT", c.config.Topic)

	messageHandler := func(client mqtt.Client, msg mqtt.Message) {
		status := string(msg.Payload())
		if err := callback(status); err != nil {
			return
		}
	}

	token := c.client.Subscribe(topic, 0, messageHandler)
	token.Wait()

	if token.Error() != nil {
		return fmt.Errorf("failed to subscribe to LWT topic: %w", token.Error())
	}

	return nil
}

func (c *Client) RequestInfo1() error {
	topic := fmt.Sprintf("cmnd/%s/INFO1", c.config.Topic)
	token := c.client.Publish(topic, 0, false, "")
	token.Wait()

	if token.Error() != nil {
		return fmt.Errorf("failed to request INFO1: %w", token.Error())
	}

	return nil
}

func (c *Client) RequestInfo2() error {
	topic := fmt.Sprintf("cmnd/%s/INFO2", c.config.Topic)
	token := c.client.Publish(topic, 0, false, "")
	token.Wait()

	if token.Error() != nil {
		return fmt.Errorf("failed to request INFO2: %w", token.Error())
	}

	return nil
}

func (c *Client) RequestSensorStatus() error {
	topic := fmt.Sprintf("cmnd/%s/STATUS", c.config.Topic)
	token := c.client.Publish(topic, 0, false, "8")
	token.Wait()

	if token.Error() != nil {
		return fmt.Errorf("failed to request sensor status: %w", token.Error())
	}

	return nil
}

func (c *Client) RestartDevice() error {
	topic := fmt.Sprintf("cmnd/%s/RESTART", c.config.Topic)
	token := c.client.Publish(topic, 0, false, "1")
	token.Wait()

	if token.Error() != nil {
		return fmt.Errorf("failed to restart device: %w", token.Error())
	}

	return nil
}

func (c *Client) Disconnect() {
	c.client.Disconnect(250)
}
