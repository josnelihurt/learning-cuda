package container

import "context"

type useCase[Input any, Output any] interface {
	Execute(ctx context.Context, input Input) (Output, error)
}
