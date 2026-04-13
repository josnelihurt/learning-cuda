package application

import (
	"context"
	"errors"
	"testing"

	"github.com/jrb/cuda-learning/webserver/pkg/domain"
	"github.com/stretchr/testify/assert"
	"github.com/stretchr/testify/mock"
)

type MockVideoRepository struct {
	mock.Mock
}

func (m *MockVideoRepository) List(ctx context.Context) ([]domain.Video, error) {
	args := m.Called(ctx)
	if args.Get(0) == nil {
		return nil, args.Error(1)
	}
	return args.Get(0).([]domain.Video), args.Error(1)
}

func (m *MockVideoRepository) GetByID(ctx context.Context, id string) (*domain.Video, error) {
	args := m.Called(ctx, id)
	if args.Get(0) == nil {
		return nil, args.Error(1)
	}
	return args.Get(0).(*domain.Video), args.Error(1)
}

func (m *MockVideoRepository) Save(ctx context.Context, video *domain.Video) error {
	args := m.Called(ctx, video)
	return args.Error(0)
}

func TestListVideosUseCase_Execute(t *testing.T) {
	tests := []struct {
		name          string
		setup         func(*MockVideoRepository)
		expectedCount int
		expectError   bool
	}{
		{
			name: "returns videos from repository",
			setup: func(repo *MockVideoRepository) {
				videos := []domain.Video{
					{ID: "video1", DisplayName: "Video 1"},
					{ID: "video2", DisplayName: "Video 2"},
				}
				repo.On("List", mock.Anything).Return(videos, nil)
			},
			expectedCount: 2,
			expectError:   false,
		},
		{
			name: "returns empty list when no videos",
			setup: func(repo *MockVideoRepository) {
				repo.On("List", mock.Anything).Return([]domain.Video{}, nil)
			},
			expectedCount: 0,
			expectError:   false,
		},
		{
			name: "returns error when repository fails",
			setup: func(repo *MockVideoRepository) {
				repo.On("List", mock.Anything).Return(nil, errors.New("repository error"))
			},
			expectError: true,
		},
	}

	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			repo := new(MockVideoRepository)
			tt.setup(repo)
			sut := NewListVideosUseCase(repo)

			result, err := sut.Execute(context.Background())

			if tt.expectError {
				assert.Error(t, err)
				assert.Nil(t, result)
			} else {
				assert.NoError(t, err)
				assert.Len(t, result, tt.expectedCount)
			}
			repo.AssertExpectations(t)
		})
	}
}
