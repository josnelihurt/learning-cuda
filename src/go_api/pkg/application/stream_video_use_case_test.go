package application

import (
	"context"
	"errors"
	"testing"
	"time"

	pb "github.com/jrb/cuda-learning/proto/gen"
	"github.com/jrb/cuda-learning/src/go_api/pkg/domain"
	"github.com/stretchr/testify/assert"
	"github.com/stretchr/testify/mock"
	"github.com/stretchr/testify/require"
	"google.golang.org/protobuf/proto"
)

type MockStreamVideoRepository struct {
	mock.Mock
}

func (m *MockStreamVideoRepository) List(ctx context.Context) ([]domain.Video, error) {
	args := m.Called(ctx)
	if args.Get(0) == nil {
		return nil, args.Error(1)
	}

	return args.Get(0).([]domain.Video), args.Error(1)
}

func (m *MockStreamVideoRepository) GetByID(ctx context.Context, id string) (*domain.Video, error) {
	args := m.Called(ctx, id)
	if args.Get(0) == nil {
		return nil, args.Error(1)
	}

	return args.Get(0).(*domain.Video), args.Error(1)
}

func (m *MockStreamVideoRepository) Save(ctx context.Context, video *domain.Video) error {
	args := m.Called(ctx, video)
	return args.Error(0)
}

type MockStreamVideoPlayer struct {
	mock.Mock
}

func (m *MockStreamVideoPlayer) Play(
	ctx context.Context,
	frameCallback func(*domain.Image, int, time.Duration) error,
) error {
	args := m.Called(ctx, frameCallback)
	return args.Error(0)
}

type MockStreamVideoPeer struct {
	mock.Mock
}

func (m *MockStreamVideoPeer) Connect(ctx context.Context) error {
	args := m.Called(ctx)
	return args.Error(0)
}

func (m *MockStreamVideoPeer) Send(payload []byte) error {
	args := m.Called(payload)
	return args.Error(0)
}

func (m *MockStreamVideoPeer) Close() error {
	args := m.Called()
	return args.Error(0)
}

func (m *MockStreamVideoPeer) Label() string {
	args := m.Called()
	return args.String(0)
}

func TestStreamVideoUseCase_Start(t *testing.T) {
	tests := []struct {
		name         string
		request      *pb.StartVideoPlaybackRequest
		setup        func(*MockStreamVideoRepository, *MockStreamVideoPlayer, *MockStreamVideoPeer, chan *pb.ProcessImageRequest, chan struct{})
		assertResult func(t *testing.T, response *pb.StartVideoPlaybackResponse, err error, sent chan *pb.ProcessImageRequest, closed chan struct{})
	}{
		{
			name: "Success_StreamsFramesOverPeer",
			request: &pb.StartVideoPlaybackRequest{
				VideoId:       "sample",
				SessionId:     "browser-session",
				Filters:       []pb.FilterType{pb.FilterType_FILTER_TYPE_GRAYSCALE},
				Accelerator:   pb.AcceleratorType_ACCELERATOR_TYPE_CPU,
				GrayscaleType: pb.GrayscaleType_GRAYSCALE_TYPE_BT709,
				TraceContext:  &pb.TraceContext{Traceparent: "trace-1"},
				ApiVersion:    "v1",
			},
			setup: func(
				repo *MockStreamVideoRepository,
				player *MockStreamVideoPlayer,
				peer *MockStreamVideoPeer,
				sent chan *pb.ProcessImageRequest,
				closed chan struct{},
			) {
				// Arrange
				repo.On("GetByID", mock.Anything, "sample").Return(&domain.Video{
					ID:   "sample",
					Path: "/tmp/sample.mp4",
				}, nil)
				peer.On("Connect", mock.Anything).Return(nil).Once()
				peer.On("Label").Return("go-video-browser-session").Maybe()
				peer.On("Send", mock.Anything).Run(func(args mock.Arguments) {
					var frameRequest pb.ProcessImageRequest
					err := proto.Unmarshal(args.Get(0).([]byte), &frameRequest)
					require.NoError(t, err)
					sent <- &frameRequest
				}).Return(nil).Once()
				peer.On("Close").Run(func(mock.Arguments) {
					close(closed)
				}).Return(nil).Once()
				player.On("Play", mock.Anything, mock.Anything).Run(func(args mock.Arguments) {
					frameCallback := args.Get(1).(func(*domain.Image, int, time.Duration) error)
					err := frameCallback(&domain.Image{
						Data:   []byte{1, 2, 3, 4, 5, 6},
						Width:  1,
						Height: 2,
					}, 0, 0)
					require.NoError(t, err)
				}).Return(nil).Once()
			},
			assertResult: func(
				t *testing.T,
				response *pb.StartVideoPlaybackResponse,
				err error,
				sent chan *pb.ProcessImageRequest,
				closed chan struct{},
			) {
				// Assert
				require.NoError(t, err)
				require.NotNil(t, response)
				assert.Equal(t, "browser-session", response.SessionId)
				assert.Equal(t, "Video playback started", response.Message)

				select {
				case frameRequest := <-sent:
					assert.Equal(t, "browser-session", frameRequest.SessionId)
					assert.Equal(t, int32(1), frameRequest.Width)
					assert.Equal(t, int32(2), frameRequest.Height)
					assert.Equal(t, int32(3), frameRequest.Channels)
					assert.Equal(t, []pb.FilterType{pb.FilterType_FILTER_TYPE_GRAYSCALE}, frameRequest.Filters)
					assert.Equal(t, pb.AcceleratorType_ACCELERATOR_TYPE_CPU, frameRequest.Accelerator)
				case <-time.After(time.Second):
					t.Fatal("expected frame to be sent to peer")
				}

				select {
				case <-closed:
				case <-time.After(time.Second):
					t.Fatal("expected peer to close after playback completes")
				}
			},
		},
		{
			name: "Error_MissingVideoID",
			request: &pb.StartVideoPlaybackRequest{
				SessionId: "browser-session",
			},
			setup: func(
				_ *MockStreamVideoRepository,
				_ *MockStreamVideoPlayer,
				_ *MockStreamVideoPeer,
				_ chan *pb.ProcessImageRequest,
				_ chan struct{},
			) {
			},
			assertResult: func(
				t *testing.T,
				response *pb.StartVideoPlaybackResponse,
				err error,
				_ chan *pb.ProcessImageRequest,
				_ chan struct{},
			) {
				// Assert
				require.ErrorIs(t, err, ErrVideoPlaybackMissingVideoID)
				assert.Nil(t, response)
			},
		},
	}

	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			// Arrange
			repo := new(MockStreamVideoRepository)
			player := new(MockStreamVideoPlayer)
			peer := new(MockStreamVideoPeer)
			sent := make(chan *pb.ProcessImageRequest, 1)
			closed := make(chan struct{})
			tt.setup(repo, player, peer, sent, closed)

			sut := NewStreamVideoUseCase(
				context.Background(),
				repo,
				func(string) (StreamVideoPlayer, error) { return player, nil },
				func(string) (StreamVideoPeer, error) { return peer, nil },
			)

			// Act
			response, err := sut.Start(context.Background(), tt.request)

			// Assert
			tt.assertResult(t, response, err, sent, closed)
			repo.AssertExpectations(t)
			player.AssertExpectations(t)
			peer.AssertExpectations(t)
		})
	}
}

func TestStreamVideoUseCase_Stop(t *testing.T) {
	// Arrange
	repo := new(MockStreamVideoRepository)
	player := new(MockStreamVideoPlayer)
	peer := new(MockStreamVideoPeer)
	started := make(chan struct{})
	closed := make(chan struct{})

	repo.On("GetByID", mock.Anything, "sample").Return(&domain.Video{
		ID:   "sample",
		Path: "/tmp/sample.mp4",
	}, nil)
	peer.On("Connect", mock.Anything).Return(nil).Once()
	peer.On("Close").Run(func(mock.Arguments) {
		close(closed)
	}).Return(nil).Once()
	player.On("Play", mock.Anything, mock.Anything).Run(func(args mock.Arguments) {
		ctx := args.Get(0).(context.Context)
		close(started)
		<-ctx.Done()
	}).Return(context.Canceled).Once()

	sut := NewStreamVideoUseCase(
		context.Background(),
		repo,
		func(string) (StreamVideoPlayer, error) { return player, nil },
		func(string) (StreamVideoPeer, error) { return peer, nil },
	)

	// Act
	startResponse, startErr := sut.Start(context.Background(), &pb.StartVideoPlaybackRequest{
		VideoId:    "sample",
		SessionId:  "browser-session",
		ApiVersion: "v1",
	})
	require.NoError(t, startErr)
	require.NotNil(t, startResponse)

	select {
	case <-started:
	case <-time.After(time.Second):
		t.Fatal("expected playback goroutine to start")
	}

	stopResponse, stopErr := sut.Stop(context.Background(), &pb.StopVideoPlaybackRequest{
		SessionId:  "browser-session",
		ApiVersion: "v1",
	})

	// Assert
	require.NoError(t, stopErr)
	require.NotNil(t, stopResponse)
	assert.True(t, stopResponse.Stopped)
	assert.Equal(t, "browser-session", stopResponse.SessionId)

	select {
	case <-closed:
	case <-time.After(time.Second):
		t.Fatal("expected peer to close after stop")
	}

	repo.AssertExpectations(t)
	player.AssertExpectations(t)
	peer.AssertExpectations(t)
}

func TestStreamVideoUseCase_Start_ErrorWhenRepositoryFails(t *testing.T) {
	// Arrange
	repo := new(MockStreamVideoRepository)
	player := new(MockStreamVideoPlayer)
	peer := new(MockStreamVideoPeer)
	repo.On("GetByID", mock.Anything, "missing").Return(nil, errors.New("not found")).Once()

	sut := NewStreamVideoUseCase(
		context.Background(),
		repo,
		func(string) (StreamVideoPlayer, error) { return player, nil },
		func(string) (StreamVideoPeer, error) { return peer, nil },
	)

	// Act
	response, err := sut.Start(context.Background(), &pb.StartVideoPlaybackRequest{
		VideoId:   "missing",
		SessionId: "browser-session",
	})

	// Assert
	require.Error(t, err)
	assert.Nil(t, response)
	assert.ErrorContains(t, err, "get video by id")
	repo.AssertExpectations(t)
	player.AssertExpectations(t)
	peer.AssertExpectations(t)
}
