package webrtc

import (
	"context"
	"errors"
	"fmt"
	"io"
	"sync"

	pb "github.com/jrb/cuda-learning/proto/gen"
	pion "github.com/pion/webrtc/v4"
)

const GoVideoDataChannelLabelPrefix = "go-video-"

type SignalingClient interface {
	SignalingStream(ctx context.Context) (pb.WebRTCSignalingService_SignalingStreamClient, error)
}

type GoPeer struct {
	signalingClient SignalingClient
	browserSession  string
	sessionID       string
	label           string

	peerConnection *pion.PeerConnection
	dataChannel    *pion.DataChannel
	signaling      pb.WebRTCSignalingService_SignalingStreamClient
	closeOnce      sync.Once
}

func NewGoPeer(browserSession string) *GoPeer {
	sessionID := GoVideoDataChannelLabelPrefix + browserSession

	return &GoPeer{
		browserSession: browserSession,
		sessionID:      sessionID,
		label:          sessionID,
	}
}

func (p *GoPeer) Connect(ctx context.Context) error {
	if p.browserSession == "" {
		return errors.New("browser session is required")
	}

	signaling, err := p.signalingClient.SignalingStream(ctx)
	if err != nil {
		return fmt.Errorf("create signaling stream: %w", err)
	}

	peerConnection, err := pion.NewPeerConnection(pion.Configuration{})
	if err != nil {
		_ = signaling.CloseSend()
		return fmt.Errorf("create peer connection: %w", err)
	}

	dataChannel, err := peerConnection.CreateDataChannel(p.label, nil)
	if err != nil {
		_ = peerConnection.Close()
		_ = signaling.CloseSend()
		return fmt.Errorf("create data channel: %w", err)
	}

	p.peerConnection = peerConnection
	p.dataChannel = dataChannel
	p.signaling = signaling

	openCh := make(chan struct{})
	answerCh := make(chan struct{})
	errCh := make(chan error, 1)
	var openOnce sync.Once

	dataChannel.OnOpen(func() {
		openOnce.Do(func() {
			close(openCh)
		})
	})

	peerConnection.OnConnectionStateChange(func(state pion.PeerConnectionState) {
		switch state {
		case pion.PeerConnectionStateFailed, pion.PeerConnectionStateClosed:
			select {
			case errCh <- fmt.Errorf("peer connection entered state %s", state.String()):
			default:
			}
		}
	})

	peerConnection.OnICECandidate(func(candidate *pion.ICECandidate) {
		if candidate == nil || p.signaling == nil {
			return
		}

		init := candidate.ToJSON()
		sdpMid := ""
		if init.SDPMid != nil {
			sdpMid = *init.SDPMid
		}
		sdpMLineIndex := int32(0)
		if init.SDPMLineIndex != nil {
			sdpMLineIndex = int32(*init.SDPMLineIndex)
		}

		if sendErr := p.signaling.Send(&pb.SignalingMessage{
			Message: &pb.SignalingMessage_IceCandidate{
				IceCandidate: &pb.SendIceCandidateRequest{
					SessionId: p.sessionID,
					Candidate: &pb.IceCandidate{
						Candidate:     init.Candidate,
						SdpMid:        sdpMid,
						SdpMlineIndex: sdpMLineIndex,
					},
				},
			},
		}); sendErr != nil {
			select {
			case errCh <- fmt.Errorf("send local ICE candidate: %w", sendErr):
			default:
			}
		}
	})

	go p.readSignaling(answerCh, errCh)

	offer, err := peerConnection.CreateOffer(nil)
	if err != nil {
		return fmt.Errorf("create offer: %w", err)
	}

	gatherComplete := pion.GatheringCompletePromise(peerConnection)
	if err := peerConnection.SetLocalDescription(offer); err != nil {
		return fmt.Errorf("set local description: %w", err)
	}
	<-gatherComplete

	if err := signaling.Send(&pb.SignalingMessage{
		Message: &pb.SignalingMessage_StartSession{
			StartSession: &pb.StartSessionRequest{
				SessionId: p.sessionID,
				SdpOffer:  peerConnection.LocalDescription().SDP,
			},
		},
	}); err != nil {
		return fmt.Errorf("send start session offer: %w", err)
	}

	select {
	case <-answerCh:
	case err := <-errCh:
		return err
	case <-ctx.Done():
		return ctx.Err()
	}

	select {
	case <-openCh:
		return nil
	case err := <-errCh:
		return err
	case <-ctx.Done():
		return ctx.Err()
	}
}

func (p *GoPeer) Send(payload []byte) error {
	if p.dataChannel == nil {
		return errors.New("data channel is not initialized")
	}
	if p.dataChannel.ReadyState() != pion.DataChannelStateOpen {
		return fmt.Errorf("data channel is not open: %s", p.dataChannel.ReadyState().String())
	}

	return p.dataChannel.Send(payload)
}

func (p *GoPeer) DataChannel() *pion.DataChannel {
	return p.dataChannel
}

func (p *GoPeer) Label() string {
	return p.label
}

func (p *GoPeer) Close() error {
	var closeErr error

	p.closeOnce.Do(func() {
		var errs []error

		if p.signaling != nil {
			if err := p.signaling.Send(&pb.SignalingMessage{
				Message: &pb.SignalingMessage_CloseSession{
					CloseSession: &pb.CloseSessionRequest{
						SessionId: p.sessionID,
					},
				},
			}); err != nil && !errors.Is(err, io.EOF) {
				errs = append(errs, fmt.Errorf("send close session: %w", err))
			}
		}

		if p.dataChannel != nil {
			if err := p.dataChannel.Close(); err != nil {
				errs = append(errs, fmt.Errorf("close data channel: %w", err))
			}
		}

		if p.peerConnection != nil {
			if err := p.peerConnection.Close(); err != nil {
				errs = append(errs, fmt.Errorf("close peer connection: %w", err))
			}
		}

		if p.signaling != nil {
			if err := p.signaling.CloseSend(); err != nil {
				errs = append(errs, fmt.Errorf("close signaling stream: %w", err))
			}
		}

		closeErr = errors.Join(errs...)
	})

	return closeErr
}

func (p *GoPeer) readSignaling(answerCh chan struct{}, errCh chan error) {
	var answerOnce sync.Once

	for {
		msg, err := p.signaling.Recv()
		if err != nil {
			if errors.Is(err, io.EOF) {
				select {
				case errCh <- errors.New("signaling stream closed"):
				default:
				}
				return
			}

			select {
			case errCh <- fmt.Errorf("receive signaling message: %w", err):
			default:
			}
			return
		}

		switch {
		case msg.GetStartSessionResponse() != nil:
			response := msg.GetStartSessionResponse()
			if response.GetSessionId() != p.sessionID {
				continue
			}

			if err := p.peerConnection.SetRemoteDescription(pion.SessionDescription{
				Type: pion.SDPTypeAnswer,
				SDP:  response.GetSdpAnswer(),
			}); err != nil {
				select {
				case errCh <- fmt.Errorf("set remote description: %w", err):
				default:
				}
				return
			}

			answerOnce.Do(func() {
				close(answerCh)
			})
		case msg.GetIceCandidate() != nil:
			candidate := msg.GetIceCandidate()
			if candidate.GetSessionId() != p.sessionID {
				continue
			}

			iceInit := pion.ICECandidateInit{
				Candidate: candidate.GetCandidate().GetCandidate(),
			}
			if sdpMid := candidate.GetCandidate().GetSdpMid(); sdpMid != "" {
				iceInit.SDPMid = &sdpMid
			}
			if lineIndex := candidate.GetCandidate().GetSdpMlineIndex(); lineIndex >= 0 {
				mLineIndex := uint16(lineIndex)
				iceInit.SDPMLineIndex = &mLineIndex
			}

			if err := p.peerConnection.AddICECandidate(iceInit); err != nil {
				select {
				case errCh <- fmt.Errorf("add remote ICE candidate: %w", err):
				default:
				}
				return
			}
		case msg.GetCloseSession() != nil:
			closeRequest := msg.GetCloseSession()
			if closeRequest.GetSessionId() != p.sessionID {
				continue
			}

			select {
			case errCh <- errors.New("remote peer closed signaling session"):
			default:
			}
			return
		}
	}
}
